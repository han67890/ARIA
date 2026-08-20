from types import SimpleNamespace

import pytest
import torch

from openrlhf.cli.evaluate_aria import ARIAEvaluator
from openrlhf.models.modeling_aria import (
    ARIA_NO_COMPRESSION_CONTEXT_CEILING,
    ARIA_NO_COMPRESSION_MAX_NEW_TOKENS,
    CLaRa,
    QCAResult,
    QuestionType,
    RAGDiagnostics,
    _ScoredDoc,
    create_paper_rag_config,
)


class _DirectTokenizer:
    model_max_length = ARIA_NO_COMPRESSION_CONTEXT_CEILING
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = []
        self.decoded_ids = None

    @staticmethod
    def encode(text, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        return list(range(1, len(text.split()) + 1))

    def __call__(
        self,
        texts,
        *,
        return_tensors,
        padding,
        add_special_tokens,
        truncation,
    ):
        assert return_tensors == "pt"
        assert padding == "longest"
        assert add_special_tokens is False
        assert truncation is False
        self.calls.append(list(texts))
        rows = [self.encode(text) for text in texts]
        width = max(len(row) for row in rows)
        ids = []
        attention = []
        for row in rows:
            padding_width = width - len(row)
            ids.append([self.pad_token_id] * padding_width + row)
            attention.append([0] * padding_width + [1] * len(row))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        self.decoded_ids = ids.detach().clone()
        return ["answer" for _ in range(ids.size(0))]


class _DirectDecoder(torch.nn.Module):
    def __init__(self, context_limit=256):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=context_limit)
        self.adapter = None
        self.generate_kwargs = None

    @property
    def device(self):
        return self.anchor.device

    def set_adapter(self, adapter):
        self.adapter = adapter

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        suffix = torch.tensor(
            [[91, 2] for _ in range(kwargs["input_ids"].size(0))],
            device=kwargs["input_ids"].device,
            dtype=torch.long,
        )
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def _direct_model(context_limit=256):
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model.decoder = _DirectDecoder(context_limit=context_limit)
    model.decoder_tokenizer = _DirectTokenizer()
    model.adapter_keys = ["query_reasoner_adapter", "decoder_adapter"]
    model.generation_top_k = 5
    return model


def _documents(count=2):
    return [
        _ScoredDoc(
            doc_id=f"doc-{index}",
            text=f"raw passage-{index}",
            corpus_index=index,
            fused_score=float(count - index),
        )
        for index in range(count)
    ]


def test_no_compression_generator_preserves_raw_order_and_uses_token_ids():
    model = _direct_model(context_limit=128)
    captured = []

    def blend(context, question, answer):
        assert answer is None
        captured.append((context, question))
        return f"system background {context} question {question}"

    model._blend_standard_prompt = blend
    decoded, prompt_lengths, document_tokens, context_limit = (
        model._generate_no_compression_context(
            ["which answer"],
            [_documents(2)],
            ARIA_NO_COMPRESSION_MAX_NEW_TOKENS,
        )
    )

    assert decoded == ["answer"]
    assert captured == [("raw passage-0\n\nraw passage-1", "which answer")]
    assert document_tokens.tolist() == [4]
    assert prompt_lengths.tolist() == [9]
    assert context_limit == 128
    assert model.decoder.adapter == "decoder_adapter"
    assert "input_ids" in model.decoder.generate_kwargs
    assert "inputs_embeds" not in model.decoder.generate_kwargs
    assert model.decoder.generate_kwargs["do_sample"] is False
    assert model.decoder.generate_kwargs["num_beams"] == 1
    assert model.decoder.generate_kwargs["max_new_tokens"] == 64
    # Only newly generated IDs are decoded; prompt IDs never leak into answers.
    assert model.decoder_tokenizer.decoded_ids.tolist() == [[91, 2]]


def test_no_compression_generator_refuses_to_truncate_overlength_context():
    model = _direct_model(context_limit=65)
    model._blend_standard_prompt = lambda *_args: "two prompt tokens"

    with pytest.raises(ValueError, match="no-truncation decoder ceiling"):
        model._generate_no_compression_context(
            ["question"],
            [_documents(1)],
            ARIA_NO_COMPRESSION_MAX_NEW_TOKENS,
        )

    assert model.decoder.generate_kwargs is None


def test_no_compression_full_path_returns_first_pass_and_skips_compressor():
    model = _direct_model(context_limit=256)
    model.query_max_length = 16
    model._rag_config = create_paper_rag_config("forward_path_off", 16)
    model._bge_projection = object()
    model._rag_diagnostics = []
    model._oracle_pool_records = []
    qca = QCAResult(
        question="question",
        question_type=QuestionType.SIMPLE,
        confidence=0.0,
        hop_count=1,
        entity_count=0,
    )
    evidence = _documents(5)
    diagnostic = RAGDiagnostics(initial_candidates=4000)

    class _Pipeline:
        @staticmethod
        def retrieve_initial_batch(questions, query_bge):
            assert len(questions) == query_bge.size(0) == 1
            return [["pool"]], [qca]

        @staticmethod
        def retrieve_scored(*_args, **_kwargs):
            return evidence, qca, diagnostic

    model.rag_pipeline = _Pipeline()
    model._prepare_query_inputs = lambda questions, max_length: {
        "input_ids": torch.ones(len(questions), 1, dtype=torch.long),
        "attention_mask": torch.ones(len(questions), 1, dtype=torch.long),
    }
    model._compr_query_reasoner_stage2 = lambda ids, mask: torch.ones(
        ids.size(0), 2
    )
    model._project_query_reps_to_bge = lambda values: values
    model._encode_subquery_for_retrieval = lambda _text: torch.ones(2)
    model._blend_standard_prompt = (
        lambda context, question, answer: f"background {context} question {question}"
    )
    model._compress_evidence = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("ARIA-NoComp must not call the compressor")
    )

    decoded, indices = model.generate_from_questions(
        ["question"],
        max_new_tokens=64,
        no_compression=True,
    )

    assert decoded == ["answer"]
    assert indices.tolist() == [[0, 1, 2, 3, 4]]
    assert diagnostic.final_candidates == 5
    assert diagnostic.second_round_candidates == 0
    assert diagnostic.evidence_memory_tokens == 0
    assert diagnostic.direct_context_document_tokens == 10
    assert diagnostic.direct_context_prompt_tokens > 0
    assert diagnostic.direct_context_ceiling == 256


def test_no_compression_evaluator_records_first_pass_indices_and_protocol():
    diagnostic = RAGDiagnostics(
        final_candidates=2,
        second_round_candidates=0,
        direct_context_document_tokens=1180,
        direct_context_prompt_tokens=1197,
        direct_context_ceiling=2048,
    )

    class _EvaluationModel:
        generation_top_k = 5

        @staticmethod
        def clear_rag_diagnostics():
            return None

        @staticmethod
        def get_rag_diagnostics():
            return [diagnostic]

        @staticmethod
        def generate_from_questions(**kwargs):
            assert kwargs["no_compression"] is True
            assert kwargs["return_first_pass_indices"] is True
            return ["answer"], torch.tensor([[4, 2, -1, -1, -1]])

    evaluator = ARIAEvaluator.__new__(ARIAEvaluator)
    evaluator.model = _EvaluationModel()
    evaluator.use_rag_pipeline = True
    evaluator.retrieval_mode = "normal"
    evaluator.no_compression = True
    evaluator.no_compression_context_limit = 2048
    evaluator.corpus_ids = [f"doc-{index}" for index in range(6)]
    evaluator.corpus_page_ids = [f"page-{index}" for index in range(6)]
    evaluator._has_explicit_page_ids = True
    evaluator._corpus_id_to_index = {
        document_id: index
        for index, document_id in enumerate(evaluator.corpus_ids)
    }

    result = evaluator.evaluate(
        questions=["question"],
        gold_answers=[["answer"]],
        example_ids=["example-0"],
        batch_size=1,
        max_new_tokens=64,
    )

    prediction = result["predictions"][0]
    assert prediction["first_pass_corpus_indices"] == [4, 2]
    assert prediction["retrieval_diagnostics"] == {
        "final_document_count": 2,
        "second_round_candidate_count": 0,
        "direct_context_document_tokens": 1180,
        "direct_context_prompt_tokens": 1197,
        "direct_context_ceiling": 2048,
    }
    assert result["mean_direct_context_document_tokens"] == pytest.approx(1180.0)
    assert result["mean_direct_context_prompt_tokens"] == pytest.approx(1197.0)
    assert result["no_compression_protocol"]["checkpoint_training_configuration"] == (
        "full"
    )
    assert result["no_compression_protocol"]["retrieval_mode"] == "normal"
    assert result["no_compression_protocol"]["passage_truncation"] is False
