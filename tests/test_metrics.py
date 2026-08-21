import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

import openrlhf.cli.evaluate_aria as evaluate_module
from openrlhf.cli.evaluate_aria import (
    ARIAEvaluator,
    QAMetrics,
    _extract_gold_document_ids,
    _extract_gold_answer,
    _map_clara_candidates_to_corpus,
    _load_oracle_qca_labels,
    _parse_qca_llm_output,
    _qca_llm_prompt,
    _qca_weighted_f1,
    _run_qca_llm_router,
    _merge_repository_candidates,
    _oracle_qca_labeled_subset,
    _repository_candidate_identity,
    _repository_top5_documents,
    _load_repository_eval_rows,
    _validate_clara_archive_dir,
    _validate_oracle_qca_conditions,
    _validate_oracle_qca_paper_panel,
    _validate_paper_answer_contract,
    _validate_qca_llm_conditions,
    aggregate_checkpoint_results,
    compare_evaluation_payloads,
    load_eval_dataset,
)
from openrlhf.cli.evaluate_aria import (
    PAPER_QCA_LLM_MODEL,
    QCA_LLM_MAX_NEW_TOKENS,
    QCA_LLM_PROMPT_SHA256,
    QCA_LLM_PROMPT_VERSION,
    QCA_LLM_PROTOCOL,
)
from openrlhf.models.modeling_aria import RAGPipelineConfig
from openrlhf.models.modeling_aria import (
    ORACLE_TOP100_PROTOCOL,
    _construct_oracle_top100_indices,
)


def test_appendix_a35_normalization_keeps_apostrophe_and_hyphen():
    assert QAMetrics.normalize_answer(" The O'Neill-style, Answer! ") == "o'neill-style answer"


def test_qca_llm_prompt_parser_and_weighted_f1_contract():
    prompt = _qca_llm_prompt("Which city is older?")
    assert "simple" in prompt and "multi-aspect" in prompt and "multi-hop" in prompt
    assert "Question: Which city is older?" in prompt
    parsed, rationale = _parse_qca_llm_output(
        "Label: multi-hop\nRationale: Two facts must be connected."
    )
    assert parsed == "multi_hop"
    assert rationale == "Two facts must be connected."
    with pytest.raises(ValueError, match="first line"):
        _parse_qca_llm_output("The label is simple.\nRationale: direct")
    with pytest.raises(ValueError, match="second line"):
        _parse_qca_llm_output("Label: simple")
    assert _qca_weighted_f1(
        ["simple", "multi_aspect", "multi_hop"],
        ["simple", "multi_aspect", "multi_hop"],
    ) == pytest.approx(1.0)


def test_qca_llm_endpoint_is_fail_closed_to_paper_setting():
    _validate_qca_llm_conditions(
        retrieval_mode="normal",
        rag_configuration="full",
        compression_rate=16,
        dataset="all",
        max_samples=None,
    )
    with pytest.raises(ValueError, match="dataset all"):
        _validate_qca_llm_conditions(
            retrieval_mode="normal",
            rag_configuration="full",
            compression_rate=16,
            dataset="nq",
            max_samples=None,
        )


class _QCALLMTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(
        self,
        prompts,
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
        width = max(len(prompt.split()) for prompt in prompts)
        return {
            "input_ids": torch.ones((len(prompts), width), dtype=torch.long),
            "attention_mask": torch.ones((len(prompts), width), dtype=torch.long),
        }

    @staticmethod
    def batch_decode(ids, skip_special_tokens):
        assert skip_special_tokens is True
        return [
            "Label: simple\nRationale: One direct fact is requested."
            for _ in range(ids.size(0))
        ]


class _QCALLMDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=32_768)
        self.adapters_disabled = False
        self.generate_kwargs = None

    class _DisableContext:
        def __init__(self, decoder):
            self.decoder = decoder

        def __enter__(self):
            self.decoder.adapters_disabled = True

        def __exit__(self, exc_type, exc, traceback):
            self.decoder.adapters_disabled = False

    def disable_adapter(self):
        return self._DisableContext(self)

    def generate(self, **kwargs):
        assert self.adapters_disabled is True
        self.generate_kwargs = kwargs
        suffix = torch.ones((kwargs["input_ids"].size(0), 2), dtype=torch.long)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def test_qca_llm_router_uses_adapter_free_greedy_decoding_and_records_prompt():
    decoder = _QCALLMDecoder()
    model = SimpleNamespace(
        decoder_model_name=PAPER_QCA_LLM_MODEL,
        config=SimpleNamespace(decoder_model_resolved_revision="a" * 40),
        decoder=decoder,
        decoder_tokenizer=_QCALLMTokenizer(),
    )
    records = _run_qca_llm_router(model, ["Who wrote it?"], batch_size=1)
    assert records[0]["protocol"] == QCA_LLM_PROTOCOL
    assert records[0]["prompt_version"] == QCA_LLM_PROMPT_VERSION
    assert records[0]["prompt_template_sha256"] == QCA_LLM_PROMPT_SHA256
    assert records[0]["parsed_type"] == "simple"
    assert records[0]["adapters_disabled"] is True
    assert decoder.adapters_disabled is False
    assert decoder.generate_kwargs["do_sample"] is False
    assert decoder.generate_kwargs["num_beams"] == 1
    assert decoder.generate_kwargs["max_new_tokens"] == QCA_LLM_MAX_NEW_TOKENS


def test_oracle_qca_label_artifacts_are_keyed_and_subset_by_example_id(tmp_path):
    json_path = tmp_path / "labels.json"
    json_path.write_text(
        json.dumps({"q-2": "multi_hop", "q-4": "multi_aspect"}),
        encoding="utf-8",
    )
    labels = _load_oracle_qca_labels(str(json_path))
    indices, reference_types = _oracle_qca_labeled_subset(
        ["q-1", "q-2", "q-3", "q-4"],
        labels,
        dataset_name="nq",
    )

    assert indices == [1, 3]
    assert reference_types == ["multi_hop", "multi_aspect"]

    jsonl_path = tmp_path / "labels.jsonl"
    jsonl_path.write_text(
        '{"example_id":"q-1","question_type":"simple"}\n'
        '{"example_id":"q-3","question_type":"multi_hop"}\n',
        encoding="utf-8",
    )
    assert _load_oracle_qca_labels(str(jsonl_path)) == {
        "q-1": "simple",
        "q-3": "multi_hop",
    }


@pytest.mark.parametrize(
    "retrieval_mode,rag_configuration,compression_rate,match",
    [
        ("oracle", "full", 16, "retrieval_mode normal"),
        ("normal", "remove_qca", 16, "rag_configuration full"),
        ("normal", "full", 32, "compression_rate 16"),
    ],
)
def test_oracle_qca_conditions_fail_closed(
    retrieval_mode, rag_configuration, compression_rate, match
):
    with pytest.raises(ValueError, match=match):
        _validate_oracle_qca_conditions(
            retrieval_mode=retrieval_mode,
            rag_configuration=rag_configuration,
            compression_rate=compression_rate,
        )
    with pytest.raises(ValueError, match="cannot be combined with --max_samples"):
        _validate_oracle_qca_conditions(
            retrieval_mode="normal",
            rag_configuration="full",
            compression_rate=16,
            max_samples=1000,
        )
    with pytest.raises(ValueError, match="requires --dataset all"):
        _validate_oracle_qca_conditions(
            retrieval_mode="normal",
            rag_configuration="full",
            compression_rate=16,
            dataset="nq",
        )


def test_oracle_qca_cli_panel_requires_the_paper_benchmark_counts():
    _validate_oracle_qca_paper_panel("musique", [f"q-{index}" for index in range(84)])
    with pytest.raises(ValueError, match="exactly 84 labeled musique"):
        _validate_oracle_qca_paper_panel(
            "musique", [f"q-{index}" for index in range(83)]
        )


def test_multi_gold_uses_best_match():
    gold = ["wrong", "Ada Lovelace"]
    assert QAMetrics.exact_match("Ada Lovelace", gold)
    assert QAMetrics.contains_exact_match("The answer is Ada Lovelace", gold)
    assert QAMetrics.f1_score("Ada Lovelace", gold) == 1.0


def test_paper_answer_contract_accepts_one_scalar_reference():
    rows = [{"question": "Who?", "answer": "Ada Lovelace"}]
    _validate_paper_answer_contract(rows, dataset_name="nq", answer_key="answer")
    assert _extract_gold_answer(rows[0], "answer") == "Ada Lovelace"


def test_paper_dataset_loader_requires_prepared_artifact():
    with pytest.raises(ValueError, match="require --eval_data_path"):
        load_eval_dataset("nq")


def test_paper_answer_contract_rejects_multi_answer_containers():
    with pytest.raises(ValueError, match="non-empty scalar"):
        _validate_paper_answer_contract(
            [{"question": "Who?", "answer": ["Ada", "Lovelace"]}],
            dataset_name="nq",
            answer_key="answer",
        )


def test_clara_candidates_merge_only_on_exact_question_alignment():
    prepared = [
        {
            "question": "Who?",
            "answer": "Ada Lovelace",
        }
    ]
    archived = [
        {
            "question": "Who?",
            "docs": ["candidate"],
        }
    ]
    merged = _merge_repository_candidates(
        prepared,
        archived,
        dataset_name="nq",
        question_key="question",
    )
    assert merged[0]["answer"] == "Ada Lovelace"
    assert merged[0]["docs"] == ["candidate"]

    archived[0]["question"] = "A different question"
    with pytest.raises(ValueError, match="does not exactly align"):
        _merge_repository_candidates(
            prepared,
            archived,
            dataset_name="nq",
            question_key="question",
        )


def test_repository_candidate_archive_is_fingerprinted_before_top5_selection():
    archived = [f"document {index}" for index in range(20)]
    validated, selected = _repository_top5_documents(archived, location="fixture.docs")
    assert validated == archived
    assert selected == archived[:5]
    assert validated is not archived


def test_repository_candidate_archive_rejects_already_truncated_rows():
    with pytest.raises(ValueError, match="exactly 20"):
        _repository_top5_documents(
            [f"document {index}" for index in range(5)],
            location="fixture.docs",
        )


def test_repository_candidate_identity_is_stable_at_document_and_page_scope():
    first = "This is a document about Ada Lovelace\nfirst passage"
    second = "This is a document about  Ada   Lovelace \nsecond passage"
    first_doc, first_page = _repository_candidate_identity(first)
    second_doc, second_page = _repository_candidate_identity(second)
    assert first_doc != second_doc
    assert first_page == second_page


def _write_external_clara_archives(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[int]]:
    documents = [
        f"This is a document about Page {index}\npassage {index}"
        for index in range(20)
    ]
    positive_indices = [0, 3]
    record = {
        "id": "fixture-0",
        "question": "Who?",
        "answer": "Ada",
        "docs": documents,
        "pos_index": positive_indices,
    }
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    archive_sha256 = {}
    for benchmark, (archive_name, member_name) in (
        evaluate_module._REPOSITORY_EVAL_ARCHIVES.items()
    ):
        archive_path = root / archive_name
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(member_name, payload)
        archive_sha256[benchmark] = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    documents_digest = hashlib.sha256()
    documents_digest.update(
        json.dumps(documents, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    documents_digest.update(b"\n")
    positives_digest = hashlib.sha256()
    positives_digest.update(
        json.dumps(
            positive_indices,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    positives_digest.update(b"\n")
    monkeypatch.setattr(
        evaluate_module,
        "_REPOSITORY_EVAL_ARCHIVE_SHA256",
        archive_sha256,
    )
    monkeypatch.setattr(
        evaluate_module,
        "_REPOSITORY_BGE_CANDIDATES_SHA256",
        {name: documents_digest.hexdigest() for name in archive_sha256},
    )
    monkeypatch.setattr(
        evaluate_module,
        "_REPOSITORY_CLARA_POSITIVE_INDICES_SHA256",
        {name: positives_digest.hexdigest() for name in archive_sha256},
    )
    return documents, positive_indices


def test_external_clara_archive_set_requires_all_four_pinned_zips(
    tmp_path, monkeypatch
):
    _write_external_clara_archives(tmp_path, monkeypatch)
    missing_path = tmp_path / "2wiki.zip"
    missing_path.unlink()
    with pytest.raises(FileNotFoundError, match="2wiki.zip"):
        _validate_clara_archive_dir(str(tmp_path))


def test_external_clara_archive_set_rejects_byte_digest_mismatch(
    tmp_path, monkeypatch
):
    _write_external_clara_archives(tmp_path, monkeypatch)
    with (tmp_path / "nq.zip").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_clara_archive_dir(str(tmp_path))


def test_external_clara_archive_set_requires_the_pinned_member(
    tmp_path, monkeypatch
):
    _write_external_clara_archives(tmp_path, monkeypatch)
    archive_path = tmp_path / "musique.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("musique/wrong.jsonl", b"{}\n")
    monkeypatch.setitem(
        evaluate_module._REPOSITORY_EVAL_ARCHIVE_SHA256,
        "musique",
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="requires exactly one member"):
        _validate_clara_archive_dir(str(tmp_path))


def test_external_clara_archive_load_checks_members_and_content_fingerprints(
    tmp_path, monkeypatch
):
    documents, positive_indices = _write_external_clara_archives(tmp_path, monkeypatch)
    paths = _validate_clara_archive_dir(str(tmp_path))
    assert set(paths) == set(evaluate_module._REPOSITORY_EVAL_ARCHIVES)

    rows = _load_repository_eval_rows("nq", str(tmp_path))
    assert rows[0]["question"] == "Who?"
    assert rows[0]["docs"] == documents
    assert rows[0]["clara_gold_candidate_indices"] == positive_indices

    monkeypatch.setitem(
        evaluate_module._REPOSITORY_BGE_CANDIDATES_SHA256,
        "nq",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="BGE candidate lists"):
        _load_repository_eval_rows("nq", str(tmp_path))


class _ClaraRecallModel:
    generation_top_k = 5

    def generate_from_text(
        self,
        questions,
        documents,
        max_new_tokens,
        return_selected_indices=False,
    ):
        assert return_selected_indices is True
        indices = torch.tensor(
            [[2, 1, 5, 6, 7] for _ in questions], dtype=torch.long
        )
        return ["answer" for _ in questions], indices


def test_clara_recall_uses_full_corpus_gold_pages_and_q_sup():
    documents = [[f"document {index}" for index in range(20)]]
    doc_ids = [[f"doc-{index}" for index in range(20)]]
    page_ids = [[f"page-{index}" for index in range(20)]]
    evaluator = ARIAEvaluator(
        model=_ClaraRecallModel(),
        corpus_docs=documents[0],
        corpus_ids=doc_ids[0],
        corpus_page_ids=page_ids[0],
        use_rag_pipeline=False,
    )

    result = evaluator.evaluate(
        questions=["question"],
        gold_answers=["answer"],
        gold_doc_ids=[["doc-0", "doc-1"]],
        documents=documents,
        clara_candidate_doc_ids=doc_ids,
        clara_candidate_page_ids=page_ids,
        batch_size=1,
    )

    assert result["recall_at_5"] == 0.5
    assert result["recall_at_5_support_count"] == 1
    assert result["clara_retrieval_provenance"]["candidate_count"] == 20
    assert result["clara_retrieval_provenance"]["hard_selection_count"] == 5
    assert result["clara_retrieval_provenance"]["support_scope"] == (
        "prepared-full-corpus-Q_sup"
    )
    assert result["predictions"][0]["retrieved_doc_ids"] == [
        "doc-2",
        "doc-1",
        "doc-5",
        "doc-6",
        "doc-7",
    ]


def test_clara_archive_candidates_require_a_unique_exact_corpus_mapping():
    documents = [["This is a document about Ada\npassage"]]
    mapped_doc_ids, mapped_page_ids, mapped_indices = _map_clara_candidates_to_corpus(
        documents,
        corpus_docs=[documents[0][0]],
        corpus_ids=["doc-ada"],
        corpus_page_ids=["https://en.wikipedia.org/wiki/Ada"],
    )
    assert mapped_doc_ids == [["doc-ada"]]
    assert mapped_page_ids == [["https://en.wikipedia.org/wiki/Ada"]]
    assert mapped_indices == [[0]]

    with pytest.raises(ValueError, match="ambiguous"):
        _map_clara_candidates_to_corpus(
            documents,
            corpus_docs=[documents[0][0], documents[0][0]],
            corpus_ids=["doc-a", "doc-b"],
            corpus_page_ids=["page-a", "page-b"],
        )


def test_clara_oracle_selects_hard_top5_from_the_shared_top100_pool():
    corpus_docs = [f"document {index}" for index in range(105)]
    corpus_ids = [f"doc-{index}" for index in range(105)]
    corpus_pages = [f"page-{index}" for index in range(105)]
    pool_record = _construct_oracle_top100_indices(
        list(range(100)),
        [2, 101],
        corpus_page_ids=corpus_pages,
    )
    evaluator = ARIAEvaluator(
        model=_ClaraRecallModel(),
        corpus_docs=corpus_docs,
        corpus_ids=corpus_ids,
        corpus_page_ids=corpus_pages,
        doc_embeddings=torch.randn(105, 1024),
        use_rag_pipeline=False,
        retrieval_mode="oracle",
    )
    result = evaluator.evaluate(
        questions=["question"],
        gold_answers=["answer"],
        gold_doc_ids=[["doc-2", "doc-101"]],
        oracle_pool_records=[pool_record],
        batch_size=1,
    )

    prediction = result["predictions"][0]
    assert result["clara_retrieval_provenance"]["candidate_count"] == 100
    assert len(prediction["oracle_pool_doc_ids"]) == 100
    assert prediction["oracle_pool_sha256"] == pool_record.pool_sha256
    assert prediction["retrieved_doc_ids"] == [
        corpus_ids[pool_record.pool_indices[index]] for index in [2, 1, 5, 6, 7]
    ]
    assert prediction["recall_at_5"] == pytest.approx(0.5)


class _RecallModel:
    compr_rate = 16
    config = SimpleNamespace(aria_rag_configuration="full")

    def __init__(self, index_rows):
        self._bge_projection = torch.nn.Linear(3, 3, bias=False)
        self._mtfrl_projection = torch.nn.Linear(3, 3, bias=False)
        self._index_rows = torch.tensor(index_rows, dtype=torch.long)
        self._diagnostics = []
        self.returned_first_pass = None
        self.oracle_gold_indices = None
        self.qca_reference_types = None
        self._oracle_records = []

    def setup_rag_pipeline(self, **kwargs):
        self.setup_arguments = kwargs

    def clear_rag_diagnostics(self):
        self._diagnostics = []
        self._oracle_records = []

    def get_rag_diagnostics(self):
        return self._diagnostics

    def get_oracle_pool_records(self):
        return self._oracle_records

    def generate_from_questions(
        self,
        questions,
        documents,
        max_new_tokens,
        return_first_pass_indices,
        oracle_gold_indices=None,
        oracle_pool_records=None,
        qca_reference_types=None,
    ):
        del documents, max_new_tokens
        self.returned_first_pass = return_first_pass_indices
        self.oracle_gold_indices = oracle_gold_indices
        self.qca_reference_types = qca_reference_types
        if qca_reference_types is None:
            self._diagnostics = [object() for _ in questions]
        else:
            self._diagnostics = [
                SimpleNamespace(
                    rule_question_type="simple",
                    oracle_question_type=question_type,
                    evidence_memory_tokens=1,
                    final_candidates=5,
                    second_round_candidates=200,
                )
                for question_type in qca_reference_types
            ]
        if oracle_pool_records is not None:
            self._oracle_records.extend(oracle_pool_records)
        elif oracle_gold_indices is not None:
            for row in oracle_gold_indices:
                injected = tuple(index for index in row if index >= 100)
                retained = tuple(range(100 - len(injected)))
                self._oracle_records.append(
                    SimpleNamespace(
                        protocol=ORACLE_TOP100_PROTOCOL,
                        pool_sha256="a" * 64,
                        pool_indices=retained + injected,
                        injected_indices=injected,
                        evicted_indices=tuple(
                            range(100 - len(injected), 100)
                        ),
                    )
                )
        return ["answer" for _ in questions], self._index_rows[: len(questions)]


def _recall_evaluator(index_rows, page_ids=None):
    corpus_ids = [f"doc-{index}" for index in range(8)]
    page_ids = page_ids or [f"page-{index}" for index in range(8)]
    model = _RecallModel(index_rows)
    evaluator = ARIAEvaluator(
        model=model,
        corpus_docs=[f"document {index}" for index in range(8)],
        corpus_ids=corpus_ids,
        corpus_page_ids=page_ids,
        doc_embeddings=torch.randn(8, 3),
        rag_config=RAGPipelineConfig(use_mtfrl=False),
    )
    return evaluator, model


def test_evaluator_uses_the_shared_bge_document_index_for_mads():
    _, model = _recall_evaluator([[1, 3, 5]])

    assert set(model.setup_arguments) == {
        "corpus_docs",
        "corpus_doc_ids",
        "corpus_page_ids",
        "doc_embeddings",
        "rag_config",
        "bm25_index",
    }


def test_evaluator_applies_and_records_oracle_qca_label_only_override():
    model = _RecallModel([[0, 1, 2, 3, 4]])
    evaluator = ARIAEvaluator(
        model=model,
        corpus_docs=[f"document {index}" for index in range(8)],
        corpus_ids=[f"doc-{index}" for index in range(8)],
        corpus_page_ids=[f"page-{index}" for index in range(8)],
        doc_embeddings=torch.randn(8, 3),
        rag_config=RAGPipelineConfig(compression_rate=16),
    )

    result = evaluator.evaluate(
        questions=["question"],
        gold_answers=["answer"],
        example_ids=["q-1"],
        qca_reference_types=["multi_hop"],
        batch_size=1,
    )

    assert model.qca_reference_types == ["multi_hop"]
    prediction = result["predictions"][0]
    assert prediction["qca_rule_type"] == "simple"
    assert prediction["qca_oracle_type"] == "multi_hop"
    assert result["oracle_qca_provenance"]["labeled_example_count"] == 1


def test_recall_at_5_maps_first_pass_indices_to_stable_corpus_ids():
    evaluator, model = _recall_evaluator(
        [[1, 3, 5, 6, 7], [0, 2, 4, 6, 7]]
    )
    result = evaluator.evaluate(
        questions=["q1", "q2"],
        gold_answers=["answer", "answer"],
        gold_doc_ids=[["doc-1", "doc-2"], ["doc-0", "doc-2", "doc-3"]],
        batch_size=2,
    )

    assert model.returned_first_pass is True
    assert result["predictions"][0]["retrieved_doc_ids"] == [
        "doc-1",
        "doc-3",
        "doc-5",
        "doc-6",
        "doc-7",
    ]
    assert result["predictions"][0]["recall_at_5"] == pytest.approx(0.5)
    assert result["predictions"][1]["recall_at_5"] == pytest.approx(2 / 3)
    assert result["recall_at_5"] == pytest.approx((0.5 + 2 / 3) / 2)

    aggregated = aggregate_checkpoint_results(
        [result, result], [42, 123], ["checkpoint-42", "checkpoint-123"]
    )
    assert aggregated["mean"]["recall_at_5"] == pytest.approx(result["recall_at_5"])
    assert aggregated["std"]["recall_at_5"] == 0.0


def test_recall_at_5_rejects_ccef_survivor_sets_smaller_than_five():
    evaluator, _ = _recall_evaluator([[1, 3, 5]])
    with pytest.raises(RuntimeError, match="exactly five survivors"):
        evaluator.evaluate(
            questions=["q"],
            gold_answers=["answer"],
            gold_doc_ids=[["doc-1", "doc-2"]],
            batch_size=1,
        )


def test_recall_at_5_rejects_padded_variable_survivors():
    evaluator, _ = _recall_evaluator([[1, 3, 5, -1, -1]])
    with pytest.raises(RuntimeError, match="exactly five survivors"):
        evaluator.evaluate(
            questions=["q"],
            gold_answers=["answer"],
            gold_doc_ids=[["doc-1"]],
            batch_size=1,
        )


def test_recall_at_5_deduplicates_pages_and_uses_gold_page_intersection():
    page_ids = ["p0", "shared", "p2", "shared", "p4", "p5", "p6", "p7"]
    evaluator, _ = _recall_evaluator([[1, 3, 2, 4, 5]], page_ids=page_ids)
    result = evaluator.evaluate(
        questions=["q"],
        gold_answers=["answer"],
        gold_doc_ids=[["doc-3", "doc-6"]],
        batch_size=1,
    )

    prediction = result["predictions"][0]
    assert prediction["retrieved_doc_ids"] == ["doc-1", "doc-2", "doc-4", "doc-5"]
    assert prediction["retrieved_page_ids"] == ["shared", "p2", "p4", "p5"]
    assert prediction["gold_page_ids"] == ["shared", "p6"]
    assert prediction["recall_at_5"] == pytest.approx(0.5)


def test_normal_recall_averages_only_rows_with_annotated_support_pages():
    evaluator, _ = _recall_evaluator(
        [[1, 3, 5, 6, 7], [0, 2, 4, 6, 7]]
    )
    result = evaluator.evaluate(
        questions=["unsupported", "supported"],
        gold_answers=["answer", "answer"],
        gold_doc_ids=[[], ["doc-2"]],
        batch_size=2,
    )

    assert result["recall_at_5_support_count"] == 1
    assert result["recall_at_5"] == pytest.approx(1.0)
    assert result["predictions"][0]["has_gold_support"] is False
    assert "recall_at_5" not in result["predictions"][0]


def test_unannotated_evaluation_does_not_fabricate_retrieval_metrics():
    evaluator, model = _recall_evaluator([[1, 3, 5, 6, 7]])
    result = evaluator.evaluate(
        questions=["q1"], gold_answers=["answer"], batch_size=1
    )

    assert model.returned_first_pass is False
    assert "recall_at_5" not in result
    assert "recall_at_5" not in result["predictions"][0]
    assert "retrieved_doc_ids" not in result["predictions"][0]


def test_evaluator_enforces_the_paper_generation_length():
    evaluator, _ = _recall_evaluator([[1, 3, 5]])

    with pytest.raises(ValueError, match="max_new_tokens=64"):
        evaluator.evaluate(
            questions=["q"],
            gold_answers=["answer"],
            max_new_tokens=65,
        )


def test_oracle_evaluation_uses_final_top5_and_materializes_fixed_pool():
    corpus_ids = [f"doc-{index}" for index in range(105)]
    corpus_pages = [f"page-{index}" for index in range(105)]
    shared_pool = _construct_oracle_top100_indices(
        list(range(100)), [2, 101], corpus_page_ids=corpus_pages
    )
    model = _RecallModel([[0, 1, 2, 3, 4]])
    evaluator = ARIAEvaluator(
        model=model,
        corpus_docs=[f"document {index}" for index in range(105)],
        corpus_ids=corpus_ids,
        corpus_page_ids=corpus_pages,
        doc_embeddings=torch.randn(105, 3),
        rag_config=RAGPipelineConfig(use_mtfrl=False),
        retrieval_mode="oracle",
    )
    result = evaluator.evaluate(
        questions=["q"],
        gold_answers=["answer"],
        gold_doc_ids=[["doc-2", "doc-101"]],
        oracle_pool_records=[shared_pool],
        batch_size=1,
    )

    assert model.returned_first_pass is False
    assert model.oracle_gold_indices == [[2, 101]]
    prediction = result["predictions"][0]
    assert len(prediction["oracle_pool_doc_ids"]) == 100
    assert prediction["oracle_pool_doc_ids"][0] == "doc-0"
    assert prediction["oracle_pool_sha256"] == shared_pool.pool_sha256
    assert prediction["oracle_injected_gold_doc_ids"] == ["doc-101"]
    assert prediction["recall_at_1"] == pytest.approx(0.0)
    assert prediction["recall_at_3"] == pytest.approx(0.5)
    assert prediction["recall_at_5"] == pytest.approx(0.5)
    assert result["recall_at_1"] == pytest.approx(0.0)
    assert result["recall_at_3"] == pytest.approx(0.5)
    assert result["recall_at_5"] == pytest.approx(0.5)


def test_oracle_evaluation_rejects_missing_corpus_level_labels():
    corpus_ids = [f"doc-{index}" for index in range(100)]
    model = _RecallModel([[0, 1, 2, 3, 4]])
    evaluator = ARIAEvaluator(
        model=model,
        corpus_docs=[f"document {index}" for index in range(100)],
        corpus_ids=corpus_ids,
        corpus_page_ids=[f"page-{index}" for index in range(100)],
        doc_embeddings=torch.randn(100, 3),
        rag_config=RAGPipelineConfig(use_mtfrl=False),
        retrieval_mode="oracle",
    )
    with pytest.raises(ValueError, match="requires corpus-level gold_doc_ids"):
        evaluator.evaluate(questions=["q"], gold_answers=["answer"])


def test_oracle_evaluation_rejects_rows_without_a_support_page():
    corpus_ids = [f"doc-{index}" for index in range(100)]
    model = _RecallModel([[0, 1, 2, 3, 4]])
    evaluator = ARIAEvaluator(
        model=model,
        corpus_docs=[f"document {index}" for index in range(100)],
        corpus_ids=corpus_ids,
        corpus_page_ids=[f"page-{index}" for index in range(100)],
        doc_embeddings=torch.randn(100, 3),
        rag_config=RAGPipelineConfig(use_mtfrl=False),
        retrieval_mode="oracle",
    )
    with pytest.raises(ValueError, match="must contain at least one support"):
        evaluator.evaluate(
            questions=["q"], gold_answers=["answer"], gold_doc_ids=[[]]
        )


def test_gold_document_ids_are_all_or_none_and_strictly_validated():
    assert _extract_gold_document_ids(
        [{"gold_doc_ids": ["doc-1"]}, {"gold_doc_ids": ["doc-2", "doc-3"]}]
    ) == [["doc-1"], ["doc-2", "doc-3"]]
    assert _extract_gold_document_ids(
        [{"gold_doc_ids": []}, {"gold_doc_ids": ["doc-2"]}]
    ) == [[], ["doc-2"]]
    assert _extract_gold_document_ids([{"question": "q"}]) is None
    with pytest.raises(ValueError, match="every row or omitted"):
        _extract_gold_document_ids(
            [{"gold_doc_ids": ["doc-1"]}, {"question": "q2"}]
        )


def _comparison_payload(benchmark_scores):
    payload = {}
    for benchmark, scores in benchmark_scores.items():
        per_seed = []
        for training_seed in (11, 22):
            predictions = []
            for index, score in enumerate(scores):
                predictions.append(
                    {
                        "example_id": f"{benchmark}-{index}",
                        "question": f"Question {benchmark} {index}?",
                        "gold_answer": f"answer-{index}",
                        "em": score,
                        "cem": score,
                        "f1": score,
                    }
                )
            per_seed.append({"seed": training_seed, "predictions": predictions})
        payload[benchmark] = {"seeds": [11, 22], "per_seed": per_seed}
    return payload


def test_headline_avg_bootstrap_weights_benchmarks_not_examples():
    # The large benchmark has three +0.8 identities while the small benchmark
    # has one -0.2 identity.  A pooled-example mean would therefore be +0.55;
    # the paper's equal-benchmark Avg is (+0.8 - 0.2) / 2 = +0.3.
    candidate = _comparison_payload(
        {
            "large": [0.9] * 3,
            "small": [0.2],
        }
    )
    baseline = _comparison_payload(
        {
            "large": [0.1] * 3,
            "small": [0.4],
        }
    )

    comparisons = compare_evaluation_payloads(
        candidate, baseline, n_resamples=99, seed=7
    )

    for metric in ("em", "cem", "f1"):
        average = comparisons["avg"][metric]
        assert average["mean_diff"] == pytest.approx(0.3)
        assert average["mean_diff"] != pytest.approx(0.55)
        assert average["ci_95_lower"] == pytest.approx(0.3)
        assert average["ci_95_upper"] == pytest.approx(0.3)
        assert average["benchmark_weighting"] == "unweighted_mean"
        assert average["resampling_unit"] == "example_identity_within_benchmark"
        assert average["n_examples_by_benchmark"] == {"large": 3, "small": 1}
        assert average["n_observations_by_benchmark"] == {"large": 6, "small": 2}
        assert average["n_seeds_by_benchmark"] == {"large": 2, "small": 2}
        assert average["benchmark_mean_differences"] == pytest.approx(
            {"large": 0.8, "small": -0.2}
        )
