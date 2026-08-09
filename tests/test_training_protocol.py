from types import SimpleNamespace

import pytest
import torch

from openrlhf.cli.train_sft import (
    _paper_warmup_steps,
    create_argument_parser,
    validate_arguments,
)
from openrlhf.cli.aria_data import (
    PHASE1_DATA_TYPES,
    Phase1FieldMap,
    _normalize_phase1_row,
)
from openrlhf.datasets.sft_dataset import make_collate_fn
from openrlhf.trainer.sft_trainer import compose_aria_training_loss


def _artifact(tmp_path, name):
    path = tmp_path / name
    path.write_text("artifact\n", encoding="utf-8")
    return str(path)


def _protocol_args(tmp_path, monkeypatch, *, stage, pretrain):
    monkeypatch.setenv("WORLD_SIZE", "8")
    dataset = tmp_path / f"{stage}-dataset"
    dataset.mkdir()
    argv = [
        "--pretrain",
        pretrain,
        "--stage",
        stage,
        "--dataset",
        str(dataset),
        "--test_url_file",
        _artifact(tmp_path, f"{stage}-test-urls.txt"),
        "--bf16",
        "--flash_attn",
    ]
    if stage == "stage2":
        checkpoint = tmp_path / "phase1-checkpoint"
        checkpoint.mkdir()
        argv.extend(
            [
                "--pretrain_checkpoint",
                str(checkpoint),
                "--corpus_path",
                _artifact(tmp_path, "corpus.json"),
                "--corpus_embeddings_path",
                _artifact(tmp_path, "corpus.pt"),
                "--bge_projection_path",
                _artifact(tmp_path, "w-bge.pt"),
            ]
        )
    args = create_argument_parser().parse_args(argv)
    validate_arguments(args)
    return args


def test_phase1_mistral_defaults_match_paper(tmp_path, monkeypatch):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage="stage1",
        pretrain="mistralai/Mistral-7B-Instruct-v0.2",
    )
    assert args.learning_rate == pytest.approx(1e-4)
    assert args.lr_warmup_steps is None
    assert args.lr_warmup_ratio == pytest.approx(0.03)
    assert args.train_batch_size == 128
    assert args.micro_train_batch_size == 16
    assert args.max_epochs == 3
    assert (args.doc_max_length, args.query_max_length) == (768, 256)
    assert (args.max_len, args.target_max_length) == (2048, 512)
    assert _paper_warmup_steps(args, 183_009) == 5_491


def test_phase2_mistral_defaults_match_paper(tmp_path, monkeypatch):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage="stage2",
        pretrain="mistralai/Mistral-7B-Instruct-v0.2",
    )
    assert args.learning_rate == pytest.approx(2e-4)
    assert args.lr_warmup_steps == 500
    assert args.train_batch_size == 32
    assert args.micro_train_batch_size == 4
    assert args.max_epochs == 5
    assert (args.doc_max_length, args.query_max_length) == (768, 256)
    assert (args.max_len, args.target_max_length) == (1024, 128)
    assert _paper_warmup_steps(args, 6_000) == 500
    assert (
        args.lambda_mse,
        args.lambda_cfrs,
        args.lambda_qr,
        args.lambda_mtfrl,
    ) == pytest.approx((0.10, 0.10, 0.05, 0.05))


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_qwen_uses_effective_batch_16_in_both_phases(
    tmp_path, monkeypatch, stage
):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage=stage,
        pretrain="Qwen/Qwen2.5-14B-Instruct",
    )
    assert args.train_batch_size == 16
    assert args.micro_train_batch_size == 2
    expected_lr = 1e-4 if stage == "stage1" else 1.6e-4
    assert args.learning_rate == pytest.approx(expected_lr)


def test_phase2_rejects_gradient_accumulation_as_a_fake_minibatch(
    tmp_path, monkeypatch
):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage="stage2",
        pretrain="mistralai/Mistral-7B-Instruct-v0.2",
    )
    args.micro_train_batch_size = 1
    with pytest.raises(ValueError, match="does not use gradient accumulation"):
        validate_arguments(args)


@pytest.mark.parametrize(
    ("source", "expected_type"),
    sorted(PHASE1_DATA_TYPES.items()),
)
def test_phase1_rows_keep_explicit_instruction_and_category(source, expected_type):
    normalized = _normalize_phase1_row(
        {
            "document": "source passage",
            "instruction": "Perform this source-provided task.",
            "target": "held-out target",
            "source_row_id": "source-1",
            "target_row_id": "target-1",
            "target_split": "held-out",
            "page_url": "https://en.wikipedia.org/wiki/Example",
        },
        0,
        source_name=source,
        field_map=Phase1FieldMap(),
    )
    assert normalized["question"] == "Perform this source-provided task."
    assert normalized["data_type"] == expected_type


def test_full_phase2_loss_uses_all_four_paper_coefficients():
    args = SimpleNamespace(
        stage="stage2",
        rag_configuration="full",
        lambda_mse=0.10,
        lambda_cfrs=0.10,
        lambda_qr=0.05,
        lambda_mtfrl=0.05,
    )
    qa = torch.tensor(2.0, requires_grad=True)
    outputs = {
        "mse_loss": torch.tensor(3.0, requires_grad=True),
        "cfrs_loss": torch.tensor(5.0, requires_grad=True),
        "qr_loss": torch.tensor(7.0, requires_grad=True),
        "mtfrl_loss": torch.tensor(11.0, requires_grad=True),
    }
    total, terms = compose_aria_training_loss(qa, outputs, args)
    assert total.item() == pytest.approx(3.7)
    assert set(terms) == {
        "qa_loss",
        "mse_loss",
        "weighted_mse_loss",
        "cfrs_loss",
        "weighted_cfrs_loss",
        "qr_loss",
        "weighted_qr_loss",
        "mtfrl_loss",
        "weighted_mtfrl_loss",
        "total_loss",
    }
    total.backward()
    assert qa.grad.item() == pytest.approx(1.0)
    assert outputs["mse_loss"].grad.item() == pytest.approx(0.10)
    assert outputs["cfrs_loss"].grad.item() == pytest.approx(0.10)
    assert outputs["qr_loss"].grad.item() == pytest.approx(0.05)
    assert outputs["mtfrl_loss"].grad.item() == pytest.approx(0.05)


def test_full_phase2_loss_fails_closed_on_missing_model_term():
    args = SimpleNamespace(
        stage="stage2",
        rag_configuration="full",
        lambda_mse=0.10,
        lambda_cfrs=0.10,
        lambda_qr=0.05,
        lambda_mtfrl=0.05,
    )
    with pytest.raises(KeyError, match="mtfrl_loss"):
        compose_aria_training_loss(
            torch.tensor(1.0),
            {
                "mse_loss": torch.tensor(0.0),
                "cfrs_loss": torch.tensor(0.0),
                "qr_loss": torch.tensor(0.0),
            },
            args,
        )


class _CharacterTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=False,
        **kwargs,
    ):
        assert isinstance(text, str)
        result = {"input_ids": [1 + (ord(char) % 251) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class _CollateModel:
    def __init__(self, stage):
        self.training_stage = stage
        self.generation_top_k = 1 if stage == "stage1" else 5
        self.stage2_retrieval_top_n = self.generation_top_k
        self.decoder_tokenizer = _CharacterTokenizer()
        self.encoder_max_length = None
        self.query_max_length = None
        self.last_phase1_prompt = None

    def _prepare_encoder_inputs(self, documents, max_length):
        self.encoder_max_length = max_length
        return {
            "input_ids": torch.ones((len(documents), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(documents), 2), dtype=torch.long),
            "memory_token_counts": torch.ones(len(documents), dtype=torch.long),
        }

    def _prepare_query_inputs(self, questions, max_length):
        self.query_max_length = max_length
        return {
            "input_ids": torch.ones((len(questions), 2), dtype=torch.long),
            "attention_mask": torch.ones((len(questions), 2), dtype=torch.long),
        }

    def _blend_prompt_and_memory_tokens(self, *, answer, **kwargs):
        self.last_phase1_prompt = kwargs
        prompt = "p" * 2_100
        return len(prompt), prompt + answer

    def _blend_prompt_and_selected_memory_tokens(self, *, query, answer):
        prompt = "system prompt Background:mem Question:" + query
        return len(prompt), prompt + answer


def test_collator_enforces_phase1_passage_input_and_target_limits():
    model = _CollateModel("stage1")
    collate = make_collate_fn(
        model,
        passage_max_len=768,
        query_max_len=256,
        input_max_len=2048,
        target_max_len=512,
        qa_loss=True,
    )
    batch = collate(
        [(["document"], "instruction", "a" * 600, "simple_qa", [0], [], ["gold"])]
    )
    assert model.encoder_max_length == 768
    assert model.last_phase1_prompt["query"] == "instruction"
    assert model.last_phase1_prompt["stage"] == "stage1_2"
    assert batch["dec_attention_mask"].sum().item() == 2048 + 512
    assert batch["labels"].ne(-100).sum().item() == 512


def test_phase2_collator_defers_realized_prompt_until_after_retrieval():
    model = _CollateModel("stage2")
    collate = make_collate_fn(
        model,
        passage_max_len=768,
        query_max_len=256,
        input_max_len=1024,
        target_max_len=128,
        qa_loss=True,
    )
    # At CR=4, five maximum-length passages can reserve 960 base memory
    # positions.  Collation must not fabricate that fixed prompt and truncate
    # the trailing question before CCEF/ACR determine the realized allocation.
    docs = [(f"document-{index} " * 768).strip() for index in range(5)]
    batch = collate(
        # P(x) is allowed to be empty; QA/MSE/CFRS still train on this row.
        [(docs, "where", "a" * 200, "qa", [], [], ["gold"])]
    )
    assert model.query_max_length == 256
    assert batch["questions"] == ["where"]
    assert batch["answers"] == ["a" * 200]
    assert batch["docs"] == [docs]
    assert "dec_input_ids" not in batch
    assert "labels" not in batch
    assert "query_position_mask" not in batch
