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
from openrlhf.models.modeling_aria import (
    CLaRa,
)
from openrlhf.trainer.sft_trainer import (
    compose_aria_training_loss,
)


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


def test_phase1_mistral_defaults_match_submission(tmp_path, monkeypatch):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage="stage1",
        pretrain="mistralai/Mistral-7B-Instruct-v0.2",
    )
    assert args.learning_rate == pytest.approx(2e-4)
    assert args.lr_warmup_steps == 500
    assert args.train_batch_size == 32
    assert args.micro_train_batch_size == 4
    assert args.max_epochs == 3
    assert (args.doc_max_length, args.query_max_length) == (768, 256)
    assert (args.max_len, args.target_max_length) == (2048, 512)
    assert _paper_warmup_steps(args, 183_009) == 500


def test_phase2_mistral_defaults_match_submission(tmp_path, monkeypatch):
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
    assert args.lambda_mse == pytest.approx(0.10)
    # Eq. (phase2_total) contains no independent CFRS, QR, or MTFRL loss,
    # so those retired knobs are not part of the public CLI contract.
    assert all(
        not hasattr(args, name)
        for name in ("lambda_cfrs", "lambda_qr", "lambda_mtfrl")
    )


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
    assert args.micro_train_batch_size == 1
    assert args.learning_rate == pytest.approx(1.6e-4)


@pytest.mark.parametrize(
    ("source", "expected_type"),
    sorted(PHASE1_DATA_TYPES.items()),
)
def test_phase1_rows_keep_category_but_drop_instruction_condition(source, expected_type):
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
    assert normalized["question"] == ""
    assert normalized["data_type"] == expected_type


def test_full_phase2_loss_is_exactly_qa_plus_point_one_hidden_state_mse():
    args = SimpleNamespace(
        stage="stage2",
        rag_configuration="full",
        lambda_mse=0.10,
        # Deliberately non-zero legacy values ensure the composer cannot
        # silently retain objectives that no longer occur in the manuscript.
        lambda_cfrs=0.70,
        lambda_qr=0.80,
        lambda_mtfrl=0.90,
    )
    qa = torch.tensor(2.0, requires_grad=True)
    outputs = {
        "mse_loss": torch.tensor(3.0, requires_grad=True),
        "cfrs_loss": torch.tensor(5.0, requires_grad=True),
        "qr_loss": torch.tensor(7.0, requires_grad=True),
        "mtfrl_loss": torch.tensor(11.0, requires_grad=True),
    }
    total, terms = compose_aria_training_loss(qa, outputs, args)
    assert total.item() == pytest.approx(2.3)
    assert set(terms) == {
        "qa_loss",
        "mse_loss",
        "weighted_mse_loss",
        "total_loss",
    }
    total.backward()
    assert qa.grad.item() == pytest.approx(1.0)
    assert outputs["mse_loss"].grad.item() == pytest.approx(0.10)
    assert outputs["cfrs_loss"].grad is None
    assert outputs["qr_loss"].grad is None
    assert outputs["mtfrl_loss"].grad is None


def test_full_phase2_loss_fails_closed_on_missing_hidden_state_mse():
    args = SimpleNamespace(
        stage="stage2",
        rag_configuration="full",
        lambda_mse=0.10,
    )
    with pytest.raises(KeyError, match="mse_loss"):
        compose_aria_training_loss(
            torch.tensor(1.0),
            {
                "cfrs_loss": torch.tensor(0.0),
                "qr_loss": torch.tensor(0.0),
                "mtfrl_loss": torch.tensor(0.0),
            },
            args,
        )


def test_phase1_does_not_add_the_phase2_mse_term():
    args = SimpleNamespace(stage="stage1", rag_configuration="full", lambda_mse=0.10)
    qa = torch.tensor(2.0, requires_grad=True)
    mse = torch.tensor(3.0, requires_grad=True)

    total, terms = compose_aria_training_loss(qa, {"mse_loss": mse}, args)

    assert total.item() == pytest.approx(2.0)
    assert terms["weighted_mse_loss"].item() == 0.0
    total.backward()
    assert qa.grad.item() == pytest.approx(1.0)
    assert mse.grad is None


def test_submission_phase2_mse_is_squared_l2_not_hidden_coordinate_mean():
    model = CLaRa.__new__(CLaRa)
    torch.nn.Module.__init__(model)
    model.decoder_tokenizer = SimpleNamespace(
        mem_token_ids_pt=torch.tensor([90, 91]),
        all_special_ids=[0, 90, 91],
    )
    hidden = torch.tensor(
        [[
            [1.0, 3.0],
            [3.0, 5.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [999.0, 999.0],
        ]],
        requires_grad=True,
    )
    input_ids = torch.tensor([[90, 91, 10, 11, 20, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, -100, 20, -100]])
    query_mask = torch.tensor([[False, False, True, True, False, False]])

    mse = model._compute_qa_lmse(
        hidden,
        input_ids,
        attention_mask,
        labels,
        query_mask,
    )
    qa = torch.tensor(2.0, requires_grad=True)
    total, _ = compose_aria_training_loss(
        qa,
        {"mse_loss": mse},
        SimpleNamespace(stage="stage2", rag_configuration="full", lambda_mse=0.1),
    )

    # Memory mean [2,4], query+answer mean [1,1]: 1^2 + 3^2 = 10.
    assert mse.item() == pytest.approx(10.0)
    assert total.item() == pytest.approx(3.0)
    total.backward()
    assert qa.grad.item() == pytest.approx(1.0)
    assert hidden.grad[0, :5].abs().sum() > 0
    assert torch.count_nonzero(hidden.grad[0, 5]) == 0


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


@pytest.mark.parametrize("target_family", sorted(set(PHASE1_DATA_TYPES.values())))
def test_submission_phase1_four_families_condition_on_memory_not_instruction(
    target_family,
):
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
        [(["document"], "instruction", "a" * 600, target_family, [0], [], ["gold"])]
    )
    assert model.encoder_max_length == 768
    assert model.last_phase1_prompt["query"] == ""
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
