import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import openrlhf.cli.train_sft as train_sft

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
    ARIA_LIKELIHOOD_REDUCTION,
    CLaRa,
    _likelihood_outputs,
    _shifted_target_token_count,
)
from openrlhf.trainer.sft_trainer import (
    accumulate_aria_eval_loss_metrics,
    compose_aria_training_loss,
    normalize_likelihood_for_global_token_mean,
)


def _artifact(tmp_path, name):
    path = tmp_path / name
    path.write_text("artifact\n", encoding="utf-8")
    return str(path)


def _protocol_args(tmp_path, monkeypatch, *, stage, pretrain, extra_args=()):
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
    argv.extend(extra_args)
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
    assert args.lambda_mse == pytest.approx(0.10)
    assert args.acr_training_gate == "soft"
    # Eq. (phase2_total) contains no independent CFRS, QR, or MTFRL loss,
    # so those retired knobs are not part of the public CLI contract.
    assert all(
        not hasattr(args, name)
        for name in ("lambda_cfrs", "lambda_qr", "lambda_mtfrl")
    )


def test_phase2_accepts_the_independent_hard_st_gate_checkpoint(
    tmp_path, monkeypatch
):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage="stage2",
        pretrain="mistralai/Mistral-7B-Instruct-v0.2",
        extra_args=("--acr_training_gate", "hard_st"),
    )

    assert args.acr_training_gate == "hard_st"


def test_phase2_checkpoint_load_forwards_the_hard_st_gate_override(
    tmp_path, monkeypatch
):
    args = _protocol_args(
        tmp_path,
        monkeypatch,
        stage="stage2",
        pretrain="mistralai/Mistral-7B-Instruct-v0.2",
        extra_args=("--acr_training_gate", "hard_st"),
    )
    (tmp_path / "stage2-test-urls.txt").write_text(
        "https://example.org/wiki/Test\n",
        encoding="utf-8",
    )
    test_digest = train_sft._url_set_sha256(
        train_sft._load_test_url_file(args.test_url_file)
    )
    phase1_config = SimpleNamespace(
        training_stage="stage1",
        decoder_model_name=args.pretrain,
        compr_rate=args.compress_rate,
        doc_max_length=args.doc_max_length,
        aria_rag_configuration="full",
        lora_target_modules=["q_proj"],
        aria_phase1_training_seed=args.seed,
        aria_phase1_test_url_sha256=test_digest,
        aria_phase1_dataset_manifest_sha256="a" * 64,
        aria_likelihood_reduction=ARIA_LIKELIHOOD_REDUCTION,
    )
    captured = {}

    def fake_load(_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(config=SimpleNamespace())

    monkeypatch.setattr(
        train_sft.CLaRaConfig,
        "from_pretrained",
        staticmethod(lambda _path: phase1_config),
    )
    monkeypatch.setattr(
        train_sft.CLaRa,
        "from_pretrained",
        staticmethod(fake_load),
    )
    monkeypatch.setattr(
        train_sft,
        "build_source_snapshot_manifest",
        lambda: {
            "scheme": "test-source-v1",
            "git_commit": "0" * 40,
            "git_dirty": False,
            "source_tree_sha256": "b" * 64,
            "source_file_count": 1,
        },
    )

    model = train_sft.setup_model(args)

    assert captured["acr_training_gate"] == "hard_st"
    assert model.config.aria_loss_weights == {"lambda_mse": pytest.approx(0.10)}
    assert model.config.aria_likelihood_reduction == ARIA_LIKELIHOOD_REDUCTION


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


def test_shifted_target_count_excludes_the_unpredicted_first_label():
    labels = torch.tensor(
        [
            [7, -100, 11, 12, -100],
            [9, 21, -100, 22, 23],
        ]
    )
    # Position zero is not predicted by a causal decoder and must not count.
    assert _shifted_target_token_count(labels).item() == 5
    outputs = _likelihood_outputs(labels, logits=torch.ones(1), mse_loss=torch.ones(()))
    assert outputs["target_token_count"].dtype == torch.long
    assert outputs["target_token_count"].item() == 5


@pytest.mark.parametrize(
    "method_name",
    ["_forward_stage_1", "_forward_stage2_batch", "_forward_clara_baseline_batch"],
)
def test_all_paper_likelihood_paths_attach_the_shared_token_count(method_name):
    source = inspect.getsource(getattr(CLaRa, method_name))
    assert "_likelihood_outputs(" in source


def test_unequal_rank_token_counts_form_exact_global_mean_and_leave_mse_unweighted():
    class _FakeDPStrategy:
        def __init__(self, global_count, dp_world_size):
            self.global_count = float(global_count)
            self.dp_world_size = int(dp_world_size)

        def scale_loss_to_global_token_mean(self, local_mean, local_count):
            return local_mean * (
                self.dp_world_size
                * local_count.to(local_mean.dtype)
                / self.global_count
            )

    strategy = _FakeDPStrategy(global_count=5, dp_world_size=2)
    rank0_mean = torch.tensor(2.0, requires_grad=True)  # two-token sum = 4
    rank1_mean = torch.tensor(5.0, requires_grad=True)  # three-token sum = 15
    rank0 = normalize_likelihood_for_global_token_mean(
        rank0_mean, {"target_token_count": torch.tensor(2)}, strategy
    )
    rank1 = normalize_likelihood_for_global_token_mean(
        rank1_mean, {"target_token_count": torch.tensor(3)}, strategy
    )
    assert ((rank0 + rank1) / 2).item() == pytest.approx(19 / 5)

    mse = torch.tensor(7.0, requires_grad=True)
    args = SimpleNamespace(stage="stage2", rag_configuration="full", lambda_mse=0.1)
    total, terms = compose_aria_training_loss(
        rank0, {"mse_loss": mse}, args
    )
    total.backward()
    # Only QA receives the token-count rank scale; L_MSE remains an example mean.
    assert terms["weighted_mse_loss"].item() == pytest.approx(0.7)
    assert mse.grad.item() == pytest.approx(0.1)


def test_deepspeed_token_mean_scaling_is_scoped_to_the_mesh_dp_group():
    # The lightweight test environment intentionally omits DeepSpeed itself,
    # so inspect this small integration boundary while testing its arithmetic
    # behavior independently above.
    path = (
        Path(__file__).parents[1]
        / "openrlhf"
        / "utils"
        / "deepspeed"
        / "deepspeed.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        child
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepspeedStrategy"
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and child.name == "scale_loss_to_global_token_mean"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert 'self.ds_device_mesh["dp"].get_group()' in method_source
    assert "dist.get_world_size(group=dp_group)" in method_source
    assert "dist.all_reduce(" in method_source
    assert "group=dp_group" in method_source
    assert "target-token count must use an integer dtype" in method_source
    assert "global target-token count must be positive and finite" in method_source


def test_eval_aggregates_qa_by_tokens_and_mse_by_examples():
    metrics = {
        "qa_loss_sum": 0.0,
        "target_tokens": 0,
        "mse_loss_sum": 0.0,
        "weighted_mse_loss_sum": 0.0,
        "samples": 0,
        "retrieval_samples": 0.0,
    }
    accumulate_aria_eval_loss_metrics(
        metrics,
        {
            "qa_loss": torch.tensor(2.0),
            "mse_loss": torch.tensor(7.0),
            "weighted_mse_loss": torch.tensor(0.7),
        },
        {"target_token_count": torch.tensor(2)},
        batch_size=3,
    )
    accumulate_aria_eval_loss_metrics(
        metrics,
        {
            "qa_loss": torch.tensor(5.0),
            "mse_loss": torch.tensor(1.0),
            "weighted_mse_loss": torch.tensor(0.1),
        },
        {"target_token_count": torch.tensor(3)},
        batch_size=1,
    )

    trainer = object.__new__(__import__(
        "openrlhf.trainer.sft_trainer", fromlist=["SFTTrainer"]
    ).SFTTrainer)
    final = trainer._calculate_final_eval_metrics(metrics, eval_gen=False)
    assert final["eval_qa_loss"] == pytest.approx(19 / 5)
    assert final["eval_mse_loss"] == pytest.approx(22 / 4)
    assert final["eval_weighted_mse_loss"] == pytest.approx(2.2 / 4)
    assert final["eval_loss"] == pytest.approx(19 / 5 + 2.2 / 4)


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


def test_phase2_hidden_mean_mse_flows_into_the_exact_two_term_objective():
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

    # Memory mean [2,4], query+answer mean [1,1]: (1^2 + 3^2) / d_h = 5.
    assert mse.item() == pytest.approx(5.0)
    assert total.item() == pytest.approx(2.5)
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
