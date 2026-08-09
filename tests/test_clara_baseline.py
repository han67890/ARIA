import pytest
import torch

from openrlhf.cli.train_sft import create_argument_parser, validate_arguments
from openrlhf.models.modeling_aria import (
    CLaRaConfig,
    _clara_st_select_candidate_memory,
)


def _hard_gather(memory: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, _, tokens, hidden = memory.shape
    gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(
        batch, indices.size(1), tokens, hidden
    )
    return memory.gather(1, gather_index)


def test_clara_selector_is_hard_forward_and_soft_backward():
    torch.manual_seed(7)
    query = torch.randn(2, 4, requires_grad=True)
    candidates = torch.randn(2, 7, 3, 4)

    selected, indices, scores, weights = _clara_st_select_candidate_memory(
        query, candidates, 5
    )

    assert indices.shape == (2, 5)
    assert scores.shape == (2, 7)
    assert weights.shape == (2, 5, 7)
    assert torch.equal(selected.detach(), _hard_gather(candidates, indices))

    slot_scale = torch.arange(1, 6, dtype=selected.dtype).view(1, 5, 1, 1)
    (selected * slot_scale).square().sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.count_nonzero(query.grad).item() > 0


def test_clara_selector_keeps_st_formula_when_candidate_count_equals_k():
    torch.manual_seed(11)
    query = torch.randn(1, 3, requires_grad=True)
    candidates = torch.randn(1, 5, 2, 3)

    selected, indices, _, weights = _clara_st_select_candidate_memory(
        query, candidates, 5
    )

    assert torch.equal(selected.detach(), _hard_gather(candidates, indices))
    assert weights.grad_fn is not None
    slot_scale = torch.arange(1, 6, dtype=selected.dtype).view(1, 5, 1, 1)
    (selected * slot_scale).sum().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.count_nonzero(query.grad).item() > 0


def test_clara_selector_fails_closed_when_candidate_pool_is_too_small():
    with pytest.raises(ValueError, match="at least k=5"):
        _clara_st_select_candidate_memory(
            torch.ones(1, 3), torch.ones(1, 4, 2, 3), 5
        )


def test_clara_selector_masks_padding_after_each_real_memory_count():
    query = torch.tensor([[1.0, 0.0]])
    candidates = torch.tensor(
        [
            [
                [[1.0, 0.0], [-1000.0, 0.0], [-1000.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            ]
        ]
    )
    counts = torch.tensor([[1, 3]])

    _, indices, scores, _ = _clara_st_select_candidate_memory(
        query,
        candidates,
        1,
        candidate_memory_counts=counts,
    )

    assert indices.tolist() == [[0]]
    assert scores[0, 0] > scores[0, 1]


def test_clara_config_preserves_peft_all_linear_sentinel():
    config = CLaRaConfig(lora_target_modules="all-linear")
    assert config.lora_target_modules == "all-linear"


def test_clara_config_does_not_confuse_literal_all_linear_list_with_sentinel():
    config = CLaRaConfig(lora_target_modules=["all-linear"])
    assert config.lora_target_modules == ["all-linear"]


def test_clara_phase2_cli_makes_all_auxiliary_coefficients_zero(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "phase2"
    checkpoint = tmp_path / "phase1"
    dataset.mkdir()
    checkpoint.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    test_urls = tmp_path / "test-urls.txt"
    corpus.write_text("{}\n", encoding="utf-8")
    test_urls.write_text("https://example.test/page\n", encoding="utf-8")
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = create_argument_parser().parse_args(
        [
            "--pretrain",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "--stage",
            "stage2",
            "--rag_configuration",
            "clara_baseline",
            "--dataset",
            str(dataset),
            "--pretrain_checkpoint",
            str(checkpoint),
            "--corpus_path",
            str(corpus),
            "--test_url_file",
            str(test_urls),
            "--train_batch_size",
            "32",
            "--micro_train_batch_size",
            "32",
            "--zero_stage",
            "2",
            "--bf16",
            "--flash_attn",
        ]
    )

    validate_arguments(args)

    assert args.lambda_mse == 0.0
    assert args.lambda_cfrs == 0.0
    assert args.lambda_qr == 0.0
    assert args.lambda_mtfrl == 0.0
