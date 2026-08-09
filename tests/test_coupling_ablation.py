from types import SimpleNamespace

import pytest
import torch

from openrlhf.cli.ablation_aria import create_rag_config
from openrlhf.cli.counterfactual_decomposition import (
    _validate_counterfactual_checkpoint_sets,
)
from openrlhf.cli.evaluate_aria import (
    _required_checkpoint_configuration,
    _validate_checkpoint_protocol,
)
from openrlhf.cli.train_sft import create_argument_parser, validate_arguments
from openrlhf.models.modeling_aria import (
    MATCHED_EVIDENCE_TOKEN_BUDGET,
    AdaptiveCompressionAllocator,
)


def test_matched_remove_all_and_fixed_forward_path_are_distinct_protocols():
    assert _required_checkpoint_configuration("remove_all_coupling") == (
        "remove_all_coupling"
    )
    assert _required_checkpoint_configuration("forward_path_off") == "full"

    config = create_rag_config("remove_all_coupling", compression_rate=16)
    assert config.use_cfrs is False
    assert config.use_acr is False
    assert config.acr_allocation_mode == "uniform_budget"
    assert config.use_mtfrl is False
    assert config.second_retrieval_mode == "static_query"
    assert config.mtfrl_second_top_k == 200

    forward_off = create_rag_config("forward_path_off", compression_rate=16)
    assert forward_off.use_cfrs is False
    assert forward_off.acr_allocation_mode == "full"
    assert forward_off.second_retrieval_mode == "disabled"


def test_training_cli_accepts_matched_remove_all_and_zeros_only_disabled_losses():
    args = create_argument_parser().parse_args(
        [
            "--pretrain",
            "mistralai/Mistral-7B-v0.1",
            "--stage",
            "stage2",
            "--rag_configuration",
            "remove_all_coupling",
        ]
    )
    # No fixture checkpoint/artifacts are supplied, so validation reaches the
    # normal filesystem prerequisite after accepting the training protocol.
    with pytest.raises(ValueError, match="--dataset is required for training"):
        validate_arguments(args)
    assert args.lambda_mse == pytest.approx(0.10)
    assert args.lambda_cfrs == 0.0
    assert args.lambda_qr == pytest.approx(0.05)
    assert args.lambda_mtfrl == 0.0


def test_training_cli_rejects_fixed_forward_path_checkpoint_training():
    args = create_argument_parser().parse_args(
        [
            "--pretrain",
            "mistralai/Mistral-7B-v0.1",
            "--stage",
            "stage2",
            "--rag_configuration",
            "forward_path_off",
        ]
    )
    with pytest.raises(ValueError, match="fixed-checkpoint inference-only"):
        validate_arguments(args)


@pytest.mark.parametrize(
    "configuration",
    ["remove_qca", "remove_ahr", "remove_igfr", "remove_mads", "remove_ccef"],
)
def test_retrieval_stage_ablation_reuses_full_checkpoint_and_cannot_train(
    configuration,
):
    assert _required_checkpoint_configuration(configuration) == "full"
    args = create_argument_parser().parse_args(
        [
            "--pretrain",
            "mistralai/Mistral-7B-v0.1",
            "--stage",
            "stage2",
            "--rag_configuration",
            configuration,
        ]
    )
    with pytest.raises(ValueError, match="fixed-checkpoint inference-only"):
        validate_arguments(args)


def test_protocol_validator_accepts_full_training_label_for_forward_path_off():
    model = SimpleNamespace(
        config=SimpleNamespace(
            decoder_model_resolved_revision="0" * 40,
            mads_semantic_model_name="BAAI/bge-large-en-v1.5",
            aria_compression_rate=16,
            aria_rag_configuration="full",
        ),
        compr_rate=16,
    )
    # Reaching the next provenance check demonstrates that the full checkpoint
    # training label was accepted for this runtime counterfactual.
    with pytest.raises(ValueError, match="aria_training_seed provenance"):
        _validate_checkpoint_protocol(
            model,
            "full-checkpoint",
            training_seed=42,
            compression_rate=16,
            expected_configuration="forward_path_off",
        )


def test_matched_control_shapes_and_release_conventions():
    remove_cfrs = create_rag_config("remove_cfrs", compression_rate=16)
    assert remove_cfrs.use_cfrs is False
    assert remove_cfrs.acr_allocation_mode == "adaptive"
    assert remove_cfrs.second_retrieval_mode == "memory_feedback"

    uniform = create_rag_config("uniform_acr", compression_rate=16)
    assert uniform.use_acr is False
    assert uniform.acr_allocation_mode == "uniform_budget"
    assert uniform.uniform_evidence_token_budget == MATCHED_EVIDENCE_TOKEN_BUDGET
    assert uniform.second_retrieval_mode == "memory_feedback"

    static = create_rag_config("static_second_retrieval", compression_rate=16)
    assert static.use_mtfrl is False
    assert static.second_retrieval_mode == "static_query"
    assert static.mtfrl_second_top_k == 200
    assert static.acr_allocation_mode == "adaptive"


def test_uniform_budget_ratio_is_score_independent_and_variable_doc_safe():
    five = AdaptiveCompressionAllocator.uniform_ratios_for_budget(
        torch.tensor([48, 48, 48, 48, 48]), 108
    )
    assert five.tolist() == pytest.approx([0.45] * 5)
    two_short = AdaptiveCompressionAllocator.uniform_ratios_for_budget(
        torch.tensor([30, 20]), 108
    )
    assert two_short.tolist() == pytest.approx([1.0, 1.0])


def test_matched_retraining_controls_are_defined_only_at_16x():
    for name in (
        "remove_cfrs",
        "uniform_acr",
        "static_second_retrieval",
        "remove_all_coupling",
    ):
        with pytest.raises(ValueError, match="defined only at 16x"):
            create_rag_config(name, compression_rate=32)


def test_counterfactual_reuses_full_paths_when_no_coupling_paths_are_omitted():
    validated = _validate_counterfactual_checkpoint_sets(
        {
            "full_aria": ["full-42", "full-123"],
            "clara_baseline": ["clara-42", "clara-123"],
        },
        seeds=[42, 123],
    )
    assert validated["aria_ret_clara_comp"] == validated["full_aria"]


def test_counterfactual_rejects_separately_trained_no_coupling_paths():
    with pytest.raises(ValueError, match="must reuse each full-ARIA checkpoint"):
        _validate_counterfactual_checkpoint_sets(
            {
                "full_aria": ["full-42", "full-123"],
                "aria_ret_clara_comp": ["no-coupling-42", "no-coupling-123"],
                "clara_baseline": ["clara-42", "clara-123"],
            },
            seeds=[42, 123],
        )
