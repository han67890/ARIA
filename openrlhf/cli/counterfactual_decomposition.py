#!/usr/bin/env python3
"""Counterfactual decomposition of ARIA retrieval and coupling gains.

For each training seed, the full and no-coupling conditions use the same full
ARIA checkpoint.  The latter disables CFRS, ACR, and MTFRL only at inference,
as specified by the paper.  Matched CLaRa remains a separately trained model.
"""

import argparse
import json
import os
import random
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from openrlhf.cli.evaluate_aria import (
    ARIAEvaluator,
    EVALUATION_ANSWER_CONTRACT,
    PAPER_COMPRESSION_RATES,
    PAPER_MAX_NEW_TOKENS,
    PAPER_TRAINING_SEEDS,
    _REPOSITORY_EVAL_ARCHIVE_SHA256,
    _corpus_id,
    _corpus_page_url,
    _corpus_sha256,
    _corpus_text,
    _extract_example_ids,
    _extract_clara_candidate_columns,
    _extract_gold_answer,
    _extract_gold_document_ids,
    _format_artifact_path,
    _map_clara_candidates_to_corpus,
    _text_sha256,
    _assert_normal_retrieval_is_not_training_index,
    _assert_protocol_fingerprints_match,
    _validate_checkpoint_protocol,
    aggregate_checkpoint_results,
    load_bge_projection,
    load_corpus,
    load_doc_embeddings,
    load_eval_dataset,
)
from openrlhf.models.modeling_aria import (
    CLaRa,
    CLaRaConfig,
    RAGPipelineConfig,
    _BM25Index,
    create_paper_rag_config,
    _tensor_is_finite_in_chunks,
)
CONFIGURATION_LABELS = {
    "full_aria": "Full ARIA",
    "aria_ret_clara_comp": "ARIA-Ret- + CLaRa-Comp",
    "clara_baseline": "CLaRa baseline",
}


def _rag_configurations(compression_rate: int) -> Dict[str, RAGPipelineConfig]:
    """Return the three explicit stage configurations used in decomposition."""
    return {
        "full_aria": create_paper_rag_config("full", compression_rate),
        "aria_ret_clara_comp": create_paper_rag_config(
            "forward_path_off", compression_rate
        ),
        "clara_baseline": create_paper_rag_config(
            "clara_baseline", compression_rate
        ),
    }


def _validate_seed_checkpoints(
    checkpoint_paths: Sequence[str], seeds: Sequence[int]
) -> List[Tuple[int, str]]:
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required")
    if len(checkpoint_paths) != len(seeds):
        raise ValueError("checkpoint_paths and seeds must have equal length")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seed identifiers must be unique")

    expanded = [os.path.expanduser(path) for path in checkpoint_paths]
    canonical = [os.path.realpath(path) if os.path.exists(path) else path for path in expanded]
    if len(canonical) != len(set(canonical)):
        raise ValueError(
            "Each training seed must reference a distinct checkpoint; duplicate paths "
            "would turn one deterministic model into a false multi-seed result"
        )
    return list(zip(seeds, expanded))


def _validate_counterfactual_checkpoint_sets(
    checkpoint_paths_by_configuration: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
) -> Dict[str, List[Tuple[int, str]]]:
    """Validate the paper's two-checkpoint, three-condition design."""
    required = {"full_aria", "clara_baseline"}
    allowed = required | {"aria_ret_clara_comp"}
    keys = set(checkpoint_paths_by_configuration)
    if not required.issubset(keys) or not keys.issubset(allowed):
        raise ValueError(
            "Counterfactual evaluation requires full_aria and clara_baseline "
            "checkpoint sets"
        )

    full_paths = checkpoint_paths_by_configuration["full_aria"]
    no_coupling_paths = checkpoint_paths_by_configuration.get(
        "aria_ret_clara_comp", full_paths
    )
    normalized_paths = {
        "full_aria": full_paths,
        "aria_ret_clara_comp": no_coupling_paths,
        "clara_baseline": checkpoint_paths_by_configuration["clara_baseline"],
    }
    validated = {
        name: _validate_seed_checkpoints(paths, seeds)
        for name, paths in normalized_paths.items()
    }

    def canonical(pairs: Sequence[Tuple[int, str]]) -> List[str]:
        return [
            os.path.realpath(path) if os.path.exists(path) else path
            for _, path in pairs
        ]

    full_canonical = canonical(validated["full_aria"])
    no_coupling_canonical = canonical(validated["aria_ret_clara_comp"])
    if no_coupling_canonical != full_canonical:
        raise ValueError(
            "Forward-path-off must reuse each full-ARIA checkpoint and disable "
            "CFRS/ACR/MTFRL only at inference; do not substitute the separately "
            "retrained remove_all_coupling checkpoints"
        )
    clara_canonical = canonical(validated["clara_baseline"])
    if set(full_canonical) & set(clara_canonical):
        raise ValueError(
            "Full ARIA and matched CLaRa must use distinct trained checkpoints"
        )
    return validated


def _set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _projection_path(
    checkpoint_path: str,
    external_template: Optional[str],
    *,
    training_seed: int,
    dataset_name: str,
    compression_rate: int,
) -> str:
    if external_template is not None:
        return _format_artifact_path(
            external_template,
            seed=training_seed,
            dataset=dataset_name,
            compression_rate=compression_rate,
        )
    bundled = os.path.join(checkpoint_path, "bge_projection.pth")
    if not os.path.isfile(bundled):
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_path!r} has no bge_projection.pth; pass "
            "--bge_projection_path to supply W_BGE explicitly"
        )
    return bundled


def _load_mtfrl_projection_strict(model: CLaRa, checkpoint_path: str) -> None:
    """Require and strictly reload the trained Phase-II MTFRL head."""
    artifact_path = os.path.join(checkpoint_path, "mtfrl_projection.pth")
    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(
            f"MTFRL is enabled, but checkpoint {checkpoint_path!r} has no "
            "mtfrl_projection.pth"
        )
    try:
        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Strict checkpoint loading requires torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Invalid MTFRL state dictionary: {artifact_path}")
    if any(
        isinstance(value, torch.Tensor) and not _tensor_is_finite_in_chunks(value)
        for value in state.values()
    ):
        raise ValueError(f"MTFRL artifact contains NaN or infinite values: {artifact_path}")
    if getattr(model, "_mtfrl_projection", None) is None:
        model.setup_mtfrl_projection(initialize_from_bge=False)
    model._mtfrl_projection.load_state_dict(state, strict=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_model(
    checkpoint_path: str,
    decoder_model: Optional[str],
    compression_rate: int,
    projection_artifact: Optional[str],
    require_mtfrl: bool,
    expected_configuration: str,
    training_seed: int,
    device: str,
) -> CLaRa:
    overrides: Dict[str, Any] = {
        "pure_inference": True,
    }
    model = CLaRa.from_pretrained(
        checkpoint_path,
        strict_aria_artifacts=True,
        external_bge_artifact=projection_artifact is not None,
        **overrides,
    )
    if decoder_model is not None and decoder_model != model.decoder_model_name:
        raise ValueError("decoder_model may only assert the checkpoint's exact backbone")
    # Always reload through the strict evaluator loader.  The historical model
    # loader swallowed incompatible projection errors and could leave a random
    # W_BGE module installed.
    if projection_artifact is not None:
        load_bge_projection(model, projection_artifact, expected_output_dim=1024)
    model._aria_protocol_fingerprint = _validate_checkpoint_protocol(
        model,
        checkpoint_path,
        training_seed,
        compression_rate,
        expected_configuration,
    )
    if require_mtfrl:
        _load_mtfrl_projection_strict(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def _decompose(config_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute paired seed-level gains, means, and population SDs."""
    seed_lists = [result["seeds"] for result in config_results.values()]
    if any(seeds != seed_lists[0] for seeds in seed_lists[1:]):
        raise ValueError("Configuration results are not aligned by training seed")

    decomposition: Dict[str, Any] = {}
    for metric in ("f1", "cem", "em"):
        full = np.asarray(
            [row[metric] for row in config_results["full_aria"]["per_seed"]],
            dtype=np.float64,
        )
        no_coupling = np.asarray(
            [row[metric] for row in config_results["aria_ret_clara_comp"]["per_seed"]],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [row[metric] for row in config_results["clara_baseline"]["per_seed"]],
            dtype=np.float64,
        )
        total_gain = full - baseline
        retrieval_gain = no_coupling - baseline
        coupling_gain = full - no_coupling
        total_mean = float(total_gain.mean())
        retrieval_mean = float(retrieval_gain.mean())
        coupling_mean = float(coupling_gain.mean())
        share_denominator = total_mean if abs(total_mean) > 1e-12 else None

        decomposition[metric] = {
            "full_aria": float(full.mean() * 100),
            "full_aria_std": float(full.std(ddof=0) * 100),
            "aria_ret_clara_comp": float(no_coupling.mean() * 100),
            "aria_ret_clara_comp_std": float(no_coupling.std(ddof=0) * 100),
            "clara_baseline": float(baseline.mean() * 100),
            "clara_baseline_std": float(baseline.std(ddof=0) * 100),
            "total_gain_pp": total_mean * 100,
            "total_gain_std_pp": float(total_gain.std(ddof=0) * 100),
            "retrieval_contribution_pp": retrieval_mean * 100,
            "retrieval_contribution_std_pp": float(retrieval_gain.std(ddof=0) * 100),
            "coupling_contribution_pp": coupling_mean * 100,
            "coupling_contribution_std_pp": float(coupling_gain.std(ddof=0) * 100),
            "retrieval_share_pct": (
                retrieval_mean / share_denominator * 100
                if share_denominator is not None else None
            ),
            "coupling_share_pct": (
                coupling_mean / share_denominator * 100
                if share_denominator is not None else None
            ),
            "per_seed_total_gain_pp": (total_gain * 100).tolist(),
            "per_seed_retrieval_contribution_pp": (retrieval_gain * 100).tolist(),
            "per_seed_coupling_contribution_pp": (coupling_gain * 100).tolist(),
        }
    return decomposition


def _print_decomposition(title: str, decomposition: Dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    for metric in ("f1", "cem"):
        row = decomposition[metric]
        print(f"\n{metric.upper()}:")
        print(
            f"  Full ARIA:              {row['full_aria']:6.2f}% "
            f"(+/-{row['full_aria_std']:.2f})"
        )
        print(
            f"  ARIA-Ret-+CLaRa-Comp:   {row['aria_ret_clara_comp']:6.2f}% "
            f"(+/-{row['aria_ret_clara_comp_std']:.2f})"
        )
        print(
            f"  CLaRa baseline:         {row['clara_baseline']:6.2f}% "
            f"(+/-{row['clara_baseline_std']:.2f})"
        )
        print(
            f"  Total gain:             {row['total_gain_pp']:6.2f} "
            f"+/- {row['total_gain_std_pp']:.2f} pp"
        )
        print(
            f"  Retrieval contribution: {row['retrieval_contribution_pp']:6.2f} "
            f"+/- {row['retrieval_contribution_std_pp']:.2f} pp"
        )
        print(
            f"  Coupling contribution:  {row['coupling_contribution_pp']:6.2f} "
            f"+/- {row['coupling_contribution_std_pp']:.2f} pp"
        )


def run_counterfactual(
    checkpoint_paths_by_configuration: Mapping[str, Sequence[str]],
    seeds: Sequence[int],
    dataset_name: str,
    corpus_path: str,
    doc_embeddings_path: str,
    bge_projection_path: Optional[str] = None,
    decoder_model: Optional[str] = None,
    compression_rate: int = 16,
    max_samples: Optional[int] = None,
    eval_data_path: Optional[str] = None,
    clara_archive_dir: Optional[str] = None,
    batch_size: int = 8,
    max_new_tokens: int = PAPER_MAX_NEW_TOKENS,
    inference_seed: int = 0,
    device: str = "cuda",
    output_dir: str = "./decomp_results",
) -> Dict[str, Any]:
    """Evaluate and decompose one benchmark across independent checkpoints."""
    seed_checkpoints_by_configuration = _validate_counterfactual_checkpoint_sets(
        checkpoint_paths_by_configuration, seeds
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset, question_key, answer_key = load_eval_dataset(
        dataset_name,
        max_samples,
        eval_data_path,
        require_clara_archive=True,
        clara_archive_dir=clara_archive_dir,
    )
    questions = [item[question_key] for item in dataset]
    gold_answers = [_extract_gold_answer(item, answer_key) for item in dataset]
    gold_document_ids = _extract_gold_document_ids(dataset)
    if gold_document_ids is None:
        raise ValueError("Counterfactual evaluation requires prepared gold_doc_ids")
    example_ids = _extract_example_ids(dataset, dataset_name)
    baseline_documents = _extract_clara_candidate_columns(dataset)

    try:
        corpus = load_corpus(corpus_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the KILT corpus required for {dataset_name}"
        ) from exc
    corpus_docs = [_corpus_text(item) for item in corpus]
    corpus_ids = [_corpus_id(item, index) for index, item in enumerate(corpus)]
    corpus_hashes = [_text_sha256(text) for text in corpus_docs]
    corpus_urls = [
        _corpus_page_url(item, index) for index, item in enumerate(corpus)
    ]
    corpus_digest = _corpus_sha256(corpus_ids, corpus_hashes, corpus_urls)
    if not corpus_docs:
        raise ValueError("The retrieval corpus is empty")
    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Corpus document IDs must be unique")
    (
        baseline_candidate_doc_ids,
        baseline_candidate_page_ids,
        _,
    ) = _map_clara_candidates_to_corpus(
        baseline_documents,
        corpus_docs=corpus_docs,
        corpus_ids=corpus_ids,
        corpus_page_ids=corpus_urls,
    )

    embedding_artifact = _format_artifact_path(
        doc_embeddings_path,
        dataset=dataset_name,
        compression_rate=compression_rate,
    )
    full_config = CLaRaConfig.from_pretrained(
        seed_checkpoints_by_configuration["full_aria"][0][1]
    )
    training_index_sha256 = getattr(
        full_config, "aria_training_retrieval_index_sha256", None
    )
    if not isinstance(training_index_sha256, str) or len(training_index_sha256) != 64:
        raise ValueError("Full ARIA checkpoint lacks its Phase-II training BGE-index fingerprint")
    doc_embeddings, evaluation_index_sha256 = load_doc_embeddings(
        embedding_artifact,
        len(corpus_docs),
        expected_ids=corpus_ids,
        expected_hashes=corpus_hashes,
        expected_page_ids=corpus_urls,
        return_index_sha256=True,
    )
    _assert_normal_retrieval_is_not_training_index(
        full_config,
        evaluation_corpus_sha256=corpus_digest,
        evaluation_index_sha256=evaluation_index_sha256,
    )
    if doc_embeddings.shape[1] != 1024:
        raise ValueError(
            "ARIA requires BGE-large-en-v1.5 document embeddings with shape "
            f"(N, 1024), got {tuple(doc_embeddings.shape)}"
        )
    bm25_index = _BM25Index().build(corpus_docs)

    config_results: Dict[str, Dict[str, Any]] = {}
    protocol_fingerprints: List[Dict[str, Any]] = []
    rag_configurations = _rag_configurations(compression_rate)
    for config_name, rag_config in rag_configurations.items():
        print(f"\n{CONFIGURATION_LABELS[config_name]} ({dataset_name})")
        checkpoint_results: List[Dict[str, Any]] = []
        seed_checkpoints = seed_checkpoints_by_configuration[config_name]
        for training_seed, checkpoint_path in seed_checkpoints:
            print(f"  Loading seed {training_seed}: {checkpoint_path}")
            _set_inference_seed(inference_seed)
            projection_artifact = (
                None
                if config_name == "clara_baseline"
                else _projection_path(
                    checkpoint_path,
                    bge_projection_path,
                    training_seed=training_seed,
                    dataset_name=dataset_name,
                    compression_rate=compression_rate,
                )
            )
            model = _load_model(
                checkpoint_path,
                decoder_model,
                compression_rate,
                projection_artifact,
                rag_config.use_mtfrl,
                {
                    "full_aria": "full",
                    # Appendix A.31 fixed-checkpoint Forward-path-off uses the
                    # full checkpoint, rho=1 and no second retrieval.
                    "aria_ret_clara_comp": "forward_path_off",
                    "clara_baseline": "clara_baseline",
                }[config_name],
                training_seed,
                device,
            )
            protocol_fingerprints.append(model._aria_protocol_fingerprint)
            evaluator = ARIAEvaluator(
                model=model,
                corpus_docs=corpus_docs,
                corpus_ids=corpus_ids,
                corpus_page_ids=corpus_urls,
                doc_embeddings=doc_embeddings,
                use_rag_pipeline=config_name != "clara_baseline",
                rag_config=rag_config,
                bm25_index=(bm25_index if config_name != "clara_baseline" else None),
            )
            checkpoint_results.append(
                evaluator.evaluate(
                    questions=questions,
                    gold_answers=gold_answers,
                    example_ids=example_ids,
                    gold_doc_ids=gold_document_ids,
                    documents=(
                        baseline_documents if config_name == "clara_baseline" else None
                    ),
                    clara_candidate_doc_ids=(
                        baseline_candidate_doc_ids
                        if config_name == "clara_baseline"
                        else None
                    ),
                    clara_candidate_page_ids=(
                        baseline_candidate_page_ids
                        if config_name == "clara_baseline"
                        else None
                    ),
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                )
            )
        config_results[config_name] = aggregate_checkpoint_results(
            checkpoint_results,
            [seed for seed, _ in seed_checkpoints],
            [path for _, path in seed_checkpoints],
        )

    _assert_protocol_fingerprints_match(protocol_fingerprints)

    decomposition = _decompose(config_results)
    _print_decomposition(
        f"DECOMPOSITION: {dataset_name} @ {compression_rate}x", decomposition
    )
    output: Dict[str, Any] = {
        "dataset": dataset_name,
        "compression_rate": compression_rate,
        "n_examples": len(questions),
        "n_checkpoints_per_configuration": len(seeds),
        "seeds": list(seeds),
        "paper_seed_protocol": set(seeds) == PAPER_TRAINING_SEEDS and len(seeds) == 5,
        "answer_contract": EVALUATION_ANSWER_CONTRACT,
        "clara_archive_sha256": dict(_REPOSITORY_EVAL_ARCHIVE_SHA256),
        "checkpoints": {
            name: [path for _, path in pairs]
            for name, pairs in seed_checkpoints_by_configuration.items()
        },
        "checkpoint_semantics": {
            "aria_ret_clara_comp": (
                "full_aria_checkpoint_with_cfrs_acr_mtfrl_disabled_at_inference"
            )
        },
        "config_results": config_results,
        "decomposition": decomposition,
    }

    output_path = os.path.join(
        output_dir, f"decomp_{dataset_name}_cr{compression_rate}.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(output), handle, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    return output


def _cross_benchmark_result(
    dataset_results: Dict[str, Dict[str, Any]],
    seeds: Sequence[int],
    checkpoint_paths_by_configuration: Mapping[str, Sequence[str]],
    compression_rate: int,
) -> Dict[str, Any]:
    """Average benchmarks per checkpoint before computing cross-seed SD."""
    config_results: Dict[str, Dict[str, Any]] = {}
    for config_name in CONFIGURATION_LABELS:
        per_checkpoint: List[Dict[str, float]] = []
        for checkpoint_index in range(len(seeds)):
            per_checkpoint.append(
                {
                    metric: float(
                        np.mean(
                            [
                                result["config_results"][config_name]["per_seed"]
                                [checkpoint_index][metric]
                                for result in dataset_results.values()
                            ]
                        )
                    )
                    for metric in ("em", "cem", "f1")
                }
            )
        config_results[config_name] = aggregate_checkpoint_results(
            per_checkpoint,
            list(seeds),
            list(checkpoint_paths_by_configuration[config_name]),
        )
    return {
        "dataset": "avg",
        "compression_rate": compression_rate,
        "benchmark_averaging": "unweighted_within_checkpoint_then_seed_std",
        "answer_contract": EVALUATION_ANSWER_CONTRACT,
        "clara_archive_sha256": dict(_REPOSITORY_EVAL_ARCHIVE_SHA256),
        "seeds": list(seeds),
        "checkpoints": {
            name: list(paths) for name, paths in checkpoint_paths_by_configuration.items()
        },
        "config_results": config_results,
        "decomposition": _decompose(config_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA counterfactual decomposition")
    parser.add_argument(
        "--full_aria_checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Independently trained full-ARIA checkpoints aligned with --seeds",
    )
    parser.add_argument(
        "--no_coupling_checkpoints",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Deprecated compatibility argument. If supplied, every path must equal "
            "the aligned --full_aria_checkpoints path; this decomposition uses "
            "fixed-checkpoint Forward-path-off."
        ),
    )
    parser.add_argument(
        "--clara_checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Separately trained matched-protocol CLaRa checkpoints",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Training-seed identities, one per checkpoint",
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["nq", "hotpotqa", "musique", "2wikimultihopqa", "all"],
    )
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Local full-KILT Dataset/JSON artifact, or explicit hf:dataset-name",
    )
    parser.add_argument("--decoder_model", type=str, default=None)
    parser.add_argument(
        "--doc_embeddings", type=str, required=True,
        help="Aligned (N,1024) corpus embedding artifact; {dataset} and {cr} work",
    )
    parser.add_argument(
        "--bge_projection_path", type=str, default=None,
        help=(
            "Optional W_BGE template ({seed}, {dataset}, {cr}); otherwise every "
            "checkpoint must contain bge_projection.pth"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./decomp_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Scalar-answer DatasetDict created by `aria-data --stage eval`",
    )
    parser.add_argument(
        "--clara_archive_dir",
        type=str,
        required=True,
        help=(
            "External directory containing the four pinned CLaRa candidate "
            "archives: nq.zip, hotpotqa.zip, musique.zip, and 2wiki.zip"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_new_tokens", type=int, default=PAPER_MAX_NEW_TOKENS
    )
    parser.add_argument("--inference_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint_paths_by_configuration: Dict[str, Sequence[str]] = {
        "full_aria": args.full_aria_checkpoints,
        "clara_baseline": args.clara_checkpoints,
    }
    if args.no_coupling_checkpoints is not None:
        checkpoint_paths_by_configuration["aria_ret_clara_comp"] = (
            args.no_coupling_checkpoints
        )
    try:
        seed_checkpoint_sets = _validate_counterfactual_checkpoint_sets(
            checkpoint_paths_by_configuration, args.seeds
        )
        checkpoint_paths_by_configuration = {
            name: [path for _, path in pairs]
            for name, pairs in seed_checkpoint_sets.items()
        }
    except ValueError as exc:
        parser.error(str(exc))
    if args.compression_rate not in PAPER_COMPRESSION_RATES:
        parser.error("--compression_rate must be one of 4, 16, 32, 64, 128")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("--batch_size and --max_new_tokens must be positive")
    if args.max_new_tokens != PAPER_MAX_NEW_TOKENS:
        parser.error("Paper-protocol decomposition requires --max_new_tokens=64")
    datasets = (
        ["nq", "hotpotqa", "musique", "2wikimultihopqa"]
        if args.dataset == "all" else [args.dataset]
    )
    all_results: Dict[str, Dict[str, Any]] = {}
    for dataset_name in datasets:
        all_results[dataset_name] = run_counterfactual(
            checkpoint_paths_by_configuration=checkpoint_paths_by_configuration,
            seeds=args.seeds,
            dataset_name=dataset_name,
            corpus_path=args.corpus_path,
            doc_embeddings_path=args.doc_embeddings,
            bge_projection_path=args.bge_projection_path,
            decoder_model=args.decoder_model,
            compression_rate=args.compression_rate,
            max_samples=args.max_samples,
            eval_data_path=args.eval_data_path,
            clara_archive_dir=args.clara_archive_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            inference_seed=args.inference_seed,
            device=args.device,
            output_dir=args.output_dir,
        )

    if len(datasets) > 1:
        average = _cross_benchmark_result(
            all_results,
            args.seeds,
            checkpoint_paths_by_configuration,
            args.compression_rate,
        )
        all_results["avg"] = average
        _print_decomposition(
            f"CROSS-BENCHMARK DECOMPOSITION @ {args.compression_rate}x",
            average["decomposition"],
        )
        output_path = os.path.join(
            args.output_dir, f"decomp_all_cr{args.compression_rate}.json"
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(all_results), handle, indent=2, ensure_ascii=False)
        print(f"\nCombined results saved to {output_path}")


if __name__ == "__main__":
    main()
