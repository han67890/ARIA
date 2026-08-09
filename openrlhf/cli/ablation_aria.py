#!/usr/bin/env python3
"""Run the component and coupling ablations reported for ARIA.

Training-seed statistics are computed from distinct checkpoints. The five
retrieval-stage ablations are fixed-checkpoint interventions and therefore use
the aligned Full-ARIA checkpoint for each seed; matched retraining labels use
their own independently trained checkpoint for each seed. One checkpoint is
never re-run under several RNG seeds and presented as a multi-seed experiment.

Example:
    python -m openrlhf.cli.ablation_aria \
        --ablation remove_igfr --dataset all --compression_rate 16 \
        --checkpoint_paths /checkpoints/aria_42 /checkpoints/aria_123 \
        --seeds 42 123 --corpus_path /data/kilt_corpus.jsonl \
        --eval_data_path /data/aria/eval \
        --doc_embeddings /artifacts/kilt_bge.pt
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
    EVALUATION_ANSWER_ALIAS_CONTRACT,
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
    _extract_gold_answers,
    _format_artifact_path,
    _required_checkpoint_configuration,
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
    COUPLING_CONTROL_PROTOCOL,
    CLaRa,
    CLaRaConfig,
    RAG_CONFIGURATION_SPECS,
    RAGPipelineConfig,
    STATIC_SECOND_QUERY_SCHEME,
    UNIFORM_BUDGET_ALLOCATION_SCHEME,
    _BM25Index,
    create_paper_rag_config,
    _tensor_is_finite_in_chunks,
)
ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = RAG_CONFIGURATION_SPECS


def create_rag_config(
    ablation_name: str,
    compression_rate: int,
    **overrides: Any,
) -> RAGPipelineConfig:
    """Create an explicit stage configuration for one paper ablation."""
    return create_paper_rag_config(
        ablation_name, compression_rate, **overrides
    )


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
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            "Each training seed must reference a distinct checkpoint path; a single "
            "checkpoint cannot be reported as a multi-seed experiment"
        )
    return list(zip(seeds, expanded))


def _set_inference_seed(seed: int) -> None:
    """Set one shared decoding seed; this is not a training-seed replicate."""
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
            "--bge_projection_path to supply the fitted W_BGE explicitly"
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


def run_ablation(
    checkpoint_paths: Sequence[str],
    seeds: Sequence[int],
    ablation_name: str,
    dataset_name: str,
    corpus_path: Optional[str],
    doc_embeddings_path: Optional[str],
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
) -> Dict[str, Any]:
    """Evaluate one ablation on one dataset, once per trained checkpoint.

    Matched-control labels require their own independently retrained checkpoint.
    Retrieval-stage, ``forward_path_off``, and ``fixed_*`` labels instead
    validate against the aligned Full-ARIA checkpoint and change only the
    inference forward path.
    """
    seed_checkpoints = _validate_seed_checkpoints(checkpoint_paths, seeds)
    print(f"\n{'=' * 60}")
    print(f"Ablation: {ablation_name}")
    print(f"  Dataset: {dataset_name}, CR: {compression_rate}x")
    print(f"  Independent checkpoints: {len(seed_checkpoints)}")
    print(f"{'=' * 60}")

    dataset, question_key, answer_key = load_eval_dataset(
        dataset_name,
        max_samples,
        eval_data_path,
        require_clara_archive=ablation_name == "clara_baseline",
        clara_archive_dir=clara_archive_dir,
    )
    questions = [item[question_key] for item in dataset]
    gold_answers = [_extract_gold_answers(item, answer_key) for item in dataset]
    example_ids = _extract_example_ids(dataset, dataset_name)
    baseline_documents: Optional[List[List[str]]] = None
    baseline_candidate_doc_ids: Optional[List[List[str]]] = None
    baseline_candidate_page_ids: Optional[List[List[str]]] = None
    baseline_gold_candidate_indices: Optional[List[List[int]]] = None
    if ablation_name == "clara_baseline":
        (
            baseline_documents,
            baseline_candidate_doc_ids,
            baseline_candidate_page_ids,
            baseline_gold_candidate_indices,
        ) = _extract_clara_candidate_columns(dataset)

    rag_config = create_rag_config(ablation_name, compression_rate)
    corpus_docs: List[str] = []
    corpus_ids: List[str] = []
    corpus_urls: List[str] = []
    corpus_digest: Optional[str] = None
    doc_embeddings: Optional[torch.Tensor] = None
    bm25_index: Optional[_BM25Index] = None
    if ablation_name != "clara_baseline":
        if corpus_path is None or doc_embeddings_path is None:
            raise ValueError("ARIA ablations require corpus_path and doc_embeddings_path")
        checkpoint_config = CLaRaConfig.from_pretrained(seed_checkpoints[0][1])
        training_index_sha256 = getattr(
            checkpoint_config, "aria_training_retrieval_index_sha256", None
        )
        if not isinstance(training_index_sha256, str) or len(training_index_sha256) != 64:
            raise ValueError("Checkpoint lacks its Phase-II training BGE-index fingerprint")
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

        embedding_artifact = _format_artifact_path(
            doc_embeddings_path,
            dataset=dataset_name,
            compression_rate=compression_rate,
        )
        doc_embeddings, evaluation_index_sha256 = load_doc_embeddings(
            embedding_artifact,
            len(corpus_docs),
            expected_ids=corpus_ids,
            expected_hashes=corpus_hashes,
            expected_page_ids=corpus_urls,
            return_index_sha256=True,
        )
        _assert_normal_retrieval_is_not_training_index(
            checkpoint_config,
            evaluation_corpus_sha256=corpus_digest,
            evaluation_index_sha256=evaluation_index_sha256,
        )
        if doc_embeddings.shape[1] != 1024:
            raise ValueError(
                "ARIA requires BGE-large-en-v1.5 document embeddings with shape "
                f"(N, 1024), got {tuple(doc_embeddings.shape)}"
            )
        bm25_index = _BM25Index().build(corpus_docs)

    checkpoint_results: List[Dict[str, Any]] = []
    protocol_fingerprints: List[Dict[str, Any]] = []
    for training_seed, checkpoint_path in seed_checkpoints:
        print(f"Loading seed {training_seed} checkpoint: {checkpoint_path}")
        _set_inference_seed(inference_seed)
        overrides: Dict[str, Any] = {
            "pure_inference": True,
        }
        model = CLaRa.from_pretrained(
            checkpoint_path,
            strict_aria_artifacts=True,
            external_bge_artifact=bge_projection_path is not None,
            **overrides,
        )
        if decoder_model is not None and decoder_model != model.decoder_model_name:
            raise ValueError(
                "decoder_model may only assert the checkpoint's exact backbone"
            )
        if ablation_name != "clara_baseline":
            projection_artifact = _projection_path(
                checkpoint_path,
                bge_projection_path,
                training_seed=training_seed,
                dataset_name=dataset_name,
                compression_rate=compression_rate,
            )
            load_bge_projection(model, projection_artifact, expected_output_dim=1024)
        protocol_fingerprints.append(
            _validate_checkpoint_protocol(
                model,
                checkpoint_path,
                training_seed,
                compression_rate,
                ablation_name,
            )
        )
        if rag_config.use_mtfrl:
            _load_mtfrl_projection_strict(model, checkpoint_path)
        model = model.to(device)
        model.eval()

        evaluator = ARIAEvaluator(
            model=model,
            corpus_docs=corpus_docs,
            corpus_ids=corpus_ids,
            corpus_page_ids=corpus_urls,
            doc_embeddings=doc_embeddings,
            use_rag_pipeline=ablation_name != "clara_baseline",
            rag_config=rag_config,
            bm25_index=bm25_index,
        )
        checkpoint_results.append(
            evaluator.evaluate(
                questions=questions,
                gold_answers=gold_answers,
                example_ids=example_ids,
                documents=baseline_documents,
                clara_candidate_doc_ids=baseline_candidate_doc_ids,
                clara_candidate_page_ids=baseline_candidate_page_ids,
                clara_gold_candidate_indices=baseline_gold_candidate_indices,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
        )

    _assert_protocol_fingerprints_match(protocol_fingerprints)

    aggregated = aggregate_checkpoint_results(
        checkpoint_results,
        [seed for seed, _ in seed_checkpoints],
        [path for _, path in seed_checkpoints],
    )
    aggregated["rag_configuration"] = ablation_name
    aggregated["answer_alias_contract"] = EVALUATION_ANSWER_ALIAS_CONTRACT
    aggregated["clara_archive_sha256"] = (
        dict(_REPOSITORY_EVAL_ARCHIVE_SHA256)
        if ablation_name == "clara_baseline"
        else None
    )
    aggregated["checkpoint_rag_configuration"] = (
        _required_checkpoint_configuration(ablation_name)
    )
    aggregated["coupling_control"] = {
        "protocol": COUPLING_CONTROL_PROTOCOL,
        "acr_allocation_mode": rag_config.acr_allocation_mode,
        "second_retrieval_mode": rag_config.second_retrieval_mode,
        "uniform_evidence_token_budget": (
            rag_config.uniform_evidence_token_budget
            if rag_config.acr_allocation_mode == "uniform_budget"
            else None
        ),
        "uniform_allocation_scheme": (
            UNIFORM_BUDGET_ALLOCATION_SCHEME
            if rag_config.acr_allocation_mode == "uniform_budget"
            else None
        ),
        "static_second_query_scheme": (
            STATIC_SECOND_QUERY_SCHEME
            if rag_config.second_retrieval_mode == "static_query"
            else None
        ),
    }
    aggregated["paper_seed_protocol"] = (
        set(seeds) == PAPER_TRAINING_SEEDS and len(seeds) == 5
    )
    return aggregated


def _cross_benchmark_average(
    dataset_results: Dict[str, Dict[str, Any]],
    checkpoint_paths: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Average benchmarks within each checkpoint, then compute seed SD."""
    per_checkpoint: List[Dict[str, float]] = []
    for checkpoint_index in range(len(checkpoint_paths)):
        per_checkpoint.append(
            {
                metric: float(
                    np.mean(
                        [
                            result["per_seed"][checkpoint_index][metric]
                            for result in dataset_results.values()
                        ]
                    )
                )
                for metric in ("em", "cem", "f1")
            }
        )
    return aggregate_checkpoint_results(
        per_checkpoint, list(seeds), list(checkpoint_paths)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA ablation experiments")
    parser.add_argument(
        "--checkpoint_paths", "--model_paths", dest="checkpoint_paths",
        type=str, nargs="+", required=True,
        help=(
            "Distinct checkpoints aligned with --seeds. Matched-control labels "
            "require their own trained checkpoints; retrieval-stage, "
            "forward_path_off, and fixed_* interventions require the corresponding "
            "full-ARIA checkpoints."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", required=True,
        help="Training-seed identities, one per checkpoint",
    )
    parser.add_argument(
        "--ablation", type=str, required=True, choices=list(ABLATION_CONFIGS)
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["nq", "hotpotqa", "musique", "2wikimultihopqa", "all"],
    )
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument(
        "--corpus_path",
        type=str,
        help="Local full-KILT Dataset/JSON artifact, or explicit hf:dataset-name",
    )
    parser.add_argument("--decoder_model", type=str, default=None)
    parser.add_argument(
        "--doc_embeddings", type=str,
        help="Aligned (N,1024) corpus embedding artifact; {dataset} and {cr} work",
    )
    parser.add_argument(
        "--bge_projection_path", type=str, default=None,
        help=(
            "Optional W_BGE artifact template ({seed}, {dataset}, {cr}); otherwise "
            "each checkpoint must contain bge_projection.pth"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Alias-complete DatasetDict created by `aria-data --stage eval`",
    )
    parser.add_argument(
        "--clara_archive_dir",
        type=str,
        default=None,
        help=(
            "External directory containing nq.zip, hotpotqa.zip, musique.zip, "
            "and 2wiki.zip; required only for --ablation clara_baseline"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_new_tokens", type=int, default=PAPER_MAX_NEW_TOKENS
    )
    parser.add_argument("--inference_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    try:
        seed_checkpoints = _validate_seed_checkpoints(args.checkpoint_paths, args.seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.compression_rate not in PAPER_COMPRESSION_RATES:
        parser.error("--compression_rate must be one of 4, 16, 32, 64, 128")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("--batch_size and --max_new_tokens must be positive")
    if args.max_new_tokens != PAPER_MAX_NEW_TOKENS:
        parser.error("Paper-protocol ablation requires --max_new_tokens=64")
    if args.ablation != "clara_baseline" and (
        args.corpus_path is None or args.doc_embeddings is None
    ):
        parser.error("non-CLaRa ablations require --corpus_path and --doc_embeddings")
    if args.ablation == "clara_baseline" and args.clara_archive_dir is None:
        parser.error("--ablation clara_baseline requires --clara_archive_dir")
    if args.ablation != "clara_baseline" and args.clara_archive_dir is not None:
        parser.error("--clara_archive_dir is valid only for --ablation clara_baseline")
    os.makedirs(args.output_dir, exist_ok=True)
    datasets = (
        ["nq", "hotpotqa", "musique", "2wikimultihopqa"]
        if args.dataset == "all" else [args.dataset]
    )
    all_results: Dict[str, Dict[str, Any]] = {}
    for dataset_name in datasets:
        result = run_ablation(
            checkpoint_paths=[path for _, path in seed_checkpoints],
            seeds=[seed for seed, _ in seed_checkpoints],
            ablation_name=args.ablation,
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
        )
        all_results[dataset_name] = result
        print(
            f"  {dataset_name}: EM={result['mean']['em'] * 100:.2f}%, "
            f"CEM={result['mean']['cem'] * 100:.2f}%, "
            f"F1={result['mean']['f1'] * 100:.2f}%"
        )

    if len(datasets) > 1:
        average = _cross_benchmark_average(
            all_results,
            [path for _, path in seed_checkpoints],
            [seed for seed, _ in seed_checkpoints],
        )
        all_results["avg"] = average
        print(
            f"\n  Avg: EM={average['mean']['em'] * 100:.2f}% "
            f"(+/-{average['std']['em'] * 100:.2f}), "
            f"CEM={average['mean']['cem'] * 100:.2f}% "
            f"(+/-{average['std']['cem'] * 100:.2f}), "
            f"F1={average['mean']['f1'] * 100:.2f}% "
            f"(+/-{average['std']['f1'] * 100:.2f})"
        )

    output_path = os.path.join(
        args.output_dir,
        f"ablation_{args.ablation}_{args.dataset}_cr{args.compression_rate}.json",
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(
                {
                    "ablation": args.ablation,
                    "compression_rate": args.compression_rate,
                    "seeds": [seed for seed, _ in seed_checkpoints],
                    "checkpoints": [path for _, path in seed_checkpoints],
                    "results": all_results,
                }
            ),
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
