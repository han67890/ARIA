#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Modified for ARIA in 2026.
# Copyright (C) 2026 Yiheng Han (ARIA modifications only).
#

"""
ARIA/CLaRa Training Script

This script handles ARIA's two paper training phases and the matched CLaRa baseline.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import re
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.compute as pc
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers.trainer import get_scheduler

from openrlhf.datasets import SFTDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.sft_trainer import SFTTrainer
from openrlhf.utils import get_strategy, get_tokenizer
from openrlhf.models.modeling_aria import (
    CLARA_DOCUMENT_REPRESENTATION_SCHEME,
    CLARA_ARCHIVE_DOCUMENT_ID_SCHEME,
    CLARA_ARCHIVE_PAGE_ID_SCHEME,
    CLARA_EVALUATION_CANDIDATE_PROTOCOL,
    CLARA_MEMORY_ALLOCATION_SCHEME,
    CLARA_PHASE2_OBJECTIVE,
    CLARA_SELECTOR_SCHEME,
    COUPLING_CONTROL_PROTOCOL,
    FIXED_CHECKPOINT_CONFIGURATIONS,
    MATCHED_EVIDENCE_TOKEN_BUDGET,
    MATCHED_RETRAINING_CONFIGURATIONS,
    RAG_CONFIGURATION_SPECS,
    STATIC_SECOND_QUERY_SCHEME,
    UNIFORM_BUDGET_ALLOCATION_SCHEME,
    CLaRaConfig,
    CLaRa,
    QR_INPUT_SCHEME,
    RAGPipelineConfig,
    create_paper_rag_config,
    _tensor_is_finite_in_chunks,
)
from openrlhf.datasets.sft_dataset import make_collate_fn
from openrlhf.utils.aria_provenance import (
    CORPUS_SHA256_SCHEME,
    CORPUS_ID_FIELDS as _ID_FIELDS,
    CORPUS_TEXT_FIELDS as _CORPUS_TEXT_FIELDS,
    CORPUS_URL_FIELDS as _URL_FIELDS,
    TEXT_SHA256_SCHEME,
    build_source_snapshot_manifest,
    canonical_page_url as _canonical_page_url,
    corpus_id as _corpus_id,
    corpus_page_url as _corpus_page_url,
    corpus_sha256 as _corpus_sha256,
    corpus_text as _corpus_text,
    text_sha256 as _text_sha256,
)
_QUERY_TEXT_FIELDS = ("query", "question", "text")
_PHASE1_TOTAL = 7_808_465
_PHASE2_ROWS_PER_EPOCH = 38_400
_PHASE2_ROWS_PER_BENCHMARK = 9_600
_PAPER_TRAINING_SEEDS = {42, 123, 456, 789, 2024}
_PAPER_PASSAGE_MAX_LENGTH = 768
_PAPER_QUERY_MAX_LENGTH = 256
_PAPER_PHASE_LENGTHS = {
    "stage1": {"input": 2_048, "target": 512},
    "stage2": {"input": 1_024, "target": 128},
}
_PAPER_PHASE2_LOSS_WEIGHTS = {
    "lambda_mse": 0.10,
    "lambda_cfrs": 0.10,
    "lambda_qr": 0.05,
    "lambda_mtfrl": 0.05,
}
_PHASE1_SOURCE_COUNTS = {
    "simpleqa": 2_000_000,
    "complexqa": 2_000_000,
    "paraphrase": 1_966_291,
    "entity_augmented": 1_842_174,
}
_PHASE1_DATA_TYPE_COUNTS = {
    "simple_qa": _PHASE1_SOURCE_COUNTS["simpleqa"],
    "complex_qa": _PHASE1_SOURCE_COUNTS["complexqa"],
    "paraphrase": _PHASE1_SOURCE_COUNTS["paraphrase"],
    "entity_augmented": _PHASE1_SOURCE_COUNTS["entity_augmented"],
}
_RAG_CONFIGURATION_SWITCHES: Dict[str, Dict[str, Any]] = RAG_CONFIGURATION_SPECS


class _ScheduledEpochSFTDataset(TorchDataset):
    """Expose the five Appendix-A.33 Phase-II samples one epoch at a time.

    All views have the same length, so the distributed sampler and scheduler
    can be constructed once while ``SFTTrainer`` switches the active view at
    each epoch boundary.
    """

    def __init__(self, dataset_dict: DatasetDict, tokenizer, max_length: int, strategy):
        names = sorted(name for name in dataset_dict if name.startswith("epoch_"))
        if names != [f"epoch_{index:03d}" for index in range(5)]:
            raise ValueError(
                "ARIA Phase II requires exactly epoch_000 ... epoch_004 in its DatasetDict"
            )
        self._epochs = [
            SFTDataset(dataset_dict[name], tokenizer, max_length, strategy)
            for name in names
        ]
        lengths = {len(dataset) for dataset in self._epochs}
        if lengths != {_PHASE2_ROWS_PER_EPOCH}:
            raise ValueError(
                f"Each Phase-II epoch must contain exactly {_PHASE2_ROWS_PER_EPOCH:,} rows; "
                f"got {sorted(lengths)}"
            )
        for name in names:
            counts = Counter(str(value) for value in dataset_dict[name]["benchmark"])
            expected_counts = {
                benchmark: _PHASE2_ROWS_PER_BENCHMARK
                for benchmark in ("nq", "hotpotqa", "musique", "2wikimultihopqa")
            }
            if dict(counts) != expected_counts:
                raise ValueError(f"{name} is not class-balanced at 9,600 rows: {counts}")
        self._active_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not 0 <= epoch < len(self._epochs):
            raise ValueError(
                f"Phase-II scheduled dataset has {len(self._epochs)} views, requested epoch {epoch}"
            )
        self._active_epoch = epoch

    def __len__(self) -> int:
        return len(self._epochs[0])

    def __getitem__(self, index: int):
        return self._epochs[self._active_epoch][index]


def _read_local_json_records(
    path: str,
    *,
    container_keys: Sequence[str],
    text_fields: Sequence[str],
) -> List[Tuple[Optional[str], Any]]:
    """Read records directly from a declared local JSON/JSONL artifact."""
    source = Path(path)
    if source.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"Expected a local .json or .jsonl file, got: {path}")

    if source.suffix.lower() == ".jsonl":
        records: List[Tuple[Optional[str], Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append((None, json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL record at {path}:{line_number}: {exc}"
                    ) from exc
        return records

    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in container_keys:
            if key in payload:
                payload = payload[key]
                break

    if isinstance(payload, list):
        return [(None, item) for item in payload]
    if isinstance(payload, dict):
        if any(field in payload for field in text_fields):
            return [(None, payload)]
        # Also support the common {document_id: text_or_record} representation.
        return [(str(record_id), item) for record_id, item in payload.items()]
    raise ValueError(f"JSON structure in {path} must be an array or object")


def _load_text_records(
    path: str,
    *,
    container_keys: Sequence[str],
    text_fields: Sequence[str],
) -> Tuple[List[str], List[str]]:
    records = _read_local_json_records(
        path, container_keys=container_keys, text_fields=text_fields
    )
    texts: List[str] = []
    record_ids: List[str] = []
    for index, (mapping_id, record) in enumerate(records):
        if isinstance(record, str):
            text = record
            record_id = mapping_id or str(index)
        elif isinstance(record, dict):
            text = next(
                (
                    record[field]
                    for field in text_fields
                    if isinstance(record.get(field), str) and record[field].strip()
                ),
                None,
            )
            if text is None:
                raise ValueError(
                    f"Record {index} in {path} has none of the text fields "
                    f"{tuple(text_fields)}"
                )
            record_id = next(
                (
                    str(record[field])
                    for field in _ID_FIELDS
                    if record.get(field) is not None
                ),
                mapping_id or str(index),
            )
        else:
            raise ValueError(
                f"Record {index} in {path} must be a string or JSON object, "
                f"got {type(record).__name__}"
            )

        text = text.strip()
        if not text:
            raise ValueError(f"Record {index} in {path} contains empty text")
        texts.append(text)
        record_ids.append(record_id)

    if not texts:
        raise ValueError(f"No records found in {path}")
    return texts, record_ids


def _url_set_sha256(urls: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(urls)).encode("utf-8")).hexdigest()


def load_corpus(path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load corpus text, stable IDs, canonical URLs, and alignment hashes."""
    records = _read_local_json_records(
        path,
        container_keys=("documents", "corpus", "passages", "data"),
        text_fields=_CORPUS_TEXT_FIELDS,
    )
    documents: List[str] = []
    document_ids: List[str] = []
    page_urls: List[str] = []
    for index, (_, record) in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                "Paper-protocol corpus rows must contain text, stable ID, and page_url"
            )
        location = f"corpus row {index}"
        text = _corpus_text(record, location=location)
        record_id = _corpus_id(record, location=location)
        documents.append(text)
        document_ids.append(record_id)
        page_urls.append(_corpus_page_url(record, location=location))
    if len(set(document_ids)) != len(document_ids):
        raise ValueError(f"Corpus document IDs must be unique: {path}")
    if len(set(page_urls)) != len(page_urls):
        raise ValueError(
            "Phase-II training retrieval corpus must be page-URL deduplicated: "
            f"{path}"
        )
    return (
        documents,
        document_ids,
        page_urls,
        [_text_sha256(text) for text in documents],
    )


def load_alignment_pairs(
    path: str,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Load provenance-linked KILT (passage, question) pairs for W_BGE fitting."""
    records = _read_local_json_records(
        path,
        container_keys=("pairs", "queries", "questions", "data"),
        text_fields=_QUERY_TEXT_FIELDS,
    )
    queries: List[str] = []
    passage_ids: List[str] = []
    page_urls: List[str] = []
    passage_hashes: List[str] = []
    for index, (mapping_id, record) in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                "W_BGE alignment rows must contain question, passage, passage ID, and page_url"
            )
        query = next(
            (
                record[field].strip()
                for field in _QUERY_TEXT_FIELDS
                if isinstance(record.get(field), str) and record[field].strip()
            ),
            None,
        )
        passage = next(
            (
                record[field].strip()
                for field in _CORPUS_TEXT_FIELDS
                if field != "text"
                and isinstance(record.get(field), str)
                and record[field].strip()
            ),
            None,
        )
        if query is None or passage is None:
            raise ValueError(
                f"W_BGE alignment row {index} requires question and passage text"
            )
        passage_id = next(
            (
                str(record[field])
                for field in ("passage_id", "doc_id", "document_id", "id")
                if record.get(field) is not None
            ),
            mapping_id,
        )
        if not passage_id:
            raise ValueError(f"W_BGE alignment row {index} requires a stable passage ID")
        raw_url = next((record[field] for field in _URL_FIELDS if record.get(field)), None)
        queries.append(query)
        passage_ids.append(passage_id)
        page_urls.append(
            _canonical_page_url(raw_url, location=f"W_BGE alignment row {index}.page_url")
        )
        passage_hashes.append(_text_sha256(passage))
    return queries, passage_ids, page_urls, passage_hashes


def _extract_tensor(payload: Any, path: str, preferred_keys: Sequence[str]) -> torch.Tensor:
    if isinstance(payload, np.ndarray):
        return torch.from_numpy(payload)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            if isinstance(value, torch.Tensor):
                return value
        tensor_values = [value for value in payload.values() if isinstance(value, torch.Tensor)]
        if len(tensor_values) == 1:
            return tensor_values[0]
    raise ValueError(
        f"Embedding artifact {path} must store a tensor directly or use one of "
        f"the keys {tuple(preferred_keys)}"
    )


def load_bge_embeddings(
    path: str,
    *,
    expected_rows: Optional[int] = None,
    expected_ids: Optional[Sequence[str]] = None,
    expected_hashes: Optional[Sequence[str]] = None,
    expected_page_ids: Optional[Sequence[str]] = None,
    expected_index_sha256: Optional[str] = None,
    return_metadata: bool = False,
) -> Any:
    """Load BGE vectors and, for a corpus, prove exact row alignment."""
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        payload: Any = np.load(path, allow_pickle=False)
    elif suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        payload = {key: archive[key] for key in archive.files}
    else:
        raise ValueError(
            f"BGE embeddings must be a local .pt/.pth/.npy/.npz file, got: {path}"
        )

    embeddings = _extract_tensor(
        payload,
        path,
        preferred_keys=(
            "doc_embeddings",
            "embeddings",
            "bge_embeddings",
            "passage_embeddings",
            "target_embeddings",
        ),
    ).detach().cpu()
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise ValueError(
            f"{path} must contain BGE-large-en-v1.5 embeddings with shape (N, 1024), "
            f"got {tuple(embeddings.shape)}"
        )
    if expected_rows is not None and embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding/corpus row mismatch: {embeddings.shape[0]} != {expected_rows}"
        )
    if not _tensor_is_finite_in_chunks(embeddings):
        raise ValueError(f"Embedding artifact contains NaN or infinity: {path}")

    if expected_ids is not None:
        if not isinstance(payload, dict):
            raise ValueError(
                "Corpus embeddings must include document_ids metadata; a bare tensor "
                "cannot prove row alignment"
            )
        artifact_ids = payload.get("document_ids", payload.get("doc_ids"))
        if isinstance(artifact_ids, np.ndarray):
            artifact_ids = artifact_ids.tolist()
        if artifact_ids is None or [str(value) for value in artifact_ids] != list(expected_ids):
            raise ValueError("Embedding document_ids do not exactly match corpus row order")
    if expected_hashes is not None:
        if not isinstance(payload, dict):
            raise ValueError("Corpus embeddings must include text_sha256 metadata")
        artifact_hashes = payload.get("text_sha256", payload.get("document_sha256"))
        if isinstance(artifact_hashes, np.ndarray):
            artifact_hashes = artifact_hashes.tolist()
        if artifact_hashes is None or list(artifact_hashes) != list(expected_hashes):
            raise ValueError("Embedding text_sha256 values do not match corpus row order")
    if expected_page_ids is not None:
        if not isinstance(payload, dict):
            raise ValueError("Corpus embeddings must include page_urls metadata")
        artifact_page_ids = payload.get("page_urls", payload.get("page_ids"))
        if isinstance(artifact_page_ids, np.ndarray):
            artifact_page_ids = artifact_page_ids.tolist()
        if artifact_page_ids is None or list(artifact_page_ids) != list(
            expected_page_ids
        ):
            raise ValueError("Embedding page_urls do not exactly match corpus row order")
    if (
        expected_ids is not None
        or expected_hashes is not None
        or expected_page_ids is not None
    ):
        if not isinstance(payload, dict):
            raise ValueError("BGE embeddings must include checkpoint provenance metadata")
        if payload.get("bge_model") != "BAAI/bge-large-en-v1.5":
            raise ValueError(
                "Embedding artifact must declare bge_model='BAAI/bge-large-en-v1.5'"
            )
        if payload.get("text_sha256_scheme") != TEXT_SHA256_SCHEME:
            raise ValueError(
                f"Embedding artifact must declare text_sha256_scheme="
                f"{TEXT_SHA256_SCHEME!r}"
            )
    embeddings = embeddings.to(dtype=torch.float32).contiguous()
    index_hasher = hashlib.sha256()
    index_hasher.update(
        json.dumps(
            {"shape": list(embeddings.shape), "dtype": "float32"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for start in range(0, embeddings.shape[0], 16_384):
        index_hasher.update(
            embeddings[start : start + 16_384].contiguous().numpy().tobytes()
        )
    computed_index_sha256 = index_hasher.hexdigest()
    declared_index_sha256 = payload.get("index_sha256") if isinstance(payload, dict) else None
    if declared_index_sha256 != computed_index_sha256:
        raise ValueError("Embedding artifact index_sha256 does not match its tensor")
    if (
        expected_index_sha256 is not None
        and computed_index_sha256 != expected_index_sha256
    ):
        raise ValueError("Embedding tensor does not match the Phase-II BGE index digest")
    provenance: Dict[str, Any] = {}
    if isinstance(payload, dict):
        artifact_format = payload.get("artifact_format")
        if artifact_format not in (None, "aria-bge-artifact-v1", "aria-bge-artifact-v2"):
            raise ValueError(f"Unsupported BGE artifact format: {artifact_format!r}")
        if artifact_format is not None:
            provenance["bge_embedding_artifact_format"] = artifact_format
        if artifact_format == "aria-bge-artifact-v2":
            source_kind = payload.get("encoder_source_kind")
            source = payload.get("encoder_source")
            if source_kind == "huggingface-hub":
                resolved = payload.get("encoder_revision_resolved")
                if (
                    not isinstance(source, str)
                    or not source
                    or not isinstance(resolved, str)
                    or re.fullmatch(r"[0-9a-fA-F]{40}", resolved) is None
                ):
                    raise ValueError(
                        "BGE v2 Hub artifact requires its source and exact resolved commit"
                    )
            elif source_kind == "local-directory":
                digest = payload.get("encoder_source_sha256")
                if (
                    not isinstance(source, str)
                    or not source
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    raise ValueError(
                        "BGE v2 local artifact requires its source and tree SHA-256"
                    )
            else:
                raise ValueError("BGE v2 artifact has an invalid encoder_source_kind")
            for key in (
                "encoder_source",
                "encoder_source_kind",
                "encoder_revision_declared",
                "encoder_revision_resolved",
                "encoder_revision_was_explicit",
                "encoder_source_sha256",
                "encoder_source_sha256_scheme",
            ):
                provenance[f"bge_{key}"] = payload.get(key)
    return (embeddings, provenance) if return_metadata else embeddings


def _load_test_url_file(path: str) -> set[str]:
    urls: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                urls.add(_canonical_page_url(line, location=f"{path}:{line_number}"))
    if not urls:
        raise ValueError("The official-test URL artifact is empty")
    return urls


def load_bge_projection(
    model: CLaRa,
    path: str,
    expected_base_model: str,
    expected_test_url_sha256: str,
) -> None:
    """Strictly load the fitted W_BGE state before optimizer construction."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("W_BGE artifact must include its fitting-protocol metadata")
    metadata = {
        key: payload.get(key)
        for key in (
            "base_model",
            "base_model_revision_declared",
            "base_model_revision_resolved",
            "bge_model",
            "sample_count",
            "epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "query_sha256",
            "passage_id_sha256",
            "passage_text_sha256",
            "test_url_sha256",
            "text_sha256_scheme",
            "qr_input_scheme",
            "bge_embedding_artifact_format",
            "bge_encoder_source",
            "bge_encoder_source_kind",
            "bge_encoder_revision_declared",
            "bge_encoder_revision_resolved",
            "bge_encoder_revision_was_explicit",
            "bge_encoder_source_sha256",
            "bge_encoder_source_sha256_scheme",
        )
    }
    expected_metadata = {
        "base_model": expected_base_model,
        "bge_model": "BAAI/bge-large-en-v1.5",
        "sample_count": 50_000,
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "test_url_sha256": expected_test_url_sha256,
        "text_sha256_scheme": TEXT_SHA256_SCHEME,
        "qr_input_scheme": QR_INPUT_SCHEME,
    }
    expected_base_revision = getattr(
        model.config, "decoder_model_resolved_revision", None
    )
    if expected_base_revision is not None:
        if metadata.get("base_model_revision_resolved") != expected_base_revision:
            raise ValueError(
                "W_BGE base-model revision does not match the loaded decoder: "
                f"{metadata.get('base_model_revision_resolved')!r} != "
                f"{expected_base_revision!r}"
            )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"W_BGE metadata {key!r} must be {expected!r}, got {metadata.get(key)!r}"
            )
    if any(
        not isinstance(metadata[key], str) or len(metadata[key]) != 64
        for key in ("query_sha256", "passage_id_sha256", "passage_text_sha256")
    ):
        raise ValueError("W_BGE metadata requires alignment-artifact fingerprints")
    if not isinstance(metadata["seed"], int):
        raise ValueError("W_BGE metadata requires its fitting seed")
    if metadata.get("bge_embedding_artifact_format") == "aria-bge-artifact-v2":
        source_kind = metadata.get("bge_encoder_source_kind")
        if source_kind == "huggingface-hub":
            resolved = metadata.get("bge_encoder_revision_resolved")
            if not isinstance(resolved, str) or re.fullmatch(
                r"[0-9a-fA-F]{40}", resolved
            ) is None:
                raise ValueError("W_BGE metadata requires its resolved BGE Hub commit")
        elif source_kind == "local-directory":
            digest = metadata.get("bge_encoder_source_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("W_BGE metadata requires its local BGE tree SHA-256")
        else:
            raise ValueError("W_BGE metadata has an invalid BGE encoder source kind")
    if isinstance(payload, dict):
        for wrapper_key in ("state_dict", "bge_projection", "W_BGE"):
            wrapped = payload.get(wrapper_key)
            if isinstance(wrapped, dict):
                payload = wrapped
                break

    weight = None
    if isinstance(payload, torch.Tensor):
        weight = payload
    elif isinstance(payload, dict):
        for key in (
            "weight",
            "_bge_projection.weight",
            "bge_projection.weight",
            "module._bge_projection.weight",
        ):
            if isinstance(payload.get(key), torch.Tensor):
                weight = payload[key]
                break
    if weight is None:
        raise ValueError(f"W_BGE state in {path} must contain a projection weight")

    expected_shape = (1024, model.hidden_size)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"W_BGE weight must have shape {expected_shape}, got {tuple(weight.shape)}"
        )
    if not torch.isfinite(weight).all():
        raise ValueError(f"W_BGE state contains NaN or infinity: {path}")

    model.setup_bge_projection(bge_dim=1024, freeze=True)
    model._bge_projection.load_state_dict(
        {"weight": weight.detach().cpu().to(dtype=torch.float32)}, strict=True
    )
    model._bge_projection.eval()
    for parameter in model._bge_projection.parameters():
        parameter.requires_grad = False
    model._bge_projection_metadata = metadata


def setup_phase2_artifacts(args: argparse.Namespace, model: CLaRa) -> None:
    """Attach the de-duplicated *training* retrieval stack before AdamW.

    This corpus intentionally excludes pages from the official test URL set.
    Its provenance is checkpointed as training-only metadata; Normal retrieval
    later runs against a separately validated full-KILT corpus/index.
    """
    corpus_docs, corpus_ids, corpus_urls, corpus_hashes = load_corpus(args.corpus_path)
    test_urls = _load_test_url_file(args.test_url_file)
    model._aria_test_url_sha256 = _url_set_sha256(test_urls)
    model.config.aria_test_url_sha256 = model._aria_test_url_sha256
    model.config.aria_text_sha256_scheme = TEXT_SHA256_SCHEME
    model._aria_corpus_metadata = {
        document_id: {"text_sha256": text_hash, "page_url": page_url}
        for document_id, text_hash, page_url in zip(
            corpus_ids, corpus_hashes, corpus_urls
        )
    }
    model.config.aria_training_corpus_sha256 = _corpus_sha256(
        corpus_ids, corpus_hashes, corpus_urls
    )
    model.config.aria_training_corpus_count = len(corpus_ids)
    model.config.aria_training_corpus_sha256_scheme = CORPUS_SHA256_SCHEME
    model.config.aria_training_corpus_scope = "page_url_deduplicated"

    with (Path(args.dataset) / "aria_manifest.json").open("r", encoding="utf-8") as handle:
        phase2_manifest = json.load(handle)
    training_retrieval = phase2_manifest.get("training_retrieval", {})
    expected_index_sha256 = training_retrieval.get("index_sha256")
    candidate_order_sha256 = training_retrieval.get("candidate_order_sha256")
    if (
        not isinstance(expected_index_sha256, str)
        or len(expected_index_sha256) != 64
        or not isinstance(candidate_order_sha256, str)
        or len(candidate_order_sha256) != 64
    ):
        raise ValueError("Phase-II manifest requires BGE index/candidate fingerprints")
    model.config.aria_training_retrieval_index_sha256 = expected_index_sha256
    model.config.aria_training_candidate_order_sha256 = candidate_order_sha256
    overlap = sorted(set(corpus_urls) & test_urls)
    if overlap:
        raise ValueError(
            f"Retrieval corpus leaks {len(overlap)} official test page URLs; "
            f"first={overlap[0]}"
        )
    rag_config = create_paper_rag_config(
        args.rag_configuration,
        args.compress_rate,
        top_k=args.generation_top_k,
    )
    model.config.aria_rag_configuration = args.rag_configuration
    model.config.aria_coupling_control_protocol = COUPLING_CONTROL_PROTOCOL
    model.config.aria_acr_allocation_mode = rag_config.acr_allocation_mode
    model.config.aria_second_retrieval_mode = rag_config.second_retrieval_mode
    model.config.aria_uniform_evidence_token_budget = (
        MATCHED_EVIDENCE_TOKEN_BUDGET
        if rag_config.acr_allocation_mode == "uniform_budget"
        else None
    )
    model.config.aria_uniform_allocation_scheme = (
        UNIFORM_BUDGET_ALLOCATION_SCHEME
        if rag_config.acr_allocation_mode == "uniform_budget"
        else None
    )
    model.config.aria_static_second_query_scheme = (
        STATIC_SECOND_QUERY_SCHEME
        if rag_config.second_retrieval_mode == "static_query"
        else None
    )
    model.config.aria_release_convention_inferred = (
        rag_config.acr_allocation_mode == "uniform_budget"
        or rag_config.second_retrieval_mode == "static_query"
    )
    model.config.aria_training_seed = int(args.seed)
    model.config.aria_compression_rate = int(args.compress_rate)
    if args.rag_configuration == "clara_baseline":
        # The Phase-II artifact currently supplies a fixed BGE top-5 candidate
        # pool.  CLaRa still executes its ST selector (N == k is permitted), and
        # never mounts any component of the ARIA retrieval/coupling pipeline.
        if model.rag_pipeline is not None:
            raise RuntimeError("Matched CLaRa must not mount the ARIA RAG pipeline")
        model._rag_config = rag_config
        model.config.clara_selector_scheme = CLARA_SELECTOR_SCHEME
        model.config.clara_document_representation_scheme = (
            CLARA_DOCUMENT_REPRESENTATION_SCHEME
        )
        model.config.clara_phase2_objective = CLARA_PHASE2_OBJECTIVE
        model.config.clara_phase2_trainable_adapters = [
            "query_reasoner_adapter",
            "decoder_adapter",
        ]
        model.config.clara_phase2_frozen_adapter = "encoder_adapter"
        model.config.clara_phase2_adapter_initialization = (
            "both-exact-copy-of-corresponding-phase1-compressor-v1"
        )
        model.config.clara_memory_allocation_scheme = CLARA_MEMORY_ALLOCATION_SCHEME
        model.config.clara_max_memory_tokens = max(
            1, int(args.doc_max_length) // int(args.compress_rate)
        )
        model.config.clara_training_candidate_count = int(args.generation_top_k)
        model.config.clara_evaluation_candidate_protocol = (
            CLARA_EVALUATION_CANDIDATE_PROTOCOL
        )
        model.config.clara_evaluation_candidate_count = 20
        model.config.clara_selection_count = int(args.generation_top_k)
        model.config.clara_archive_document_id_scheme = (
            CLARA_ARCHIVE_DOCUMENT_ID_SCHEME
        )
        model.config.clara_archive_page_id_scheme = CLARA_ARCHIVE_PAGE_ID_SCHEME
        model.train()
        return

    doc_embeddings = load_bge_embeddings(
        args.corpus_embeddings_path,
        expected_rows=len(corpus_docs),
        expected_ids=corpus_ids,
        expected_hashes=corpus_hashes,
        expected_page_ids=corpus_urls,
        expected_index_sha256=expected_index_sha256,
    )
    if len(corpus_docs) < args.generation_top_k:
        raise ValueError(
            f"Corpus has {len(corpus_docs)} documents but Phase II top-k is "
            f"{args.generation_top_k}"
        )

    # Every full Phase-II run loads the explicit local projection state strictly.
    load_bge_projection(
        model,
        args.bge_projection_path,
        args.pretrain,
        model._aria_test_url_sha256,
    )
    model.train()
    model.setup_rag_pipeline(
        corpus_docs=corpus_docs,
        corpus_doc_ids=corpus_ids,
        corpus_page_ids=corpus_urls,
        doc_embeddings=doc_embeddings,
        rag_config=rag_config,
        initialize_missing_mtfrl=True,
    )
    if rag_config.use_mtfrl and model._mtfrl_projection is None:
        raise RuntimeError("Full ARIA setup requires a registered MTFRL projection head")


def fit_bge_projection_only(args: argparse.Namespace) -> None:
    """Run only the paper's fixed 50k-query W_BGE alignment stage."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("--fit_bge_projection_only must be launched as a single process")

    torch.manual_seed(args.seed)
    model = setup_model(args)
    query_texts, passage_ids, page_urls, passage_hashes = load_alignment_pairs(
        args.bge_fit_queries_path
    )
    if len(query_texts) != 50_000:
        raise ValueError(
            f"Appendix A.1 requires exactly 50,000 W_BGE alignment queries, "
            f"got {len(query_texts)}"
        )
    test_urls = _load_test_url_file(args.test_url_file)
    overlap = sorted(set(page_urls) & test_urls)
    if overlap:
        raise ValueError(
            f"W_BGE alignment includes {len(overlap)} held-out test pages; first={overlap[0]}"
        )
    target_embeddings, target_bge_provenance = load_bge_embeddings(
        args.bge_fit_embeddings_path,
        expected_rows=len(query_texts),
        expected_ids=passage_ids,
        expected_hashes=passage_hashes,
        return_metadata=True,
    )

    # fit_bge_projection creates a new layer on CPU when absent, so register it
    # before moving the complete model to the selected accelerator.
    model.setup_bge_projection(bge_dim=1024, freeze=False)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)

    losses = model.fit_bge_projection(
        query_texts,
        target_embeddings,
        max_length=args.query_max_length,
        epochs=2,
        batch_size=128,
        learning_rate=5e-4,
        seed=args.seed,
        require_paper_sample_count=True,
    )
    output_path = Path(args.bge_projection_save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in model._bge_projection.state_dict().items()
    }
    state = {
        "state_dict": state_dict,
        "base_model": args.pretrain,
        "base_model_revision_declared": model.config.decoder_model_revision,
        "base_model_revision_resolved": (
            model.config.decoder_model_resolved_revision
        ),
        "bge_model": "BAAI/bge-large-en-v1.5",
        "sample_count": len(query_texts),
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "seed": args.seed,
        "query_max_length": int(args.query_max_length),
        "text_sha256_scheme": TEXT_SHA256_SCHEME,
        "qr_input_scheme": QR_INPUT_SCHEME,
        "query_sha256": hashlib.sha256(
            "\n".join(_text_sha256(query) for query in query_texts).encode("utf-8")
        ).hexdigest(),
        "passage_id_sha256": hashlib.sha256(
            "\n".join(passage_ids).encode("utf-8")
        ).hexdigest(),
        "passage_text_sha256": hashlib.sha256(
            "\n".join(passage_hashes).encode("utf-8")
        ).hexdigest(),
        "test_url_count": len(test_urls),
        "test_url_sha256": _url_set_sha256(test_urls),
        **target_bge_provenance,
    }
    torch.save(state, output_path)
    print(f"Saved fitted W_BGE projection to {output_path}; epoch losses: {losses}")


def create_clara_config(args: argparse.Namespace) -> CLaRaConfig:
    """Create CLaRa configuration from command line arguments."""
    is_clara = args.rag_configuration == "clara_baseline"
    return CLaRaConfig(
        decoder_model_name=args.pretrain,
        decoder_model_revision=args.pretrain_revision,
        compr_rate=args.compress_rate,
        doc_max_length=args.doc_max_length,
        compr_n_layers=5,
        compr_use_mlp=False,
        compr_model_name=None,
        lora=True,  # LoRA on decoder and compressor
        lora_compressor=False,  # For BERT-style compressors only
        load_adapters=True,
        kbtc_training=False,
        # Appendix A.37 counts only the three LoRA adapters and MTFRL head as
        # trainable in Phase II. Keeping the full embedding matrix out of AdamW
        # also avoids unintended decoupled weight decay on frozen backbone rows.
        optimize_mem_tokens=False,
        different_mem_tokens=True,
        generation_top_k=args.generation_top_k,
        device_map=None,
        lora_r=16,
        lora_target_modules="all-linear" if is_clara else ["q_proj"],
        aria_rag_configuration=args.rag_configuration,
        training_form="both_separately",
        training_stage=args.stage,
        sep=True,
        attn_implementation="flash_attention_2" if args.flash_attn else "sdpa",
        stage2_retrieval_top_n=args.stage2_retrieval_top_n,
        pure_inference=args.pure_inference
    )


def setup_model(args: argparse.Namespace) -> CLaRa:
    """Setup CLaRa model from arguments."""
    cfg = create_clara_config(args)

    if args.pretrain_checkpoint is not None:
        print(f"Loading model from checkpoint: {args.pretrain_checkpoint}")
        if args.stage == "stage2":
            phase1_config = CLaRaConfig.from_pretrained(args.pretrain_checkpoint)
            current_test_digest = _url_set_sha256(
                _load_test_url_file(args.test_url_file)
            )
            expected_phase1 = {
                "training_stage": "stage1",
                "decoder_model_name": args.pretrain,
                "compr_rate": int(args.compress_rate),
                "doc_max_length": int(args.doc_max_length),
                "aria_rag_configuration": (
                    "clara_baseline"
                    if args.rag_configuration == "clara_baseline"
                    else "full"
                ),
                "lora_target_modules": (
                    "all-linear"
                    if args.rag_configuration == "clara_baseline"
                    else ["q_proj"]
                ),
                "aria_phase1_training_seed": int(args.seed),
                "aria_phase1_test_url_sha256": current_test_digest,
            }
            for key, expected in expected_phase1.items():
                actual = getattr(phase1_config, key, None)
                if actual != expected:
                    raise ValueError(
                        f"Phase-I checkpoint metadata {key!r} must be {expected!r}, "
                        f"got {actual!r}"
                    )
            if args.pretrain_revision is not None:
                recorded_revisions = {
                    str(value)
                    for value in (
                        getattr(phase1_config, "decoder_model_revision", None),
                        getattr(
                            phase1_config,
                            "decoder_model_resolved_revision",
                            None,
                        ),
                    )
                    if value is not None
                }
                if args.pretrain_revision not in recorded_revisions:
                    raise ValueError(
                        "--pretrain_revision does not match Phase-I checkpoint "
                        f"provenance: {args.pretrain_revision!r} not in "
                        f"{sorted(recorded_revisions)}"
                    )
            phase1_manifest_digest = getattr(
                phase1_config, "aria_phase1_dataset_manifest_sha256", None
            )
            if not isinstance(phase1_manifest_digest, str) or len(phase1_manifest_digest) != 64:
                raise ValueError(
                    "Phase-I checkpoint requires its dataset-manifest fingerprint"
                )
        model = CLaRa.from_pretrained(
            args.pretrain_checkpoint,
            strict_aria_artifacts=args.stage == "stage2",
            strict_source_training_stage="stage1" if args.stage == "stage2" else None,
            initialize_query_reasoner_adapter=args.stage == "stage2",
            training_stage=args.stage,
            generation_top_k=args.generation_top_k,
            doc_max_length=args.doc_max_length,
            compr_rate=args.compress_rate,
            aria_rag_configuration=args.rag_configuration,
            lora_target_modules=(
                "all-linear"
                if args.rag_configuration == "clara_baseline"
                else ["q_proj"]
            ),
        )
    else:
        print("Initializing new model")
        model = CLaRa(cfg)

    source_manifest = build_source_snapshot_manifest()
    model._aria_source_snapshot_manifest = source_manifest
    model.config.aria_source_snapshot_scheme = source_manifest["scheme"]
    model.config.aria_source_git_commit = source_manifest["git_commit"]
    model.config.aria_source_git_dirty = source_manifest["git_dirty"]
    model.config.aria_source_tree_sha256 = source_manifest[
        "source_tree_sha256"
    ]
    model.config.aria_source_file_count = source_manifest["source_file_count"]
    model.config.aria_passage_max_length = int(args.doc_max_length)
    model.config.aria_query_max_length = int(args.query_max_length)
    model.config.aria_input_max_length = int(args.max_len)
    model.config.aria_target_max_length = int(args.target_max_length)
    model.config.aria_loss_weights = {
        name: float(getattr(args, name)) for name in _PAPER_PHASE2_LOSS_WEIGHTS
    }
    if args.stage == "stage2" and args.rag_configuration == "clara_baseline":
        model.configure_clara_phase2_trainable_parameters()
    if not args.fit_bge_projection_only:
        model.config.aria_training_schedule = {
            "learning_rate": float(args.learning_rate),
            "scheduler": args.lr_scheduler,
            "warmup_steps": args.lr_warmup_steps,
            "warmup_ratio": float(args.lr_warmup_ratio),
            "epochs": int(args.max_epochs),
            "effective_batch_size": int(args.train_batch_size),
            "micro_batch_size_per_rank": int(args.micro_train_batch_size),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "gradient_accumulation_steps": 1,
            "max_gradient_norm": float(args.max_norm),
        }
    return model


def _active_adapter_names(decoder: Any) -> List[str]:
    active = getattr(decoder, "active_adapters", None)
    active = active() if callable(active) else active
    if isinstance(active, str):
        active = [active]
    if not isinstance(active, (list, tuple)) or not active:
        raise RuntimeError("multi-adapter checkpointing requires active adapter names")
    return [str(name) for name in active]


def _adapters_are_enabled(decoder: Any) -> bool:
    marker = getattr(decoder, "_aria_test_adapters_enabled", None)
    if isinstance(marker, bool):
        return marker
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError as exc:
        raise RuntimeError("adapter checkpointing requires PEFT") from exc
    disabled_flags = {
        bool(module.disable_adapters)
        for module in decoder.modules()
        if isinstance(module, BaseTunerLayer)
    }
    if not disabled_flags:
        raise RuntimeError("adapter checkpointing found no PEFT tuner layers")
    if len(disabled_flags) != 1:
        raise RuntimeError("PEFT adapter layers have inconsistent enabled states")
    return not disabled_flags.pop()


def _restore_adapter_state(decoder: Any, names: Sequence[str], enabled: bool) -> None:
    decoder.set_adapter(list(names))
    toggle_name = "enable_adapters" if enabled else "disable_adapters"
    toggle = getattr(decoder, toggle_name, None)
    if not callable(toggle):
        raise RuntimeError(f"adapter checkpointing requires decoder.{toggle_name}()")
    toggle()


def _adapter_checkpoint_contexts(decoder: Any):
    """Bind one checkpoint recomputation to its forward-pass LoRA adapter."""
    forward_adapters = _active_adapter_names(decoder)
    forward_enabled = _adapters_are_enabled(decoder)

    @contextmanager
    def recompute_context():
        previous_adapters = _active_adapter_names(decoder)
        previous_enabled = _adapters_are_enabled(decoder)
        _restore_adapter_state(decoder, forward_adapters, forward_enabled)
        try:
            yield
        finally:
            _restore_adapter_state(decoder, previous_adapters, previous_enabled)

    return nullcontext(), recompute_context()


def enable_gradient_checkpointing(model: CLaRa) -> None:
    """Enable checkpointing while preserving Phase-II adapter identity."""
    decoder = model.decoder
    decoder.config.use_cache = False
    kwargs = {"use_reentrant": False}
    adapter_keys = list(getattr(model, "adapter_keys", ()))
    adapter_context_required = bool(adapter_keys)
    if adapter_context_required:
        if not callable(getattr(decoder, "set_adapter", None)):
            raise RuntimeError("adapter checkpointing requires decoder.set_adapter")
        _active_adapter_names(decoder)
        _adapters_are_enabled(decoder)
        set_checkpointing = getattr(decoder, "_set_gradient_checkpointing", None)
        if set_checkpointing is not None:
            try:
                old_checkpointing_api = "value" in inspect.signature(
                    set_checkpointing
                ).parameters
            except (TypeError, ValueError):
                old_checkpointing_api = True
            if old_checkpointing_api:
                raise RuntimeError(
                    "ARIA adapter training requires the Transformers "
                    "checkpoint context API; this decoder exposes the unsafe legacy API"
                )
        kwargs["context_fn"] = lambda: _adapter_checkpoint_contexts(decoder)
    try:
        decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=kwargs
        )
    except TypeError as exc:
        # Transformers 4.41 accepts checkpointing but some remote-code models
        # do not expose the keyword argument.
        if adapter_context_required:
            raise RuntimeError(
                "ARIA adapter checkpointing cannot safely ignore context_fn"
            ) from exc
        decoder.gradient_checkpointing_enable()
    if hasattr(decoder, "enable_input_require_grads"):
        decoder.enable_input_require_grads()
    if not bool(getattr(decoder, "is_gradient_checkpointing", False)):
        raise RuntimeError("decoder failed to enable gradient checkpointing")


def setup_datasets(args: argparse.Namespace, tokenizer, strategy, model: CLaRa):
    """Setup training and evaluation datasets."""
    # Paper-protocol training consumes aria_data.py artifacts with their exact
    # Phase-I counts and five-view Phase-II schedule.
    manifest_path = Path(args.dataset) / "aria_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Paper-protocol dataset requires manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    test_urls = _load_test_url_file(args.test_url_file)
    test_url_sha256 = _url_set_sha256(test_urls)
    if (
        manifest.get("test_url_count") != len(test_urls)
        or manifest.get("test_url_sha256") != test_url_sha256
    ):
        raise ValueError(
            "Dataset manifest and training run use different official test-URL sets"
        )
    candidate = load_from_disk(args.dataset)

    if args.stage == "stage2":
        if not isinstance(candidate, DatasetDict):
            raise ValueError(
                "Phase II requires the five-view DatasetDict produced by aria_data.py"
            )
        if (
            manifest.get("phase") != "phase2"
            or manifest.get("samples_per_epoch") != _PHASE2_ROWS_PER_EPOCH
            or manifest.get("epochs") != 5
        ):
            raise ValueError(
                "Phase-II manifest must record the five-epoch paper schedule"
            )
        epoch_seeds = manifest.get("epoch_seed_schedule")
        if (
            not isinstance(epoch_seeds, list)
            or len(epoch_seeds) != 5
            or len(set(epoch_seeds)) != 5
        ):
            raise ValueError("Phase-II manifest must declare five distinct epoch seeds")
        training_retrieval = manifest.get("training_retrieval")
        if (
            not isinstance(training_retrieval, dict)
            or training_retrieval.get("model") != "BAAI/bge-large-en-v1.5"
            or training_retrieval.get("top_k") != 5
            or training_retrieval.get("corpus_scope") != "page_url_deduplicated"
            or not isinstance(training_retrieval.get("index_sha256"), str)
            or len(training_retrieval["index_sha256"]) != 64
            or not isinstance(training_retrieval.get("candidate_order_sha256"), str)
            or len(training_retrieval["candidate_order_sha256"]) != 64
        ):
            raise ValueError(
                "Phase-II manifest requires the fixed BGE top-5 retrieval provenance"
            )
        model.config.aria_training_retrieval_index_sha256 = training_retrieval[
            "index_sha256"
        ]
        model.config.aria_training_candidate_order_sha256 = training_retrieval[
            "candidate_order_sha256"
        ]
        model.config.aria_dataset_manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        model.config.aria_epoch_seed_schedule = [int(value) for value in epoch_seeds]
        shard_manifest = {
            record.get("split"): record
            for record in manifest.get("epoch_shards", [])
            if isinstance(record, dict)
        }
        corpus_metadata = getattr(model, "_aria_corpus_metadata", {})
        corpus_ids = set(corpus_metadata)
        if not corpus_ids:
            raise RuntimeError("Phase-II dataset validation requires the configured corpus")
        all_view_ids: set[str] = set()
        for split_name in sorted(candidate):
            if not split_name.startswith("epoch_"):
                continue
            split = candidate[split_name]
            epoch_index = int(split_name.split("_")[1])
            record = shard_manifest.get(split_name)
            if (
                record is None
                or record.get("epoch") != epoch_index
                or record.get("seed") != epoch_seeds[epoch_index]
                or record.get("count") != len(split)
                or record.get("fingerprint") != getattr(split, "_fingerprint", None)
            ):
                raise ValueError(f"{split_name} does not match its manifest record")
            if set(split["sampling_epoch"]) != {epoch_index}:
                raise ValueError(f"{split_name} contains an invalid sampling_epoch")
            if set(split["epoch_seed"]) != {epoch_seeds[epoch_index]}:
                raise ValueError(f"{split_name} contains an invalid epoch_seed")
            view_ids = [str(value) for value in split["view_id"]]
            if len(view_ids) != len(set(view_ids)) or all_view_ids.intersection(view_ids):
                raise ValueError(f"{split_name} contains duplicate Phase-II view_id values")
            all_view_ids.update(view_ids)

            unknown: set[str] = set()
            for row_index, row in enumerate(split):
                doc_ids = [str(value) for value in row["doc_ids"]]
                docs = row["docs"]
                page_urls = row["page_url"]
                if not (len(doc_ids) == len(docs) == len(page_urls) == 5):
                    raise ValueError(f"{split_name} row {row_index} has misaligned candidates")
                for doc_id, text, page_url in zip(doc_ids, docs, page_urls):
                    expected = corpus_metadata.get(doc_id)
                    if expected is None:
                        unknown.add(doc_id)
                        continue
                    if expected["text_sha256"] != _text_sha256(text):
                        raise ValueError(
                            f"{split_name} row {row_index} text does not match corpus ID {doc_id!r}"
                        )
                    if expected["page_url"] != _canonical_page_url(
                        page_url,
                        location=f"{split_name} row {row_index}.page_url",
                    ):
                        raise ValueError(
                            f"{split_name} row {row_index} URL does not match corpus ID {doc_id!r}"
                        )
                for gold_id in row["gold_doc_ids"]:
                    if str(gold_id) not in corpus_ids:
                        unknown.add(str(gold_id))
            if unknown:
                first = sorted(unknown)[0]
                raise ValueError(
                    f"{split_name} contains {len(unknown)} gold IDs absent from corpus; first={first}"
                )
        train_dataset = _ScheduledEpochSFTDataset(
            candidate, tokenizer, args.max_len, strategy
        )
    else:
        if not isinstance(candidate, Dataset):
            raise ValueError("Phase I requires the Dataset produced by aria_data.py")
        if (
            manifest.get("phase") != "phase1"
            or manifest.get("objective") != "conditional_generation"
            or len(candidate) != _PHASE1_TOTAL
        ):
            raise ValueError(
                f"Phase I requires all {_PHASE1_TOTAL:,} rows and a "
                "conditional_generation manifest; legacy paraphrase-only artifacts "
                "must be rebuilt with aria_data.py"
            )
        source_records = manifest.get("sources")
        manifest_counts = (
            {
                name: int(record.get("final_count", -1))
                for name, record in source_records.items()
            }
            if isinstance(source_records, dict)
            else {}
        )
        if manifest_counts != _PHASE1_SOURCE_COUNTS:
            raise ValueError("Phase-I manifest source counts do not match Appendix A.19")
        manifest_data_types = (
            {
                str(record.get("data_type")): int(record.get("final_count", -1))
                for record in source_records.values()
            }
            if isinstance(source_records, dict)
            else {}
        )
        if manifest_data_types != _PHASE1_DATA_TYPE_COUNTS:
            raise ValueError(
                "Phase-I manifest must preserve the four conditional-generation categories"
            )
        value_counts = pc.value_counts(candidate.data.column("pretraining_source"))
        actual_counts = {
            str(value): int(count)
            for value, count in zip(
                value_counts.field("values").to_pylist(),
                value_counts.field("counts").to_pylist(),
            )
        }
        if actual_counts != _PHASE1_SOURCE_COUNTS:
            raise ValueError(
                f"Phase-I Arrow rows do not match source counts: {actual_counts}"
            )
        data_type_counts = pc.value_counts(candidate.data.column("data_type"))
        actual_data_type_counts = {
            str(value): int(count)
            for value, count in zip(
                data_type_counts.field("values").to_pylist(),
                data_type_counts.field("counts").to_pylist(),
            )
        }
        if actual_data_type_counts != _PHASE1_DATA_TYPE_COUNTS:
            raise ValueError(
                "Phase-I Arrow rows do not match conditional-generation category counts: "
                f"{actual_data_type_counts}"
            )
        if manifest.get("dataset_fingerprint") != getattr(candidate, "_fingerprint", None):
            raise ValueError("Phase-I dataset fingerprint does not match its manifest")
        model.config.aria_phase1_training_seed = int(args.seed)
        model.config.aria_phase1_test_url_sha256 = test_url_sha256
        model.config.aria_phase1_dataset_manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        model.config.aria_phase1_base_model = args.pretrain
        model.config.aria_phase1_base_model_resolved_revision = (
            model.config.decoder_model_resolved_revision
        )
        model.config.aria_phase1_compression_rate = int(args.compress_rate)
        train_dataset = SFTDataset(
            candidate,
            tokenizer,
            args.max_len,
            strategy,
        )

    # Training dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        collate_fn=make_collate_fn(
            model,
            qa_loss=args.qa_loss,
            passage_max_len=args.doc_max_length,
            query_max_len=args.query_max_length,
            input_max_len=args.max_len,
            target_max_len=args.target_max_length,
        ),
        # One optimizer step is one physical global minibatch (accumulation=1).
        # Drop the deterministic tail so no partial paper minibatch crosses an
        # epoch boundary; Phase I has 7,808,465 rows, not a batch multiple.
        drop_last_multiple=(
            args.micro_train_batch_size * strategy.accumulated_gradient
        ),
    )

    # Evaluation dataset (optional)
    eval_dataloader = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,
            strategy,
            dataset_split=args.eval_split,
        )
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            args.max_len,
            strategy,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            collate_fn=make_collate_fn(
                model,
                qa_loss=args.qa_loss,
                passage_max_len=args.doc_max_length,
                query_max_len=args.query_max_length,
                input_max_len=args.max_len,
                target_max_len=args.target_max_length,
            ),
            drop_last=False,
        )

    return train_dataset, train_dataloader, eval_dataloader


def _training_step_counts(
    train_dataloader,
    accumulated_gradient: int,
    max_epochs: int,
) -> Tuple[int, int]:
    """Return exact optimizer steps per epoch and across training.

    The scheduler must be based on the sampler-backed loader, rather than the
    raw dataset length: a distributed sampler may deterministically drop a tail
    to keep gradient accumulation inside each epoch.
    """
    if accumulated_gradient <= 0:
        raise ValueError("accumulated_gradient must be positive")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    micro_steps_per_epoch = len(train_dataloader)
    if micro_steps_per_epoch % accumulated_gradient != 0:
        raise ValueError(
            "Training dataloader length must be divisible by gradient accumulation: "
            f"{micro_steps_per_epoch} micro-steps vs {accumulated_gradient} accumulation steps"
        )
    num_update_steps_per_epoch = micro_steps_per_epoch // accumulated_gradient
    if num_update_steps_per_epoch == 0:
        raise ValueError("Training dataloader does not contain one complete optimizer step")
    return num_update_steps_per_epoch, max_epochs * num_update_steps_per_epoch


def _paper_warmup_steps(args: argparse.Namespace, max_steps: int) -> int:
    """Resolve the phase-specific schedule exactly as reported in the paper."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.stage == "stage1":
        if args.lr_warmup_steps is not None:
            raise ValueError("Phase I warmup is ratio-based")
        return math.ceil(max_steps * args.lr_warmup_ratio)
    if args.lr_warmup_steps is None:
        raise ValueError("Phase II requires an absolute warmup-step count")
    return int(args.lr_warmup_steps)


def _adapter_state(decoder: Any, adapter_name: str) -> Dict[str, torch.Tensor]:
    """Return one adapter state using the checkpoint serializer's public API."""
    getter = getattr(decoder, "get_adapter_state_dict", None)
    if not callable(getter):
        raise RuntimeError("Paper-protocol LoRA auditing requires get_adapter_state_dict()")
    state = getter(adapter_name)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"LoRA adapter {adapter_name!r} has no serialized state")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(f"LoRA adapter {adapter_name!r} contains a non-tensor value")
    return state


def _assert_fresh_lora_adapter(decoder: Any, adapter_name: str) -> None:
    """Check PEFT's paper-specified Kaiming-A/zero-B initialization contract."""
    state = _adapter_state(decoder, adapter_name)
    a_tensors = [value for key, value in state.items() if "lora_A" in key]
    b_tensors = [value for key, value in state.items() if "lora_B" in key]
    if not a_tensors or not b_tensors:
        raise RuntimeError(f"LoRA adapter {adapter_name!r} lacks A/B matrices")
    if any(not torch.isfinite(value).all() for value in a_tensors + b_tensors):
        raise RuntimeError(f"LoRA adapter {adapter_name!r} contains NaN or infinity")
    if any(torch.count_nonzero(value).item() == 0 for value in a_tensors):
        raise RuntimeError(
            f"LoRA adapter {adapter_name!r} A matrix is zero; expected Kaiming-uniform"
        )
    if any(torch.count_nonzero(value).item() != 0 for value in b_tensors):
        raise RuntimeError(
            f"LoRA adapter {adapter_name!r} B matrix is nonzero at fresh initialization"
        )


def _assert_phase1_adapter_copy(
    decoder: Any,
    target_adapters: Sequence[str] = ("query_reasoner_adapter",),
) -> None:
    """Require each declared Phase-II LoRA to copy the compressor exactly."""
    compressor = _adapter_state(decoder, "encoder_adapter")
    for adapter_name in target_adapters:
        target_state = _adapter_state(decoder, adapter_name)
        if set(compressor) != set(target_state):
            raise RuntimeError(
                f"Phase-II {adapter_name} and compressor LoRA tensors have "
                "different key sets"
            )
        for key, source in compressor.items():
            target = target_state[key]
            if source.shape != target.shape or not torch.equal(
                source.detach(), target.detach()
            ):
                raise RuntimeError(
                    f"Phase-II {adapter_name} must exactly copy the corresponding "
                    f"Phase-I encoder_adapter; mismatch at {key!r}"
                )


def _validate_lora_configuration(model: CLaRa, required_adapters: Sequence[str]) -> None:
    if int(getattr(model.config, "lora_r", -1)) != 16:
        raise RuntimeError("ARIA requires LoRA rank 16")
    is_clara = getattr(model.config, "aria_rag_configuration", None) == "clara_baseline"
    expected_targets: Any = "all-linear" if is_clara else ["q_proj"]
    if getattr(model.config, "lora_target_modules", None) != expected_targets:
        raise RuntimeError(
            "CLaRa requires all-linear LoRA placement"
            if is_clara
            else "ARIA requires q_proj-only LoRA placement"
        )
    peft_configs = getattr(model.decoder, "peft_config", {})
    for adapter_name in required_adapters:
        config = peft_configs.get(adapter_name) if hasattr(peft_configs, "get") else None
        if config is None:
            raise RuntimeError(f"Missing required LoRA adapter {adapter_name!r}")
        if (
            int(getattr(config, "r", -1)) != 16
            or float(getattr(config, "lora_alpha", -1.0)) != 32.0
            or not math.isclose(
                float(getattr(config, "lora_dropout", -1.0)),
                0.10,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or getattr(config, "bias", None) != "none"
        ):
            raise RuntimeError(
                f"LoRA adapter {adapter_name!r} must use r=16, alpha=32, "
                "dropout=0.10, bias='none'"
            )
        configured_targets = getattr(config, "target_modules", None)
        if is_clara:
            if configured_targets != "all-linear":
                raise RuntimeError(
                    f"CLaRa adapter {adapter_name!r} must target all linear layers"
                )
        elif set(configured_targets or ()) != {"q_proj"}:
            raise RuntimeError(
                f"ARIA adapter {adapter_name!r} must target q_proj only"
            )


def _validate_feedback_initialization(model: CLaRa) -> None:
    projection = getattr(model, "_mtfrl_projection", None)
    if projection is None:
        raise RuntimeError("Full Phase II requires the trainable P_fb projection")
    scheme = getattr(model.config, "mtfrl_initialization_scheme", None)
    if scheme != "xavier-uniform-zero-bias-v1":
        raise RuntimeError(
            "P_fb must record Xavier-uniform weights and zero biases; "
            f"got initialization scheme {scheme!r}"
        )
    weights: List[torch.Tensor] = []
    biases: List[torch.Tensor] = []
    for module in projection.modules():
        if isinstance(module, torch.nn.Linear):
            weights.append(module.weight)
            if module.bias is not None:
                biases.append(module.bias)
    if not weights or not biases:
        raise RuntimeError("P_fb must contain initialized Linear weights and biases")
    if any(not torch.isfinite(value).all() for value in weights + biases):
        raise RuntimeError("P_fb contains NaN or infinity at initialization")
    if any(torch.count_nonzero(value).item() == 0 for value in weights):
        raise RuntimeError("P_fb contains an all-zero weight matrix instead of Xavier weights")
    if any(torch.count_nonzero(value).item() != 0 for value in biases):
        raise RuntimeError("P_fb biases must be zero initialized")


def _validate_trainable_parameter_contract(args: argparse.Namespace, model: CLaRa) -> None:
    """Fail closed if AdamW would update anything outside the paper modules."""
    if args.stage == "stage1":
        required_adapters = ("encoder_adapter",)
        allowed_markers = required_adapters
        _validate_lora_configuration(model, required_adapters)
        _assert_fresh_lora_adapter(model.decoder, "encoder_adapter")
    elif args.rag_configuration == "clara_baseline":
        # The matched baseline has a separate method-specific training contract.
        required_adapters = (
            "encoder_adapter",
            "query_reasoner_adapter",
            "decoder_adapter",
        )
        allowed_markers = ("query_reasoner_adapter", "decoder_adapter")
        _validate_lora_configuration(model, required_adapters)
        _assert_phase1_adapter_copy(
            model.decoder,
            ("query_reasoner_adapter", "decoder_adapter"),
        )
    else:
        required_adapters = (
            "encoder_adapter",
            "query_reasoner_adapter",
            "decoder_adapter",
        )
        _validate_lora_configuration(model, required_adapters)
        _assert_phase1_adapter_copy(model.decoder)
        _assert_fresh_lora_adapter(model.decoder, "decoder_adapter")
        use_mtfrl = _RAG_CONFIGURATION_SWITCHES[args.rag_configuration].get(
            "use_mtfrl", True
        )
        if use_mtfrl:
            _validate_feedback_initialization(model)
            allowed_markers = required_adapters + ("_mtfrl_projection",)
        else:
            allowed_markers = required_adapters

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("Paper-protocol optimizer has no trainable parameters")
    unexpected = sorted(
        name
        for name in trainable_names
        if not any(marker in name for marker in allowed_markers)
    )
    if unexpected:
        raise RuntimeError(
            "AdamW would update parameters outside the paper contract; first entries: "
            + ", ".join(unexpected[:8])
        )
    for marker in allowed_markers:
        if not any(marker in name for name in trainable_names):
            raise RuntimeError(f"Paper trainable module {marker!r} is absent from AdamW")

    model.config.aria_optimizer = "AdamW"
    model.config.aria_adam_betas = [float(value) for value in args.adam_betas]
    model.config.aria_adam_epsilon = float(args.adam_eps)
    model.config.aria_weight_decay = float(args.l2)
    model.config.aria_trainable_parameter_names = sorted(trainable_names)


def _validate_optimizer_parameter_contract(model: CLaRa, optimizer: Any) -> None:
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual_list = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    ]
    actual = {id(parameter) for parameter in actual_list}
    if len(actual) != len(actual_list):
        raise RuntimeError("AdamW contains duplicate parameter references")
    if actual != expected:
        raise RuntimeError(
            "AdamW parameter groups must contain every and only requires_grad=True parameter"
        )


def setup_training_components(args: argparse.Namespace, model: CLaRa, train_dataloader, strategy):
    """Setup optimizer, scheduler and other training components."""
    _validate_trainable_parameter_contract(args, model)

    # Configure optimizer
    optimizer = strategy.create_optimizer(
        model,
        lr=args.learning_rate,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.l2,
    )
    _validate_optimizer_parameter_contract(model, optimizer)

    # Configure scheduler
    num_update_steps_per_epoch, max_steps = _training_step_counts(
        train_dataloader,
        strategy.accumulated_gradient,
        args.max_epochs,
    )

    warmup_steps = _paper_warmup_steps(args, max_steps)
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    # Prepare models with strategy
    model, optimizer, scheduler = strategy.prepare((model, optimizer, scheduler))

    return model, optimizer, scheduler, num_update_steps_per_epoch


def load_checkpoint_if_exists(args: argparse.Namespace, strategy, model: CLaRa) -> int:
    """Load checkpoint if it exists and return consumed samples."""
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model, args.ckpt_path)
        consumed_samples = states.get("consumed_samples", 0)
        strategy.print(f"Loaded checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    return consumed_samples


def train(args: argparse.Namespace):
    """Main training function."""
    # Configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()
    if strategy.accumulated_gradient != 1:
        raise RuntimeError(
            "Paper minibatch B must be materialized in one forward pass; "
            "DeepSpeed gradient accumulation must equal 1"
        )

    # Setup model
    model = setup_model(args)

    # Full Phase II artifacts must be loaded before optimizer construction:
    # setup_rag_pipeline registers the trainable MTFRL head on the model.
    if args.stage == "stage2":
        setup_phase2_artifacts(args, model)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # Configure tokenizer
    tokenizer = get_tokenizer(
        args.pretrain,
        model,
        "right",
        strategy,
        use_fast=not args.disable_fast_tokenizer
    )
    strategy.print(model)

    # Setup datasets
    train_dataset, train_dataloader, eval_dataloader = setup_datasets(
        args, tokenizer, strategy, model
    )

    # Setup training components
    model, optimizer, scheduler, num_update_steps_per_epoch = setup_training_components(
        args, model, train_dataloader, strategy
    )

    # Load checkpoint if exists
    consumed_samples = load_checkpoint_if_exists(args, strategy, model)

    # Ensure save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    # Start training
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # Save final model
    strategy.save_model(model, tokenizer, args.save_path)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="ARIA two-phase training")

    # Model and checkpoint arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--pretrain", type=str, required=True, help="Base model path")
    model_group.add_argument(
        "--pretrain_revision",
        type=str,
        default=None,
        help=(
            "Hugging Face branch, tag, or exact commit for --pretrain. "
            "The resolved commit is recorded in the checkpoint."
        ),
    )
    model_group.add_argument("--pretrain_checkpoint", type=str, default=None,
                           help="CLaRa checkpoint to continue training from")
    model_group.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2"],
                           help="Training stage")
    model_group.add_argument(
        "--generation_top_k", type=int, default=None,
        help="1 in Phase I; final evidence count 5 in Phase II",
    )
    model_group.add_argument("--pure_inference", action="store_true", default=False,
                           help="Pure inference mode")

    # CLaRa specific arguments
    clara_group = parser.add_argument_group("ARIA Configuration")
    clara_group.add_argument(
        "--doc_max_length",
        "--passage_max_length",
        dest="doc_max_length",
        type=int,
        default=_PAPER_PASSAGE_MAX_LENGTH,
        help="Maximum passage length (paper: 768 tokens)",
    )
    clara_group.add_argument("--compress_rate", type=int, default=16, help="Document compression rate")
    clara_group.add_argument("--qa_loss", action="store_true", default=True,
                            help="Use QA loss for joint training")
    clara_group.add_argument("--stage2_mips", action="store_true", default=False,
                            help="Use MIPS for stage2 retrieval")
    clara_group.add_argument("--stage2_retrieval_top_n", type=int, default=5,
                            help="Phase-II first-five candidate ceiling (paper: 5)")
    clara_group.add_argument("--lambda_mse", type=float, default=0.1,
                            help="Phase-II MSE coefficient lambda (paper: 0.10)")
    clara_group.add_argument(
        "--lambda_cfrs",
        type=float,
        default=0.1,
        help="Phase-II CFRS coefficient mu (paper: 0.10)",
    )
    clara_group.add_argument(
        "--lambda_qr",
        type=float,
        default=0.05,
        help="Phase-II QR alignment coefficient nu (paper: 0.05)",
    )
    clara_group.add_argument(
        "--lambda_mtfrl",
        type=float,
        default=0.05,
        help="Phase-II MTFRL coefficient xi (paper: 0.05)",
    )
    clara_group.add_argument(
        "--rag_configuration",
        choices=sorted(_RAG_CONFIGURATION_SWITCHES),
        default="full",
        help=(
            "Full ARIA or one explicit separately trained configuration. The "
            "matched controls are remove_cfrs, uniform_acr, "
            "static_second_retrieval, and remove_all_coupling; fixed_* and "
            "forward_path_off labels are inference-only."
        ),
    )
    clara_group.add_argument("--mse_loss", action="store_true", default=False,
                            help=argparse.SUPPRESS)
    clara_group.add_argument("--do_eval_gen", action="store_true", default=False,
                            help="Evaluate generation during eval")

    # Checkpoint and saving
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument("--save_path", type=str, default="./ckpt", help="Model save path")
    checkpoint_group.add_argument("--save_steps", type=int, default=-1, help="Save every N steps")
    checkpoint_group.add_argument("--save_hf_ckpt", action="store_true", default=False,
                                help="Save HuggingFace checkpoint")
    checkpoint_group.add_argument("--disable_ds_ckpt", action="store_true", default=False,
                                help="Disable DeepSpeed checkpoint")
    checkpoint_group.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft",
                                help="Checkpoint path to load")
    checkpoint_group.add_argument("--load_checkpoint", action="store_true", default=False,
                                help="Load from checkpoint")
    checkpoint_group.add_argument("--max_ckpt_num", type=int, default=3, help="Max checkpoint number")
    checkpoint_group.add_argument("--max_ckpt_mem", type=int, default=1e8, help="Max checkpoint memory")

    # Training configuration
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Paper default is selected by phase: 3 in Phase I, 5 in Phase II",
    )
    training_group.add_argument("--learning_rate", type=float, default=None,
                              help="Paper default: Phase I 1e-4; Phase II 2e-4 (Mistral/Llama) or 1.6e-4 (Qwen)")
    training_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=None,
        help="Absolute warmup steps (Phase II paper default: 500; omit in Phase I)",
    )
    training_group.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.03,
        help="Phase-I warmup fraction (paper: 3%% of optimizer steps)",
    )
    training_group.add_argument("--lr_scheduler", type=str, default="cosine",
                              help="Learning rate scheduler")
    training_group.add_argument("--l2", type=float, default=0, help="Weight decay")
    training_group.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95),
                              help="Adam optimizer betas")
    training_group.add_argument(
        "--adam_eps", type=float, default=1e-8, help="AdamW epsilon (paper: 1e-8)"
    )
    training_group.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    training_group.add_argument("--pretrain_mode", action="store_true", default=False,
                              help="Use pretrain loss")

    # DeepSpeed and distributed training
    distributed_group = parser.add_argument_group("Distributed Training")
    distributed_group.add_argument(
        "--micro_train_batch_size",
        type=int,
        default=None,
        help="Per-rank batch; defaults to effective_batch/WORLD_SIZE (no accumulation)",
    )
    distributed_group.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="Effective global batch selected by phase/backbone paper protocol",
    )
    distributed_group.add_argument("--local_rank", type=int, default=-1,
                                 help="Local rank for DeepSpeed")
    distributed_group.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    distributed_group.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    distributed_group.add_argument("--gradient_checkpointing", action="store_true", default=False,
                                 help="Enable gradient checkpointing")
    distributed_group.add_argument("--flash_attn", action="store_true", default=False,
                                 help="Enable FlashAttention2")
    distributed_group.add_argument("--ds_tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")
    # Dataset configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument("--dataset", type=str, default=None, help="Training dataset path")
    dataset_group.add_argument("--dataset_probs", type=str, default=None,
                             help="Dataset sampling probabilities")
    dataset_group.add_argument("--eval_dataset", type=str, default=None, help="Evaluation dataset path")
    dataset_group.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    dataset_group.add_argument("--eval_split", type=str, default="train", help="Evaluation split")
    dataset_group.add_argument("--max_samples", type=int, default=None,
                             help="Optional debug truncation outside paper-protocol training")
    dataset_group.add_argument(
        "--max_len",
        "--input_max_length",
        dest="max_len",
        type=int,
        default=None,
        help="Maximum prompt/input length (paper: Phase I 2048, Phase II 1024)",
    )
    dataset_group.add_argument(
        "--target_max_length",
        type=int,
        default=None,
        help="Maximum supervised target length (paper: Phase I 512, Phase II 128)",
    )
    dataset_group.add_argument(
        "--query_max_length",
        type=int,
        default=_PAPER_QUERY_MAX_LENGTH,
        help="Maximum Query Reasoner length (paper: 256)",
    )

    artifact_group = parser.add_argument_group("Retrieval Artifacts")
    artifact_group.add_argument("--corpus_path", help="Fixed retrieval corpus JSON/JSONL")
    artifact_group.add_argument("--corpus_embeddings_path", help="Aligned (N,1024) BGE document embeddings")
    artifact_group.add_argument("--bge_projection_path", help="Frozen fitted W_BGE checkpoint")
    artifact_group.add_argument(
        "--test_url_file", help="Canonical official-test page URLs, one per line"
    )
    artifact_group.add_argument("--fit_bge_projection_only", action="store_true")
    artifact_group.add_argument("--bge_fit_queries_path", help="Exactly 50,000 alignment queries")
    artifact_group.add_argument("--bge_fit_embeddings_path", help="Exactly 50,000 aligned BGE targets")
    artifact_group.add_argument("--bge_projection_save_path", default="./artifacts/w_bge.pt")

    # Logging and monitoring
    logging_group = parser.add_argument_group("Logging and Monitoring")
    logging_group.add_argument("--logging_steps", type=int, default=1, help="Log every N steps")
    logging_group.add_argument("--eval_steps", type=int, default=-1, help="Evaluate every N steps")
    logging_group.add_argument("--use_wandb", type=str, default=None, help="Wandb project name")
    logging_group.add_argument("--wandb_org", type=str, default=None, help="Wandb organization")
    logging_group.add_argument("--wandb_group", type=str, default=None, help="Wandb group")
    logging_group.add_argument("--wandb_project", type=str, default="CLaRa", help="Wandb project")
    logging_group.add_argument("--wandb_run_name", type=str,
                             default="clara_%s" % datetime.now().strftime("%m%dT%H:%M"),
                             help="Wandb run name")
    logging_group.add_argument("--use_tensorboard", type=str, default=None,
                             help="TensorBoard logging path")

    # Additional arguments
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=42, help="Random seed")
    misc_group.add_argument("--disable_fast_tokenizer", action="store_true", default=False,
                          help="Disable fast tokenizer")
    misc_group.add_argument("--use_ms", action="store_true", default=False,
                          help="Use ModelScope")

    return parser


def validate_arguments(args: argparse.Namespace):
    """Validate command line arguments."""
    # Validate training stage
    if args.stage not in ["stage1", "stage2"]:
        raise ValueError(f"Invalid stage: {args.stage}")
    if args.generation_top_k is None:
        args.generation_top_k = 1 if args.stage == "stage1" else 5

    is_qwen = "qwen" in args.pretrain.lower()
    phase_lengths = _PAPER_PHASE_LENGTHS[args.stage]
    if args.max_epochs is None:
        args.max_epochs = 3 if args.stage == "stage1" else 5
    if args.train_batch_size is None:
        args.train_batch_size = (
            16 if is_qwen else (128 if args.stage == "stage1" else 32)
        )
    if args.max_len is None:
        args.max_len = phase_lengths["input"]
    if args.target_max_length is None:
        args.target_max_length = phase_lengths["target"]
    if args.stage == "stage2" and args.lr_warmup_steps is None:
        args.lr_warmup_steps = 500
    if args.learning_rate is None:
        args.learning_rate = (
            1e-4
            if args.stage == "stage1"
            else (1.6e-4 if is_qwen else 2e-4)
        )
    if args.stage == "stage2" and args.rag_configuration == "clara_baseline":
        # Appendix A.37 uses answer CE only.  Keep the shared trainer interface,
        # but make every auxiliary coefficient exactly zero in the effective
        # run configuration and serialized checkpoint.
        for name in _PAPER_PHASE2_LOSS_WEIGHTS:
            setattr(args, name, 0.0)
    elif args.stage == "stage2":
        # Disabled training operations have exactly zero coefficients in the
        # serialized effective objective, while the remaining Eq. (13) terms
        # retain their paper weights.
        if args.rag_configuration in {"remove_cfrs", "remove_all_coupling"}:
            args.lambda_cfrs = 0.0
        if args.rag_configuration in {
            "static_second_retrieval",
            "remove_all_coupling",
            "remove_mtfrl",
        }:
            args.lambda_mtfrl = 0.0

    if args.stage == "stage2" and args.rag_configuration in FIXED_CHECKPOINT_CONFIGURATIONS:
        raise ValueError(
            f"{args.rag_configuration} is a fixed-checkpoint inference-only "
            "intervention: train --rag_configuration full and apply it at evaluation"
        )
    if (
        args.stage == "stage2"
        and args.rag_configuration in MATCHED_RETRAINING_CONFIGURATIONS
        and args.compress_rate != 16
    ):
        raise ValueError(
            "the paper's budget/topology-matched retraining controls are defined "
            "only at 16x compression"
        )

    # Validate compression parameters
    if args.compress_rate not in {4, 16, 32, 64, 128}:
        raise ValueError("Paper compression rate must be one of 4, 16, 32, 64, 128")

    if args.doc_max_length != _PAPER_PASSAGE_MAX_LENGTH:
        raise ValueError("Paper protocol requires passage/doc_max_length=768")
    if args.query_max_length != _PAPER_QUERY_MAX_LENGTH:
        raise ValueError("Paper protocol requires query_max_length=256")
    if args.max_len != phase_lengths["input"]:
        raise ValueError(
            f"Paper protocol requires Phase {'I' if args.stage == 'stage1' else 'II'} "
            f"input_max_length={phase_lengths['input']}"
        )
    if args.target_max_length != phase_lengths["target"]:
        raise ValueError(
            f"Paper protocol requires Phase {'I' if args.stage == 'stage1' else 'II'} "
            f"target_max_length={phase_lengths['target']}"
        )

    if args.lr_warmup_steps is not None and args.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    for name in _PAPER_PHASE2_LOSS_WEIGHTS:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if tuple(float(value) for value in args.adam_betas) != (0.9, 0.95):
        raise ValueError("Paper protocol requires AdamW betas=(0.9, 0.95)")
    if not math.isclose(args.adam_eps, 1e-8, rel_tol=0.0, abs_tol=1e-20):
        raise ValueError("Paper protocol requires AdamW epsilon=1e-8")
    if not math.isclose(args.l2, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Paper protocol requires zero weight decay")
    if not math.isclose(args.max_norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Paper protocol requires maximum gradient norm 1.0")

    if args.pretrain_revision is not None and not args.pretrain_revision.strip():
        raise ValueError("--pretrain_revision must be a non-empty revision")
    if args.fit_bge_projection_only:
        if args.stage != "stage2":
            raise ValueError("W_BGE fitting requires --stage stage2 so the QR adapter exists")
        if args.pretrain_checkpoint is not None:
            raise ValueError(
                "W_BGE must be fitted before ARIA training, without --pretrain_checkpoint"
            )
        for name in ("bge_fit_queries_path", "bge_fit_embeddings_path", "test_url_file"):
            value = getattr(args, name)
            if not value or not os.path.exists(value):
                raise ValueError(f"--{name} must name an existing local artifact")
        return

    if not args.dataset:
        raise ValueError("--dataset is required for training")
    if args.seed not in _PAPER_TRAINING_SEEDS:
        raise ValueError(
            "Paper-protocol training seed must be one of 42, 123, 456, 789, 2024"
        )
    if not os.path.exists(args.dataset):
        raise ValueError(f"Dataset path does not exist: {args.dataset}")
    if args.max_samples is not None:
        raise ValueError("Paper-protocol training requires the complete dataset")
    if not args.test_url_file or not os.path.isfile(args.test_url_file):
        raise ValueError("Paper-protocol training requires an existing --test_url_file")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 0:
        raise ValueError("WORLD_SIZE must be positive")
    if args.lr_scheduler != "cosine":
        raise ValueError("Paper protocol requires the cosine learning-rate scheduler")
    if not args.bf16:
        raise ValueError("Paper protocol requires BF16 training")
    # The release launchers use ZeRO-2 and FlashAttention 2 as execution
    # optimizations.  Neither changes the paper method, so alternative
    # DeepSpeed stages and Transformers SDPA remain supported.
    if args.stage == "stage2" and args.disable_fast_tokenizer:
        raise ValueError("Phase-II Eq. (4) requires a fast tokenizer for query offsets")

    expected_batch = 16 if is_qwen else (128 if args.stage == "stage1" else 32)
    if args.train_batch_size != expected_batch:
        raise ValueError(
            f"Paper protocol requires global batch {expected_batch} for this backbone"
        )
    if args.micro_train_batch_size is None:
        if expected_batch % world_size != 0:
            raise ValueError(
                f"Effective batch {expected_batch} is not divisible by WORLD_SIZE={world_size}"
            )
        args.micro_train_batch_size = expected_batch // world_size
    if args.micro_train_batch_size <= 0:
        raise ValueError("micro_train_batch_size must be positive")
    expected_lr = (
        1e-4
        if args.stage == "stage1"
        else (1.6e-4 if is_qwen else 2e-4)
    )
    if not math.isclose(args.learning_rate, expected_lr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Paper protocol requires learning_rate={expected_lr:g}")
    data_parallel_micro_batch = world_size * args.micro_train_batch_size
    if data_parallel_micro_batch != expected_batch:
        raise ValueError(
            "Paper minibatch B is the global effective batch and does not use "
            f"gradient accumulation: WORLD_SIZE ({world_size}) * micro batch "
            f"({args.micro_train_batch_size}) must equal {expected_batch}"
        )

    if args.stage == "stage1":
        if args.rag_configuration not in {"full", "clara_baseline"}:
            raise ValueError(
                "Phase I permits full ARIA or the matched all-linear CLaRa compressor"
            )
        if args.max_epochs != 3:
            raise ValueError("Paper-protocol Phase I requires exactly 3 epochs")
        if args.lr_warmup_steps is not None:
            raise ValueError(
                "Paper-protocol Phase I uses a 3% warmup ratio, not absolute warmup steps"
            )
        if not math.isclose(args.lr_warmup_ratio, 0.03, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Paper-protocol Phase I requires lr_warmup_ratio=0.03")
        if args.generation_top_k != 1:
            raise ValueError("Phase I uses exactly one document per reconstruction pair")
    if args.stage == "stage2":
        if args.max_epochs != 5:
            raise ValueError("Paper-protocol Phase II requires exactly 5 epochs")
        if args.generation_top_k != 5:
            raise ValueError("Paper-protocol Phase II uses a top-5 candidate ceiling")
        if args.stage2_retrieval_top_n != 5:
            raise ValueError(
                "Paper-protocol Phase II keeps a first-five ceiling; CCEF may "
                "produce 1-5 actual survivors via its threshold"
            )
        if args.lr_warmup_steps != 500:
            raise ValueError("Paper-protocol Phase II requires exactly 500 warmup steps")
        expected_loss_weights = dict(_PAPER_PHASE2_LOSS_WEIGHTS)
        if args.rag_configuration == "clara_baseline":
            expected_loss_weights = {
                name: 0.0 for name in _PAPER_PHASE2_LOSS_WEIGHTS
            }
        else:
            if args.rag_configuration in {"remove_cfrs", "remove_all_coupling"}:
                expected_loss_weights["lambda_cfrs"] = 0.0
            if args.rag_configuration in {
                "static_second_retrieval", "remove_all_coupling"
            }:
                expected_loss_weights["lambda_mtfrl"] = 0.0
        for name, expected in expected_loss_weights.items():
            if not math.isclose(
                getattr(args, name), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Paper-protocol Phase II requires {name}={expected:g}"
                )
        if not args.pretrain_checkpoint:
            raise ValueError("Phase II must initialize from a Phase-I checkpoint")
        required_artifacts = ["corpus_path", "test_url_file"]
        if args.rag_configuration != "clara_baseline":
            required_artifacts.extend(
                ["corpus_embeddings_path", "bge_projection_path"]
            )
        for name in required_artifacts:
            value = getattr(args, name)
            if not value or not os.path.exists(value):
                raise ValueError(f"Phase II requires existing --{name}")

    if args.pretrain_checkpoint and not os.path.exists(args.pretrain_checkpoint):
        raise ValueError(f"Pretrain checkpoint path does not exist: {args.pretrain_checkpoint}")


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args)

    # Handle ModelScope patch
    if args.use_ms:
        try:
            from modelscope.utils.hf_util import patch_hub
            patch_hub()
            print("ModelScope hub patched successfully")
        except ImportError:
            print("Warning: ModelScope not available, skipping hub patch")

    # Print configuration
    print("=" * 60)
    print("ARIA Training Configuration")
    print("=" * 60)
    print(f"Training stage: {args.stage}")
    print(f"Base model: {args.pretrain}")
    print(f"Document max length: {args.doc_max_length}")
    print(f"Compression rate: {args.compress_rate}")
    print(f"Generation top-k: {args.generation_top_k}")
    print(f"Dataset: {args.dataset}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size (micro/global): {args.micro_train_batch_size}/{args.train_batch_size}")
    print("=" * 60)

    if args.fit_bge_projection_only:
        fit_bge_projection_only(args)
        return

    # Start training
    train(args)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
