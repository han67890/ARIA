#!/usr/bin/env python3
"""Prepare the datasets used by ARIA's two-phase training protocol.

This module materializes the four Phase-I sources and four Phase-II benchmark
pools from explicit local artifacts or already-cached Hugging Face datasets.
It validates schemas, exact paper counts, independently held-out paraphrase
targets, and train/test URL separation before writing the training datasets.

Source URI syntax
-----------------
``local:/absolute/or/relative/path``
    A Dataset/DatasetDict saved with ``save_to_disk`` or a JSON(L), Parquet,
    CSV, or Arrow file.
``hf:namespace/dataset``
    A Hugging Face dataset that is already present in the local datasets
    cache.  ``DownloadConfig(local_files_only=True)`` is always used.

The command writes protocol manifests next to the datasets.  These manifests
record source mappings, exact counts, test-URL provenance, and the fixed
Phase-II epoch seed schedule so every run can be verified directly from its
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit, urlunsplit

from datasets import (
    Dataset,
    DatasetDict,
    DownloadConfig,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from openrlhf.utils.musique_augmentation import (
    MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
    MUSIQUE_PARTIAL_CHAIN_TARGET,
    build_musique_partial_rows,
    validate_musique_partial_metadata,
)
from openrlhf.utils.aria_provenance import (
    EVALUATION_ANSWER_ALIAS_CONTRACT,
    EVALUATION_GOLD_DOCUMENT_CONTRACT,
)


PROTOCOL_VERSION = "aria-paper-v1"
_MUSIQUE_NLP = None

# Table A19.  The exact total is 7,808,465 (the main text rounds it to 7.8M).
PHASE1_SOURCE_COUNTS: Mapping[str, int] = {
    "simpleqa": 2_000_000,
    "complexqa": 2_000_000,
    "paraphrase": 1_966_291,
    "entity_augmented": 1_842_174,
}
PHASE1_TOTAL = sum(PHASE1_SOURCE_COUNTS.values())
PHASE1_DATA_TYPES: Mapping[str, str] = {
    "simpleqa": "simple_qa",
    "complexqa": "complex_qa",
    "paraphrase": "paraphrase",
    "entity_augmented": "entity_augmented",
}

# Table A20, including the augmented MuSiQue partition (Appendix A.33).
PHASE2_POOL_COUNTS: Mapping[str, int] = {
    "nq": 58_622,
    "hotpotqa": 90_185,
    "musique": 168_745,
    "2wikimultihopqa": 167_454,
}
PHASE2_BENCHMARKS: Tuple[str, ...] = tuple(PHASE2_POOL_COUNTS)
PHASE2_SAMPLES_PER_BENCHMARK = 9_600
PHASE2_SAMPLES_PER_EPOCH = PHASE2_SAMPLES_PER_BENCHMARK * len(PHASE2_BENCHMARKS)
PHASE2_CANDIDATE_DOCS = 5

EVALUATION_COUNTS: Mapping[str, int] = {
    "nq": 6_489,
    "hotpotqa": 7_384,
    "musique": 2_417,
    "2wikimultihopqa": 12_576,
}

# Appendix A.33 reports these artifact counts. The repository's explicit v2
# partial-state builder preserves every k-1 prefix and deterministically adds
# non-leaking subset/frontier states to reconcile the reported 70,845 rows.
MUSIQUE_AUGMENTATION_COUNTS: Mapping[str, int] = {
    "original": 19_938,
    "subquestion": 52_107,
    "partial_chain": 70_845,
    "entity_variant": 25_855,
}


@dataclass(frozen=True)
class SourceSpec:
    """An explicit, non-downloading dataset source."""

    uri: str
    split: str = "train"
    config: Optional[str] = None


@dataclass(frozen=True)
class Phase1FieldMap:
    input_key: str = "document"
    instruction_key: str = "instruction"
    target_key: str = "target"
    source_id_key: str = "source_row_id"
    target_id_key: str = "target_row_id"
    target_split_key: str = "target_split"
    page_url_key: str = "page_url"


@dataclass(frozen=True)
class Phase2FieldMap:
    question_key: str = "question"
    answer_key: str = "answer"
    # Paper evaluation requires this explicit benchmark-provided alias set.
    # It is deliberately separate from ``answer_key`` so a scalar answer can
    # never be silently promoted to a complete alias annotation.
    gold_answers_key: str = "gold_answers"
    docs_key: str = "docs"
    pos_index_key: str = "pos_index"
    # Every source must provide the complete corpus-level annotation
    # independently of the ranked candidate list.  Inferring P(x) from
    # candidate-local ``pos_index`` silently loses supports outside D1.
    gold_doc_ids_key: str = "gold_doc_ids"
    source_id_key: str = "source_row_id"
    page_url_key: str = "page_url"
    doc_ids_key: str = "doc_ids"
    doc_text_key: str = "text"
    doc_page_url_key: str = "page_url"
    doc_id_key: str = "id"
    augmentation_type_key: str = "augmentation_type"
    augmentation_parent_id_key: str = "augmentation_parent_id"
    construction_method_key: str = "construction_method"
    entity_preserved_key: str = "entity_preserved"
    rouge_l_key: str = "rouge_l"
    decomposition_preserved_key: str = "decomposition_preserved"
    answer_preserved_key: str = "answer_preserved"
    partial_state_id_key: str = "partial_state_id"
    partial_state_protocol_key: str = "partial_state_protocol"
    partial_state_kind_key: str = "partial_state_kind"
    partial_hop_count_key: str = "partial_hop_count"
    partial_known_hop_indices_key: str = "partial_known_hop_indices"
    partial_frontier_hop_index_key: str = "partial_frontier_hop_index"
    partial_mandatory_prefix_key: str = "partial_mandatory_prefix"
    partial_prompt_sha256_key: str = "partial_prompt_sha256"
    partial_selection_sha256_key: str = "partial_selection_sha256"
    needs_candidate_retrieval_key: str = "needs_candidate_retrieval"


def _require_nonempty_string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _require_field(row: Mapping[str, Any], key: str, *, location: str) -> Any:
    if key not in row:
        raise ValueError(f"{location} requires field {key!r}")
    return row[key]


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_page_url(value: Any, *, location: str) -> str:
    """Canonicalize a Wikipedia/page URL for exact train/test deduplication."""

    url = _require_nonempty_string(value, location=location)
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(
            f"{location} must be an absolute http(s) page URL, got {url!r}"
        )
    path = re.sub(r"/{2,}", "/", parts.path)
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _url_set_sha256(urls: Set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(urls)).encode("utf-8")).hexdigest()


def _select_split(data: Any, split: str, *, source: str) -> Dataset:
    if isinstance(data, Dataset):
        return data
    if isinstance(data, DatasetDict):
        if split not in data:
            raise ValueError(
                f"Source {source!r} requires split {split!r}; available splits are "
                f"{list(data.keys())}"
            )
        return data[split]
    raise TypeError(f"Source {source!r} must resolve to a Dataset or DatasetDict")


def load_explicit_source(spec: SourceSpec) -> Dataset:
    """Load the declared local or cached-HF source exclusively."""

    if not spec.uri:
        raise ValueError("An explicit source URI is required")

    if spec.uri.startswith("local:"):
        raw_path = spec.uri[len("local:") :]
        path = Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local source does not exist: {path}")

        if path.is_dir():
            try:
                return _select_split(load_from_disk(str(path)), spec.split, source=spec.uri)
            except (FileNotFoundError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Directory {path} is not a valid Dataset/DatasetDict saved with save_to_disk"
                ) from exc

        extension = path.suffix.lower()
        builders = {
            ".json": "json",
            ".jsonl": "json",
            ".parquet": "parquet",
            ".csv": "csv",
            ".arrow": "arrow",
        }
        if extension not in builders:
            raise ValueError(
                f"Local dataset extension must be one of {sorted(builders)}; "
                f"received {extension!r}"
            )
        data = load_dataset(
            builders[extension],
            data_files={spec.split: str(path)},
            split=spec.split,
        )
        return _select_split(data, spec.split, source=spec.uri)

    if spec.uri.startswith("hf:"):
        dataset_name = spec.uri[len("hf:") :].strip()
        if not dataset_name:
            raise ValueError("hf: source URI must include a dataset name")
        try:
            return load_dataset(
                dataset_name,
                spec.config,
                split=spec.split,
                download_config=DownloadConfig(local_files_only=True),
                download_mode="reuse_dataset_if_exists",
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Hugging Face source {dataset_name!r} ({spec.split!r}) is not available "
                "locally. ARIA data preparation never downloads data; pre-cache it or use local:."
            ) from exc

    raise ValueError(
        f"Source {spec.uri!r} must use an explicit local: or hf: prefix"
    )


def _iter_url_values(value: Any, *, key: str, location: str) -> Iterable[str]:
    if isinstance(value, str):
        yield canonicalize_page_url(value, location=location)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{location} must not be an empty URL list")
        for index, item in enumerate(value):
            item_location = f"{location}[{index}]"
            if isinstance(item, str):
                yield canonicalize_page_url(item, location=item_location)
            elif isinstance(item, Mapping):
                yield canonicalize_page_url(
                    _require_field(item, key, location=item_location),
                    location=f"{item_location}.{key}",
                )
            else:
                raise ValueError(f"{item_location} must be a URL string or mapping")
        return
    raise ValueError(f"{location} must be a URL string or list")


def load_test_url_set(
    sources: Sequence[SourceSpec],
    *,
    page_url_key: str = "page_url",
) -> Set[str]:
    """Load the union of official test page URLs used by both training phases."""

    if not sources:
        raise ValueError(
            "At least one --test-url-source is required for Phase-I/Phase-II train/test deduplication"
        )

    urls: Set[str] = set()
    for spec in sources:
        # A newline-delimited local URL list provides a canonical aggregate.
        if spec.uri.startswith("local:"):
            raw_path = spec.uri[len("local:") :]
            path = Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()
            if path.is_file() and path.suffix.lower() in {".txt", ".urls"}:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if line.strip():
                            urls.add(
                                canonicalize_page_url(
                                    line.strip(), location=f"{path}:{line_number}"
                                )
                            )
                continue

        dataset = load_explicit_source(spec)
        for row_index, row in enumerate(dataset):
            value = _require_field(
                row,
                page_url_key,
                location=f"test URL source {spec.uri} row {row_index}",
            )
            urls.update(
                _iter_url_values(
                    value,
                    key=page_url_key,
                    location=f"test URL source {spec.uri} row {row_index}.{page_url_key}",
                )
            )

    if not urls:
        raise ValueError("The supplied test URL sources contain no page URLs")
    return urls


def _phase1_row_overlaps_test(
    row: Mapping[str, Any],
    *,
    field_map: Phase1FieldMap,
    test_urls: Set[str],
    source_name: str,
    row_index: int,
) -> bool:
    value = _require_field(
        row,
        field_map.page_url_key,
        location=f"Phase-I {source_name} row {row_index}",
    )
    page_url = canonicalize_page_url(
        value,
        location=f"Phase-I {source_name} row {row_index}.{field_map.page_url_key}",
    )
    return page_url in test_urls


def _normalize_phase1_row(
    row: Mapping[str, Any],
    row_index: int,
    *,
    source_name: str,
    field_map: Phase1FieldMap,
) -> Dict[str, Any]:
    location = f"Phase-I {source_name} row {row_index}"
    document = _require_nonempty_string(
        _require_field(row, field_map.input_key, location=location),
        location=f"{location}.{field_map.input_key}",
    )
    instruction = _require_nonempty_string(
        _require_field(row, field_map.instruction_key, location=location),
        location=f"{location}.{field_map.instruction_key}",
    )
    target = _require_nonempty_string(
        _require_field(row, field_map.target_key, location=location),
        location=f"{location}.{field_map.target_key}",
    )
    source_row_id = str(
        _require_field(row, field_map.source_id_key, location=location)
    ).strip()
    target_row_id = str(
        _require_field(row, field_map.target_id_key, location=location)
    ).strip()
    if not source_row_id or not target_row_id:
        raise ValueError(f"{location} source/target IDs must be non-empty")
    if source_row_id == target_row_id:
        raise ValueError(
            f"{location} reuses source_row_id as target_row_id; target must be independently held out"
        )

    target_split = _require_nonempty_string(
        _require_field(row, field_map.target_split_key, location=location),
        location=f"{location}.{field_map.target_split_key}",
    )
    normalized_split = target_split.lower().replace("_", "-")
    if normalized_split not in {"heldout", "held-out"}:
        raise ValueError(
            f"{location}.{field_map.target_split_key} must explicitly mark a held-out target"
        )

    input_hash = _text_hash(document)
    target_hash = _text_hash(target)
    if input_hash == target_hash:
        raise ValueError(
            f"{location} has target identical to its input passage; conditional "
            "generation requires a distinct target y"
        )

    page_url = canonicalize_page_url(
        _require_field(row, field_map.page_url_key, location=location),
        location=f"{location}.{field_map.page_url_key}",
    )
    return {
        "docs": [document],
        "question": instruction,
        "answer": target,
        "pos_index": [0],
        "data_type": PHASE1_DATA_TYPES[source_name],
        "pretraining_source": source_name,
        "source_row_id": source_row_id,
        "target_row_id": target_row_id,
        "target_split": "held-out",
        "page_url": page_url,
        "input_sha256": input_hash,
        "target_sha256": target_hash,
        "protocol_version": PROTOCOL_VERSION,
    }


def prepare_phase1_data(
    source_specs: Mapping[str, SourceSpec],
    field_maps: Mapping[str, Phase1FieldMap],
    output_dir: str,
    test_urls: Set[str],
) -> Dataset:
    """Validate and merge the four Phase-I conditional-generation sources."""

    if set(source_specs) != set(PHASE1_SOURCE_COUNTS):
        missing = sorted(set(PHASE1_SOURCE_COUNTS) - set(source_specs))
        extra = sorted(set(source_specs) - set(PHASE1_SOURCE_COUNTS))
        raise ValueError(f"Phase-I requires exactly four sources; missing={missing}, extra={extra}")

    normalized_sources: List[Dataset] = []
    source_manifest: Dict[str, Any] = {}
    for source_name, expected_count in PHASE1_SOURCE_COUNTS.items():
        spec = source_specs[source_name]
        field_map = field_maps[source_name]
        raw = load_explicit_source(spec)

        # Remove page-URL overlaps first; the resulting source directly matches
        # the exact Table A19 count.
        deduplicated = raw.filter(
            lambda row, idx: not _phase1_row_overlaps_test(
                row,
                field_map=field_map,
                test_urls=test_urls,
                source_name=source_name,
                row_index=idx,
            ),
            with_indices=True,
            desc=f"Deduplicating Phase-I {source_name} against test URLs",
        )
        removed = len(raw) - len(deduplicated)
        if len(deduplicated) != expected_count:
            raise ValueError(
                f"Phase-I {source_name} must contain exactly {expected_count:,} rows after "
                f"test-URL deduplication; got {len(deduplicated):,} "
                f"({len(raw):,} raw, {removed:,} removed), as specified by Table A19."
            )

        normalized = deduplicated.map(
            lambda row, idx: _normalize_phase1_row(
                row,
                idx,
                source_name=source_name,
                field_map=field_map,
            ),
            with_indices=True,
            remove_columns=deduplicated.column_names,
            desc=f"Validating Phase-I {source_name}",
        )
        if len(normalized) != expected_count:
            raise RuntimeError(f"Phase-I normalization changed the {source_name} row count")
        source_ids = normalized["source_row_id"]
        target_ids = normalized["target_row_id"]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(f"Phase-I {source_name} source_row_id values must be unique")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"Phase-I {source_name} target_row_id values must be unique")
        normalized_sources.append(normalized)
        source_manifest[source_name] = {
            "source": asdict(spec),
            "field_map": asdict(field_map),
            "data_type": PHASE1_DATA_TYPES[source_name],
            "raw_count": len(raw),
            "test_url_overlaps_removed": removed,
            "final_count": len(normalized),
            "expected_count": expected_count,
            "fingerprint": getattr(normalized, "_fingerprint", None),
        }

    merged = concatenate_datasets(normalized_sources)
    if len(merged) != PHASE1_TOTAL:
        raise RuntimeError(f"Expected {PHASE1_TOTAL:,} total Phase-I rows, got {len(merged):,}")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Phase-I output directory must be empty before materialization: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(output))

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "phase1",
        "objective": "conditional_generation",
        "paper_total": PHASE1_TOTAL,
        "actual_total": len(merged),
        "documents_per_example": 1,
        "answer_type": "string",
        "test_url_count": len(test_urls),
        "test_url_sha256": _url_set_sha256(test_urls),
        "sources": source_manifest,
        "dataset_fingerprint": getattr(merged, "_fingerprint", None),
    }
    with (output / "aria_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return merged


def _normalize_answers(value: Any, *, location: str) -> Tuple[str, List[str]]:
    if isinstance(value, str):
        answer = _require_nonempty_string(value, location=location)
        return answer, [answer]
    if isinstance(value, (list, tuple)) and value:
        answers = [
            _require_nonempty_string(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
        return answers[0], answers
    raise ValueError(f"{location} must be a non-empty string or list of strings")


def _normalize_pos_indices(
    value: Any,
    *,
    n_docs: int,
    location: str,
    allow_empty: bool = False,
) -> Set[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{location} must be a list of gold document indices")
    if not value and not allow_empty:
        raise ValueError(f"{location} must be a non-empty list of gold document indices")
    result: Set[int] = set()
    for position, index in enumerate(value):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{location}[{position}] must be an integer")
        if index < 0 or index >= n_docs:
            raise ValueError(f"{location}[{position}]={index} is outside [0, {n_docs})")
        if index in result:
            raise ValueError(f"{location} contains duplicate index {index}")
        result.add(index)
    return result


def _normalize_gold_doc_ids(
    value: Any, *, location: str, allow_empty: bool = False
) -> List[str]:
    """Validate an explicit, complete corpus-level positive-document set."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{location} must be a list of corpus document IDs")
    if not value and not allow_empty:
        raise ValueError(f"{location} must be a non-empty list of corpus document IDs")
    document_ids = [str(item).strip() for item in value]
    if any(not document_id for document_id in document_ids):
        raise ValueError(f"{location} must contain only non-empty corpus document IDs")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{location} contains duplicate corpus document IDs")
    return document_ids


def _extract_phase2_documents(
    row: Mapping[str, Any],
    *,
    field_map: Phase2FieldMap,
    test_urls: Set[str],
    location: str,
    require_positive_in_top_k: bool = False,
) -> Tuple[List[str], List[str], List[str], List[int], List[str], int, int]:
    raw_docs = _require_field(row, field_map.docs_key, location=location)
    if not isinstance(raw_docs, (list, tuple)) or not raw_docs:
        raise ValueError(f"{location}.{field_map.docs_key} must be a non-empty list")
    positives = _normalize_pos_indices(
        _require_field(row, field_map.pos_index_key, location=location),
        n_docs=len(raw_docs),
        location=f"{location}.{field_map.pos_index_key}",
        allow_empty=True,
    )

    all_strings = all(isinstance(doc, str) for doc in raw_docs)
    all_mappings = all(isinstance(doc, Mapping) for doc in raw_docs)
    if not all_strings and not all_mappings:
        raise ValueError(
            f"{location}.{field_map.docs_key} must contain either all strings or all mappings"
        )

    aligned_urls: Optional[Sequence[Any]] = None
    aligned_ids: Optional[Sequence[Any]] = None
    if all_strings:
        raw_urls = _require_field(row, field_map.page_url_key, location=location)
        if not isinstance(raw_urls, (list, tuple)) or len(raw_urls) != len(raw_docs):
            raise ValueError(
                f"{location}.{field_map.page_url_key} must align one-to-one with string docs"
            )
        aligned_urls = raw_urls
        raw_ids = _require_field(row, field_map.doc_ids_key, location=location)
        if not isinstance(raw_ids, (list, tuple)) or len(raw_ids) != len(raw_docs):
            raise ValueError(
                f"{location}.{field_map.doc_ids_key} must align one-to-one with string docs"
            )
        aligned_ids = raw_ids

    docs: List[str] = []
    page_urls: List[str] = []
    doc_ids: List[str] = []
    retained_positive: Set[int] = set()
    all_gold_doc_ids: List[str] = []
    overlaps_removed = 0
    duplicates_removed = 0
    page_to_retained_index: Dict[str, int] = {}

    for old_index, raw_doc in enumerate(raw_docs):
        doc_location = f"{location}.{field_map.docs_key}[{old_index}]"
        if isinstance(raw_doc, str):
            text = _require_nonempty_string(raw_doc, location=doc_location)
            if aligned_urls is None:
                raise RuntimeError(f"Internal schema error at {doc_location}: URL alignment is absent")
            raw_url = aligned_urls[old_index]
            if aligned_ids is None:
                raise RuntimeError(f"Internal schema error at {doc_location}: ID alignment is absent")
            raw_doc_id = aligned_ids[old_index]
            if raw_doc_id is None:
                raise ValueError(f"{location}.{field_map.doc_ids_key}[{old_index}] is null")
            doc_id = _require_nonempty_string(
                str(raw_doc_id),
                location=f"{location}.{field_map.doc_ids_key}[{old_index}]",
            )
        elif isinstance(raw_doc, Mapping):
            text = _require_nonempty_string(
                _require_field(raw_doc, field_map.doc_text_key, location=doc_location),
                location=f"{doc_location}.{field_map.doc_text_key}",
            )
            raw_url = _require_field(
                raw_doc, field_map.doc_page_url_key, location=doc_location
            )
            raw_doc_id = _require_field(raw_doc, field_map.doc_id_key, location=doc_location)
            if raw_doc_id is None:
                raise ValueError(f"{doc_location}.{field_map.doc_id_key} is null")
            doc_id = _require_nonempty_string(
                str(raw_doc_id),
                location=f"{doc_location}.{field_map.doc_id_key}",
            )
        else:
            raise ValueError(f"{doc_location} must be a string or a mapping")

        page_url = canonicalize_page_url(
            raw_url,
            location=f"{doc_location}.{field_map.doc_page_url_key}",
        )
        if page_url in test_urls:
            overlaps_removed += 1
            continue

        if old_index in positives:
            # Preserve every annotated support that remains in the fixed
            # training corpus, independently of later top-5 truncation.
            all_gold_doc_ids.append(doc_id)

        retained_index = page_to_retained_index.get(page_url)
        if retained_index is not None:
            # Ranked candidate lists follow the evaluation protocol: the first
            # (highest-ranked) passage represents a page. A support annotation
            # on a lower-ranked passage promotes that retained page occurrence.
            duplicates_removed += 1
            if old_index in positives:
                retained_positive.add(retained_index)
            continue

        new_index = len(docs)
        page_to_retained_index[page_url] = new_index
        docs.append(text)
        page_urls.append(page_url)
        doc_ids.append(doc_id)
        if old_index in positives:
            retained_positive.add(new_index)

    if len(docs) < PHASE2_CANDIDATE_DOCS:
        raise ValueError(
            f"{location} has only {len(docs)} non-test candidates after URL filtering; "
            f"{PHASE2_CANDIDATE_DOCS} real candidates are required and documents are never duplicated as padding"
        )

    docs = docs[:PHASE2_CANDIDATE_DOCS]
    page_urls = page_urls[:PHASE2_CANDIDATE_DOCS]
    doc_ids = doc_ids[:PHASE2_CANDIDATE_DOCS]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"{location} contains duplicate stable document IDs")
    final_positive = sorted(
        index for index in retained_positive if index < PHASE2_CANDIDATE_DOCS
    )
    if require_positive_in_top_k and not final_positive:
        raise ValueError(
            f"{location} has no gold document among its first {PHASE2_CANDIDATE_DOCS} candidates"
        )
    return (
        docs,
        page_urls,
        doc_ids,
        final_positive,
        all_gold_doc_ids,
        overlaps_removed,
        duplicates_removed,
    )


def _normalize_augmentation_type(value: Any, *, location: str) -> str:
    augmentation_type = _require_nonempty_string(value, location=location).lower().replace("-", "_")
    aliases = {
        "sub_question": "subquestion",
        "subquestion_decomposition": "subquestion",
        "partial_chain_completion": "partial_chain",
        "entity_variant_augmentation": "entity_variant",
    }
    return aliases.get(augmentation_type, augmentation_type)


def _normalize_phase2_row(
    row: Mapping[str, Any],
    row_index: int,
    *,
    benchmark: str,
    field_map: Phase2FieldMap,
    test_urls: Set[str],
    require_musique_augmentation: bool = True,
    evaluation_mode: bool = False,
) -> Dict[str, Any]:
    location = (
        f"Evaluation {benchmark} row {row_index}"
        if evaluation_mode
        else f"Phase-II {benchmark} row {row_index}"
    )
    question = _require_nonempty_string(
        _require_field(row, field_map.question_key, location=location),
        location=f"{location}.{field_map.question_key}",
    )
    answer, gold_answers = _normalize_answers(
        _require_field(row, field_map.answer_key, location=location),
        location=f"{location}.{field_map.answer_key}",
    )
    source_row_id = str(
        _require_field(row, field_map.source_id_key, location=location)
    ).strip()
    if not source_row_id:
        raise ValueError(f"{location}.{field_map.source_id_key} must be non-empty")

    (
        docs,
        page_urls,
        doc_ids,
        pos_index,
        all_gold_doc_ids,
        overlaps_removed,
        duplicates_removed,
    ) = _extract_phase2_documents(
        row,
        field_map=field_map,
        test_urls=test_urls,
        location=location,
        require_positive_in_top_k=False,
    )
    explicit_gold_doc_ids = _normalize_gold_doc_ids(
        _require_field(row, field_map.gold_doc_ids_key, location=location),
        location=f"{location}.{field_map.gold_doc_ids_key}",
        allow_empty=True,
    )
    # ``pos_index`` remains useful for checking candidate-local annotations,
    # but it cannot define the full-corpus support set: a gold page may be
    # absent from the source candidate list altogether.  Require the explicit
    # annotation for both training and evaluation, and use the local labels
    # only as a consistency check.
    missing_candidate_labels = sorted(
        set(all_gold_doc_ids) - set(explicit_gold_doc_ids)
    )
    if missing_candidate_labels:
        raise ValueError(
            f"{location}.{field_map.gold_doc_ids_key} omits candidate positives: "
            f"{missing_candidate_labels[:3]}"
        )
    if evaluation_mode:
        if field_map.gold_answers_key not in row:
            raise ValueError(
                f"{location} requires field {field_map.gold_answers_key!r} with "
                "a non-empty list of benchmark-provided gold aliases"
            )
        raw_gold_answers = _require_field(
            row, field_map.gold_answers_key, location=location
        )
        if not isinstance(raw_gold_answers, (list, tuple)) or not raw_gold_answers:
            raise ValueError(
                f"{location}.{field_map.gold_answers_key} must be a non-empty "
                "list of benchmark-provided gold aliases"
            )
        explicit_aliases = [
            _require_nonempty_string(
                value,
                location=f"{location}.{field_map.gold_answers_key}[{index}]",
            )
            for index, value in enumerate(raw_gold_answers)
        ]
        # The primary answer and the explicit alias field both originate in the
        # benchmark annotation.  Preserve their union without synthesizing any
        # spelling variants.
        gold_answers = list(dict.fromkeys([*gold_answers, *explicit_aliases]))
    augmentation_type = "original"
    augmentation_metadata: Dict[str, Any] = {
        "augmentation_parent_id": "",
        "construction_method": "original",
        "entity_preserved": False,
        "decomposition_preserved": False,
        "answer_preserved": False,
        "rouge_l": -1.0,
        "partial_state_id": "",
        "partial_state_protocol": "",
        "partial_state_kind": "",
        "partial_hop_count": -1,
        "partial_known_hop_indices": [],
        "partial_frontier_hop_index": -1,
        "partial_mandatory_prefix": False,
        "partial_prompt_sha256": "",
        "partial_selection_sha256": "",
    }
    if benchmark == "musique" and require_musique_augmentation:
        augmentation_type = _normalize_augmentation_type(
            _require_field(row, field_map.augmentation_type_key, location=location),
            location=f"{location}.{field_map.augmentation_type_key}",
        )
        if augmentation_type != "original":
            parent_id = _require_nonempty_string(
                _require_field(row, field_map.augmentation_parent_id_key, location=location),
                location=f"{location}.{field_map.augmentation_parent_id_key}",
            )
            method = _require_nonempty_string(
                _require_field(row, field_map.construction_method_key, location=location),
                location=f"{location}.{field_map.construction_method_key}",
            )
            expected_method = {
                "subquestion": "deterministic_template",
                "partial_chain": MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
                "entity_variant": "gpt-5.5_entity_variant",
            }.get(augmentation_type)
            if method != expected_method:
                raise ValueError(
                    f"{location} construction_method must be {expected_method!r} for "
                    f"{augmentation_type!r}, got {method!r}"
                )
            augmentation_metadata.update(
                augmentation_parent_id=parent_id,
                construction_method=method,
            )
        if augmentation_type == "partial_chain":
            if _require_field(
                row, field_map.needs_candidate_retrieval_key, location=location
            ) is not False:
                raise ValueError(
                    f"{location} requires fresh BGE top-5 candidates before Phase II"
                )
            partial_values = {
                "augmentation_parent_id": augmentation_metadata[
                    "augmentation_parent_id"
                ],
                "partial_state_id": _require_field(
                    row, field_map.partial_state_id_key, location=location
                ),
                "partial_state_protocol": _require_field(
                    row, field_map.partial_state_protocol_key, location=location
                ),
                "partial_state_kind": _require_field(
                    row, field_map.partial_state_kind_key, location=location
                ),
                "partial_hop_count": _require_field(
                    row, field_map.partial_hop_count_key, location=location
                ),
                "partial_known_hop_indices": _require_field(
                    row, field_map.partial_known_hop_indices_key, location=location
                ),
                "partial_frontier_hop_index": _require_field(
                    row, field_map.partial_frontier_hop_index_key, location=location
                ),
            }
            validate_musique_partial_metadata(partial_values)
            if source_row_id != partial_values["partial_state_id"]:
                raise ValueError(
                    f"{location} source_row_id must equal its partial_state_id"
                )
            mandatory_prefix = _require_field(
                row, field_map.partial_mandatory_prefix_key, location=location
            )
            if not isinstance(mandatory_prefix, bool):
                raise ValueError(f"{location}.partial_mandatory_prefix must be boolean")
            prompt_sha256 = _require_nonempty_string(
                _require_field(
                    row, field_map.partial_prompt_sha256_key, location=location
                ),
                location=f"{location}.partial_prompt_sha256",
            )
            if prompt_sha256 != hashlib.sha256(question.encode("utf-8")).hexdigest():
                raise ValueError(f"{location} partial prompt SHA-256 mismatch")
            selection_sha256 = _require_nonempty_string(
                _require_field(
                    row, field_map.partial_selection_sha256_key, location=location
                ),
                location=f"{location}.partial_selection_sha256",
            )
            if re.fullmatch(r"[0-9a-f]{64}", selection_sha256) is None:
                raise ValueError(
                    f"{location}.partial_selection_sha256 must be lowercase SHA-256"
                )
            augmentation_metadata.update(
                partial_state_id=str(partial_values["partial_state_id"]),
                partial_state_protocol=str(partial_values["partial_state_protocol"]),
                partial_state_kind=str(partial_values["partial_state_kind"]),
                partial_hop_count=int(partial_values["partial_hop_count"]),
                partial_known_hop_indices=list(
                    partial_values["partial_known_hop_indices"]
                ),
                partial_frontier_hop_index=int(
                    partial_values["partial_frontier_hop_index"]
                ),
                partial_mandatory_prefix=mandatory_prefix,
                partial_prompt_sha256=prompt_sha256,
                partial_selection_sha256=selection_sha256,
                answer_preserved=True,
            )
        if augmentation_type == "entity_variant":
            entity_preserved = _require_field(
                row, field_map.entity_preserved_key, location=location
            )
            decomposition_preserved = _require_field(
                row, field_map.decomposition_preserved_key, location=location
            )
            answer_preserved = _require_field(
                row, field_map.answer_preserved_key, location=location
            )
            rouge_l = _require_field(row, field_map.rouge_l_key, location=location)
            if entity_preserved is not True:
                raise ValueError(
                    f"{location} must satisfy exact spaCy named-entity preservation"
                )
            if decomposition_preserved is not True or answer_preserved is not True:
                raise ValueError(
                    f"{location} must preserve the original decomposition and answer"
                )
            if isinstance(rouge_l, bool) or not isinstance(rouge_l, (int, float)) or rouge_l < 0.4:
                raise ValueError(f"{location} entity variant requires ROUGE-L >= 0.4")
            augmentation_metadata.update(
                entity_preserved=True,
                decomposition_preserved=True,
                answer_preserved=True,
                rouge_l=float(rouge_l),
            )

    result: Dict[str, Any] = {
        "question": question,
        "answer": answer,
        "gold_answers": gold_answers,
        "docs": docs,
        # Singular field name retained intentionally: it is an aligned list with
        # exactly one canonical page URL per candidate document.
        "page_url": page_urls,
        "doc_ids": doc_ids,
        "gold_doc_ids": explicit_gold_doc_ids,
        "pos_index": pos_index,
        "data_type": "qa",
        "benchmark": benchmark,
        "source_row_id": source_row_id,
        "candidate_count": PHASE2_CANDIDATE_DOCS,
        "test_url_overlaps_removed": overlaps_removed,
        "duplicate_candidate_urls_removed": duplicates_removed,
        "protocol_version": PROTOCOL_VERSION,
        "augmentation_type": augmentation_type,
        **augmentation_metadata,
    }
    return result


def _musique_entities(text: str) -> Set[str]:
    global _MUSIQUE_NLP
    if _MUSIQUE_NLP is None:
        try:
            import spacy

            _MUSIQUE_NLP = spacy.load("en_core_web_sm")
        except Exception as exc:
            raise RuntimeError(
                "MuSiQue validation requires spaCy en_core_web_sm"
            ) from exc
    return {
        f"{entity.label_}:{entity.text.casefold().strip()}"
        for entity in _MUSIQUE_NLP(text).ents
        if entity.text.strip()
    }


def _rouge_l_f1(left: str, right: str) -> float:
    left_tokens = left.casefold().split()
    right_tokens = right.casefold().split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    previous = [0] * (len(right_tokens) + 1)
    for left_token in left_tokens:
        current = [0]
        for index, right_token in enumerate(right_tokens, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision = lcs / len(left_tokens)
    recall = lcs / len(right_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _validate_musique_augmentation(dataset: Dataset) -> Dict[str, int]:
    counts: Dict[str, int] = {key: 0 for key in MUSIQUE_AUGMENTATION_COUNTS}
    unexpected: Set[str] = set()
    for value in dataset["augmentation_type"]:
        if value in counts:
            counts[value] += 1
        else:
            unexpected.add(value)
    if unexpected:
        raise ValueError(f"MuSiQue has unexpected augmentation_type values: {sorted(unexpected)}")
    if counts != dict(MUSIQUE_AUGMENTATION_COUNTS):
        raise ValueError(
            "MuSiQue augmentation counts do not match Appendix A.33: "
            f"expected={dict(MUSIQUE_AUGMENTATION_COUNTS)}, actual={counts}"
        )
    originals = {
        str(row["source_row_id"]): row
        for row in dataset
        if row["augmentation_type"] == "original"
    }
    partial_state_ids = [
        str(row["partial_state_id"])
        for row in dataset
        if row["augmentation_type"] == "partial_chain"
    ]
    if len(partial_state_ids) != MUSIQUE_PARTIAL_CHAIN_TARGET:
        raise ValueError(
            "MuSiQue partial-state protocol requires exactly "
            f"{MUSIQUE_PARTIAL_CHAIN_TARGET:,} rows"
        )
    if len(set(partial_state_ids)) != len(partial_state_ids):
        raise ValueError("MuSiQue partial_state_id values must be globally unique")
    selected_digest = hashlib.sha256(
        ("\n".join(sorted(partial_state_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    selection_digests = {
        str(row["partial_selection_sha256"])
        for row in dataset
        if row["augmentation_type"] == "partial_chain"
    }
    if selection_digests != {selected_digest}:
        raise ValueError(
            "MuSiQue partial rows do not share their exact selected-state digest"
        )
    for row_index, row in enumerate(dataset):
        augmentation_type = row["augmentation_type"]
        if augmentation_type == "original":
            continue
        parent_id = str(row["augmentation_parent_id"])
        parent = originals.get(parent_id)
        if parent is None:
            raise ValueError(
                f"MuSiQue augmented row {row_index} references unknown original {parent_id!r}"
            )
        if list(row["gold_answers"]) != list(parent["gold_answers"]):
            raise ValueError(
                f"MuSiQue augmented row {row_index} must preserve its parent answers"
            )
        if augmentation_type == "entity_variant":
            if _musique_entities(row["question"]) != _musique_entities(parent["question"]):
                raise ValueError(
                    f"MuSiQue entity variant row {row_index} must satisfy recomputed "
                    "NER preservation"
                )
            if _rouge_l_f1(row["question"], parent["question"]) < 0.4:
                raise ValueError(
                    f"MuSiQue entity variant row {row_index} must satisfy recomputed "
                    "ROUGE-L >= 0.4"
                )
    return counts


def _validate_musique_audit_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Require the two human checks reported in Appendix A.33."""
    required = {
        "human_filter_sample_fraction": 0.05,
        "human_ai_kappa": 0.81,
        "answer_audit_sample_size": 200,
        "answer_preserved_count": 196,
    }
    normalized: Dict[str, Any] = {}
    for key, expected in required.items():
        if key not in manifest:
            raise ValueError(f"MuSiQue verification manifest requires {key!r}")
        value = manifest[key]
        if isinstance(expected, float):
            if not isinstance(value, (int, float)) or abs(float(value) - expected) > 1e-9:
                raise ValueError(f"MuSiQue audit {key} must be {expected}, got {value}")
            normalized[key] = float(value)
        else:
            if value != expected:
                raise ValueError(f"MuSiQue audit {key} must be {expected}, got {value}")
            normalized[key] = int(value)
    normalized["answer_preservation_rate"] = 196 / 200
    return normalized


def _benchmark_seed(epoch_seed: int, benchmark: str) -> int:
    digest = hashlib.sha256(f"{epoch_seed}:{benchmark}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def prepare_musique_partial_chains(
    source_spec: SourceSpec,
    output_dir: str,
    *,
    parent_id_key: str = "source_row_id",
    question_key: str = "question",
    answer_key: str = "answer",
    decomposition_key: str = "question_decomposition",
) -> Dataset:
    """Materialize exactly 70,845 partial queries before candidate retrieval."""
    raw = load_explicit_source(source_spec)
    rows, selection_manifest = build_musique_partial_rows(
        raw,
        target_count=MUSIQUE_PARTIAL_CHAIN_TARGET,
        parent_id_key=parent_id_key,
        question_key=question_key,
        answer_key=answer_key,
        decomposition_key=decomposition_key,
    )
    if len(rows) != MUSIQUE_PARTIAL_CHAIN_TARGET:
        raise RuntimeError("MuSiQue partial-chain builder returned the wrong count")
    dataset = Dataset.from_list(rows)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"MuSiQue partial output directory must be empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "musique_partial_chain_generation",
        "source": asdict(source_spec),
        "source_fields": {
            "parent_id": parent_id_key,
            "question": question_key,
            "answer": answer_key,
            "decomposition": decomposition_key,
        },
        "count": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "partial_chain": selection_manifest,
        "candidate_retrieval_contract": {
            "required": True,
            "model": "BAAI/bge-large-en-v1.5",
            "top_k": PHASE2_CANDIDATE_DOCS,
            "scope": "page_url_deduplicated_phase2_training_corpus",
            "reason": "the augmented query differs from its parent",
            "completion_marker": "set needs_candidate_retrieval=false only after attaching fresh aligned candidates",
        },
    }
    with (output / "aria_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return dataset


def prepare_phase2_data(
    source_specs: Mapping[str, SourceSpec],
    field_maps: Mapping[str, Phase2FieldMap],
    output_dir: str,
    test_urls: Set[str],
    epoch_seeds: Sequence[int],
    musique_audit_manifest: Mapping[str, Any],
    training_retrieval_index_sha256: str,
) -> DatasetDict:
    """Create five protocol-verifiable, class-balanced Phase-II epoch shards."""

    if set(source_specs) != set(PHASE2_BENCHMARKS):
        missing = sorted(set(PHASE2_BENCHMARKS) - set(source_specs))
        extra = sorted(set(source_specs) - set(PHASE2_BENCHMARKS))
        raise ValueError(f"Phase-II requires all four benchmark sources; missing={missing}, extra={extra}")
    if len(epoch_seeds) != 5 or len(set(epoch_seeds)) != 5:
        raise ValueError("Phase-II requires exactly five distinct fixed epoch seeds")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", training_retrieval_index_sha256):
        raise ValueError("training retrieval index digest must be a 64-character SHA-256")
    musique_audit = _validate_musique_audit_manifest(musique_audit_manifest)

    pools: Dict[str, Dataset] = {}
    pool_manifest: Dict[str, Any] = {}
    for benchmark in PHASE2_BENCHMARKS:
        spec = source_specs[benchmark]
        field_map = field_maps[benchmark]
        raw = load_explicit_source(spec)
        expected_pool_count = PHASE2_POOL_COUNTS[benchmark]
        if len(raw) != expected_pool_count:
            raise ValueError(
                f"Phase-II {benchmark} pool must contain exactly {expected_pool_count:,} rows "
                f"(Table A20); got {len(raw):,}"
            )
        normalized = raw.map(
            lambda row, idx: _normalize_phase2_row(
                row,
                idx,
                benchmark=benchmark,
                field_map=field_map,
                test_urls=test_urls,
            ),
            with_indices=True,
            remove_columns=raw.column_names,
            desc=f"Validating Phase-II {benchmark}",
        )
        if len(normalized) < PHASE2_SAMPLES_PER_BENCHMARK:
            raise ValueError(
                f"Phase-II {benchmark} has {len(normalized):,} valid rows, fewer than "
                f"the required {PHASE2_SAMPLES_PER_BENCHMARK:,}; sampling with replacement is forbidden"
            )
        source_row_ids = normalized["source_row_id"]
        if len(set(source_row_ids)) != len(source_row_ids):
            raise ValueError(
                f"Phase-II {benchmark}.{field_map.source_id_key} must uniquely identify every pool row"
            )

        augmentation_counts = None
        if benchmark == "musique":
            augmentation_counts = _validate_musique_augmentation(normalized)
        pools[benchmark] = normalized
        pool_manifest[benchmark] = {
            "source": asdict(spec),
            "field_map": asdict(field_map),
            "count": len(normalized),
            "expected_count": expected_pool_count,
            "fingerprint": getattr(normalized, "_fingerprint", None),
            "augmentation_counts": augmentation_counts,
        }

    candidate_order_hasher = hashlib.sha256()
    for benchmark in sorted(pools):
        pool = pools[benchmark]
        for source_row_id, document_ids in zip(
            pool["source_row_id"], pool["doc_ids"]
        ):
            candidate_order_hasher.update(
                json.dumps(
                    [benchmark, str(source_row_id), [str(value) for value in document_ids]],
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            candidate_order_hasher.update(b"\n")

    epoch_shards: Dict[str, Dataset] = {}
    epoch_manifest: List[Dict[str, Any]] = []
    for epoch_index, epoch_seed in enumerate(epoch_seeds):
        benchmark_shards: List[Dataset] = []
        benchmark_records: Dict[str, Any] = {}
        for benchmark in PHASE2_BENCHMARKS:
            pool = pools[benchmark]
            sample_seed = _benchmark_seed(int(epoch_seed), benchmark)
            sample_indices = random.Random(sample_seed).sample(
                range(len(pool)), PHASE2_SAMPLES_PER_BENCHMARK
            )
            if len(sample_indices) != len(set(sample_indices)):
                raise RuntimeError(f"Internal error: {benchmark} epoch {epoch_index} sampling repeated a row")
            shard = pool.select(sample_indices)
            shard = shard.add_column("sampling_epoch", [epoch_index] * len(shard))
            shard = shard.add_column("epoch_seed", [int(epoch_seed)] * len(shard))
            shard = shard.add_column("benchmark_sample_seed", [sample_seed] * len(shard))
            shard = shard.add_column(
                "view_id",
                [f"epoch-{epoch_index}:{benchmark}:{row_id}" for row_id in shard["source_row_id"]],
            )
            benchmark_shards.append(shard)
            benchmark_records[benchmark] = {
                "pool_count": len(pool),
                "sample_count": len(shard),
                "sample_seed": sample_seed,
            }

        merged = concatenate_datasets(benchmark_shards)
        permutation = list(range(len(merged)))
        random.Random(int(epoch_seed)).shuffle(permutation)
        merged = merged.select(permutation)
        if len(merged) != PHASE2_SAMPLES_PER_EPOCH:
            raise RuntimeError(
                f"Epoch {epoch_index} must contain {PHASE2_SAMPLES_PER_EPOCH:,} rows, got {len(merged):,}"
            )
        epoch_name = f"epoch_{epoch_index:03d}"
        epoch_shards[epoch_name] = merged
        epoch_manifest.append(
            {
                "epoch": epoch_index,
                "split": epoch_name,
                "seed": int(epoch_seed),
                "count": len(merged),
                "benchmarks": benchmark_records,
                "fingerprint": getattr(merged, "_fingerprint", None),
            }
        )

    scheduled_views = sum(len(shard) for shard in epoch_shards.values())
    if scheduled_views != PHASE2_SAMPLES_PER_EPOCH * 5:
        raise RuntimeError(f"Expected 192,000 scheduled Phase-II views, got {scheduled_views:,}")

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Phase-II output directory must be empty before materialization: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    # The output root is a DatasetDict with one epoch_XXX split per training
    # epoch. The separate schedule_once Dataset represents one pass over all
    # 192k scheduled views (max_epochs=1).
    epoch_dataset = DatasetDict(epoch_shards)
    epoch_dataset.save_to_disk(str(output))
    schedule_once = concatenate_datasets(list(epoch_shards.values()))
    schedule_once.save_to_disk(str(output / "schedule_once"))
    DatasetDict(pools).save_to_disk(str(output / "validated_pools"))

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "phase2",
        "sampling": "uniform_without_replacement_within_each_benchmark_and_epoch",
        "samples_per_benchmark_per_epoch": PHASE2_SAMPLES_PER_BENCHMARK,
        "samples_per_epoch": PHASE2_SAMPLES_PER_EPOCH,
        "epochs": 5,
        "epoch_seed_schedule": [int(seed) for seed in epoch_seeds],
        "scheduled_example_views": scheduled_views,
        "candidate_documents_per_example": PHASE2_CANDIDATE_DOCS,
        "training_retrieval": {
            "model": "BAAI/bge-large-en-v1.5",
            "top_k": 5,
            "index_sha256": training_retrieval_index_sha256.lower(),
            "candidate_order_sha256": candidate_order_hasher.hexdigest(),
            "corpus_scope": "page_url_deduplicated",
            "contract": "source candidate order is training-index BGE top-5 before scheduled sampling",
        },
        "musique_human_audit": musique_audit,
        "test_url_count": len(test_urls),
        "test_url_sha256": _url_set_sha256(test_urls),
        "pool_sources": pool_manifest,
        "epoch_shards": epoch_manifest,
        "consumption": {
            "preferred": "select epoch_000 ... epoch_004 once each while resuming the same run",
            "alternative": "consume schedule_once exactly once (trainer max_epochs=1)",
            "warning": "use each epoch shard once, or use schedule_once for one trainer epoch",
        },
    }
    with (output / "aria_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return epoch_dataset


def prepare_evaluation_data(
    source_specs: Mapping[str, SourceSpec],
    field_maps: Mapping[str, Phase2FieldMap],
    output_dir: str,
) -> DatasetDict:
    """Validate official evaluation splits without applying train/test filtering."""

    datasets: Dict[str, Dataset] = {}
    split_manifest: Dict[str, Any] = {}
    for benchmark in PHASE2_BENCHMARKS:
        raw = load_explicit_source(source_specs[benchmark])
        expected_count = EVALUATION_COUNTS[benchmark]
        if len(raw) != expected_count:
            raise ValueError(
                f"Evaluation {benchmark} must contain exactly {expected_count:,} official rows; "
                f"got {len(raw):,}"
            )
        normalized = raw.map(
            lambda row, idx: _normalize_phase2_row(
                row,
                idx,
                benchmark=benchmark,
                field_map=field_maps[benchmark],
                test_urls=set(),
                require_musique_augmentation=False,
                evaluation_mode=True,
            ),
            with_indices=True,
            remove_columns=raw.column_names,
            desc=f"Validating evaluation {benchmark}",
        )
        datasets[benchmark] = normalized
        split_manifest[benchmark] = {
            "source": asdict(source_specs[benchmark]),
            "field_map": asdict(field_maps[benchmark]),
            "count": len(normalized),
            "expected_count": expected_count,
            "fingerprint": getattr(normalized, "_fingerprint", None),
        }

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Evaluation output directory must be empty before materialization: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    result = DatasetDict(datasets)
    result.save_to_disk(str(output))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "evaluation",
        "retrieval": "full_kilt_corpus_at_evaluation_time",
        "retrieval_gold_contract": EVALUATION_GOLD_DOCUMENT_CONTRACT,
        "answer_alias_contract": EVALUATION_ANSWER_ALIAS_CONTRACT,
        "splits": split_manifest,
    }
    with (output / "aria_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return result


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    phase1_group = parser.add_argument_group("Phase-I explicit sources")
    for source_name in PHASE1_SOURCE_COUNTS:
        flag = source_name.replace("_", "-")
        dest = source_name
        phase1_group.add_argument(f"--phase1-{flag}-source", dest=f"phase1_{dest}_source")
        phase1_group.add_argument(
            f"--phase1-{flag}-split", dest=f"phase1_{dest}_split", default="train"
        )
        phase1_group.add_argument(f"--phase1-{flag}-config", dest=f"phase1_{dest}_config")
        for option, default in asdict(Phase1FieldMap()).items():
            option_flag = option.replace("_", "-")
            phase1_group.add_argument(
                f"--phase1-{flag}-{option_flag}",
                dest=f"phase1_{dest}_{option}",
                default=default,
            )

    phase2_group = parser.add_argument_group("Phase-II/evaluation explicit sources")
    for benchmark in PHASE2_BENCHMARKS:
        flag = benchmark.replace("_", "-")
        dest = benchmark.replace("-", "_")
        phase2_group.add_argument(f"--phase2-{flag}-source", dest=f"phase2_{dest}_source")
        phase2_group.add_argument(
            f"--phase2-{flag}-split", dest=f"phase2_{dest}_split", default="train"
        )
        phase2_group.add_argument(f"--phase2-{flag}-config", dest=f"phase2_{dest}_config")
        for option, default in asdict(Phase2FieldMap()).items():
            option_flag = option.replace("_", "-")
            phase2_group.add_argument(
                f"--phase2-{flag}-{option_flag}",
                dest=f"phase2_{dest}_{option}",
                default=default,
            )


def _collect_phase1_args(
    args: argparse.Namespace,
) -> Tuple[Dict[str, SourceSpec], Dict[str, Phase1FieldMap]]:
    specs: Dict[str, SourceSpec] = {}
    maps: Dict[str, Phase1FieldMap] = {}
    missing: List[str] = []
    for source_name in PHASE1_SOURCE_COUNTS:
        source = getattr(args, f"phase1_{source_name}_source")
        if not source:
            missing.append(f"--phase1-{source_name.replace('_', '-')}-source")
            continue
        specs[source_name] = SourceSpec(
            uri=source,
            split=getattr(args, f"phase1_{source_name}_split"),
            config=getattr(args, f"phase1_{source_name}_config"),
        )
        maps[source_name] = Phase1FieldMap(
            **{
                key: getattr(args, f"phase1_{source_name}_{key}")
                for key in asdict(Phase1FieldMap())
            }
        )
    if missing:
        raise ValueError("Phase-I requires source arguments: " + ", ".join(missing))
    return specs, maps


def _collect_phase2_args(
    args: argparse.Namespace,
    *,
    split_override: Optional[str] = None,
) -> Tuple[Dict[str, SourceSpec], Dict[str, Phase2FieldMap]]:
    specs: Dict[str, SourceSpec] = {}
    maps: Dict[str, Phase2FieldMap] = {}
    missing: List[str] = []
    for benchmark in PHASE2_BENCHMARKS:
        dest = benchmark.replace("-", "_")
        source = getattr(args, f"phase2_{dest}_source")
        if not source:
            missing.append(f"--phase2-{benchmark}-source")
            continue
        specs[benchmark] = SourceSpec(
            uri=source,
            split=split_override or getattr(args, f"phase2_{dest}_split"),
            config=getattr(args, f"phase2_{dest}_config"),
        )
        maps[benchmark] = Phase2FieldMap(
            **{
                key: getattr(args, f"phase2_{dest}_{key}")
                for key in asdict(Phase2FieldMap())
            }
        )
    if missing:
        raise ValueError("Phase-II requires source arguments: " + ", ".join(missing))
    return specs, maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare paper-faithful ARIA datasets")
    parser.add_argument(
        "--stage",
        choices=("phase1", "phase2", "eval", "musique-partial", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./data/aria")
    parser.add_argument(
        "--epoch-seeds",
        type=int,
        nargs=5,
        default=None,
        metavar=("E0", "E1", "E2", "E3", "E4"),
        help=(
            "Five distinct Phase-II sampling seeds recorded in the data manifest."
        ),
    )
    parser.add_argument(
        "--test-url-source",
        action="append",
        default=[],
        help="Repeatable explicit local:/hf: source containing the official test page_url set",
    )
    parser.add_argument("--test-url-split", default="test")
    parser.add_argument("--test-url-config")
    parser.add_argument("--test-url-key", default="page_url")
    parser.add_argument(
        "--musique-audit-manifest",
        help="JSON record for the Appendix A.33 human filter and 200-example answer check",
    )
    parser.add_argument(
        "--musique-decomposition-key",
        default="question_decomposition",
        help="Ordered 2-4-hop decomposition field used by --stage musique-partial",
    )
    parser.add_argument(
        "--training-retrieval-index-sha256",
        "--normal-retrieval-index-sha256",
        dest="training_retrieval_index_sha256",
        help=(
            "SHA-256 of the page-URL-deduplicated Phase-II BGE index used to "
            "produce ordered training top-5 candidates"
        ),
    )
    parser.add_argument(
        "--eval-split",
        default="validation",
        help="Split loaded from each Phase-II source when --stage eval/all is requested",
    )
    _add_source_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.output_dir)

    if args.stage == "musique-partial":
        if not args.phase2_musique_source:
            raise ValueError(
                "--stage musique-partial requires --phase2-musique-source"
            )
        prepare_musique_partial_chains(
            SourceSpec(
                uri=args.phase2_musique_source,
                split=args.phase2_musique_split,
                config=args.phase2_musique_config,
            ),
            str(root / "musique_partial"),
            parent_id_key=args.phase2_musique_source_id_key,
            question_key=args.phase2_musique_question_key,
            answer_key=args.phase2_musique_answer_key,
            decomposition_key=args.musique_decomposition_key,
        )
        return

    test_urls: Set[str] = set()
    if args.stage in {"phase1", "phase2", "all"}:
        test_url_specs = [
            SourceSpec(uri=uri, split=args.test_url_split, config=args.test_url_config)
            for uri in args.test_url_source
        ]
        test_urls = load_test_url_set(test_url_specs, page_url_key=args.test_url_key)

    if args.stage in {"phase1", "all"}:
        specs, maps = _collect_phase1_args(args)
        prepare_phase1_data(specs, maps, str(root / "phase1"), test_urls)

    if args.stage in {"phase2", "all"}:
        if args.epoch_seeds is None:
            raise ValueError(
                "Phase II requires five distinct values through --epoch-seeds; "
                "training-run seeds are configured independently"
            )
        if not args.musique_audit_manifest:
            raise ValueError("Phase II requires --musique-audit-manifest")
        if not args.training_retrieval_index_sha256:
            raise ValueError("Phase II requires --training-retrieval-index-sha256")
        audit_path = Path(args.musique_audit_manifest)
        if not audit_path.is_file():
            raise FileNotFoundError(
                f"MuSiQue verification manifest must be an existing file: {audit_path}"
            )
        with audit_path.open("r", encoding="utf-8") as handle:
            musique_audit = json.load(handle)
        if not isinstance(musique_audit, Mapping):
            raise ValueError("MuSiQue verification manifest must contain one JSON object")
        specs, maps = _collect_phase2_args(args)
        prepare_phase2_data(
            specs,
            maps,
            str(root / "phase2"),
            test_urls,
            epoch_seeds=args.epoch_seeds,
            musique_audit_manifest=musique_audit,
            training_retrieval_index_sha256=args.training_retrieval_index_sha256,
        )

    if args.stage in {"eval", "all"}:
        specs, maps = _collect_phase2_args(args, split_override=args.eval_split)
        prepare_evaluation_data(specs, maps, str(root / "eval"))


if __name__ == "__main__":
    main()
