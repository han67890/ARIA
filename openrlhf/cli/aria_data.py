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
from openrlhf.utils.aria_provenance import (
    EVALUATION_ANSWER_CONTRACT,
    EVALUATION_GOLD_DOCUMENT_CONTRACT,
    file_sha256,
)


PROTOCOL_VERSION = "aria-paper-v1"
PAPER_PHASE2_EPOCH_SEEDS: Tuple[int, ...] = (42, 123, 456, 789, 2024)
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

# Appendix A.33 historical derived-artifact counts.
MUSIQUE_AUGMENTATION_COUNTS: Mapping[str, int] = {
    "original": 19_938,
    "subquestion": 52_107,
    "partial_chain": 70_845,
    "entity_variant": 25_855,
}
MUSIQUE_DERIVED_MANIFEST_PROTOCOL = "aria-musique-derived-source-v1"


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
    decomposition_key: str = "question_decomposition"


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
    _instruction = _require_nonempty_string(
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
        # Phase I retains the submission's four target families but conditions the
        # decoder only on F(d), never on the source task instruction.
        "question": "",
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
    raw_answer = _require_field(row, field_map.answer_key, location=location)
    if evaluation_mode:
        answer = _require_nonempty_string(
            raw_answer,
            location=f"{location}.{field_map.answer_key}",
        )
        gold_answers = [answer]
    else:
        answer, gold_answers = _normalize_answers(
            raw_answer,
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
    augmentation_type = "original"
    augmentation_metadata: Dict[str, Any] = {
        "augmentation_parent_id": "",
        "construction_method": "original",
        "entity_preserved": False,
        "decomposition_preserved": False,
        "answer_preserved": False,
        "rouge_l": -1.0,
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
            augmentation_metadata.update(
                augmentation_parent_id=parent_id,
                construction_method=method,
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
    if not evaluation_mode:
        result["gold_answers"] = gold_answers
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


def _canonical_manifest_value(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "MuSiQue derived rows must contain JSON-serializable values"
        ) from exc


def _update_canonical_row_digest(hasher: Any, row: Mapping[str, Any]) -> None:
    hasher.update(_canonical_manifest_value(dict(row)).encode("utf-8"))
    hasher.update(b"\n")


def _validate_musique_derived_manifest(
    raw: Dataset,
    field_map: Phase2FieldMap,
    manifest_path: str,
) -> Dict[str, Any]:
    """Verify the artifact-owner-supplied historical MuSiQue derived source."""
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MuSiQue derived manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            manifest = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("MuSiQue derived manifest must be valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("MuSiQue derived manifest must be a JSON object")
    if manifest.get("protocol") != MUSIQUE_DERIVED_MANIFEST_PROTOCOL:
        raise ValueError(
            "MuSiQue derived manifest protocol must be "
            f"{MUSIQUE_DERIVED_MANIFEST_PROTOCOL!r}"
        )
    if manifest.get("total_count") != PHASE2_POOL_COUNTS["musique"]:
        raise ValueError("MuSiQue derived manifest total_count is not 168,745")
    if manifest.get("family_counts") != dict(MUSIQUE_AUGMENTATION_COUNTS):
        raise ValueError("MuSiQue derived manifest family_counts do not match the paper")
    if manifest.get("source_columns") != list(raw.column_names):
        raise ValueError(
            "MuSiQue derived manifest source_columns do not match the source schema"
        )
    if manifest.get("field_map") != asdict(field_map):
        raise ValueError(
            "MuSiQue derived manifest field_map does not match the configured schema"
        )

    source_hasher = hashlib.sha256()
    family_hashers = {
        family: hashlib.sha256() for family in MUSIQUE_AUGMENTATION_COUNTS
    }
    parent_link_hasher = hashlib.sha256()
    family_counts = {family: 0 for family in MUSIQUE_AUGMENTATION_COUNTS}
    originals: Dict[str, Tuple[str, str, Set[str]]] = {}
    seen_source_ids: Set[str] = set()

    for row_index, raw_row in enumerate(raw):
        row = dict(raw_row)
        family = _normalize_augmentation_type(
            _require_field(
                row,
                field_map.augmentation_type_key,
                location=f"MuSiQue derived row {row_index}",
            ),
            location=f"MuSiQue derived row {row_index}.augmentation_type",
        )
        if family not in family_counts:
            raise ValueError(
                f"MuSiQue derived row {row_index} has unknown family {family!r}"
            )
        source_id = str(
            _require_field(
                row,
                field_map.source_id_key,
                location=f"MuSiQue derived row {row_index}",
            )
        ).strip()
        if not source_id:
            raise ValueError(f"MuSiQue derived row {row_index} has an empty source ID")
        if source_id in seen_source_ids:
            raise ValueError(
                f"MuSiQue derived row {row_index} repeats source ID {source_id!r}"
            )
        seen_source_ids.add(source_id)
        decomposition = _require_field(
            row,
            field_map.decomposition_key,
            location=f"MuSiQue derived row {row_index}",
        )
        if (
            not isinstance(decomposition, Sequence)
            or isinstance(decomposition, (str, bytes, bytearray))
            or not decomposition
        ):
            raise ValueError(
                f"MuSiQue derived row {row_index} requires a non-empty decomposition"
            )
        answer = _require_field(
            row,
            field_map.answer_key,
            location=f"MuSiQue derived row {row_index}",
        )
        parent_id = ""
        if family == "original":
            hop_answers = {
                _canonical_manifest_value(hop["answer"])
                for hop in decomposition
                if isinstance(hop, Mapping) and "answer" in hop
            }
            originals[source_id] = (
                _canonical_manifest_value(answer),
                _canonical_manifest_value(decomposition),
                hop_answers,
            )
        else:
            parent_id = _require_nonempty_string(
                _require_field(
                    row,
                    field_map.augmentation_parent_id_key,
                    location=f"MuSiQue derived row {row_index}",
                ),
                location=f"MuSiQue derived row {row_index}.augmentation_parent_id",
            )
        family_counts[family] += 1
        _update_canonical_row_digest(source_hasher, row)
        _update_canonical_row_digest(family_hashers[family], row)
        parent_link_hasher.update(
            _canonical_manifest_value(
                [family, source_id, parent_id, answer, decomposition]
            ).encode("utf-8")
        )
        parent_link_hasher.update(b"\n")

    if len(originals) != MUSIQUE_AUGMENTATION_COUNTS["original"]:
        raise ValueError("MuSiQue original family count does not match the manifest")
    if family_counts != dict(MUSIQUE_AUGMENTATION_COUNTS):
        raise ValueError(
            "MuSiQue derived source family counts do not match the paper: "
            f"{family_counts}"
        )
    # A second bounded-memory pass verifies every derived-parent contract after
    # the complete original-ID table has been collected.
    for row_index, raw_row in enumerate(raw):
        row = dict(raw_row)
        family = _normalize_augmentation_type(
            row[field_map.augmentation_type_key],
            location=f"MuSiQue derived row {row_index}.augmentation_type",
        )
        if family == "original":
            continue
        parent_id = str(row[field_map.augmentation_parent_id_key]).strip()
        parent_contract = originals.get(parent_id)
        if parent_contract is None:
            raise ValueError(
                f"MuSiQue derived row {row_index} references unknown parent {parent_id!r}"
            )
        parent_answer, parent_decomposition, parent_hop_answers = parent_contract
        row_decomposition = _canonical_manifest_value(row[field_map.decomposition_key])
        if row_decomposition != parent_decomposition:
            raise ValueError(
                f"MuSiQue derived row {row_index} changes its parent decomposition"
            )
        row_answer = _canonical_manifest_value(row[field_map.answer_key])
        if family in {"partial_chain", "entity_variant"}:
            if row_answer != parent_answer:
                raise ValueError(
                    f"MuSiQue derived row {row_index} changes its parent answer"
                )
        elif family == "subquestion" and row_answer not in parent_hop_answers:
            raise ValueError(
                f"MuSiQue subquestion row {row_index} answer is not an annotated hop answer"
            )

    computed_family_digests = {
        family: hasher.hexdigest() for family, hasher in family_hashers.items()
    }
    declared_family_digests = manifest.get("family_content_sha256")
    if declared_family_digests != computed_family_digests:
        raise ValueError("MuSiQue derived manifest family content digests do not match")
    if manifest.get("source_content_sha256") != source_hasher.hexdigest():
        raise ValueError("MuSiQue derived manifest source content digest does not match")
    if manifest.get("parent_link_sha256") != parent_link_hasher.hexdigest():
        raise ValueError("MuSiQue derived manifest parent-link digest does not match")
    return {
        "protocol": MUSIQUE_DERIVED_MANIFEST_PROTOCOL,
        "path": str(path.resolve()),
        "manifest_sha256": file_sha256(path),
        "source_content_sha256": source_hasher.hexdigest(),
        "family_content_sha256": computed_family_digests,
        "parent_link_sha256": parent_link_hasher.hexdigest(),
        "family_counts": family_counts,
        "source_columns": list(raw.column_names),
        "field_map": asdict(field_map),
    }


def _validate_musique_augmentation(dataset: Dataset) -> Dict[str, int]:
    """Validate normalized rows after the raw historical artifact check."""
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
        if augmentation_type in {"partial_chain", "entity_variant"} and list(
            row["gold_answers"]
        ) != list(parent["gold_answers"]):
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


def _benchmark_seed(epoch_seed: int, benchmark: str) -> int:
    digest = hashlib.sha256(f"{epoch_seed}:{benchmark}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _validate_phase2_epoch_seeds(epoch_seeds: Sequence[int]) -> Tuple[int, ...]:
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in epoch_seeds):
        raise ValueError("Phase-II epoch seeds must be integers")
    normalized = tuple(int(seed) for seed in epoch_seeds)
    if normalized != PAPER_PHASE2_EPOCH_SEEDS:
        raise ValueError(
            "Paper Phase-II requires epoch seed schedule "
            f"{PAPER_PHASE2_EPOCH_SEEDS}, got {normalized}"
        )
    return normalized


def prepare_phase2_data(
    source_specs: Mapping[str, SourceSpec],
    field_maps: Mapping[str, Phase2FieldMap],
    output_dir: str,
    test_urls: Set[str],
    epoch_seeds: Sequence[int],
    training_retrieval_index_sha256: str,
    musique_derived_manifest_path: str,
) -> DatasetDict:
    """Create five protocol-verifiable, benchmark-balanced Phase-II epoch shards."""

    if set(source_specs) != set(PHASE2_BENCHMARKS):
        missing = sorted(set(PHASE2_BENCHMARKS) - set(source_specs))
        extra = sorted(set(source_specs) - set(PHASE2_BENCHMARKS))
        raise ValueError(f"Phase-II requires all four benchmark sources; missing={missing}, extra={extra}")
    epoch_seeds = _validate_phase2_epoch_seeds(epoch_seeds)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", training_retrieval_index_sha256):
        raise ValueError("training retrieval index digest must be a 64-character SHA-256")
    if not isinstance(musique_derived_manifest_path, str) or not (
        musique_derived_manifest_path.strip()
    ):
        raise ValueError(
            "Phase II requires --phase2-musique-derived-manifest from the "
            "versioned historical source"
        )
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
        musique_derived_manifest = None
        if benchmark == "musique":
            musique_derived_manifest = _validate_musique_derived_manifest(
                raw,
                field_map,
                musique_derived_manifest_path,
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
            "derived_artifact_manifest": musique_derived_manifest,
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
        "answer_contract": EVALUATION_ANSWER_CONTRACT,
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
        choices=("phase1", "phase2", "eval", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./data/aria")
    parser.add_argument(
        "--epoch-seeds",
        type=int,
        nargs=5,
        default=list(PAPER_PHASE2_EPOCH_SEEDS),
        metavar=("E0", "E1", "E2", "E3", "E4"),
        help=(
            "Paper Phase-II epoch-view schedule: 42 123 456 789 2024. "
            "The schedule is recorded in the data manifest."
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
        "--phase2-musique-derived-manifest",
        help=(
            "Artifact-owner-supplied aria-musique-derived-source-v1 JSON "
            "manifest for the complete 168,745-row historical derived source"
        ),
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
        if not args.training_retrieval_index_sha256:
            raise ValueError("Phase II requires --training-retrieval-index-sha256")
        if not args.phase2_musique_derived_manifest:
            raise ValueError(
                "Phase II requires --phase2-musique-derived-manifest"
            )
        specs, maps = _collect_phase2_args(args)
        prepare_phase2_data(
            specs,
            maps,
            str(root / "phase2"),
            test_urls,
            epoch_seeds=args.epoch_seeds,
            training_retrieval_index_sha256=args.training_retrieval_index_sha256,
            musique_derived_manifest_path=args.phase2_musique_derived_manifest,
        )

    if args.stage in {"eval", "all"}:
        specs, maps = _collect_phase2_args(args, split_override=args.eval_split)
        prepare_evaluation_data(specs, maps, str(root / "eval"))


if __name__ == "__main__":
    main()
