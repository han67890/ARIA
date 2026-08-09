#!/usr/bin/env python3
"""End-to-end ARIA evaluation on the four QA benchmarks used in the paper.

The command evaluates one independently trained checkpoint per training seed,
preserving the paper's multi-seed experiment protocol.

Examples:
    # One checkpoint (no multi-seed claim)
    python -m openrlhf.cli.evaluate_aria \
        --model_path /checkpoints/aria_seed42 \
        --dataset nq --compression_rate 16 \
        --eval_data_path /data/aria/eval \
        --corpus_path /data/kilt_corpus.jsonl \
        --doc_embeddings /artifacts/kilt_bge.pt

    # Five independently trained checkpoints supplied explicitly
    python -m openrlhf.cli.evaluate_aria \
        --model_paths /checkpoints/aria_42 /checkpoints/aria_123 \
                      /checkpoints/aria_456 /checkpoints/aria_789 \
                      /checkpoints/aria_2024 \
        --seeds 42 123 456 789 2024 \
        --dataset all --compression_rate 16 \
        --eval_data_path /data/aria/eval \
        --corpus_path /data/kilt_corpus.jsonl \
        --doc_embeddings /artifacts/kilt_bge.pt

    # The equivalent checkpoint-template form
    python -m openrlhf.cli.evaluate_aria \
        --model_path_template '/checkpoints/aria_seed{seed}_cr{cr}' \
        --seeds 42 123 456 789 2024 \
        --dataset all --compression_rate 16 \
        --eval_data_path /data/aria/eval \
        --corpus_path /data/kilt_corpus.jsonl \
        --doc_embeddings /artifacts/kilt_bge.pt
"""

import argparse
import hashlib
import json
import os
import random
import re
import string
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm

from openrlhf.models.modeling_aria import (
    CLARA_ARCHIVE_DOCUMENT_ID_SCHEME,
    CLARA_ARCHIVE_PAGE_ID_SCHEME,
    CLARA_DOCUMENT_REPRESENTATION_SCHEME,
    CLARA_EVALUATION_CANDIDATE_PROTOCOL,
    CLARA_MEMORY_ALLOCATION_SCHEME,
    CLARA_PHASE2_OBJECTIVE,
    CLARA_SELECTOR_SCHEME,
    COUPLING_CONTROL_PROTOCOL,
    CLaRa,
    CLaRaConfig,
    MATCHED_EVIDENCE_TOKEN_BUDGET,
    MTFRL_INITIALIZATION_SCHEME,
    CFRS_RECONSTRUCTION_SCHEME,
    ORACLE_TOP100_PROTOCOL,
    QR_INPUT_SCHEME,
    RAG_CONFIGURATION_SPECS,
    RAGPipelineConfig,
    STATIC_SECOND_QUERY_SCHEME,
    UNIFORM_BUDGET_ALLOCATION_SCHEME,
    _BM25Index,
    create_paper_rag_config,
    _page_deduplicate_ranked_indices,
    required_checkpoint_configuration,
    _tensor_is_finite_in_chunks,
)
from openrlhf.utils.aria_provenance import (
    CORPUS_SHA256_SCHEME,
    EVALUATION_ANSWER_ALIAS_CONTRACT,
    EVALUATION_GOLD_DOCUMENT_CONTRACT,
    SOURCE_SNAPSHOT_SCHEME,
    TEXT_SHA256_SCHEME,
    corpus_id as _shared_corpus_id,
    corpus_page_url as _shared_corpus_page_url,
    corpus_sha256 as _shared_corpus_sha256,
    corpus_text as _shared_corpus_text,
    file_sha256,
    text_sha256 as _shared_text_sha256,
)
# ---------------------------------------------------------------------------
# Evaluation metrics (Appendix A.35)
# ---------------------------------------------------------------------------


class QAMetrics:
    """ARIA's EM, CEM, and token-level F1 metrics.

    Normalisation follows Appendix A.35 exactly and in order: Unicode NFKC,
    lowercase, standalone English-article removal, removal of ASCII punctuation
    except apostrophes and hyphens, then whitespace collapse.
    """

    # Hyphenated forms and apostrophe-linked forms (for example ``the-best``,
    # ``the's``, and ``l'an``) are not standalone. Quotes such as ``'the'`` do
    # still delimit a standalone token. Curly apostrophes survive the strictly
    # ASCII punctuation step, so they receive the same token-boundary treatment.
    _ARTICLE_RE = re.compile(
        r"(?<![\w-])(?<!\w['’])(?:a|an|the)(?![\w-])(?!['’]\w)"
    )
    _PUNCTUATION_TO_DELETE = "".join(
        character for character in string.punctuation if character not in {"'", "-"}
    )
    _PUNCTUATION_TABLE = str.maketrans("", "", _PUNCTUATION_TO_DELETE)

    @staticmethod
    def _as_text(value: Any) -> str:
        return "" if value is None else str(value)

    @classmethod
    def normalize_answer(cls, text: str) -> str:
        """Apply the paper's Appendix A.35 normalisation pipeline."""
        normalized = unicodedata.normalize("NFKC", cls._as_text(text))
        normalized = normalized.lower()
        normalized = cls._ARTICLE_RE.sub(" ", normalized)
        normalized = normalized.translate(cls._PUNCTUATION_TABLE)
        return " ".join(normalized.split())

    @classmethod
    def gold_answers(cls, ground_truth: Any) -> List[str]:
        """Convert common dataset answer containers into a list of gold strings."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, Mapping):
            # Hugging Face QA datasets commonly use one of these containers.
            answers: List[str] = []
            for key in ("text", "answer", "answers", "aliases", "normalized_aliases"):
                if key in ground_truth:
                    answers.extend(cls.gold_answers(ground_truth[key]))
            if answers:
                return list(dict.fromkeys(answers))
            return [cls._as_text(ground_truth)]
        if isinstance(ground_truth, Sequence) and not isinstance(
            ground_truth, (bytes, bytearray)
        ):
            answers: List[str] = []
            for value in ground_truth:
                answers.extend(cls.gold_answers(value))
            return answers
        return [cls._as_text(ground_truth)]

    @classmethod
    def _exact_match_single(cls, prediction: str, ground_truth: str) -> bool:
        return cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth)

    @classmethod
    def exact_match(cls, prediction: str, ground_truth: Any) -> bool:
        """Exact match against the best of all supplied gold answers."""
        golds = cls.gold_answers(ground_truth)
        return any(cls._exact_match_single(prediction, gold) for gold in golds)

    @classmethod
    def _contains_exact_match_single(cls, prediction: str, ground_truth: str) -> bool:
        pred_norm = cls.normalize_answer(prediction)
        gold_norm = cls.normalize_answer(ground_truth)
        # Require a substantive gold answer before substring matching.
        if not gold_norm:
            return pred_norm == gold_norm
        return gold_norm in pred_norm

    @classmethod
    def contains_exact_match(cls, prediction: str, ground_truth: Any) -> bool:
        """Substring CEM against the best of all supplied gold answers."""
        golds = cls.gold_answers(ground_truth)
        return any(cls._contains_exact_match_single(prediction, gold) for gold in golds)

    @classmethod
    def _f1_single(cls, prediction: str, ground_truth: str) -> float:
        pred_tokens = cls.normalize_answer(prediction).split()
        gold_tokens = cls.normalize_answer(ground_truth).split()

        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)

        num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return (2.0 * precision * recall) / (precision + recall)

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: Any) -> float:
        """Token F1 against the best of all supplied gold answers."""
        golds = cls.gold_answers(ground_truth)
        if not golds:
            return 0.0
        return max(cls._f1_single(prediction, gold) for gold in golds)

    @classmethod
    def compute_all(cls, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """Compute each metric independently against its best matching gold."""
        golds = cls.gold_answers(ground_truth)
        if not golds:
            return {"em": 0.0, "cem": 0.0, "f1": 0.0}
        return {
            "em": float(max(cls._exact_match_single(prediction, gold) for gold in golds)),
            "cem": float(
                max(cls._contains_exact_match_single(prediction, gold) for gold in golds)
            ),
            "f1": float(max(cls._f1_single(prediction, gold) for gold in golds)),
        }


# ---------------------------------------------------------------------------
# Dataset and artifact loading
# ---------------------------------------------------------------------------


DATASET_CONFIGS = {
    "nq": {
        "path": "nq_open",
        "split": "validation",
        "question_key": "question",
        "answer_key": "answer",
    },
    "hotpotqa": {
        "path": "hotpotqa/hotpot_qa",
        "config": "fullwiki",
        "split": "validation",
        "question_key": "question",
        "answer_key": "answer",
    },
    "musique": {
        "path": "musique/musique",
        "split": "validation",
        "question_key": "question",
        "answer_key": "answer",
    },
    "2wikimultihopqa": {
        "path": "2wikimultihopqa",
        "split": "validation",
        "question_key": "question",
        "answer_key": "answer",
    },
}

EVALUATION_COUNTS = {
    "nq": 6_489,
    "hotpotqa": 7_384,
    "musique": 2_417,
    "2wikimultihopqa": 12_576,
}

_REPOSITORY_EVAL_ARCHIVES = {
    "nq": ("nq.zip", "nq/eval_processed_no_pos.jsonl"),
    "hotpotqa": ("hotpotqa.zip", "hotpotqa/eval_processed_no_pos.jsonl"),
    "musique": ("musique.zip", "musique/eval_processed_no_pos.jsonl"),
    "2wikimultihopqa": ("2wiki.zip", "2wiki/eval_processed_no_pos.jsonl"),
}
_REPOSITORY_EVAL_ARCHIVE_SHA256 = {
    "nq": "7d26d5c29694cd81cccfcac4fd29c16ae7f245b4c554623cbe3c6ec8c3a0ad41",
    "hotpotqa": "f46d7cfc23199f6cdff5e3ce1872ff150e6d940eb83b343cf37431cd740fa4db",
    "musique": "85a55afc5c6067d00eef1888e13b598039a515f787791b23fbb495c35827e264",
    "2wikimultihopqa": "39cf44bcfa24938c40617ef5bba90235642bf02f537297f4055b3f6bc756846c",
}
_REPOSITORY_BGE_CANDIDATES_SHA256 = {
    "nq": "2e6126e5e7ab59401a0870a256d54dc7e188d97a5a8d3b9e53e14457001793f2",
    "hotpotqa": "6a8747f07642b438effb601c9f8e20fb5659eb21fe80bc3b8205df6646907776",
    "musique": "d4b89b500dc3c6c7c324f4992a56c1c02e960ae06da08a7dad390d42aa17f136",
    "2wikimultihopqa": "b8f6023a4df21bcdb36f5c6bd299e1bfc5bcb0ed1906de528fb784a138f2962f",
}
_REPOSITORY_CLARA_POSITIVE_INDICES_SHA256 = {
    "nq": "3fa5b5c2bee7bca30936d3d6f8704cd095e45143f8d4a947a6dc036bef0ecfe1",
    "hotpotqa": "3702710243f744979c8b3f3f2a59db724ec5e79267337b947c08089752a648b0",
    "musique": "02d3522d942e844c01dda8716892b24412c0793ef1fdd1814c365a8ec6bd1f35",
    "2wikimultihopqa": "e5bab2ed3d54d188fe3d91a4881c5d641b35ac25fbcfbdc36ab62458dfa2eba5",
}
_REPOSITORY_BGE_CANDIDATE_COUNT = 20
_PAPER_NORMAL_TOP_K = 5
PAPER_COMPRESSION_RATES = {4, 16, 32, 64, 128}
PAPER_TRAINING_SEEDS = {42, 123, 456, 789, 2024}
PAPER_MAX_NEW_TOKENS = 64
PAPER_BGE_MODEL = "BAAI/bge-large-en-v1.5"
# Backward-compatible public spelling retained for downstream callers.
EVALUATION_GOLD_CONTRACT = EVALUATION_GOLD_DOCUMENT_CONTRACT
_ANSWER_ALIAS_FIELDS = (
    "gold_answers",
    "answers",
    "answer_aliases",
    "aliases",
    "normalized_aliases",
    "acceptable_answers",
)


_REPOSITORY_DOCUMENT_HEADER = re.compile(
    r"^This is a document about\s+(.+?)\s*(?:\r?\n|$)"
)


def _repository_candidate_identity(document: str) -> Tuple[str, str]:
    """Derive stable archive-local document/page IDs from canonical text."""
    match = _REPOSITORY_DOCUMENT_HEADER.match(document.strip())
    if match is None:
        raise ValueError("Repository CLaRa document lacks its canonical page-title header")
    normalized_title = " ".join(match.group(1).split()).casefold()
    if not normalized_title:
        raise ValueError("Repository CLaRa document has an empty page title")
    document_id = "archive-text-sha256:" + hashlib.sha256(
        document.encode("utf-8")
    ).hexdigest()
    page_id = "archive-page-title-sha256:" + hashlib.sha256(
        normalized_title.encode("utf-8")
    ).hexdigest()
    return document_id, page_id


def _repository_top5_documents(documents: Any, *, location: str) -> Tuple[List[str], List[str]]:
    """Validate an archived ranked candidate list and its legacy first-five view.

    The upstream-derived evaluation archives retain the first 20 BGE-ranked
    candidates for each query.  Their published fingerprints cover that complete
    list.  Matched CLaRa evaluation now passes all 20 to its learned ST selector;
    the second return value remains available to audit the fixed first-five view.
    """
    if (
        not isinstance(documents, list)
        or len(documents) != _REPOSITORY_BGE_CANDIDATE_COUNT
        or any(not isinstance(text, str) or not text.strip() for text in documents)
    ):
        raise ValueError(
            f"{location} must contain exactly "
            f"{_REPOSITORY_BGE_CANDIDATE_COUNT} non-empty ranked BGE documents"
        )
    archived = list(documents)
    return archived, archived[:_PAPER_NORMAL_TOP_K]


def _validate_clara_archive_dir(clara_archive_dir: str) -> Dict[str, Path]:
    """Validate the complete external four-archive CLaRa artifact set.

    The large upstream-derived ZIPs are intentionally not distributed in this
    source repository.  Matched CLaRa is a four-benchmark protocol, so an
    external directory is accepted only when every pinned byte-identical ZIP
    and every required member is present.  This prevents a partial or silently
    substituted archive set from being reported as the matched control.
    """
    if not isinstance(clara_archive_dir, str) or not clara_archive_dir.strip():
        raise ValueError(
            "Matched CLaRa requires --clara_archive_dir containing all four "
            "pinned candidate ZIP archives"
        )
    archive_root = Path(os.path.expanduser(clara_archive_dir)).resolve()
    if not archive_root.is_dir():
        raise FileNotFoundError(
            f"CLaRa archive directory does not exist or is not a directory: {archive_root}"
        )

    paths: Dict[str, Path] = {}
    for benchmark, (archive_name, member_name) in _REPOSITORY_EVAL_ARCHIVES.items():
        archive_path = archive_root / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"CLaRa archive directory requires {archive_name!r} for {benchmark}: "
                f"{archive_path}"
            )
        actual_sha256 = file_sha256(archive_path)
        expected_sha256 = _REPOSITORY_EVAL_ARCHIVE_SHA256[benchmark]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"CLaRa archive {archive_path} has SHA-256 {actual_sha256}, expected "
                f"{expected_sha256}"
            )
        try:
            with zipfile.ZipFile(archive_path) as archive:
                if archive.namelist().count(member_name) != 1:
                    raise ValueError(
                        f"CLaRa archive {archive_path} requires exactly one member "
                        f"{member_name!r}"
                    )
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid CLaRa ZIP archive: {archive_path}") from exc
        paths[benchmark] = archive_path
    return paths


def _load_repository_eval_rows(
    dataset_name: str,
    clara_archive_dir: str,
) -> List[Dict[str, Any]]:
    """Read fingerprinted candidate rows from an external pinned archive.

    The archives contain ranked BGE top-20 candidate lists but no benchmark
    alias sets. Full ARIA ignores them and performs full-corpus retrieval;
    matched CLaRa applies its learned hard-forward/soft-backward top-5 selector
    to all retained candidates after an alias-complete split is exactly joined.
    """
    _, member_name = _REPOSITORY_EVAL_ARCHIVES[dataset_name]
    archive_path = _validate_clara_archive_dir(clara_archive_dir)[dataset_name]

    rows: List[Dict[str, Any]] = []
    documents_hasher = hashlib.sha256()
    positives_hasher = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive:
        if member_name not in archive.namelist():
            raise ValueError(
                f"Evaluation archive {archive_path} requires member {member_name!r}"
            )
        with archive.open(member_name) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"{archive_path}:{member_name}:{line_number} must be a JSON object"
                    )
                question = record.get("question")
                if not isinstance(question, str) or not question.strip() or "answer" not in record:
                    raise ValueError(
                        f"{archive_path}:{member_name}:{line_number} requires question and answer"
                    )
                row: Dict[str, Any] = {
                    "question": question,
                    "answer": record["answer"],
                    "id": str(
                        record.get("id")
                        or record.get("_id")
                        or f"{dataset_name}:{line_number - 1}"
                    ),
                }
                if record.get("docs") is not None:
                    archived_documents, _ = _repository_top5_documents(
                        record["docs"],
                        location=f"{archive_path}:{member_name}:{line_number}.docs",
                    )
                    row["docs"] = archived_documents
                    candidate_identities = [
                        _repository_candidate_identity(document)
                        for document in archived_documents
                    ]
                    row["clara_candidate_doc_ids"] = [
                        identity[0] for identity in candidate_identities
                    ]
                    row["clara_candidate_page_ids"] = [
                        identity[1] for identity in candidate_identities
                    ]
                    raw_positive_indices = record.get("pos_index")
                    if not isinstance(raw_positive_indices, list):
                        raise ValueError(
                            f"{archive_path}:{member_name}:{line_number}.pos_index "
                            "must be a list"
                        )
                    positive_indices: List[int] = []
                    for position, value in enumerate(raw_positive_indices):
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 0
                            or value >= len(archived_documents)
                        ):
                            raise ValueError(
                                f"{archive_path}:{member_name}:{line_number}.pos_index"
                                f"[{position}] is outside the archived candidate pool"
                            )
                        positive_indices.append(value)
                    if len(positive_indices) != len(set(positive_indices)):
                        raise ValueError("Repository CLaRa positive indices must be unique")
                    row["clara_gold_candidate_indices"] = positive_indices
                    documents_hasher.update(
                        json.dumps(
                            archived_documents,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    )
                    documents_hasher.update(b"\n")
                    positives_hasher.update(
                        json.dumps(
                            positive_indices,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    )
                    positives_hasher.update(b"\n")
                for alias_key in _ANSWER_ALIAS_FIELDS:
                    if record.get(alias_key) is not None:
                        row[alias_key] = record[alias_key]
                rows.append(row)
    if documents_hasher.hexdigest() != _REPOSITORY_BGE_CANDIDATES_SHA256[dataset_name]:
        raise ValueError(
            f"External {dataset_name} BGE candidate lists must match the pinned fingerprint"
        )
    if (
        positives_hasher.hexdigest()
        != _REPOSITORY_CLARA_POSITIVE_INDICES_SHA256[dataset_name]
    ):
        raise ValueError(
            f"External {dataset_name} CLaRa positive indices must match the pinned fingerprint"
        )
    return rows


def _validate_paper_answer_alias_contract(
    dataset: Any,
    *,
    dataset_name: str,
    answer_key: str,
) -> None:
    """Require one explicit benchmark-provided alias list on every row.

    The external upstream-derived archives have only a scalar ``answer``. Treating
    that scalar as a complete alias set would silently change Appendix A.35's
    max-over-aliases metric, so those archives are candidate artifacts only.
    """
    for index, item in enumerate(dataset):
        raw_aliases = item.get(EVALUATION_ANSWER_ALIAS_CONTRACT["field"])
        if (
            not isinstance(raw_aliases, Sequence)
            or isinstance(raw_aliases, (str, bytes, bytearray))
            or not raw_aliases
        ):
            raise ValueError(
                f"Paper-protocol {dataset_name} row {index} requires an explicit "
                "non-empty gold_answers list of benchmark-provided aliases. The "
                "external *_no_pos archives contain only scalar answers and may be "
                "used only for their retained CLaRa candidates; pass an "
                "--eval_data_path artifact produced by `aria-data --stage eval`."
            )
        aliases: List[str] = []
        for alias_index, alias in enumerate(raw_aliases):
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(
                    f"Paper-protocol {dataset_name} row {index}.gold_answers"
                    f"[{alias_index}] must be a non-empty benchmark-provided string"
                )
            aliases.append(alias)
        if len(aliases) != len(set(aliases)):
            raise ValueError(
                f"Paper-protocol {dataset_name} row {index}.gold_answers must "
                "not contain duplicate aliases"
            )
        primary_answers = QAMetrics.gold_answers(item.get(answer_key))
        if not primary_answers or any(answer not in aliases for answer in primary_answers):
            raise ValueError(
                f"Paper-protocol {dataset_name} row {index}.gold_answers must "
                "include every benchmark-provided primary answer"
            )


def _merge_repository_candidates(
    dataset: Any,
    repository_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    question_key: str,
) -> List[Dict[str, Any]]:
    """Attach fingerprinted CLaRa candidates to an alias-complete split.

    The two artifacts are joined only by exact official row order and exact
    question text.  No fuzzy matching or inferred alias mapping is permitted.
    """
    if len(dataset) != len(repository_rows):
        raise ValueError(
            f"Prepared {dataset_name} split and external candidate archive must "
            f"have equal counts, got {len(dataset)} and {len(repository_rows)}"
        )
    candidate_fields = (
        "docs",
        "clara_candidate_doc_ids",
        "clara_candidate_page_ids",
        "clara_gold_candidate_indices",
    )
    merged: List[Dict[str, Any]] = []
    for index, (item, repository_row) in enumerate(zip(dataset, repository_rows)):
        if item.get(question_key) != repository_row.get("question"):
            raise ValueError(
                f"Prepared {dataset_name} row {index} does not exactly align with "
                "the external CLaRa candidate archive question"
            )
        row = dict(item)
        for field in candidate_fields:
            if field not in repository_row:
                raise ValueError(
                    f"Bundled {dataset_name} candidate row {index} lacks {field!r}"
                )
            row[field] = repository_row[field]
        merged.append(row)
    return merged


def _extract_clara_candidate_columns(
    dataset: Sequence[Mapping[str, Any]],
) -> Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[int]]]:
    """Materialize the validated external CLaRa candidate columns."""
    documents_by_row: List[List[str]] = []
    document_ids_by_row: List[List[str]] = []
    page_ids_by_row: List[List[str]] = []
    positive_indices_by_row: List[List[int]] = []
    for row_index, item in enumerate(dataset):
        documents = item.get("docs")
        if (
            not isinstance(documents, list)
            or len(documents) != _REPOSITORY_BGE_CANDIDATE_COUNT
        ):
            raise ValueError(
                "CLaRa baseline requires the complete external BGE top-20 pool; "
                f"row {row_index} has no valid docs field"
            )
        document_ids = item.get("clara_candidate_doc_ids")
        page_ids = item.get("clara_candidate_page_ids")
        positive_indices = item.get("clara_gold_candidate_indices")
        if not (
            isinstance(document_ids, list)
            and isinstance(page_ids, list)
            and isinstance(positive_indices, list)
            and len(document_ids) == len(documents)
            and len(page_ids) == len(documents)
        ):
            raise ValueError(
                f"CLaRa row {row_index} lacks stable external-archive candidate identities"
            )
        documents_by_row.append(documents)
        document_ids_by_row.append(document_ids)
        page_ids_by_row.append(page_ids)
        positive_indices_by_row.append(positive_indices)
    return (
        documents_by_row,
        document_ids_by_row,
        page_ids_by_row,
        positive_indices_by_row,
    )


def load_eval_dataset(
    dataset_name: str,
    max_samples: Optional[int] = None,
    eval_data_path: Optional[str] = None,
    require_clara_archive: bool = False,
    clara_archive_dir: Optional[str] = None,
):
    """Load an alias-complete paper split and optional external CLaRa candidates."""
    cfg = DATASET_CONFIGS[dataset_name]
    dataset = None
    if eval_data_path is None:
        raise ValueError(
            "Paper-protocol answer metrics require --eval_data_path with explicit "
            "benchmark-provided gold_answers lists. External *_no_pos archives "
            "contain scalar answers only and are CLaRa candidate artifacts."
        )
    prepared_path = Path(os.path.expanduser(eval_data_path))
    manifest_path = prepared_path / "aria_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Prepared evaluation artifact requires manifest: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("phase") != "evaluation":
        raise ValueError("--eval_data_path is not an ARIA evaluation artifact")
    if manifest.get("retrieval_gold_contract") != EVALUATION_GOLD_CONTRACT:
        raise ValueError(
            "--eval_data_path must be regenerated with explicit corpus-level "
            "gold_doc_ids covering all annotated positives independently of its "
            "candidate list"
        )
    if manifest.get("answer_alias_contract") != EVALUATION_ANSWER_ALIAS_CONTRACT:
        raise ValueError(
            "--eval_data_path must be regenerated with an explicit non-empty "
            "gold_answers list of benchmark-provided aliases on every row"
        )
    prepared = load_from_disk(str(prepared_path))
    if isinstance(prepared, Mapping):
        if dataset_name not in prepared:
            raise ValueError(
                f"Prepared evaluation artifact has no {dataset_name!r} split"
            )
        dataset = prepared[dataset_name]
    else:
        if len(DATASET_CONFIGS) != 1:
            raise ValueError(
                "--eval_data_path must point to the DatasetDict produced by aria-data"
            )
        dataset = prepared
    split_record = manifest.get("splits", {}).get(dataset_name)
    if (
        not isinstance(split_record, Mapping)
        or split_record.get("count") != len(dataset)
        or split_record.get("fingerprint") != getattr(dataset, "_fingerprint", None)
    ):
        raise ValueError(
            f"Prepared evaluation split {dataset_name!r} does not match its manifest"
        )
    repository_rows = None
    if require_clara_archive:
        if clara_archive_dir is None:
            raise ValueError(
                "Matched CLaRa requires --clara_archive_dir containing all four "
                "pinned candidate ZIP archives"
            )
        repository_rows = _load_repository_eval_rows(
            dataset_name,
            clara_archive_dir,
        )

    if len(dataset) != EVALUATION_COUNTS[dataset_name]:
        raise ValueError(
            f"Official {dataset_name} evaluation split must contain exactly "
            f"{EVALUATION_COUNTS[dataset_name]:,} rows, got {len(dataset):,}"
        )
    _validate_paper_answer_alias_contract(
        dataset,
        dataset_name=dataset_name,
        answer_key=cfg["answer_key"],
    )
    if require_clara_archive:
        dataset = _merge_repository_candidates(
            dataset,
            repository_rows,
            dataset_name=dataset_name,
            question_key=cfg["question_key"],
        )
    if max_samples is not None:
        count = min(max(max_samples, 0), len(dataset))
        dataset = dataset[:count] if isinstance(dataset, list) else dataset.select(range(count))
    return dataset, cfg["question_key"], cfg["answer_key"]


def load_corpus(corpus_path: str):
    """Load an explicitly declared full KILT Wikipedia corpus artifact."""
    if corpus_path.startswith("hf:"):
        dataset_name = corpus_path[len("hf:") :].strip()
        if not dataset_name:
            raise ValueError("hf: corpus paths must include a dataset name")
        return load_dataset(dataset_name, split="full")

    path = Path(os.path.expanduser(corpus_path))
    if not path.exists():
        raise FileNotFoundError(f"KILT corpus artifact does not exist: {path}")
    if path.is_dir():
        corpus = load_from_disk(str(path))
        if isinstance(corpus, DatasetDict):
            if "full" not in corpus:
                raise ValueError("KILT DatasetDict corpus must contain a 'full' split")
            corpus = corpus["full"]
        if not isinstance(corpus, Dataset):
            raise ValueError(f"Saved corpus artifact must contain a Dataset: {path}")
        return corpus
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("KILT corpus must be a saved Dataset or JSON/JSONL file")
    return load_dataset("json", data_files=str(path), split="train")


def _extract_example_ids(dataset: Any, dataset_name: str) -> List[str]:
    """Return stable, dataset-namespaced identities for paired comparisons."""
    ids: List[str] = []
    seen: set[str] = set()
    for index, item in enumerate(dataset):
        raw_id = None
        for key in ("id", "_id", "question_id", "qid", "example_id"):
            if key in item and item[key] is not None:
                raw_id = item[key]
                break
        base = f"{dataset_name}:{raw_id if raw_id is not None else index}"
        # Assign deterministic unique bootstrap identities to repeated source IDs.
        example_id = base if base not in seen else f"{base}#{index}"
        seen.add(example_id)
        ids.append(example_id)
    return ids


def _extract_gold_answers(item: Mapping[str, Any], answer_key: str) -> List[str]:
    """Collect the primary answer and any dataset-provided answer aliases."""
    values: List[Any] = [item[answer_key]]
    for alias_key in _ANSWER_ALIAS_FIELDS:
        if alias_key in item and item[alias_key] is not None:
            values.append(item[alias_key])
    return list(dict.fromkeys(QAMetrics.gold_answers(values)))


def _extract_gold_document_ids(dataset: Any) -> Optional[List[List[str]]]:
    """Read optional corpus-level retrieval labels from a prepared eval split.

    Repository archives lack positive labels aligned to stable full-KILT corpus
    IDs.  A prepared artifact may provide the canonical ``gold_doc_ids`` field,
    but the field must then be present for every row. Empty lists are retained:
    Appendix A.35 averages Recall@k only over ``Q_sup``, the rows with at least
    one annotated supporting page.
    """
    extracted: List[Optional[List[str]]] = []
    for index, item in enumerate(dataset):
        if "gold_doc_ids" not in item or item["gold_doc_ids"] is None:
            extracted.append(None)
            continue
        raw_ids = item["gold_doc_ids"]
        if not isinstance(raw_ids, Sequence) or isinstance(
            raw_ids, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"Evaluation row {index}.gold_doc_ids must be a list of corpus IDs"
            )
        document_ids = [str(value).strip() for value in raw_ids]
        if (
            any(not value for value in document_ids)
            or len(document_ids) != len(set(document_ids))
        ):
            raise ValueError(
                f"Evaluation row {index}.gold_doc_ids must contain unique non-empty IDs"
            )
        extracted.append(document_ids)

    if all(value is None for value in extracted):
        return None
    if any(value is None for value in extracted):
        raise ValueError(
            "Evaluation gold_doc_ids must be supplied for every row or omitted for every row"
        )
    return [value for value in extracted if value is not None]


def _corpus_text(item: Mapping[str, Any]) -> str:
    return _shared_corpus_text(item, location="corpus row")


def _corpus_id(item: Mapping[str, Any], index: int) -> str:
    return _shared_corpus_id(item, location=f"corpus row {index}")


def _corpus_page_url(item: Mapping[str, Any], index: int) -> str:
    return _shared_corpus_page_url(item, location=f"corpus row {index}")


def _corpus_sha256(
    document_ids: Sequence[str],
    text_hashes: Sequence[str],
    page_urls: Sequence[str],
) -> str:
    return _shared_corpus_sha256(document_ids, text_hashes, page_urls)


def _format_artifact_path(
    path: str,
    *,
    seed: Optional[int] = None,
    dataset: Optional[str] = None,
    compression_rate: Optional[int] = None,
) -> str:
    """Expand the declared artifact-template fields exactly."""
    try:
        return os.path.expanduser(
            path.format(
                seed=seed,
                dataset=dataset,
                compression_rate=compression_rate,
                cr=compression_rate,
            )
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid artifact/checkpoint path template {path!r}: {exc}") from exc


def _load_artifact(path: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Artifact does not exist or is not a file: {path}")

    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Reading a .safetensors artifact requires the safetensors package"
            ) from exc
        return load_file(path, device="cpu")

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This evaluator requires a PyTorch version supporting safe "
            "torch.load(..., weights_only=True) artifact loading"
        ) from exc


def _extract_tensor(artifact: Any, *, artifact_name: str) -> torch.Tensor:
    if isinstance(artifact, torch.Tensor):
        return artifact.detach().cpu()
    if isinstance(artifact, np.ndarray):
        return torch.from_numpy(np.asarray(artifact))
    if isinstance(artifact, Mapping):
        for key in (artifact_name, "doc_embeddings", "embeddings", "tensor", "arr_0"):
            if key in artifact:
                return _extract_tensor(artifact[key], artifact_name=artifact_name)
        tensor_values = [
            value
            for value in artifact.values()
            if isinstance(value, (torch.Tensor, np.ndarray))
        ]
        if len(tensor_values) == 1:
            return _extract_tensor(tensor_values[0], artifact_name=artifact_name)
    raise ValueError(f"Artifact must contain tensor {artifact_name!r}")


def _text_sha256(text: str) -> str:
    return _shared_text_sha256(text)


def _metadata_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def load_doc_embeddings(
    path: str,
    expected_documents: int,
    expected_ids: Optional[Sequence[str]] = None,
    expected_hashes: Optional[Sequence[str]] = None,
    expected_index_sha256: Optional[str] = None,
    *,
    expected_page_ids: Optional[Sequence[str]] = None,
    return_index_sha256: bool = False,
) -> Any:
    """Load and validate the dense BGE corpus matrix used by AHR and MTFRL."""
    artifact = _load_artifact(path)
    embeddings = _extract_tensor(artifact, artifact_name="doc_embeddings").contiguous()
    if embeddings.ndim != 2:
        raise ValueError(
            f"doc_embeddings must have shape (documents, dense_dim), got "
            f"{tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] != expected_documents:
        raise ValueError(
            "doc_embeddings row count must match the corpus exactly: "
            f"{embeddings.shape[0]} != {expected_documents}"
        )
    if embeddings.shape[1] != 1024:
        raise ValueError(
            "ARIA requires BGE-large-en-v1.5 document embeddings with dimension 1024; "
            f"got {embeddings.shape[1]}"
        )
    embeddings = embeddings.float().contiguous()
    if not _tensor_is_finite_in_chunks(embeddings):
        raise ValueError("doc_embeddings contains NaN or infinite values")
    if expected_ids is not None:
        if not isinstance(artifact, Mapping):
            raise ValueError("doc_embeddings must include document_ids alignment metadata")
        artifact_ids = artifact.get("document_ids", artifact.get("doc_ids"))
        if isinstance(artifact_ids, np.ndarray):
            artifact_ids = artifact_ids.tolist()
        if artifact_ids is None or [str(value) for value in artifact_ids] != list(expected_ids):
            raise ValueError("doc_embeddings document_ids do not match corpus row order")
    if expected_hashes is not None:
        if not isinstance(artifact, Mapping):
            raise ValueError("doc_embeddings must include text_sha256 alignment metadata")
        artifact_hashes = artifact.get("text_sha256", artifact.get("document_sha256"))
        if isinstance(artifact_hashes, np.ndarray):
            artifact_hashes = artifact_hashes.tolist()
        if artifact_hashes is None or list(artifact_hashes) != list(expected_hashes):
            raise ValueError("doc_embeddings text_sha256 values do not match corpus row order")
    if expected_page_ids is not None:
        if not isinstance(artifact, Mapping):
            raise ValueError("doc_embeddings must include page_urls alignment metadata")
        artifact_page_ids = artifact.get("page_urls", artifact.get("page_ids"))
        if isinstance(artifact_page_ids, np.ndarray):
            artifact_page_ids = artifact_page_ids.tolist()
        if artifact_page_ids is None or list(artifact_page_ids) != list(
            expected_page_ids
        ):
            raise ValueError("doc_embeddings page_urls do not match corpus row order")
    if (
        expected_ids is not None
        or expected_hashes is not None
        or expected_page_ids is not None
    ):
        if not isinstance(artifact, Mapping) or _metadata_scalar(
            artifact.get("bge_model")
        ) != "BAAI/bge-large-en-v1.5":
            raise ValueError(
                "doc_embeddings must declare bge_model='BAAI/bge-large-en-v1.5'"
            )
        if _metadata_scalar(artifact.get("text_sha256_scheme")) != TEXT_SHA256_SCHEME:
            raise ValueError(
                f"doc_embeddings must declare text_sha256_scheme="
                f"{TEXT_SHA256_SCHEME!r}"
            )
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
    declared_index_sha256 = (
        _metadata_scalar(artifact.get("index_sha256"))
        if isinstance(artifact, Mapping)
        else None
    )
    if declared_index_sha256 != computed_index_sha256:
        raise ValueError("doc_embeddings index_sha256 does not match its tensor")
    if (
        expected_index_sha256 is not None
        and computed_index_sha256 != expected_index_sha256
    ):
        raise ValueError("doc_embeddings do not match the explicitly expected BGE index")
    if return_index_sha256:
        return embeddings, computed_index_sha256
    return embeddings


def load_bge_projection(
    model: CLaRa,
    path: str,
    expected_output_dim: int,
) -> None:
    """Load an explicit W_BGE artifact, overriding any bundled projection."""
    artifact = _load_artifact(path)
    if not isinstance(artifact, Mapping):
        raise ValueError("W_BGE artifact must contain fitting-protocol metadata")
    expected_metadata = {
        "base_model": model.decoder_model_name,
        "bge_model": "BAAI/bge-large-en-v1.5",
        "sample_count": 50_000,
        "epochs": 2,
        "batch_size": 128,
        "learning_rate": 5e-4,
        "text_sha256_scheme": TEXT_SHA256_SCHEME,
        "qr_input_scheme": QR_INPUT_SCHEME,
    }
    decoder_revision = getattr(
        model.config, "decoder_model_resolved_revision", None
    )
    if decoder_revision is not None:
        expected_metadata["base_model_revision_resolved"] = decoder_revision
    for key, expected in expected_metadata.items():
        actual = _metadata_scalar(artifact.get(key))
        if actual != expected:
            raise ValueError(
                f"W_BGE metadata {key!r} must be {expected!r}, got {actual!r}"
            )
    for key in (
        "query_sha256",
        "passage_id_sha256",
        "passage_text_sha256",
        "test_url_sha256",
    ):
        value = _metadata_scalar(artifact.get(key))
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"W_BGE artifact is missing the {key} fingerprint")
    if not isinstance(_metadata_scalar(artifact.get("seed")), int):
        raise ValueError("W_BGE artifact is missing its fitting seed")
    candidate: Any = artifact
    if isinstance(candidate, Mapping):
        for container_key in ("state_dict", "bge_projection", "_bge_projection"):
            if container_key in candidate and isinstance(candidate[container_key], Mapping):
                candidate = candidate[container_key]
                break

    weight: Optional[torch.Tensor] = None
    if isinstance(candidate, (torch.Tensor, np.ndarray)):
        weight = _extract_tensor(candidate, artifact_name="weight")
    elif isinstance(candidate, Mapping):
        for key, value in candidate.items():
            if str(key).endswith("weight") and isinstance(value, (torch.Tensor, np.ndarray)):
                possible = _extract_tensor(value, artifact_name="weight")
                if possible.ndim == 2 and possible.shape[0] == expected_output_dim:
                    weight = possible
                    break

    if weight is None or weight.ndim != 2:
        raise ValueError(f"BGE projection artifact {path} requires a 2-D weight")
    if weight.shape[0] != expected_output_dim:
        raise ValueError(
            "BGE projection output dimension must match doc_embeddings: "
            f"{weight.shape[0]} != {expected_output_dim}"
        )
    if weight.shape[1] != model.hidden_size:
        raise ValueError(
            "BGE projection input dimension must match the model hidden size: "
            f"{weight.shape[1]} != {model.hidden_size}"
        )
    if not weight.is_floating_point():
        weight = weight.float()
    if not torch.isfinite(weight).all().item():
        raise ValueError("BGE projection contains NaN or infinite values")

    model.setup_bge_projection(bge_dim=expected_output_dim)
    model._bge_projection.load_state_dict({"weight": weight.contiguous()}, strict=True)
    model._bge_projection_metadata = {
        key: _metadata_scalar(value)
        for key, value in artifact.items()
        if key != "state_dict"
    }


# ---------------------------------------------------------------------------
# Evaluator and seed/checkpoint aggregation
# ---------------------------------------------------------------------------


class ARIAEvaluator:
    """End-to-end evaluation harness for one trained checkpoint."""

    def __init__(
        self,
        model: CLaRa,
        corpus_docs: List[str],
        corpus_ids: Optional[List[str]] = None,
        doc_embeddings: Optional[torch.Tensor] = None,
        use_rag_pipeline: bool = True,
        rag_config: Optional[RAGPipelineConfig] = None,
        bm25_index: Optional[_BM25Index] = None,
        retrieval_mode: str = "normal",
        corpus_page_ids: Optional[List[str]] = None,
    ):
        if retrieval_mode not in {"normal", "oracle"}:
            raise ValueError("retrieval_mode must be 'normal' or 'oracle'")
        self.model = model
        self.use_rag_pipeline = use_rag_pipeline
        self.retrieval_mode = retrieval_mode
        self.corpus_ids = list(corpus_ids) if corpus_ids is not None else None
        self._has_explicit_page_ids = corpus_page_ids is not None
        self.corpus_page_ids = (
            list(corpus_page_ids)
            if corpus_page_ids is not None
            else (list(self.corpus_ids) if self.corpus_ids is not None else None)
        )
        self._corpus_id_to_index = (
            {doc_id: index for index, doc_id in enumerate(self.corpus_ids)}
            if self.corpus_ids is not None
            else None
        )
        if retrieval_mode == "oracle" and (
            not use_rag_pipeline
            or self.corpus_ids is None
            or not self._has_explicit_page_ids
        ):
            raise ValueError(
                "Oracle evaluation requires full RAG, stable corpus IDs, and page IDs"
            )

        if use_rag_pipeline:
            if not corpus_docs:
                raise ValueError("Full RAG evaluation requires a non-empty corpus")
            if doc_embeddings is None:
                raise ValueError(
                    "Full ARIA evaluation requires dense doc_embeddings aligned "
                    "with the retrieval corpus"
                )
            if doc_embeddings.ndim != 2 or doc_embeddings.shape[0] != len(corpus_docs):
                raise ValueError(
                    "doc_embeddings must be a 2-D matrix aligned one-to-one with corpus_docs"
                )
            if self.corpus_ids is not None and len(self.corpus_ids) != len(corpus_docs):
                raise ValueError(
                    "corpus_ids must be aligned one-to-one with corpus_docs"
                )
            if self.corpus_ids is not None and len(set(self.corpus_ids)) != len(
                self.corpus_ids
            ):
                raise ValueError("corpus_ids must be unique")
            if self.corpus_page_ids is not None and len(self.corpus_page_ids) != len(
                corpus_docs
            ):
                raise ValueError(
                    "corpus_page_ids must be aligned one-to-one with corpus_docs"
                )
            if self.corpus_page_ids is not None and any(
                not isinstance(page_id, str) or not page_id.strip()
                for page_id in self.corpus_page_ids
            ):
                raise ValueError("corpus_page_ids must contain non-empty strings")
            projection = getattr(model, "_bge_projection", None)
            if projection is None:
                raise ValueError(
                    "Full RAG evaluation requires W_BGE. Bundle bge_projection.pth "
                    "with the checkpoint or pass --bge_projection_path."
                )
            if projection.out_features != doc_embeddings.shape[1]:
                raise ValueError(
                    "W_BGE output dimension does not match doc_embeddings: "
                    f"{projection.out_features} != {doc_embeddings.shape[1]}"
                )

            cfg = rag_config or RAGPipelineConfig(
                top_k=5,
                use_cfrs=True,
                cfrs_weight=0.3,
                use_acr=True,
                acr_min_token_ratio=0.25,
                acr_max_token_ratio=1.0,
                use_mtfrl=True,
                mtfrl_second_top_k=200,
                igfr_gap_threshold=0.50,
                igfr_max_iterations=None,
                ccef_discount_alpha=0.5,
                ccef_filter_threshold=0.30,
                compression_rate=getattr(model, "compr_rate", None),
            )
            if cfg.use_mtfrl:
                mtfrl_projection = getattr(model, "_mtfrl_projection", None)
                if mtfrl_projection is None:
                    raise ValueError(
                        "Full RAG with MTFRL requires the trained projection head; "
                        "the checkpoint must contain mtfrl_projection.pth"
                    )
                output_layers = [
                    module
                    for module in mtfrl_projection.modules()
                    if isinstance(module, torch.nn.Linear)
                ]
                if (
                    not output_layers
                    or output_layers[-1].out_features != doc_embeddings.shape[1]
                ):
                    raise ValueError(
                        "MTFRL projection output dimension must match doc_embeddings"
                    )
            self.model.setup_rag_pipeline(
                corpus_docs=corpus_docs,
                corpus_doc_ids=corpus_ids,
                corpus_page_ids=self.corpus_page_ids,
                doc_embeddings=doc_embeddings,
                rag_config=cfg,
                bm25_index=bm25_index,
            )

    def evaluate(
        self,
        questions: List[str],
        gold_answers: List[Any],
        example_ids: Optional[List[Hashable]] = None,
        gold_doc_ids: Optional[List[List[str]]] = None,
        documents: Optional[List[List[str]]] = None,
        clara_candidate_doc_ids: Optional[List[List[str]]] = None,
        clara_candidate_page_ids: Optional[List[List[str]]] = None,
        clara_gold_candidate_indices: Optional[List[List[int]]] = None,
        batch_size: int = 8,
        max_new_tokens: int = PAPER_MAX_NEW_TOKENS,
    ) -> Dict[str, Any]:
        """Evaluate one checkpoint and retain identity-aligned per-example scores."""
        if len(questions) != len(gold_answers):
            raise ValueError("questions and gold_answers must have equal length")
        if gold_doc_ids is not None:
            if not self.use_rag_pipeline:
                raise ValueError("Recall@5 requires full-corpus RAG evaluation")
            if self.corpus_ids is None:
                raise ValueError("Recall@5 requires stable corpus_ids")
            if not self._has_explicit_page_ids or self.corpus_page_ids is None:
                raise ValueError("Recall@5 requires stable corpus_page_ids")
            if len(gold_doc_ids) != len(questions):
                raise ValueError(
                    "gold_doc_ids must be aligned one-to-one with questions"
                )
            corpus_id_set = set(self.corpus_ids)
            normalized_gold_doc_ids: List[List[str]] = []
            normalized_gold_page_ids: List[List[str]] = []
            for index, values in enumerate(gold_doc_ids):
                if self.retrieval_mode == "oracle" and not values:
                    raise ValueError(
                        f"Oracle gold_doc_ids[{index}] must contain at least one support"
                    )
                normalized = [str(value).strip() for value in values]
                if (
                    any(not value for value in normalized)
                    or len(normalized) != len(set(normalized))
                ):
                    raise ValueError(
                        f"gold_doc_ids[{index}] must contain unique non-empty IDs"
                    )
                unknown = sorted(set(normalized) - corpus_id_set)
                if unknown:
                    raise ValueError(
                        f"gold_doc_ids[{index}] contains IDs absent from the evaluation "
                        f"corpus: {unknown[:3]}"
                    )
                normalized_gold_doc_ids.append(normalized)
                page_ids_in_annotation_order = [
                    self.corpus_page_ids[self._corpus_id_to_index[document_id]]
                    for document_id in normalized
                ]
                normalized_gold_page_ids.append(
                    list(dict.fromkeys(page_ids_in_annotation_order))
                )
            gold_doc_ids = normalized_gold_doc_ids
            gold_page_ids: Optional[List[List[str]]] = normalized_gold_page_ids
        else:
            gold_page_ids = None
        if self.retrieval_mode == "oracle" and gold_doc_ids is None:
            raise ValueError(
                "Oracle evaluation requires corpus-level gold_doc_ids for every question"
            )
        if self.use_rag_pipeline and documents is not None:
            raise ValueError(
                "Paper-protocol evaluation must retrieve from the full KILT corpus; "
                "pre-retrieved documents cannot be supplied"
            )
        if documents is not None and len(documents) != len(questions):
            raise ValueError("documents must be aligned one-to-one with questions")
        clara_metadata = (
            clara_candidate_doc_ids,
            clara_candidate_page_ids,
            clara_gold_candidate_indices,
        )
        if any(value is not None for value in clara_metadata) and not all(
            value is not None for value in clara_metadata
        ):
            raise ValueError(
                "CLaRa Recall@5 requires candidate document IDs, page IDs, and "
                "positive indices together"
            )
        clara_gold_page_ids: Optional[List[List[str]]] = None
        if all(value is not None for value in clara_metadata):
            if self.use_rag_pipeline or documents is None or gold_doc_ids is not None:
                raise ValueError(
                    "Archive-local CLaRa recall metadata is valid only for the "
                    "matched non-RAG candidate path"
                )
            if (
                clara_candidate_doc_ids is None
                or clara_candidate_page_ids is None
                or clara_gold_candidate_indices is None
            ):
                raise RuntimeError("CLaRa recall metadata validation is inconsistent")
            if not (
                len(clara_candidate_doc_ids)
                == len(clara_candidate_page_ids)
                == len(clara_gold_candidate_indices)
                == len(documents)
                == len(questions)
            ):
                raise ValueError("CLaRa recall metadata must align one-to-one with questions")
            clara_gold_page_ids = []
            for row_index, (
                row_documents,
                row_doc_ids,
                row_page_ids,
                row_positive_indices,
            ) in enumerate(
                zip(
                    documents,
                    clara_candidate_doc_ids,
                    clara_candidate_page_ids,
                    clara_gold_candidate_indices,
                )
            ):
                if not (
                    len(row_documents)
                    == len(row_doc_ids)
                    == len(row_page_ids)
                    == _REPOSITORY_BGE_CANDIDATE_COUNT
                ):
                    raise ValueError(
                        f"CLaRa recall row {row_index} must align exactly 20 candidates"
                    )
                if any(not isinstance(value, str) or not value for value in row_doc_ids):
                    raise ValueError("CLaRa candidate document IDs must be non-empty strings")
                if any(not isinstance(value, str) or not value for value in row_page_ids):
                    raise ValueError("CLaRa candidate page IDs must be non-empty strings")
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value >= len(row_documents)
                    for value in row_positive_indices
                ) or len(row_positive_indices) != len(set(row_positive_indices)):
                    raise ValueError(
                        f"CLaRa positive indices are invalid at row {row_index}"
                    )
                clara_gold_page_ids.append(
                    list(
                        dict.fromkeys(
                            row_page_ids[value] for value in row_positive_indices
                        )
                    )
                )
        clara_recall_enabled = clara_gold_page_ids is not None
        clara_retrieval_provenance: Optional[Dict[str, Any]] = None
        if clara_recall_enabled:
            if (
                clara_candidate_doc_ids is None
                or clara_candidate_page_ids is None
                or clara_gold_candidate_indices is None
            ):
                raise RuntimeError("CLaRa provenance metadata is unexpectedly absent")

            def _ordered_rows_sha256(rows: Sequence[Sequence[Any]]) -> str:
                hasher = hashlib.sha256()
                for row in rows:
                    hasher.update(
                        json.dumps(
                            list(row),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    hasher.update(b"\n")
                return hasher.hexdigest()

            clara_retrieval_provenance = {
                "protocol": CLARA_EVALUATION_CANDIDATE_PROTOCOL,
                "release_convention": True,
                "paper_specification_scope": (
                    "paper-specifies-st-top-k-but-not-a-unique-candidate-pool-size"
                ),
                "candidate_source": "retained-repository-bge-top20",
                "candidate_count": _REPOSITORY_BGE_CANDIDATE_COUNT,
                "hard_selection_count": int(self.model.generation_top_k),
                "document_id_scheme": CLARA_ARCHIVE_DOCUMENT_ID_SCHEME,
                "page_id_scheme": CLARA_ARCHIVE_PAGE_ID_SCHEME,
                "page_deduplicated_recall": True,
                "support_scope": "Q_sup",
                "example_count": len(questions),
                "candidate_document_order_sha256": _ordered_rows_sha256(
                    clara_candidate_doc_ids
                ),
                "candidate_page_order_sha256": _ordered_rows_sha256(
                    clara_candidate_page_ids
                ),
                "positive_indices_sha256": _ordered_rows_sha256(
                    clara_gold_candidate_indices
                ),
            }
        if example_ids is None:
            example_ids = list(range(len(questions)))
        if len(example_ids) != len(questions):
            raise ValueError("example_ids must be aligned one-to-one with questions")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_new_tokens != PAPER_MAX_NEW_TOKENS:
            raise ValueError(
                "Paper-protocol evaluation requires max_new_tokens=64"
            )

        all_metrics: List[Dict[str, float]] = []
        retrieval_cutoffs = (1, 3, 5) if self.retrieval_mode == "oracle" else (5,)
        retrieval_recalls: Dict[int, List[float]] = {
            cutoff: [] for cutoff in retrieval_cutoffs
        }
        evidence_memory_tokens: List[int] = []
        predictions: List[Dict[str, Any]] = []

        for start in tqdm(range(0, len(questions), batch_size), desc="Evaluating"):
            batch_questions = questions[start : start + batch_size]
            batch_docs = documents[start : start + batch_size] if documents is not None else None
            retrieved_indices: Optional[Union[torch.Tensor, Sequence[Sequence[int]]]] = None
            batch_diagnostics: Optional[Sequence[Any]] = None

            # Clear diagnostics so each batch verifies exact full-RAG retrieval.
            if (
                self.use_rag_pipeline
                and batch_docs is None
                and hasattr(self.model, "clear_rag_diagnostics")
            ):
                self.model.clear_rag_diagnostics()

            if not self.use_rag_pipeline and batch_docs is not None:
                if clara_recall_enabled:
                    pred_texts, retrieved_indices = self.model.generate_from_text(
                        questions=batch_questions,
                        documents=batch_docs,
                        max_new_tokens=max_new_tokens,
                        return_selected_indices=True,
                    )
                else:
                    pred_texts = self.model.generate_from_text(
                        questions=batch_questions,
                        documents=batch_docs,
                        max_new_tokens=max_new_tokens,
                    )
            else:
                generation_kwargs: Dict[str, Any] = dict(
                    questions=batch_questions,
                    documents=batch_docs,
                    max_new_tokens=max_new_tokens,
                    # Appendix A.35 reports Normal Recall@5 on the first-pass
                    # CCEF survivors, before MTFRL changes the evidence set.
                    return_first_pass_indices=(
                        gold_doc_ids is not None and self.retrieval_mode == "normal"
                    ),
                )
                if self.retrieval_mode == "oracle":
                    if (
                        self._corpus_id_to_index is None
                        or gold_doc_ids is None
                        or gold_page_ids is None
                    ):
                        raise RuntimeError("Oracle corpus mapping was not initialized")
                    generation_kwargs["oracle_gold_indices"] = []
                    for document_row, page_row in zip(
                        gold_doc_ids[start : start + len(batch_questions)],
                        gold_page_ids[start : start + len(batch_questions)],
                    ):
                        representative_by_page: Dict[str, int] = {}
                        for document_id in document_row:
                            corpus_index = self._corpus_id_to_index[document_id]
                            representative_by_page.setdefault(
                                self.corpus_page_ids[corpus_index], corpus_index
                            )
                        generation_kwargs["oracle_gold_indices"].append(
                            [representative_by_page[page_id] for page_id in page_row]
                        )
                pred_texts, retrieved_indices = self.model.generate_from_questions(
                    **generation_kwargs
                )
            if len(pred_texts) != len(batch_questions):
                raise RuntimeError(
                    "Model returned a different number of predictions than questions: "
                    f"{len(pred_texts)} != {len(batch_questions)}"
                )
            if (
                self.use_rag_pipeline
                and batch_docs is None
                and hasattr(self.model, "get_rag_diagnostics")
            ):
                batch_diagnostics = self.model.get_rag_diagnostics()
                if len(batch_diagnostics) != len(batch_questions):
                    raise RuntimeError(
                        "Full RAG evaluation requires one retrieval diagnostic per "
                        f"example; got {len(batch_diagnostics)} for "
                        f"{len(batch_questions)} examples"
                    )

            batch_retrieved_ids: Optional[List[List[str]]] = None
            batch_retrieved_page_ids: Optional[List[List[str]]] = None
            oracle_pool_records: Optional[List[Any]] = None
            if self.retrieval_mode == "oracle":
                getter = getattr(self.model, "get_oracle_pool_records", None)
                if getter is None:
                    raise RuntimeError("Oracle model must expose its constructed pool records")
                oracle_pool_records = getter()
                if len(oracle_pool_records) != len(batch_questions):
                    raise RuntimeError(
                        "Oracle evaluation requires one pool record per question"
                    )
            if clara_recall_enabled:
                if not isinstance(retrieved_indices, torch.Tensor):
                    raise RuntimeError(
                        "CLaRa Recall@5 requires tensor-valued hard selector indices"
                    )
                if (
                    retrieved_indices.ndim != 2
                    or retrieved_indices.shape
                    != (len(batch_questions), self.model.generation_top_k)
                    or retrieved_indices.dtype == torch.bool
                    or torch.is_floating_point(retrieved_indices)
                ):
                    raise RuntimeError(
                        "CLaRa hard selector indices must have integer shape (B, top_k)"
                    )
                if (
                    clara_candidate_doc_ids is None
                    or clara_candidate_page_ids is None
                ):
                    raise RuntimeError("CLaRa candidate identities are unexpectedly absent")
                local_rows = retrieved_indices.detach().cpu().tolist()
                batch_retrieved_ids = []
                batch_retrieved_page_ids = []
                batch_candidate_doc_ids = clara_candidate_doc_ids[
                    start : start + len(batch_questions)
                ]
                batch_candidate_page_ids = clara_candidate_page_ids[
                    start : start + len(batch_questions)
                ]
                for row_index, index_row in enumerate(local_rows):
                    if (
                        len(index_row) != self.model.generation_top_k
                        or len(index_row) != len(set(index_row))
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, (int, np.integer))
                            or value < 0
                            or value >= len(batch_candidate_doc_ids[row_index])
                            for value in index_row
                        )
                    ):
                        raise RuntimeError(
                            f"CLaRa returned invalid hard top-k indices at row {row_index}"
                        )
                    deduplicated = _page_deduplicate_ranked_indices(
                        [int(value) for value in index_row],
                        batch_candidate_page_ids[row_index],
                        limit=5,
                    )
                    if not deduplicated:
                        raise RuntimeError("CLaRa page-deduplicated hard top-k is empty")
                    batch_retrieved_ids.append(
                        [batch_candidate_doc_ids[row_index][value] for value in deduplicated]
                    )
                    batch_retrieved_page_ids.append(
                        [batch_candidate_page_ids[row_index][value] for value in deduplicated]
                    )
            if gold_doc_ids is not None:
                if isinstance(retrieved_indices, torch.Tensor):
                    if retrieved_indices.ndim != 2:
                        raise RuntimeError(
                            "Model retrieval indices must be a two-dimensional batch"
                        )
                    if retrieved_indices.dtype == torch.bool or torch.is_floating_point(
                        retrieved_indices
                    ):
                        raise RuntimeError(
                            "Model retrieval indices must use an integer dtype"
                        )
                    index_rows = retrieved_indices.detach().cpu().tolist()
                elif isinstance(retrieved_indices, Sequence) and not isinstance(
                    retrieved_indices, (str, bytes)
                ):
                    index_rows = []
                    for row in retrieved_indices:
                        if isinstance(row, torch.Tensor):
                            if row.ndim != 1 or row.dtype == torch.bool or torch.is_floating_point(row):
                                raise RuntimeError(
                                    "Each retrieval-index row must be a one-dimensional integer sequence"
                                )
                            row = row.detach().cpu().tolist()
                        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                            raise RuntimeError(
                                "Each retrieval-index row must be an integer sequence"
                            )
                        index_rows.append(list(row))
                else:
                    raise RuntimeError(
                        "Model retrieval indices must be a batch of integer sequences"
                    )
                if len(index_rows) != len(batch_questions):
                    raise RuntimeError(
                        "Model returned a different number of retrieval rows than questions"
                    )
                batch_retrieved_ids = []
                batch_retrieved_page_ids = []
                for row_index, index_row in enumerate(index_rows):
                    if -1 in index_row:
                        first_padding = index_row.index(-1)
                        if any(value != -1 for value in index_row[first_padding:]):
                            raise RuntimeError(
                                "Padded retrieval indices must use trailing -1 values only"
                            )
                        index_row = index_row[:first_padding]
                    if not 1 <= len(index_row) <= 5:
                        raise RuntimeError(
                            "Paper CCEF returns between one and five survivors; "
                            f"batch row {row_index} returned {len(index_row)}"
                        )
                    if any(
                        isinstance(value, bool) or not isinstance(value, (int, np.integer))
                        for value in index_row
                    ):
                        raise RuntimeError(
                            f"Model returned a non-integer corpus index in batch row {row_index}"
                        )
                    index_row = [int(value) for value in index_row]
                    if any(
                        value < 0 or value >= len(self.corpus_ids)
                        for value in index_row
                    ):
                        raise RuntimeError(
                            f"Model returned an out-of-range corpus index in batch row {row_index}"
                        )
                    # D_q^(k) is defined after retaining only the first ranked
                    # occurrence of each page ID. Exact duplicate indices and
                    # different passages from the same page collapse here.
                    index_row = list(
                        _page_deduplicate_ranked_indices(
                            index_row,
                            self.corpus_page_ids,
                            limit=5,
                        )
                    )
                    if not index_row:
                        raise RuntimeError(
                            f"Page-deduplicated Recall@5 row {row_index} is empty"
                        )
                    batch_retrieved_ids.append(
                        [self.corpus_ids[value] for value in index_row]
                    )
                    batch_retrieved_page_ids.append(
                        [self.corpus_page_ids[value] for value in index_row]
                    )

            for offset, prediction in enumerate(pred_texts):
                index = start + offset
                golds = QAMetrics.gold_answers(gold_answers[index])
                metrics = QAMetrics.compute_all(prediction, golds)
                all_metrics.append(metrics)
                prediction_record: Dict[str, Any] = {
                    "example_id": str(example_ids[index]),
                    "question": batch_questions[offset],
                    "prediction": str(prediction),
                    "gold_answers": golds,
                    **metrics,
                }
                if batch_diagnostics is not None:
                    diagnostic = batch_diagnostics[offset]
                    raw_token_count = getattr(
                        diagnostic, "evidence_memory_tokens", None
                    )
                    if raw_token_count is not None:
                        token_count = int(raw_token_count)
                        if token_count <= 0:
                            raise RuntimeError(
                                "RAG diagnostics require a positive realized "
                                "evidence-token count"
                            )
                        evidence_memory_tokens.append(token_count)
                        prediction_record["retrieval_diagnostics"] = {
                            "final_document_count": int(
                                getattr(diagnostic, "final_candidates", 0)
                            ),
                            "second_round_candidate_count": int(
                                getattr(diagnostic, "second_round_candidates", 0)
                            ),
                            "evidence_memory_tokens": token_count,
                        }
                if gold_doc_ids is not None:
                    if (
                        batch_retrieved_ids is None
                        or batch_retrieved_page_ids is None
                        or gold_page_ids is None
                    ):
                        raise RuntimeError("Recall@5 retrieval IDs were not materialized")
                    retrieved_ids = batch_retrieved_ids[offset]
                    retrieved_pages = batch_retrieved_page_ids[offset]
                    gold_ids = gold_doc_ids[index]
                    gold_pages = gold_page_ids[index]
                    prediction_record.update(
                        {
                            "gold_doc_ids": gold_ids,
                            "gold_page_ids": gold_pages,
                            "retrieved_doc_ids": retrieved_ids,
                            "retrieved_page_ids": retrieved_pages,
                            "has_gold_support": bool(gold_pages),
                        }
                    )
                    if gold_pages:
                        gold_page_set = set(gold_pages)
                        for cutoff in retrieval_cutoffs:
                            recall = len(
                                set(retrieved_pages[:cutoff]) & gold_page_set
                            ) / len(gold_page_set)
                            retrieval_recalls[cutoff].append(recall)
                            prediction_record[f"recall_at_{cutoff}"] = recall
                    if self.retrieval_mode == "oracle":
                        if oracle_pool_records is None or self.corpus_ids is None:
                            raise RuntimeError("Oracle pool provenance was not materialized")
                        pool_record = oracle_pool_records[offset]
                        pool_ids = [
                            self.corpus_ids[value]
                            for value in pool_record.pool_indices
                        ]
                        pool_pages = [
                            self.corpus_page_ids[value]
                            for value in pool_record.pool_indices
                        ]
                        if len(pool_pages) != len(set(pool_pages)):
                            raise RuntimeError(
                                "Oracle top-100 pool contains duplicate page IDs"
                            )
                        if not set(retrieved_ids).issubset(pool_ids):
                            raise RuntimeError(
                                "Oracle Recall@5 result escaped the fixed top-100 pool"
                            )
                        prediction_record.update(
                            {
                                "oracle_pool_protocol": pool_record.protocol,
                                "oracle_pool_sha256": pool_record.pool_sha256,
                                "oracle_pool_doc_ids": pool_ids,
                                "oracle_pool_page_ids": pool_pages,
                                "oracle_injected_gold_doc_ids": [
                                    self.corpus_ids[value]
                                    for value in pool_record.injected_indices
                                ],
                                "oracle_injected_gold_page_ids": [
                                    self.corpus_page_ids[value]
                                    for value in pool_record.injected_indices
                                ],
                                "oracle_evicted_doc_ids": [
                                    self.corpus_ids[value]
                                    for value in pool_record.evicted_indices
                                ],
                                "oracle_evicted_page_ids": [
                                    self.corpus_page_ids[value]
                                    for value in pool_record.evicted_indices
                                ],
                            }
                        )
                elif clara_recall_enabled:
                    if (
                        batch_retrieved_ids is None
                        or batch_retrieved_page_ids is None
                        or clara_gold_candidate_indices is None
                        or clara_gold_page_ids is None
                        or clara_candidate_doc_ids is None
                    ):
                        raise RuntimeError(
                            "CLaRa Recall@5 identities were not materialized"
                        )
                    retrieved_ids = batch_retrieved_ids[offset]
                    retrieved_pages = batch_retrieved_page_ids[offset]
                    positive_indices = clara_gold_candidate_indices[index]
                    gold_ids = list(
                        dict.fromkeys(
                            clara_candidate_doc_ids[index][value]
                            for value in positive_indices
                        )
                    )
                    gold_pages = clara_gold_page_ids[index]
                    prediction_record.update(
                        {
                            "clara_candidate_count": _REPOSITORY_BGE_CANDIDATE_COUNT,
                            "clara_gold_candidate_indices": positive_indices,
                            "gold_doc_ids": gold_ids,
                            "gold_page_ids": gold_pages,
                            "retrieved_doc_ids": retrieved_ids,
                            "retrieved_page_ids": retrieved_pages,
                            "has_gold_support": bool(gold_pages),
                        }
                    )
                    if gold_pages:
                        recall = len(
                            set(retrieved_pages[:5]) & set(gold_pages)
                        ) / len(set(gold_pages))
                        retrieval_recalls[5].append(recall)
                        prediction_record["recall_at_5"] = recall
                predictions.append(prediction_record)

        if not all_metrics:
            empty_result = {
                "em": 0.0,
                "cem": 0.0,
                "f1": 0.0,
                "count": 0,
                "predictions": [],
            }
            if gold_doc_ids is not None or clara_recall_enabled:
                for cutoff in retrieval_cutoffs:
                    empty_result[f"recall_at_{cutoff}"] = 0.0
                empty_result["recall_at_5_support_count"] = 0
            if clara_retrieval_provenance is not None:
                empty_result["clara_retrieval_provenance"] = (
                    clara_retrieval_provenance
                )
            return empty_result
        result = {
            metric: float(np.mean([row[metric] for row in all_metrics]))
            for metric in ("em", "cem", "f1")
        } | {"count": len(all_metrics), "predictions": predictions}
        if evidence_memory_tokens:
            if len(evidence_memory_tokens) != len(all_metrics):
                raise RuntimeError(
                    "realized evidence-token diagnostics must cover every RAG example"
                )
            result["mean_evidence_memory_tokens"] = float(
                np.mean(evidence_memory_tokens)
            )
        if gold_doc_ids is not None or clara_recall_enabled:
            support_page_rows = (
                gold_page_ids if gold_doc_ids is not None else clara_gold_page_ids
            ) or []
            expected_support_count = sum(bool(page_row) for page_row in support_page_rows)
            if any(
                len(retrieval_recalls[cutoff]) != expected_support_count
                for cutoff in retrieval_cutoffs
            ):
                raise RuntimeError("Recall@k count does not match Q_sup supporting examples")
            result["recall_at_5_support_count"] = expected_support_count
            if expected_support_count:
                for cutoff in retrieval_cutoffs:
                    result[f"recall_at_{cutoff}"] = float(
                        np.mean(retrieval_recalls[cutoff])
                    )
        if clara_retrieval_provenance is not None:
            result["clara_retrieval_provenance"] = clara_retrieval_provenance
        return result

    def evaluate_multi_seed(
        self,
        questions: List[str],
        gold_answers: List[Any],
        seeds: Optional[List[Optional[int]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility wrapper that permits exactly one checkpoint/seed.

        Multi-seed aggregation requires independently trained checkpoint objects
        and is implemented by the CLI. Re-running ``self.model`` five times is
        intentionally rejected.
        """
        if seeds is None:
            seeds = [None]
        if len(seeds) != 1:
            raise ValueError(
                "ARIAEvaluator owns one checkpoint and cannot represent multiple "
                "training seeds. Supply --model_paths or --model_path_template to "
                "the evaluation CLI."
            )
        result = self.evaluate(questions, gold_answers, **kwargs)
        return aggregate_checkpoint_results([result], seeds, ["<in-memory-checkpoint>"])

    def get_rag_diagnostics(self):
        return self.model.get_rag_diagnostics()


def aggregate_checkpoint_results(
    results: List[Dict[str, Any]],
    seeds: List[Optional[int]],
    checkpoint_paths: List[str],
) -> Dict[str, Any]:
    """Aggregate one evaluation result per independently trained checkpoint."""
    if not results:
        raise ValueError("At least one checkpoint result is required")
    if not (len(results) == len(seeds) == len(checkpoint_paths)):
        raise ValueError("results, seeds, and checkpoint_paths must have equal length")

    per_seed: List[Dict[str, Any]] = []
    for result, seed, checkpoint_path in zip(results, seeds, checkpoint_paths):
        per_seed.append(
            {
                "seed": seed,
                "checkpoint": checkpoint_path,
                **result,
            }
        )

    metric_names = ["em", "cem", "f1"]
    evidence_presence = ["mean_evidence_memory_tokens" in result for result in results]
    if any(evidence_presence) and not all(evidence_presence):
        raise ValueError(
            "mean_evidence_memory_tokens must be present for every checkpoint "
            "result or omitted from all"
        )
    if all(evidence_presence):
        metric_names.append("mean_evidence_memory_tokens")
    for recall_metric in ("recall_at_1", "recall_at_3", "recall_at_5"):
        recall_presence = [recall_metric in result for result in results]
        if any(recall_presence) and not all(recall_presence):
            raise ValueError(
                f"{recall_metric} must be present for every checkpoint result "
                "or omitted from all"
            )
        if all(recall_presence):
            metric_names.append(recall_metric)

    mean_metrics: Dict[str, float] = {}
    std_metrics: Dict[str, float] = {}
    for metric in metric_names:
        values = np.asarray([result[metric] for result in results], dtype=np.float64)
        mean_metrics[metric] = float(values.mean())
        # Population SD is the convention used by the existing paper tables.
        std_metrics[metric] = float(values.std(ddof=0))

    aggregated = {
        "mean": mean_metrics,
        "std": std_metrics,
        "per_seed": per_seed,
        "seeds": seeds,
        "checkpoints": checkpoint_paths,
        "n_checkpoints": len(results),
    }
    clara_provenance = [
        result.get("clara_retrieval_provenance") for result in results
    ]
    if any(value is not None for value in clara_provenance):
        if any(value is None for value in clara_provenance) or any(
            value != clara_provenance[0] for value in clara_provenance[1:]
        ):
            raise ValueError(
                "All matched CLaRa checkpoints must share exact evaluation-candidate provenance"
            )
        aggregated["clara_retrieval_provenance"] = clara_provenance[0]
    return aggregated


# ---------------------------------------------------------------------------
# Paired bootstrap significance testing
# ---------------------------------------------------------------------------


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42,
    example_ids: Optional[Sequence[Hashable]] = None,
    seed_ids: Optional[Sequence[Hashable]] = None,
) -> Dict[str, Any]:
    """Two-sided paired bootstrap pooled across seeds by example identity.

    ``scores_a`` and ``scores_b`` may be one-dimensional aligned observations or
    two-dimensional ``(training_seed, example)`` matrices. For multi-seed input,
    observations are first pooled within each example identity (mean over seeds),
    then example identities are resampled with replacement. This preserves the
    pairing and avoids treating five outputs for one query as five independent
    queries. Flat multi-seed input should supply both ``example_ids`` and
    ``seed_ids``.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    array_a = np.asarray(scores_a, dtype=np.float64)
    array_b = np.asarray(scores_b, dtype=np.float64)
    if array_a.shape != array_b.shape:
        raise ValueError(
            f"Paired score arrays must have identical shape: {array_a.shape} != {array_b.shape}"
        )
    if array_a.ndim not in (1, 2):
        raise ValueError("scores must be a 1-D vector or a 2-D (seed, example) matrix")
    if array_a.size == 0:
        raise ValueError("scores must not be empty")
    if not np.isfinite(array_a).all() or not np.isfinite(array_b).all():
        raise ValueError("scores contain NaN or infinite values")

    original_shape = array_a.shape
    if array_a.ndim == 2:
        n_seed_rows, n_examples = array_a.shape
        if example_ids is None:
            example_ids = np.tile(np.arange(n_examples), n_seed_rows).tolist()
        elif len(example_ids) == n_examples:
            example_ids = np.tile(np.asarray(example_ids, dtype=object), n_seed_rows).tolist()
        if seed_ids is None:
            seed_ids = np.repeat(np.arange(n_seed_rows), n_examples).tolist()
        elif len(seed_ids) == n_seed_rows:
            seed_ids = np.repeat(np.asarray(seed_ids, dtype=object), n_examples).tolist()
    elif seed_ids is not None and example_ids is None:
        raise ValueError("Flat multi-seed scores require explicit example_ids")

    flat_a = array_a.reshape(-1)
    flat_b = array_b.reshape(-1)
    if example_ids is None:
        example_ids = list(range(flat_a.size))
    if len(example_ids) != flat_a.size:
        raise ValueError("example_ids must contain one identity per paired observation")
    if seed_ids is not None and len(seed_ids) != flat_a.size:
        raise ValueError("seed_ids must contain one seed per paired observation")

    if seed_ids is not None:
        seen_pairs: set[Tuple[str, str]] = set()
        for example_id, training_seed in zip(example_ids, seed_ids):
            pair = (str(example_id), str(training_seed))
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate (example_id, seed_id) observation encountered: {pair}"
                )
            seen_pairs.add(pair)

    grouped_differences: Dict[str, List[float]] = defaultdict(list)
    for example_id, difference in zip(example_ids, flat_a - flat_b):
        grouped_differences[str(example_id)].append(float(difference))

    # Equal weight per query identity, with seed observations pooled inside it.
    identity_differences = np.asarray(
        [np.mean(values) for values in grouped_differences.values()], dtype=np.float64
    )
    mean_diff = float(identity_differences.mean())

    rng = np.random.default_rng(seed)
    n_identities = identity_differences.size
    bootstrap_differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sampled = rng.integers(0, n_identities, size=n_identities)
        bootstrap_differences[index] = identity_differences[sampled].mean()

    # A genuine two-sided p-value: twice the smaller bootstrap tail, with the
    # standard plus-one finite-resampling correction and a cap at one.
    lower_tail = (np.count_nonzero(bootstrap_differences <= 0.0) + 1) / (
        n_resamples + 1
    )
    upper_tail = (np.count_nonzero(bootstrap_differences >= 0.0) + 1) / (
        n_resamples + 1
    )
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))

    unique_seed_count = (
        len({str(value) for value in seed_ids}) if seed_ids is not None else 1
    )
    return {
        "mean_diff": mean_diff,
        "ci_95_lower": float(np.percentile(bootstrap_differences, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_differences, 97.5)),
        "p_value": float(p_value),
        "alternative": "two-sided",
        "n_resamples": int(n_resamples),
        "n_examples": int(n_identities),
        "n_observations": int(flat_a.size),
        "n_seeds": int(unique_seed_count),
        "input_shape": list(original_shape),
        "resampling_unit": "example_identity",
        "seed_pooling": "mean_within_example",
    }


def _stratified_benchmark_bootstrap_test(
    identity_differences_by_benchmark: Mapping[str, Sequence[float]],
    *,
    n_observations_by_benchmark: Mapping[str, int],
    n_seeds_by_benchmark: Mapping[str, int],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap the paper's unweighted mean across benchmark panels.

    Training-seed observations must already be pooled within each example
    identity.  Each bootstrap replicate resamples identities independently
    *within* every benchmark, computes one mean per benchmark, and then gives
    those benchmark means equal weight.  This deliberately avoids allowing a
    larger evaluation set to dominate the headline ``Avg`` statistic.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not identity_differences_by_benchmark:
        raise ValueError("At least one benchmark panel is required")

    benchmarks = sorted(identity_differences_by_benchmark)
    panels: Dict[str, np.ndarray] = {}
    for benchmark in benchmarks:
        differences = np.asarray(
            identity_differences_by_benchmark[benchmark], dtype=np.float64
        )
        if differences.ndim != 1 or differences.size == 0:
            raise ValueError(
                f"Benchmark {benchmark!r} must contain a non-empty 1-D identity panel"
            )
        if not np.isfinite(differences).all():
            raise ValueError(
                f"Benchmark {benchmark!r} contains NaN or infinite differences"
            )
        panels[benchmark] = differences

    missing_observation_counts = set(benchmarks) - set(n_observations_by_benchmark)
    missing_seed_counts = set(benchmarks) - set(n_seeds_by_benchmark)
    if missing_observation_counts or missing_seed_counts:
        raise ValueError(
            "Benchmark metadata is incomplete: "
            f"missing_observations={sorted(missing_observation_counts)}, "
            f"missing_seeds={sorted(missing_seed_counts)}"
        )

    benchmark_mean_differences = {
        benchmark: float(panels[benchmark].mean()) for benchmark in benchmarks
    }
    mean_diff = float(np.mean(list(benchmark_mean_differences.values())))

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        replicate_benchmark_means: List[float] = []
        for benchmark in benchmarks:
            differences = panels[benchmark]
            sampled = rng.integers(0, differences.size, size=differences.size)
            replicate_benchmark_means.append(float(differences[sampled].mean()))
        bootstrap_differences[index] = float(np.mean(replicate_benchmark_means))

    lower_tail = (np.count_nonzero(bootstrap_differences <= 0.0) + 1) / (
        n_resamples + 1
    )
    upper_tail = (np.count_nonzero(bootstrap_differences >= 0.0) + 1) / (
        n_resamples + 1
    )
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))

    n_examples_by_benchmark = {
        benchmark: int(panels[benchmark].size) for benchmark in benchmarks
    }
    observations = {
        benchmark: int(n_observations_by_benchmark[benchmark])
        for benchmark in benchmarks
    }
    seeds = {
        benchmark: int(n_seeds_by_benchmark[benchmark]) for benchmark in benchmarks
    }
    return {
        "mean_diff": mean_diff,
        "ci_95_lower": float(np.percentile(bootstrap_differences, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_differences, 97.5)),
        "p_value": float(p_value),
        "alternative": "two-sided",
        "n_resamples": int(n_resamples),
        "n_benchmarks": len(benchmarks),
        "benchmarks": benchmarks,
        "benchmark_mean_differences": benchmark_mean_differences,
        "n_examples": int(sum(n_examples_by_benchmark.values())),
        "n_examples_by_benchmark": n_examples_by_benchmark,
        "n_observations": int(sum(observations.values())),
        "n_observations_by_benchmark": observations,
        "n_seeds_by_benchmark": seeds,
        "resampling_unit": "example_identity_within_benchmark",
        "seed_pooling": "mean_within_example",
        "benchmark_weighting": "unweighted_mean",
    }


def compare_evaluation_payloads(
    candidate_results: Mapping[str, Any],
    baseline_results: Mapping[str, Any],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run aligned per-benchmark and paper-level ``Avg`` significance tests."""
    comparisons: Dict[str, Any] = {}
    avg_identity_differences: Dict[str, Dict[str, np.ndarray]] = {
        metric: {} for metric in ("em", "cem", "f1")
    }
    n_observations_by_benchmark: Dict[str, int] = {}
    n_seeds_by_benchmark: Dict[str, int] = {}
    dataset_names = [name for name in candidate_results if name != "avg"]
    for dataset_name in dataset_names:
        if dataset_name not in baseline_results:
            raise ValueError(f"Baseline results have no {dataset_name!r} dataset")
        candidate_runs = candidate_results[dataset_name].get("per_seed", [])
        baseline_runs = baseline_results[dataset_name].get("per_seed", [])
        if not candidate_runs or not baseline_runs:
            raise ValueError(
                f"Both result files must retain per-seed predictions for {dataset_name}"
            )

        def validate_panel(
            result: Mapping[str, Any],
            runs: Sequence[Mapping[str, Any]],
            label: str,
        ) -> set[str]:
            declared_seeds = {str(value) for value in result.get("seeds", [])}
            run_seeds = [str(run.get("seed")) for run in runs]
            if not declared_seeds or set(run_seeds) != declared_seeds:
                raise ValueError(
                    f"{label} {dataset_name} run seeds do not match declared seeds"
                )
            if len(run_seeds) != len(set(run_seeds)):
                raise ValueError(f"{label} {dataset_name} contains duplicate seed runs")
            expected_ids: Optional[set[str]] = None
            canonical_content: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
            for run in runs:
                predictions = run.get("predictions", [])
                ids = {str(prediction.get("example_id")) for prediction in predictions}
                if len(ids) != len(predictions):
                    raise ValueError(
                        f"{label} {dataset_name} contains duplicate example IDs in one seed"
                    )
                if expected_ids is None:
                    expected_ids = ids
                elif ids != expected_ids:
                    raise ValueError(
                        f"{label} {dataset_name} is not a complete seed x example panel"
                    )
                for prediction in predictions:
                    example_id = str(prediction.get("example_id"))
                    signature = (
                        str(prediction.get("question")),
                        tuple(
                            sorted(
                                QAMetrics.normalize_answer(answer)
                                for answer in QAMetrics.gold_answers(
                                    prediction.get("gold_answers", [])
                                )
                            )
                        ),
                    )
                    previous = canonical_content.setdefault(example_id, signature)
                    if previous != signature:
                        raise ValueError(
                            f"{label} reuses {example_id!r} for different content"
                        )
            if not expected_ids:
                raise ValueError(f"{label} {dataset_name} prediction panel is empty")
            return declared_seeds

        candidate_seeds = validate_panel(
            candidate_results[dataset_name], candidate_runs, "candidate"
        )
        baseline_seeds = validate_panel(
            baseline_results[dataset_name], baseline_runs, "baseline"
        )
        if candidate_seeds != baseline_seeds:
            raise ValueError(
                f"Candidate and baseline seed sets differ for {dataset_name}: "
                f"{sorted(candidate_seeds)} != {sorted(baseline_seeds)}"
            )

        def index_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
            indexed: Dict[Tuple[str, str], Mapping[str, Any]] = {}
            for run in runs:
                training_seed = str(run.get("seed"))
                for prediction in run.get("predictions", []):
                    key = (training_seed, str(prediction.get("example_id")))
                    if key in indexed:
                        raise ValueError(f"Duplicate saved prediction identity: {key}")
                    indexed[key] = prediction
            return indexed

        candidate_index = index_runs(candidate_runs)
        baseline_index = index_runs(baseline_runs)
        if not candidate_index or candidate_index.keys() != baseline_index.keys():
            missing_candidate = sorted(baseline_index.keys() - candidate_index.keys())[:3]
            missing_baseline = sorted(candidate_index.keys() - baseline_index.keys())[:3]
            raise ValueError(
                f"Saved predictions are not paired for {dataset_name}; "
                f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
            )
        for key in candidate_index:
            candidate_prediction = candidate_index[key]
            baseline_prediction = baseline_index[key]
            candidate_signature = (
                str(candidate_prediction.get("question")),
                tuple(
                    sorted(
                        QAMetrics.normalize_answer(answer)
                        for answer in QAMetrics.gold_answers(
                            candidate_prediction.get("gold_answers", [])
                        )
                    )
                ),
            )
            baseline_signature = (
                str(baseline_prediction.get("question")),
                tuple(
                    sorted(
                        QAMetrics.normalize_answer(answer)
                        for answer in QAMetrics.gold_answers(
                            baseline_prediction.get("gold_answers", [])
                        )
                    )
                ),
            )
            if candidate_signature != baseline_signature:
                raise ValueError(
                    f"Saved prediction content differs for paired identity {key}"
                )
        keys = sorted(candidate_index)
        example_ids = [example_id for _, example_id in keys]
        seed_ids = [training_seed for training_seed, _ in keys]
        comparisons[dataset_name] = {
            metric: paired_bootstrap_test(
                [float(candidate_index[key][metric]) for key in keys],
                [float(baseline_index[key][metric]) for key in keys],
                n_resamples=n_resamples,
                seed=seed,
                example_ids=example_ids,
                seed_ids=seed_ids,
            )
            for metric in ("em", "cem", "f1")
        }
        n_observations_by_benchmark[dataset_name] = len(keys)
        n_seeds_by_benchmark[dataset_name] = len(candidate_seeds)
        for metric in ("em", "cem", "f1"):
            grouped_differences: Dict[str, List[float]] = defaultdict(list)
            for key in keys:
                grouped_differences[key[1]].append(
                    float(candidate_index[key][metric])
                    - float(baseline_index[key][metric])
                )
            avg_identity_differences[metric][dataset_name] = np.asarray(
                [np.mean(values) for values in grouped_differences.values()],
                dtype=np.float64,
            )

    if dataset_names:
        comparisons["avg"] = {
            metric: _stratified_benchmark_bootstrap_test(
                avg_identity_differences[metric],
                n_observations_by_benchmark=n_observations_by_benchmark,
                n_seeds_by_benchmark=n_seeds_by_benchmark,
                n_resamples=n_resamples,
                seed=seed,
            )
            for metric in ("em", "cem", "f1")
        }
    return comparisons


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _canonical_checkpoint(path: str) -> str:
    expanded = os.path.expanduser(path)
    return os.path.realpath(expanded) if os.path.exists(expanded) else expanded


def _resolve_seed_checkpoints(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> List[Tuple[Optional[int], str]]:
    seeds: Optional[List[int]] = args.seeds

    if args.model_path is not None:
        if seeds is not None and len(seeds) != 1:
            parser.error(
                "--model_path identifies one checkpoint and accepts at most one --seeds "
                "value; use --model_paths or --model_path_template for multi-seed evaluation"
            )
        resolved = [(seeds[0] if seeds else None, os.path.expanduser(args.model_path))]
    elif args.model_paths is not None:
        if seeds is None:
            parser.error("--model_paths requires matching explicit --seeds values")
        if len(seeds) != len(args.model_paths):
            parser.error("--model_paths and --seeds must have the same number of values")
        resolved = [
            (training_seed, os.path.expanduser(path))
            for training_seed, path in zip(seeds, args.model_paths)
        ]
    else:
        if not seeds:
            parser.error("--model_path_template requires explicit --seeds values")
        if "{seed" not in args.model_path_template:
            parser.error("--model_path_template must include the {seed} template field")
        resolved = [
            (
                training_seed,
                _format_artifact_path(
                    args.model_path_template,
                    seed=training_seed,
                    compression_rate=args.compression_rate,
                ),
            )
            for training_seed in seeds
        ]

    seed_values = [seed for seed, _ in resolved if seed is not None]
    if len(seed_values) != len(set(seed_values)):
        parser.error("Training seed identifiers must be unique")
    canonical_paths = [_canonical_checkpoint(path) for _, path in resolved]
    if len(canonical_paths) != len(set(canonical_paths)):
        parser.error(
            "Each training seed must reference a distinct checkpoint; duplicate paths "
            "would re-run one deterministic model and invalidate seed statistics"
        )
    return resolved


def _set_inference_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _required_checkpoint_configuration(runtime_configuration: str) -> str:
    """Compatibility wrapper around the shared protocol registry."""
    return required_checkpoint_configuration(runtime_configuration)


def _assert_normal_retrieval_is_not_training_index(
    checkpoint_config: CLaRaConfig,
    *,
    evaluation_corpus_sha256: str,
    evaluation_index_sha256: str,
) -> None:
    """Reject accidentally evaluating on the de-duplicated training index."""
    training_corpus_sha256 = getattr(
        checkpoint_config, "aria_training_corpus_sha256", None
    )
    training_index_sha256 = getattr(
        checkpoint_config, "aria_training_retrieval_index_sha256", None
    )
    if evaluation_corpus_sha256 == training_corpus_sha256:
        raise ValueError(
            "Normal evaluation requires the full KILT corpus, not the "
            "page-URL-deduplicated Phase-II training corpus"
        )
    if evaluation_index_sha256 == training_index_sha256:
        raise ValueError(
            "Normal evaluation requires a full-KILT BGE index distinct from the "
            "Phase-II training index"
        )


def _validate_checkpoint_protocol(
    model: CLaRa,
    checkpoint_path: str,
    training_seed: Optional[int],
    compression_rate: int,
    expected_configuration: str,
) -> Dict[str, Any]:
    """Verify immutable *training* metadata instead of trusting CLI labels.

    The checkpoint is deliberately not asked to identify the Normal-evaluation
    corpus.  Its retrieval provenance belongs to the page-URL-deduplicated
    Phase-II training corpus, whereas evaluation validates a separate full-KILT
    corpus/index pair at load time.
    """
    config = model.config
    decoder_revision = getattr(
        config, "decoder_model_resolved_revision", None
    )
    if not isinstance(decoder_revision, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}", decoder_revision
    ) is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} requires an exact resolved base-model revision"
        )
    mads_model = getattr(config, "mads_semantic_model_name", PAPER_BGE_MODEL)
    configuration_uses_mads = expected_configuration not in {
        "remove_mads",
        "clara_baseline",
    }
    if configuration_uses_mads and mads_model != PAPER_BGE_MODEL:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} requires MADS model "
            f"{PAPER_BGE_MODEL!r}, got {mads_model!r}"
        )
    actual_rate = int(getattr(config, "aria_compression_rate", model.compr_rate))
    if actual_rate != compression_rate or int(model.compr_rate) != compression_rate:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} was trained at CR={actual_rate}, "
            f"not requested CR={compression_rate}"
        )
    checkpoint_configuration = _required_checkpoint_configuration(
        expected_configuration
    )
    actual_configuration = getattr(config, "aria_rag_configuration", None)
    if actual_configuration != checkpoint_configuration:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} was trained as "
            f"{actual_configuration!r}, expected {checkpoint_configuration!r} "
            f"for runtime configuration {expected_configuration!r}"
        )
    trained_rag_config = create_paper_rag_config(
        checkpoint_configuration, compression_rate
    )
    expected_control_metadata = {
        "aria_coupling_control_protocol": COUPLING_CONTROL_PROTOCOL,
        "aria_acr_allocation_mode": trained_rag_config.acr_allocation_mode,
        "aria_second_retrieval_mode": trained_rag_config.second_retrieval_mode,
        "aria_uniform_evidence_token_budget": (
            MATCHED_EVIDENCE_TOKEN_BUDGET
            if trained_rag_config.acr_allocation_mode == "uniform_budget"
            else None
        ),
        "aria_uniform_allocation_scheme": (
            UNIFORM_BUDGET_ALLOCATION_SCHEME
            if trained_rag_config.acr_allocation_mode == "uniform_budget"
            else None
        ),
        "aria_static_second_query_scheme": (
            STATIC_SECOND_QUERY_SCHEME
            if trained_rag_config.second_retrieval_mode == "static_query"
            else None
        ),
        "aria_release_convention_inferred": (
            trained_rag_config.acr_allocation_mode == "uniform_budget"
            or trained_rag_config.second_retrieval_mode == "static_query"
        ),
    }
    if checkpoint_configuration in {
        "remove_cfrs",
        "uniform_acr",
        "static_second_retrieval",
        "remove_all_coupling",
    }:
        for key, expected in expected_control_metadata.items():
            actual = getattr(config, key, None)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint coupling metadata {key!r} must be {expected!r}, "
                    f"got {actual!r}"
                )
    recorded_seed = getattr(config, "aria_training_seed", None)
    if recorded_seed is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} requires aria_training_seed provenance"
        )
    if int(recorded_seed) not in PAPER_TRAINING_SEEDS:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} records a non-paper seed: {recorded_seed}"
        )
    if training_seed is not None and int(recorded_seed) != int(training_seed):
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} records seed {recorded_seed}, "
            f"not CLI seed {training_seed}"
        )
    manifest_digest = getattr(config, "aria_dataset_manifest_sha256", None)
    epoch_schedule = getattr(config, "aria_epoch_seed_schedule", None)
    if (
        not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
        or not isinstance(epoch_schedule, list)
        or len(epoch_schedule) != 5
        or len(set(epoch_schedule)) != 5
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} requires complete Phase-II dataset provenance"
        )
    if model.training_stage != "stage2" or model.generation_top_k != 5:
        raise ValueError(
            "Paper evaluation requires a Phase-II checkpoint with a top-5 candidate ceiling"
        )
    canonical_architecture = {
        "doc_max_length": 768,
        "query_max_length": 256,
        "stage2_input_max_length": 1024,
        "max_new_tokens": PAPER_MAX_NEW_TOKENS,
        "lora": True,
        "lora_r": 16,
        "lora_r_compressor": 16,
        "sep": True,
        "different_mem_tokens": True,
        "optimize_mem_tokens": False,
        "compr_model_name": None,
        "compr_n_layers": 5,
        "compr_use_mlp": False,
        "training_form": "both_separately",
        "stage2_retrieval_top_n": 5,
        "aria_text_sha256_scheme": TEXT_SHA256_SCHEME,
        "qr_input_scheme": QR_INPUT_SCHEME,
        "mtfrl_initialization_scheme": MTFRL_INITIALIZATION_SCHEME,
        "cfrs_reconstruction_scheme": CFRS_RECONSTRUCTION_SCHEME,
        "cfrs_reconstruction_chunk_tokens": 128,
    }
    for key, expected in canonical_architecture.items():
        if getattr(config, key, None) != expected:
            raise ValueError(
                f"Checkpoint architecture {key!r} must be {expected!r}, "
                f"got {getattr(config, key, None)!r}"
            )
    expected_lora_targets: Any = (
        "all-linear" if checkpoint_configuration == "clara_baseline" else ["q_proj"]
    )
    if getattr(config, "lora_target_modules", None) != expected_lora_targets:
        raise ValueError(
            "CLaRa checkpoints require all-linear LoRA placement"
            if checkpoint_configuration == "clara_baseline"
            else "ARIA checkpoints require q_proj-only LoRA placement"
        )
    if checkpoint_configuration == "clara_baseline":
        expected_clara_metadata = {
            "clara_selector_scheme": CLARA_SELECTOR_SCHEME,
            "clara_document_representation_scheme": (
                CLARA_DOCUMENT_REPRESENTATION_SCHEME
            ),
            "clara_phase2_objective": CLARA_PHASE2_OBJECTIVE,
            "clara_phase2_trainable_adapters": [
                "query_reasoner_adapter",
                "decoder_adapter",
            ],
            "clara_phase2_frozen_adapter": "encoder_adapter",
            "clara_phase2_adapter_initialization": (
                "both-exact-copy-of-corresponding-phase1-compressor-v1"
            ),
            "clara_memory_allocation_scheme": CLARA_MEMORY_ALLOCATION_SCHEME,
            "clara_max_memory_tokens": max(1, 768 // compression_rate),
            "clara_training_candidate_count": 5,
            "clara_evaluation_candidate_protocol": (
                CLARA_EVALUATION_CANDIDATE_PROTOCOL
            ),
            "clara_evaluation_candidate_count": 20,
            "clara_selection_count": 5,
            "clara_archive_document_id_scheme": CLARA_ARCHIVE_DOCUMENT_ID_SCHEME,
            "clara_archive_page_id_scheme": CLARA_ARCHIVE_PAGE_ID_SCHEME,
            "aria_loss_weights": {
                "lambda_mse": 0.0,
                "lambda_cfrs": 0.0,
                "lambda_qr": 0.0,
                "lambda_mtfrl": 0.0,
            },
        }
        for key, expected in expected_clara_metadata.items():
            if getattr(config, key, None) != expected:
                raise ValueError(
                    f"CLaRa checkpoint metadata {key!r} must be {expected!r}, "
                    f"got {getattr(config, key, None)!r}"
                )
        adapter_keys = getattr(model, "adapter_keys", None)
        if set(adapter_keys or ()) != {
            "encoder_adapter",
            "query_reasoner_adapter",
            "decoder_adapter",
        }:
            raise ValueError(
                "CLaRa checkpoint requires exactly encoder, query-reasoner, and "
                "decoder adapters"
            )
        trainable_names = getattr(config, "aria_trainable_parameter_names", None)
        if (
            not isinstance(trainable_names, list)
            or not trainable_names
            or any(
                "query_reasoner_adapter" not in name and "decoder_adapter" not in name
                for name in trainable_names
            )
            or not all(
                any(adapter in name for name in trainable_names)
                for adapter in ("query_reasoner_adapter", "decoder_adapter")
            )
        ):
            raise ValueError(
                "CLaRa checkpoint optimizer provenance must contain only QR and "
                "generator adapter parameters"
            )
    if checkpoint_configuration not in {
        "static_second_retrieval",
        "remove_all_coupling",
        "clara_baseline",
    }:
        if getattr(config, "mtfrl_initialization_rank", None) is not None:
            raise ValueError(
                "Paper MTFRL uses Xavier-uniform initialization, not a low-rank factor"
            )
        decoder_hidden_size = int(model.decoder.config.hidden_size)
        if decoder_hidden_size % 2 != 0:
            raise ValueError("Decoder hidden size must be even for the H/2 MTFRL head")
        expected_mtfrl_width = decoder_hidden_size // 2
        if getattr(config, "mtfrl_hidden_width", None) != expected_mtfrl_width:
            raise ValueError(
                "Paper MTFRL checkpoint hidden width must be H/2: "
                f"expected {expected_mtfrl_width}, got "
                f"{getattr(config, 'mtfrl_hidden_width', None)!r}"
            )
    optional_training_lengths = {
        "aria_passage_max_length": 768,
        "aria_query_max_length": 256,
        "aria_input_max_length": 1024,
        "aria_target_max_length": 128,
    }
    for key, expected in optional_training_lengths.items():
        actual = getattr(config, key, None)
        if actual is not None and actual != expected:
            raise ValueError(
                f"Checkpoint training protocol {key!r} must be {expected}, got {actual!r}"
            )
    test_url_digest = getattr(config, "aria_test_url_sha256", None)
    if not isinstance(test_url_digest, str) or len(test_url_digest) != 64:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} requires the official-test URL fingerprint"
        )
    if checkpoint_configuration != "clara_baseline":
        projection_metadata = getattr(model, "_bge_projection_metadata", None) or {}
        expected_projection_metadata = {
            "base_model": model.decoder_model_name,
            "base_model_revision_resolved": decoder_revision,
            "bge_model": "BAAI/bge-large-en-v1.5",
            "sample_count": 50_000,
            "epochs": 2,
            "batch_size": 128,
            "learning_rate": 5e-4,
            "test_url_sha256": test_url_digest,
            "text_sha256_scheme": TEXT_SHA256_SCHEME,
            "qr_input_scheme": QR_INPUT_SCHEME,
        }
        for key, expected in expected_projection_metadata.items():
            if projection_metadata.get(key) != expected:
                raise ValueError(
                    f"W_BGE metadata {key!r} must be {expected!r}, "
                    f"got {projection_metadata.get(key)!r}"
                )
        if not isinstance(projection_metadata.get("seed"), int):
            raise ValueError("W_BGE requires its fitting seed")
        for key in (
            "query_sha256",
            "passage_id_sha256",
            "passage_text_sha256",
            "test_url_sha256",
        ):
            value = projection_metadata.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"W_BGE requires the {key} fingerprint")
        bge_artifact_format = projection_metadata.get(
            "bge_embedding_artifact_format"
        )
        if bge_artifact_format == "aria-bge-artifact-v2":
            source_kind = projection_metadata.get("bge_encoder_source_kind")
            if source_kind == "huggingface-hub":
                resolved = projection_metadata.get(
                    "bge_encoder_revision_resolved"
                )
                if not isinstance(resolved, str) or re.fullmatch(
                    r"[0-9a-fA-F]{40}", resolved
                ) is None:
                    raise ValueError("W_BGE requires its resolved BGE Hub commit")
            elif source_kind == "local-directory":
                source_digest = projection_metadata.get(
                    "bge_encoder_source_sha256"
                )
                if not isinstance(source_digest, str) or re.fullmatch(
                    r"[0-9a-f]{64}", source_digest
                ) is None:
                    raise ValueError("W_BGE requires its local BGE tree SHA-256")
            else:
                raise ValueError("W_BGE has an invalid BGE encoder source kind")
        projection = getattr(model, "_bge_projection", None)
        if projection is None:
            raise ValueError("Paper ARIA evaluation requires W_BGE in the checkpoint")
        projection_sha256 = hashlib.sha256(
            projection.weight.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        w_bge_fingerprint: Optional[Tuple[Any, ...]] = (
            projection_metadata.get("base_model"),
            projection_metadata.get("base_model_revision_resolved"),
            projection_metadata.get("bge_model"),
            projection_metadata.get("sample_count"),
            projection_metadata.get("epochs"),
            projection_metadata.get("batch_size"),
            projection_metadata.get("learning_rate"),
            projection_metadata.get("seed"),
            projection_metadata.get("query_sha256"),
            projection_metadata.get("passage_id_sha256"),
            projection_metadata.get("passage_text_sha256"),
            projection_metadata.get("test_url_sha256"),
            projection_metadata.get("text_sha256_scheme"),
            projection_metadata.get("qr_input_scheme"),
            bge_artifact_format,
            projection_metadata.get("bge_encoder_source"),
            projection_metadata.get("bge_encoder_source_kind"),
            projection_metadata.get("bge_encoder_revision_declared"),
            projection_metadata.get("bge_encoder_revision_resolved"),
            projection_metadata.get("bge_encoder_revision_was_explicit"),
            projection_metadata.get("bge_encoder_source_sha256"),
            projection_metadata.get("bge_encoder_source_sha256_scheme"),
            projection_sha256,
        )
    else:
        w_bge_fingerprint = None

    phase1_seed = getattr(config, "aria_phase1_training_seed", None)
    phase1_manifest = getattr(config, "aria_phase1_dataset_manifest_sha256", None)
    phase1_test_digest = getattr(config, "aria_phase1_test_url_sha256", None)
    if phase1_seed is None or int(phase1_seed) != int(recorded_seed):
        raise ValueError("Phase-I and Phase-II training seeds are not aligned")
    if (
        not isinstance(phase1_manifest, str)
        or len(phase1_manifest) != 64
        or phase1_test_digest != test_url_digest
        or getattr(config, "aria_phase1_base_model", None) != model.decoder_model_name
        or getattr(config, "aria_phase1_base_model_resolved_revision", None)
        != decoder_revision
        or int(getattr(config, "aria_phase1_compression_rate", -1)) != compression_rate
    ):
        raise ValueError("Checkpoint requires aligned Phase-I provenance")

    training_index_digest = getattr(
        config, "aria_training_retrieval_index_sha256", None
    )
    training_candidate_digest = getattr(
        config, "aria_training_candidate_order_sha256", None
    )
    if (
        not isinstance(training_index_digest, str)
        or len(training_index_digest) != 64
        or not isinstance(training_candidate_digest, str)
        or len(training_candidate_digest) != 64
    ):
        raise ValueError("Checkpoint requires fixed BGE top-5 training provenance")
    recorded_corpus_sha256 = getattr(config, "aria_training_corpus_sha256", None)
    recorded_corpus_count = getattr(config, "aria_training_corpus_count", None)
    recorded_corpus_scheme = getattr(
        config, "aria_training_corpus_sha256_scheme", None
    )
    recorded_corpus_scope = getattr(config, "aria_training_corpus_scope", None)
    if (
        not isinstance(recorded_corpus_sha256, str)
        or len(recorded_corpus_sha256) != 64
        or not isinstance(recorded_corpus_count, int)
        or recorded_corpus_count <= 0
        or recorded_corpus_scheme != CORPUS_SHA256_SCHEME
        or recorded_corpus_scope != "page_url_deduplicated"
    ):
        raise ValueError(
            "Checkpoint requires its page-URL-deduplicated training corpus fingerprint"
        )
    source_scheme = getattr(config, "aria_source_snapshot_scheme", None)
    source_tree = getattr(config, "aria_source_tree_sha256", None)
    source_commit = getattr(config, "aria_source_git_commit", None)
    source_dirty = getattr(config, "aria_source_git_dirty", None)
    source_file_count = getattr(config, "aria_source_file_count", None)
    if (
        source_scheme != SOURCE_SNAPSHOT_SCHEME
        or not isinstance(source_tree, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_tree) is None
        or (
            source_commit is not None
            and (
                not isinstance(source_commit, str)
                or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
            )
        )
        or (source_dirty is not None and not isinstance(source_dirty, bool))
        or isinstance(source_file_count, bool)
        or not isinstance(source_file_count, int)
        or source_file_count <= 0
    ):
        raise ValueError("Checkpoint requires an exact embedded source snapshot")

    return {
        "core": (
            model.decoder_model_name,
            decoder_revision,
            mads_model,
            int(model.doc_max_length),
            int(getattr(config, "lora_r", -1)),
            int(getattr(config, "lora_r_compressor", -1)),
            bool(getattr(config, "sep", False)),
            bool(getattr(config, "different_mem_tokens", False)),
            bool(getattr(config, "optimize_mem_tokens", False)),
            getattr(config, "compr_model_name", None),
            getattr(config, "compr_n_layers", None),
            bool(getattr(config, "compr_use_mlp", False)),
            getattr(config, "compr_linear_type", None),
            bool(getattr(config, "compr_rms_norm", False)),
            getattr(config, "training_form", None),
            compression_rate,
            phase1_manifest,
            phase1_test_digest,
            manifest_digest,
            tuple(int(value) for value in epoch_schedule),
            test_url_digest,
            training_index_digest,
            training_candidate_digest,
            recorded_corpus_sha256,
            recorded_corpus_count,
            recorded_corpus_scheme,
            recorded_corpus_scope,
            source_scheme,
            source_tree,
            source_commit,
            source_dirty,
            source_file_count,
        ),
        "training_retrieval": {
            "corpus_sha256": recorded_corpus_sha256,
            "corpus_count": recorded_corpus_count,
            "corpus_sha256_scheme": recorded_corpus_scheme,
            "corpus_scope": recorded_corpus_scope,
            "index_sha256": training_index_digest,
            "candidate_order_sha256": training_candidate_digest,
        },
        "w_bge": w_bge_fingerprint,
        "source": {
            "scheme": source_scheme,
            "tree_sha256": source_tree,
            "git_commit": source_commit,
            "git_dirty": source_dirty,
            "file_count": source_file_count,
        },
    }


def _assert_protocol_fingerprints_match(
    fingerprints: Sequence[Mapping[str, Any]],
) -> None:
    """Require matched data/backbone settings across a reported experiment."""
    if not fingerprints:
        raise ValueError("No checkpoint protocol fingerprints were collected")
    core = fingerprints[0]["core"]
    if any(fingerprint.get("core") != core for fingerprint in fingerprints[1:]):
        raise ValueError(
            "Reported checkpoints do not share the same data/backbone protocol"
        )
    w_bge = [fingerprint.get("w_bge") for fingerprint in fingerprints]
    concrete_w_bge = [value for value in w_bge if value is not None]
    if concrete_w_bge and any(
        value != concrete_w_bge[0] for value in concrete_w_bge[1:]
    ):
        raise ValueError("Reported ARIA checkpoints do not share one frozen W_BGE")


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


def _cross_benchmark_average(
    dataset_results: Dict[str, Dict[str, Any]],
    seed_checkpoints: List[Tuple[Optional[int], str]],
) -> Dict[str, Any]:
    """Compute the paper's unweighted benchmark average for each seed first."""
    metric_names = ["em", "cem", "f1"]
    for recall_metric in ("recall_at_1", "recall_at_3", "recall_at_5"):
        if dataset_results and all(
            recall_metric in run
            for result in dataset_results.values()
            for run in result.get("per_seed", [])
        ):
            metric_names.append(recall_metric)
    per_checkpoint: List[Dict[str, Any]] = []
    dataset_names = list(dataset_results)
    for checkpoint_index, _ in enumerate(seed_checkpoints):
        per_checkpoint.append(
            {
                metric: float(
                    np.mean(
                        [
                            dataset_results[name]["per_seed"][checkpoint_index][metric]
                            for name in dataset_names
                        ]
                    )
                )
                for metric in metric_names
            }
        )
    return aggregate_checkpoint_results(
        per_checkpoint,
        [seed for seed, _ in seed_checkpoints],
        [path for _, path in seed_checkpoints],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIA End-to-End Evaluation")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--model_path", type=str, help="One checkpoint (not a multi-seed experiment)"
    )
    checkpoint_group.add_argument(
        "--model_paths",
        "--seed_checkpoints",
        dest="model_paths",
        type=str,
        nargs="+",
        help="Distinct independently trained checkpoint paths, aligned with --seeds",
    )
    checkpoint_group.add_argument(
        "--model_path_template",
        type=str,
        help="Checkpoint template containing {seed}; {cr} is also supported",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Training seed identities corresponding one-to-one with checkpoints",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["nq", "hotpotqa", "musique", "2wikimultihopqa", "all"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--retrieval_mode",
        choices=["normal", "oracle"],
        default="normal",
        help=(
            "Normal full-corpus retrieval, or deterministic Oracle reranking over "
            "a BGE top-100 pool with missing positives injected at its tail."
        ),
    )
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help=(
            "Alias-complete DatasetDict created by `aria-data --stage eval`; "
            "external scalar-answer ZIPs are CLaRa candidate artifacts only"
        ),
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        help="Local full-KILT Dataset/JSON artifact, or explicit hf:dataset-name",
    )
    parser.add_argument(
        "--clara_archive_dir",
        type=str,
        default=None,
        help=(
            "External directory containing the four pinned CLaRa candidate ZIPs "
            "(nq.zip, hotpotqa.zip, musique.zip, 2wiki.zip). Required only for "
            "--rag_configuration clara_baseline; archives are not bundled."
        ),
    )
    parser.add_argument(
        "--decoder_model",
        type=str,
        default=None,
        help="Optional decoder base-model override (defaults to checkpoint config)",
    )
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_new_tokens", type=int, default=PAPER_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--inference_seed",
        type=int,
        default=0,
        help="Shared generation RNG seed; this is not a training-seed replicate",
    )
    parser.add_argument(
        "--doc_embeddings",
        type=str,
        default=None,
        help=(
            "Dense corpus embedding artifact (.pt/.pth/.npz) containing the matrix, "
            "document_ids, and text_sha256; required for full RAG. {dataset} works."
        ),
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        default=None,
        help="Saved evaluator JSON for paired significance testing",
    )
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--bge_projection_path",
        type=str,
        default=None,
        help=(
            "Optional explicit W_BGE artifact override. Supports {seed}, {dataset}, "
            "and {cr}; otherwise the projection must be bundled in each checkpoint."
        ),
    )
    parser.add_argument("--no_rag_pipeline", action="store_true")
    parser.add_argument(
        "--rag_configuration",
        choices=sorted(RAG_CONFIGURATION_SPECS),
        default=None,
        help=(
            "Explicit training/runtime protocol. Use remove_all_coupling for "
            "the independently retrained 108-token/static-D2 control and "
            "forward_path_off for the full-checkpoint 184-token/no-D2 intervention."
        ),
    )
    parser.add_argument("--no_cfrs", action="store_true")
    parser.add_argument("--no_acr", action="store_true")
    parser.add_argument("--no_mtfrl", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.compression_rate not in PAPER_COMPRESSION_RATES:
        parser.error("--compression_rate must be one of 4, 16, 32, 64, 128")
    if args.max_samples is not None and args.max_samples < 0:
        parser.error("--max_samples must be non-negative")
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("--batch_size and --max_new_tokens must be positive")
    if args.max_new_tokens != PAPER_MAX_NEW_TOKENS:
        parser.error("Paper-protocol evaluation requires --max_new_tokens=64")
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap_resamples must be positive")
    if args.no_rag_pipeline:
        parser.error(
            "End-to-end ARIA evaluation uses the retrieval pipeline. Use QAMetrics "
            "on existing predictions for metrics-only evaluation."
        )
    disabled_couplings = (args.no_cfrs, args.no_acr, args.no_mtfrl)
    if args.rag_configuration is not None and any(disabled_couplings):
        parser.error(
            "--rag_configuration cannot be combined with legacy --no_* switches"
        )
    expected_configuration_by_switches = {
        (False, False, False): "full",
        (True, False, False): "fixed_remove_cfrs",
        (False, True, False): "fixed_uniform_acr",
        (False, False, True): "fixed_remove_mtfrl",
        (True, True, True): "forward_path_off",
    }
    expected_configuration = args.rag_configuration or (
        expected_configuration_by_switches.get(disabled_couplings)
    )
    if expected_configuration is None:
        parser.error(
            "Select an explicit --rag_configuration, or one supported legacy "
            "fixed-checkpoint switch combination"
        )
    is_clara_baseline = expected_configuration == "clara_baseline"
    if is_clara_baseline:
        if args.clara_archive_dir is None:
            parser.error(
                "--rag_configuration clara_baseline requires --clara_archive_dir"
            )
        if args.retrieval_mode != "normal":
            parser.error("Matched CLaRa supports only --retrieval_mode normal")
    elif args.clara_archive_dir is not None:
        parser.error("--clara_archive_dir is valid only for matched CLaRa evaluation")
    if not is_clara_baseline and (
        args.corpus_path is None or args.doc_embeddings is None
    ):
        parser.error(
            "Full-ARIA evaluation requires --corpus_path and --doc_embeddings"
        )

    seed_checkpoints = _resolve_seed_checkpoints(parser, args)
    first_checkpoint_config = CLaRaConfig.from_pretrained(seed_checkpoints[0][1])
    training_index_sha256 = getattr(
        first_checkpoint_config, "aria_training_retrieval_index_sha256", None
    )
    if not isinstance(training_index_sha256, str) or len(training_index_sha256) != 64:
        parser.error("checkpoint requires its Phase-II training BGE-index fingerprint")
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_to_eval = (
        ["nq", "hotpotqa", "musique", "2wikimultihopqa"]
        if args.dataset == "all"
        else [args.dataset]
    )
    all_results: Dict[str, Dict[str, Any]] = {}
    evaluation_retrieval_provenance: Dict[str, Dict[str, Any]] = {}

    for dataset_name in datasets_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'=' * 60}")

        dataset, question_key, answer_key = load_eval_dataset(
            dataset_name,
            args.max_samples,
            args.eval_data_path,
            require_clara_archive=is_clara_baseline,
            clara_archive_dir=args.clara_archive_dir,
        )
        questions = [item[question_key] for item in dataset]
        gold_answers = [_extract_gold_answers(item, answer_key) for item in dataset]
        gold_document_ids = _extract_gold_document_ids(dataset)
        if args.retrieval_mode == "oracle" and gold_document_ids is None:
            raise ValueError(
                "--retrieval_mode oracle requires prepared evaluation rows with "
                "corpus-level gold_doc_ids"
            )
        example_ids = _extract_example_ids(dataset, dataset_name)
        clara_documents: Optional[List[List[str]]] = None
        clara_candidate_doc_ids: Optional[List[List[str]]] = None
        clara_candidate_page_ids: Optional[List[List[str]]] = None
        clara_gold_candidate_indices: Optional[List[List[int]]] = None
        if is_clara_baseline:
            (
                clara_documents,
                clara_candidate_doc_ids,
                clara_candidate_page_ids,
                clara_gold_candidate_indices,
            ) = _extract_clara_candidate_columns(dataset)
        print(f"Loaded {len(questions)} examples")

        corpus_docs: List[str] = []
        corpus_ids: List[str] = []
        corpus_urls: List[str] = []
        corpus_digest: Optional[str] = None
        doc_embeddings: Optional[torch.Tensor] = None
        bm25_index: Optional[_BM25Index] = None
        if not is_clara_baseline:
            try:
                corpus = load_corpus(args.corpus_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Full RAG requires a loadable KILT corpus ({dataset_name})"
                ) from exc
            corpus_docs = [_corpus_text(item) for item in corpus]
            corpus_ids = [_corpus_id(item, index) for index, item in enumerate(corpus)]
            corpus_hashes = [_text_sha256(text) for text in corpus_docs]
            corpus_urls = [
                _corpus_page_url(item, index) for index, item in enumerate(corpus)
            ]
            corpus_digest = _corpus_sha256(corpus_ids, corpus_hashes, corpus_urls)
            if len(corpus_ids) != len(set(corpus_ids)):
                raise ValueError("Corpus document IDs must be unique")
            embeddings_path = _format_artifact_path(
                args.doc_embeddings,
                dataset=dataset_name,
                compression_rate=args.compression_rate,
            )
            doc_embeddings, evaluation_index_sha256 = load_doc_embeddings(
                embeddings_path,
                len(corpus_docs),
                expected_ids=corpus_ids,
                expected_hashes=corpus_hashes,
                expected_page_ids=corpus_urls,
                return_index_sha256=True,
            )
            evaluation_retrieval_provenance[dataset_name] = {
                "retrieval_mode": args.retrieval_mode,
                "corpus_role": (
                    "oracle_bge_top100_source_full_kilt"
                    if args.retrieval_mode == "oracle"
                    else "normal_evaluation_full_kilt"
                ),
                "corpus_sha256": corpus_digest,
                "corpus_count": len(corpus_ids),
                "corpus_unique_page_count": len(set(corpus_urls)),
                "corpus_sha256_scheme": CORPUS_SHA256_SCHEME,
                "page_id_scheme": "canonical-page-url-v1",
                "index_sha256": evaluation_index_sha256,
                "bge_model": "BAAI/bge-large-en-v1.5",
                "text_sha256_scheme": TEXT_SHA256_SCHEME,
                "mads_semantic_source": "shared_bge_document_embeddings",
            }
            if args.retrieval_mode == "normal":
                _assert_normal_retrieval_is_not_training_index(
                    first_checkpoint_config,
                    evaluation_corpus_sha256=corpus_digest,
                    evaluation_index_sha256=evaluation_index_sha256,
                )
            # BM25 is immutable after build, so all independently trained
            # checkpoints for this benchmark can safely share one full index.
            bm25_index = _BM25Index().build(corpus_docs)

        rag_config = create_paper_rag_config(
            expected_configuration, args.compression_rate
        )

        checkpoint_results: List[Dict[str, Any]] = []
        checkpoint_times: List[float] = []
        protocol_fingerprints: List[Dict[str, Any]] = []
        for training_seed, checkpoint_path in seed_checkpoints:
            seed_label = f"seed {training_seed}" if training_seed is not None else "single"
            print(f"\nLoading {seed_label} checkpoint: {checkpoint_path}")
            _set_inference_seed(args.inference_seed)
            model_overrides: Dict[str, Any] = {
                "pure_inference": True,
            }
            model = CLaRa.from_pretrained(
                checkpoint_path,
                strict_aria_artifacts=True,
                external_bge_artifact=args.bge_projection_path is not None,
                **model_overrides,
            )
            if (
                args.decoder_model is not None
                and args.decoder_model != model.decoder_model_name
            ):
                raise ValueError(
                    "--decoder_model must match the backbone recorded by the checkpoint"
                )

            if not is_clara_baseline:
                projection_path: Optional[str] = None
                if args.bge_projection_path is not None:
                    projection_path = _format_artifact_path(
                        args.bge_projection_path,
                        seed=training_seed,
                        dataset=dataset_name,
                        compression_rate=args.compression_rate,
                    )
                else:
                    bundled_projection = os.path.join(
                        os.path.expanduser(checkpoint_path), "bge_projection.pth"
                    )
                    if os.path.isfile(bundled_projection):
                        projection_path = bundled_projection

                # Load every local projection through the strict artifact validator.
                if projection_path is not None:
                    load_bge_projection(
                        model,
                        projection_path,
                        expected_output_dim=int(doc_embeddings.shape[1]),
                    )

            protocol_fingerprints.append(
                _validate_checkpoint_protocol(
                    model,
                    checkpoint_path,
                    training_seed,
                    args.compression_rate,
                    expected_configuration,
                )
            )

            model = model.to(args.device)
            model.eval()
            evaluator = ARIAEvaluator(
                model=model,
                corpus_docs=corpus_docs,
                corpus_ids=corpus_ids,
                corpus_page_ids=corpus_urls,
                doc_embeddings=doc_embeddings,
                use_rag_pipeline=not is_clara_baseline,
                rag_config=rag_config,
                bm25_index=bm25_index,
                retrieval_mode=args.retrieval_mode,
            )

            start_time = time.time()
            checkpoint_result = evaluator.evaluate(
                questions=questions,
                gold_answers=gold_answers,
                example_ids=example_ids,
                gold_doc_ids=(None if is_clara_baseline else gold_document_ids),
                documents=clara_documents,
                clara_candidate_doc_ids=clara_candidate_doc_ids,
                clara_candidate_page_ids=clara_candidate_page_ids,
                clara_gold_candidate_indices=clara_gold_candidate_indices,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            checkpoint_times.append(time.time() - start_time)
            checkpoint_results.append(checkpoint_result)

            del evaluator
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _assert_protocol_fingerprints_match(protocol_fingerprints)

        results = aggregate_checkpoint_results(
            checkpoint_results,
            [seed for seed, _ in seed_checkpoints],
            [path for _, path in seed_checkpoints],
        )
        results["elapsed_seconds_per_checkpoint"] = checkpoint_times
        all_results[dataset_name] = results

        print(
            f"\n{dataset_name} Results "
            f"({len(seed_checkpoints)} independent checkpoint(s), "
            f"{sum(checkpoint_times):.1f}s):"
        )
        reported_metrics = ["em", "cem", "f1"]
        reported_metrics.extend(
            metric
            for metric in ("recall_at_1", "recall_at_3", "recall_at_5")
            if metric in results["mean"]
        )
        for metric in reported_metrics:
            print(
                f"  Avg {metric.upper():<11}: {results['mean'][metric] * 100:.2f}% "
                f"± {results['std'][metric] * 100:.2f}%"
            )

    if args.dataset == "all" and len(all_results) > 1:
        average_result = _cross_benchmark_average(all_results, seed_checkpoints)
        all_results["avg"] = average_result
        print(f"\n{'=' * 60}")
        print("Cross-Benchmark Average")
        print(f"{'=' * 60}")
        for metric in average_result["mean"]:
            print(
                f"  Avg {metric.upper():<11}: {average_result['mean'][metric] * 100:.2f}% "
                f"± {average_result['std'][metric] * 100:.2f}%"
            )

    result_prefix = "aria" if expected_configuration == "full" else expected_configuration
    output_path = os.path.join(
        args.output_dir, f"{result_prefix}_{args.dataset}_cr{args.compression_rate}.json"
    )
    significance = None
    if args.baseline_results is not None:
        with open(os.path.expanduser(args.baseline_results), "r", encoding="utf-8") as handle:
            baseline_payload = json.load(handle)
        baseline_results = baseline_payload.get("results", baseline_payload)
        if not isinstance(baseline_results, Mapping):
            raise ValueError("--baseline_results must contain an evaluator result object")
        significance = compare_evaluation_payloads(
            all_results,
            baseline_results,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )

    payload = {
        "metadata": {
            "dataset": args.dataset,
            "retrieval_mode": args.retrieval_mode,
            "compression_rate": args.compression_rate,
            "training_seeds": [seed for seed, _ in seed_checkpoints],
            "checkpoints": [path for _, path in seed_checkpoints],
            "inference_seed": args.inference_seed,
            "normalization": "ARIA Appendix A.35",
            "answer_alias_contract": EVALUATION_ANSWER_ALIAS_CONTRACT,
            "rag_configuration": expected_configuration,
            "checkpoint_rag_configuration": _required_checkpoint_configuration(
                expected_configuration
            ),
            "coupling_control_protocol": COUPLING_CONTROL_PROTOCOL,
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
            "release_convention_inferred": (
                rag_config.acr_allocation_mode == "uniform_budget"
                or rag_config.second_retrieval_mode == "static_query"
            ),
            "evaluation_retrieval_provenance": evaluation_retrieval_provenance,
            "clara_archive_sha256": (
                dict(_REPOSITORY_EVAL_ARCHIVE_SHA256)
                if is_clara_baseline
                else None
            ),
            "oracle_protocol": (
                {
                    "name": ORACLE_TOP100_PROTOCOL,
                    "pool_size": 100,
                    "base_order": "BGE score descending, corpus index ascending on ties",
                    "page_deduplication": "retain first ranked occurrence of each canonical page URL",
                    "positive_insertion": "missing gold pages at tail in annotation order",
                    "eviction": "lowest-ranked non-gold pages first",
                    "candidate_acquisition_scope": "replaces AHR and IGFR",
                    "first_ranking_stage": "MADS then CCEF",
                    "mtfrl_scope": "same fixed top-100 pool",
                    "reported_recall_scope": "final page-deduplicated CFRS order at k=1,3,5",
                }
                if args.retrieval_mode == "oracle"
                else None
            ),
            "paper_seed_protocol": (
                {seed for seed, _ in seed_checkpoints if seed is not None}
                == PAPER_TRAINING_SEEDS
                and len(seed_checkpoints) == 5
            ),
        },
        "results": all_results,
        "significance_vs_baseline": significance,
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(
            _json_safe(payload),
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
