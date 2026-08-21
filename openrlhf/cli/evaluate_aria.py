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
import math
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

from openrlhf.cli.aria_data import PAPER_PHASE2_EPOCH_SEEDS
from openrlhf.models.modeling_aria import (
    ARIA_NO_COMPRESSION_CONFIGURATION,
    ARIA_NO_COMPRESSION_CONTEXT_CEILING,
    ARIA_NO_COMPRESSION_CONTEXT_POLICY,
    ARIA_NO_COMPRESSION_SCHEME,
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
    CFRS_RECONSTRUCTION_SCHEME,
    MATCHED_EVIDENCE_TOKEN_BUDGET,
    MTFRL_INITIALIZATION_SCHEME,
    ORACLE_QCA_PROTOCOL,
    ORACLE_TOP100_PROTOCOL,
    OraclePoolRecord,
    QuestionType,
    QR_INPUT_SCHEME,
    RAG_CONFIGURATION_SPECS,
    RAGPipelineConfig,
    RETRIEVAL_STRAIGHT_THROUGH_SCHEME,
    STATIC_SECOND_QUERY_SCHEME,
    UNIFORM_BUDGET_ALLOCATION_SCHEME,
    _BM25Index,
    _SemanticAgent,
    _base_decoder_only,
    _chunked_inner_product_topk_unique_pages,
    _construct_oracle_top100_indices,
    create_paper_rag_config,
    _page_deduplicate_ranked_indices,
    required_checkpoint_configuration,
    _tensor_is_finite_in_chunks,
)

from openrlhf.utils.aria_provenance import (
    CORPUS_SHA256_SCHEME,
    EVALUATION_ANSWER_CONTRACT,
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

PAPER_QCA_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QCA_LLM_PROTOCOL = "zero-shot-base-mistral-label-only-v1"
QCA_LLM_PROMPT_VERSION = "qca-three-label-zero-shot-v1"
QCA_LLM_MAX_NEW_TOKENS = 64
QCA_LLM_PROMPT_TEMPLATE = """[INST] Classify the question into exactly one query-complexity label.

Labels:
- simple: a direct question that can be answered without combining multiple facts or comparing multiple aspects.
- multi-aspect: a question asking for multiple attributes, criteria, entities, or parallel aspects, without requiring a reasoning chain across facts.
- multi-hop: a question that requires connecting two or more facts in sequence to reach the answer.

Do not answer the question. Return exactly a first line in the form
Label: <simple|multi-aspect|multi-hop>
followed by a second line in the form
Rationale: <brief explanation>

Question: {question}
[/INST]"""
QCA_LLM_PROMPT_SHA256 = hashlib.sha256(
    QCA_LLM_PROMPT_TEMPLATE.encode("utf-8")
).hexdigest()

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
ORACLE_QCA_PANEL_COUNTS: Mapping[str, int] = {
    "nq": 225,
    "hotpotqa": 257,
    "musique": 84,
    "2wikimultihopqa": 434,
}
PAPER_MAX_NEW_TOKENS = 64
PAPER_BGE_MODEL = "BAAI/bge-large-en-v1.5"
ORACLE_QUERY_EMBEDDING_PROTOCOL = "frozen-bge-direct-query-cls-l2-v1"
# Backward-compatible public spelling retained for downstream callers.
EVALUATION_GOLD_CONTRACT = EVALUATION_GOLD_DOCUMENT_CONTRACT
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

    The archives contain ranked BGE top-20 candidate lists. Full ARIA ignores
    them and performs full-corpus retrieval;
    matched CLaRa applies its learned hard-forward/soft-backward top-5 selector
    to all retained candidates after the prepared split is exactly joined.
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


def _validate_paper_answer_contract(
    dataset: Any,
    *,
    dataset_name: str,
    answer_key: str,
) -> None:
    """Require the old-paper scalar reference answer on every prepared row."""
    for index, item in enumerate(dataset):
        answer = item.get(answer_key)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"Paper-protocol {dataset_name} row {index} requires an explicit "
                f"non-empty scalar {answer_key!r} reference answer"
            )


def _merge_repository_candidates(
    dataset: Any,
    repository_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    question_key: str,
) -> List[Dict[str, Any]]:
    """Attach fingerprinted CLaRa candidates to the prepared evaluation split.

    The two artifacts are joined only by exact official row order and exact
    question text.  No fuzzy matching or inferred alias mapping is permitted.
    """
    if len(dataset) != len(repository_rows):
        raise ValueError(
            f"Prepared {dataset_name} split and external candidate archive must "
            f"have equal counts, got {len(dataset)} and {len(repository_rows)}"
        )
    candidate_fields = ("docs",)
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
) -> List[List[str]]:
    """Materialize the validated external CLaRa candidate text columns."""
    documents_by_row: List[List[str]] = []
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
        documents_by_row.append(documents)
    return documents_by_row


def _map_clara_candidates_to_corpus(
    documents_by_row: Sequence[Sequence[str]],
    *,
    corpus_docs: Sequence[str],
    corpus_ids: Sequence[str],
    corpus_page_ids: Sequence[str],
) -> Tuple[List[List[str]], List[List[str]], List[List[int]]]:
    """Map archived candidate text to unique full-corpus document/page identities.

    Exact outer-stripped text is the join key. Missing or ambiguous matches fail
    closed because archive-local positions cannot define corpus-level Recall@5.
    """
    if not (
        len(corpus_docs) == len(corpus_ids) == len(corpus_page_ids)
        and corpus_docs
    ):
        raise ValueError("CLaRa candidate mapping requires an aligned non-empty corpus")
    indices_by_text_hash: Dict[str, List[int]] = defaultdict(list)
    for corpus_index, document in enumerate(corpus_docs):
        indices_by_text_hash[_text_sha256(document)].append(corpus_index)

    mapped_doc_ids: List[List[str]] = []
    mapped_page_ids: List[List[str]] = []
    mapped_indices: List[List[int]] = []
    for row_index, documents in enumerate(documents_by_row):
        row_indices: List[int] = []
        for candidate_index, document in enumerate(documents):
            candidates = indices_by_text_hash.get(_text_sha256(document), [])
            exact = [
                index
                for index in candidates
                if corpus_docs[index].strip() == document.strip()
            ]
            if len(exact) != 1:
                reason = "missing" if not exact else "ambiguous"
                raise ValueError(
                    "CLaRa archive candidate cannot be mapped uniquely to the full "
                    f"corpus ({reason}) at row {row_index}, candidate {candidate_index}"
                )
            row_indices.append(exact[0])
        mapped_indices.append(row_indices)
        mapped_doc_ids.append([corpus_ids[index] for index in row_indices])
        mapped_page_ids.append([corpus_page_ids[index] for index in row_indices])
    return mapped_doc_ids, mapped_page_ids, mapped_indices


def load_eval_dataset(
    dataset_name: str,
    max_samples: Optional[int] = None,
    eval_data_path: Optional[str] = None,
    require_clara_archive: bool = False,
    clara_archive_dir: Optional[str] = None,
):
    """Load a scalar-answer paper split and optional external CLaRa candidates."""
    cfg = DATASET_CONFIGS[dataset_name]
    dataset = None
    if eval_data_path is None:
        raise ValueError(
            "Paper-protocol answer metrics require --eval_data_path with one "
            "explicit scalar benchmark answer per row."
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
    if manifest.get("answer_contract") != EVALUATION_ANSWER_CONTRACT:
        raise ValueError(
            "--eval_data_path must be regenerated with the scalar-answer paper contract"
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
    _validate_paper_answer_contract(
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


def _load_oracle_qca_labels(path: str) -> Dict[str, str]:
    """Load explicit example-ID-to-QCA-label annotations from JSON or JSONL."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Oracle-QCA label artifact does not exist: {resolved}")

    def unique_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Oracle-QCA JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    records: List[Tuple[Any, Any]] = []
    if resolved.suffix.casefold() == ".jsonl":
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=unique_object)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid Oracle-QCA JSONL row {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"Oracle-QCA JSONL row {line_number} must be an object"
                    )
                if "example_id" not in row or "question_type" not in row:
                    raise ValueError(
                        "Oracle-QCA JSONL rows require example_id and question_type"
                    )
                records.append((row["example_id"], row["question_type"]))
    else:
        with resolved.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle, object_pairs_hook=unique_object)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid Oracle-QCA JSON artifact: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(
                "Oracle-QCA JSON must map example_id directly to question_type"
            )
        records = list(payload.items())

    labels: Dict[str, str] = {}
    for raw_example_id, raw_type in records:
        if not isinstance(raw_example_id, str) or not raw_example_id.strip():
            raise ValueError("Oracle-QCA example_id values must be non-empty strings")
        example_id = raw_example_id.strip()
        if example_id in labels:
            raise ValueError(f"Duplicate Oracle-QCA example_id {example_id!r}")
        try:
            question_type = QuestionType(raw_type).value
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(value.value for value in QuestionType)
            raise ValueError(
                f"Oracle-QCA label for {example_id!r} must be one of {allowed}; "
                f"got {raw_type!r}"
            ) from exc
        labels[example_id] = question_type
    if not labels:
        raise ValueError("Oracle-QCA label artifact must contain at least one label")
    return labels


def _oracle_qca_labeled_subset(
    example_ids: Sequence[Hashable],
    labels: Mapping[str, str],
    *,
    dataset_name: str,
) -> Tuple[List[int], List[str]]:
    """Return the dataset rows with explicit labels, retaining dataset order."""
    indices: List[int] = []
    reference_types: List[str] = []
    for index, raw_example_id in enumerate(example_ids):
        example_id = str(raw_example_id)
        if example_id in labels:
            indices.append(index)
            reference_types.append(labels[example_id])
    if not indices:
        raise ValueError(
            f"Oracle-QCA labels do not match any {dataset_name} evaluation example_id"
        )
    return indices, reference_types


def _validate_oracle_qca_paper_panel(
    dataset_name: str,
    matched_example_ids: Sequence[Hashable],
) -> None:
    expected_count = ORACLE_QCA_PANEL_COUNTS[dataset_name]
    if len(matched_example_ids) != expected_count:
        raise ValueError(
            f"Oracle-QCA requires exactly {expected_count} labeled {dataset_name} "
            f"examples; matched {len(matched_example_ids)}"
        )


def _validate_oracle_qca_conditions(
    *,
    retrieval_mode: str,
    rag_configuration: str,
    compression_rate: int,
    max_samples: Optional[int] = None,
    dataset: Optional[str] = None,
) -> None:
    """Fail closed outside the paper's Normal/full/16x Oracle-QCA endpoint."""
    if retrieval_mode != "normal":
        raise ValueError("Oracle-QCA requires --retrieval_mode normal")
    if rag_configuration != "full":
        raise ValueError("Oracle-QCA requires --rag_configuration full")
    if compression_rate != 16:
        raise ValueError("Oracle-QCA requires --compression_rate 16")
    if max_samples is not None:
        raise ValueError("Oracle-QCA cannot be combined with --max_samples")
    if dataset is not None and dataset != "all":
        raise ValueError("Oracle-QCA paper endpoint requires --dataset all")


def _validate_qca_llm_conditions(
    *,
    retrieval_mode: str,
    rag_configuration: str,
    compression_rate: int,
    dataset: str,
    max_samples: Optional[int],
) -> None:
    """Fail closed outside the paper's Mistral QCA sensitivity setting."""
    if retrieval_mode != "normal":
        raise ValueError("QCA-LLM requires --retrieval_mode normal")
    if rag_configuration != "full":
        raise ValueError("QCA-LLM requires --rag_configuration full")
    if compression_rate != 16:
        raise ValueError("QCA-LLM requires --compression_rate 16")
    if dataset != "all":
        raise ValueError("QCA-LLM paper endpoints require --dataset all")
    if max_samples is not None:
        raise ValueError("QCA-LLM cannot be combined with --max_samples")


def _qca_llm_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QCA-LLM requires a non-empty question")
    return QCA_LLM_PROMPT_TEMPLATE.format(question=question.strip())


def _parse_qca_llm_output(raw_output: str) -> Tuple[str, str]:
    """Parse the strict two-line zero-shot router response."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("QCA-LLM returned an empty response")
    lines = raw_output.strip().splitlines()
    match = re.fullmatch(
        r"Label:\s*(simple|multi[-_]aspect|multi[-_]hop)\s*",
        lines[0],
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            "QCA-LLM first line must be exactly 'Label: <class>'"
        )
    if len(lines) < 2 or not lines[1].startswith("Rationale:"):
        raise ValueError("QCA-LLM second line must start with 'Rationale:'")
    rationale = "\n".join(
        [lines[1][len("Rationale:") :].strip(), *lines[2:]]
    ).strip()
    if not rationale:
        raise ValueError("QCA-LLM rationale must be non-empty")
    parsed = match.group(1).casefold().replace("-", "_")
    return QuestionType(parsed).value, rationale


def _run_qca_llm_router(
    model: CLaRa,
    questions: Sequence[str],
    *,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Run the zero-shot base decoder once and return auditable route records."""
    if model.decoder_model_name != PAPER_QCA_LLM_MODEL:
        raise ValueError(
            f"QCA-LLM requires base model {PAPER_QCA_LLM_MODEL!r}"
        )
    revision = getattr(
        model.config, "decoder_model_resolved_revision", None
    )
    if not isinstance(revision, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}", revision
    ) is None:
        raise ValueError("QCA-LLM requires an exact resolved base-model revision")
    if batch_size <= 0:
        raise ValueError("QCA-LLM batch_size must be positive")
    prompts = [_qca_llm_prompt(question) for question in questions]
    records: List[Dict[str, Any]] = []
    decoder = model.decoder
    tokenizer = model.decoder_tokenizer
    parameter = next(decoder.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    context_limit = int(
        getattr(getattr(decoder, "config", None), "max_position_embeddings", 32_768)
    )
    if context_limit <= QCA_LLM_MAX_NEW_TOKENS:
        raise ValueError("QCA-LLM decoder context cannot reserve 64 output tokens")

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            truncation=False,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise RuntimeError("QCA-LLM tokenizer omitted input IDs or attention mask")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1)
        if torch.any(prompt_lengths > context_limit - QCA_LLM_MAX_NEW_TOKENS):
            raise ValueError("QCA-LLM prompt exceeds the reserved decoder context")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), _base_decoder_only(decoder):
            output_ids = decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                top_p=None,
                temperature=None,
                max_new_tokens=QCA_LLM_MAX_NEW_TOKENS,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        prompt_width = input_ids.size(1)
        if output_ids.ndim != 2 or output_ids.shape[0] != len(batch_prompts):
            raise RuntimeError("QCA-LLM decoder returned malformed token IDs")
        raw_outputs = tokenizer.batch_decode(
            output_ids[:, prompt_width:], skip_special_tokens=True
        )
        if len(raw_outputs) != len(batch_prompts):
            raise RuntimeError("QCA-LLM decoder output count does not match prompts")
        latency_ms = elapsed_ms / len(batch_prompts)
        for local_index, (prompt, raw_output) in enumerate(
            zip(batch_prompts, raw_outputs)
        ):
            parsed_type, rationale = _parse_qca_llm_output(raw_output)
            records.append(
                {
                    "protocol": QCA_LLM_PROTOCOL,
                    "question": questions[start + local_index],
                    "prompt": prompt,
                    "prompt_version": QCA_LLM_PROMPT_VERSION,
                    "prompt_template_sha256": QCA_LLM_PROMPT_SHA256,
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "raw_output": raw_output,
                    "parsed_type": parsed_type,
                    "rationale": rationale,
                    "latency_ms": latency_ms,
                    "base_model": model.decoder_model_name,
                    "base_model_revision_resolved": revision.lower(),
                    "adapters_disabled": True,
                    "decoding": "greedy-eos-or-64",
                }
            )
    return records


def _qca_weighted_f1(
    reference_types: Sequence[str], predicted_types: Sequence[str]
) -> float:
    if not reference_types or len(reference_types) != len(predicted_types):
        raise ValueError("QCA weighted F1 requires aligned non-empty labels")
    labels = [value.value for value in QuestionType]
    if any(value not in labels for value in (*reference_types, *predicted_types)):
        raise ValueError("QCA weighted F1 received an invalid label")
    total = len(reference_types)
    weighted = 0.0
    for label in labels:
        true_positive = sum(
            reference == label and predicted == label
            for reference, predicted in zip(reference_types, predicted_types)
        )
        false_positive = sum(
            reference != label and predicted == label
            for reference, predicted in zip(reference_types, predicted_types)
        )
        false_negative = sum(
            reference == label and predicted != label
            for reference, predicted in zip(reference_types, predicted_types)
        )
        support = true_positive + false_negative
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
        weighted += support * f1
    return weighted / total


def _extract_gold_answer(item: Mapping[str, Any], answer_key: str) -> str:
    """Read the prepared split's single old-paper reference answer."""
    answer = item.get(answer_key)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Evaluation row requires a non-empty scalar {answer_key!r}")
    return answer


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


def _bge_encoder_spec_from_artifact(artifact: Any) -> Dict[str, Any]:
    """Resolve the immutable encoder used to build a v2 BGE corpus index."""
    if not isinstance(artifact, Mapping) or artifact.get("artifact_format") != (
        "aria-bge-artifact-v2"
    ):
        raise ValueError(
            "Oracle pool construction requires a v2 BGE artifact with verified "
            "encoder provenance"
        )
    if artifact.get("bge_model") != PAPER_BGE_MODEL:
        raise ValueError(f"Oracle requires bge_model={PAPER_BGE_MODEL!r}")
    source = artifact.get("encoder_source")
    kind = artifact.get("encoder_source_kind")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Oracle BGE artifact requires encoder_source")
    if kind == "huggingface-hub":
        revision = artifact.get("encoder_revision_resolved")
        if source != PAPER_BGE_MODEL or not isinstance(revision, str) or re.fullmatch(
            r"[0-9a-fA-F]{40}", revision
        ) is None:
            raise ValueError(
                "Oracle BGE Hub provenance requires the paper model and an exact commit"
            )
        return {
            "source": source,
            "source_kind": kind,
            "revision": revision.lower(),
            "source_sha256": None,
        }
    if kind == "local-directory":
        source_sha256 = artifact.get("encoder_source_sha256")
        if not isinstance(source_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", source_sha256
        ) is None:
            raise ValueError("Oracle local BGE provenance requires a directory SHA-256")
        if not Path(source).expanduser().is_dir():
            raise FileNotFoundError(
                f"Oracle BGE encoder directory is unavailable: {source}"
            )
        return {
            "source": str(Path(source).expanduser().resolve()),
            "source_kind": kind,
            "revision": None,
            "source_sha256": source_sha256,
        }
    raise ValueError("Oracle BGE artifact has an unsupported encoder_source_kind")


def _encode_oracle_queries(
    questions: Sequence[str], encoder_spec: Mapping[str, Any]
) -> torch.Tensor:
    """Encode questions once with the same frozen BGE encoder as the corpus."""
    if not questions or any(not isinstance(value, str) or not value.strip() for value in questions):
        raise ValueError("Oracle BGE query encoding requires non-empty questions")
    agent = _SemanticAgent(
        str(encoder_spec["source"]),
        encoder_spec.get("revision"),
    )
    embeddings = torch.from_numpy(agent._embed(list(questions))).float()
    if embeddings.ndim != 2 or embeddings.shape[1] != 1024:
        raise RuntimeError("Oracle BGE queries must have shape (batch, 1024)")
    if not torch.isfinite(embeddings).all().item():
        raise RuntimeError("Oracle BGE query embeddings contain non-finite values")
    return torch.nn.functional.normalize(embeddings, dim=-1, eps=1e-12)


def _build_shared_oracle_pool_records(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    *,
    corpus_ids: Sequence[str],
    corpus_page_ids: Sequence[str],
    gold_doc_ids: Sequence[Sequence[str]],
) -> List[OraclePoolRecord]:
    """Build the page-unique top-100 pool shared by ARIA and CLaRa."""
    if query_embeddings.ndim != 2 or query_embeddings.shape[0] != len(gold_doc_ids):
        raise ValueError("Oracle query embeddings must align with gold_doc_ids")
    if not (
        doc_embeddings.ndim == 2
        and len(corpus_ids) == len(corpus_page_ids) == doc_embeddings.shape[0]
    ):
        raise ValueError("Oracle corpus identities and embeddings must align")
    if len(set(corpus_page_ids)) < 100:
        raise ValueError("Oracle top-100 requires at least 100 unique corpus pages")
    corpus_id_to_index = {document_id: index for index, document_id in enumerate(corpus_ids)}
    if len(corpus_id_to_index) != len(corpus_ids):
        raise ValueError("Oracle corpus document IDs must be unique")
    normalized_documents = torch.nn.functional.normalize(
        doc_embeddings.detach().float().cpu(), dim=-1, eps=1e-12
    )
    base_rows = _chunked_inner_product_topk_unique_pages(
        torch.nn.functional.normalize(
            query_embeddings.detach().float().cpu(), dim=-1, eps=1e-12
        ),
        normalized_documents,
        corpus_page_ids,
        100,
    )
    records: List[OraclePoolRecord] = []
    for row_index, document_ids in enumerate(gold_doc_ids):
        representative_by_page: Dict[str, int] = {}
        for document_id in document_ids:
            if document_id not in corpus_id_to_index:
                raise ValueError(
                    f"Oracle gold_doc_ids[{row_index}] contains an unknown corpus ID"
                )
            corpus_index = corpus_id_to_index[document_id]
            representative_by_page.setdefault(corpus_page_ids[corpus_index], corpus_index)
        if not representative_by_page:
            raise ValueError(
                f"Oracle gold_doc_ids[{row_index}] must contain at least one support"
            )
        records.append(
            _construct_oracle_top100_indices(
                base_rows[row_index].tolist(),
                list(representative_by_page.values()),
                corpus_page_ids=corpus_page_ids,
            )
        )
    return records


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
    return_encoder_spec: bool = False,
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
    if return_index_sha256 and return_encoder_spec:
        return embeddings, computed_index_sha256, _bge_encoder_spec_from_artifact(artifact)
    if return_index_sha256:
        return embeddings, computed_index_sha256
    if return_encoder_spec:
        return embeddings, _bge_encoder_spec_from_artifact(artifact)
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
        no_compression: bool = False,
    ):
        if retrieval_mode not in {"normal", "oracle"}:
            raise ValueError("retrieval_mode must be 'normal' or 'oracle'")
        if no_compression and retrieval_mode != "normal":
            raise ValueError("ARIA-NoComp supports only Normal retrieval")
        if no_compression and not use_rag_pipeline:
            raise ValueError("ARIA-NoComp requires the full five-stage RAG pipeline")
        if no_compression and corpus_ids is None:
            raise ValueError("ARIA-NoComp requires stable corpus_ids for provenance")
        self.model = model
        self.use_rag_pipeline = use_rag_pipeline
        self.retrieval_mode = retrieval_mode
        self.no_compression = bool(no_compression)
        self.corpus_docs = list(corpus_docs)
        self.doc_embeddings = doc_embeddings
        self.rag_config: Optional[RAGPipelineConfig] = None
        self.no_compression_context_limit = (
            model._resolve_no_compression_context_limit()
            if self.no_compression
            else None
        )
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
        if self.corpus_ids is not None and len(self.corpus_ids) != len(self.corpus_docs):
            raise ValueError("corpus_ids must be aligned one-to-one with corpus_docs")
        if self.corpus_page_ids is not None and len(self.corpus_page_ids) != len(
            self.corpus_docs
        ):
            raise ValueError("corpus_page_ids must be aligned one-to-one with corpus_docs")
        if self.corpus_ids is not None and len(set(self.corpus_ids)) != len(self.corpus_ids):
            raise ValueError("corpus_ids must be unique")
        if self.corpus_page_ids is not None and any(
            not isinstance(page_id, str) or not page_id.strip()
            for page_id in self.corpus_page_ids
        ):
            raise ValueError("corpus_page_ids must contain non-empty strings")
        if not use_rag_pipeline and (
            not self.corpus_docs
            or self.corpus_ids is None
            or not self._has_explicit_page_ids
        ):
            raise ValueError(
                "CLaRa evaluation requires the full corpus and stable document/page IDs"
            )
        if retrieval_mode == "oracle" and (
            not self.corpus_docs
            or self.corpus_ids is None
            or not self._has_explicit_page_ids
            or doc_embeddings is None
        ):
            raise ValueError(
                "Oracle evaluation requires the full corpus, aligned BGE embeddings, "
                "and stable document/page IDs"
            )
        if retrieval_mode == "oracle" and not use_rag_pipeline and (
            doc_embeddings is None
            or doc_embeddings.ndim != 2
            or doc_embeddings.shape != (len(self.corpus_docs), 1024)
            or not _tensor_is_finite_in_chunks(doc_embeddings)
        ):
            raise ValueError(
                "CLaRa Oracle requires finite (corpus_size, 1024) BGE embeddings"
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
            if self.corpus_ids is not None and len(set(self.corpus_ids)) != len(
                self.corpus_ids
            ):
                raise ValueError("corpus_ids must be unique")
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
            self.rag_config = cfg
            if self.no_compression and (
                not all(
                    (cfg.use_qca, cfg.use_ahr, cfg.use_igfr, cfg.use_mads, cfg.use_ccef)
                )
                or cfg.use_cfrs
                or cfg.acr_allocation_mode != "full"
                or cfg.second_retrieval_mode != "disabled"
            ):
                raise ValueError(
                    "ARIA-NoComp requires all five retrieval stages and no "
                    "CFRS/ACR/MTFRL runtime path"
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
        gold_answers: List[str],
        example_ids: Optional[List[Hashable]] = None,
        gold_doc_ids: Optional[List[List[str]]] = None,
        documents: Optional[List[List[str]]] = None,
        clara_candidate_doc_ids: Optional[List[List[str]]] = None,
        clara_candidate_page_ids: Optional[List[List[str]]] = None,
        oracle_pool_records: Optional[Sequence[OraclePoolRecord]] = None,
        qca_reference_types: Optional[List[str]] = None,
        qca_llm_records: Optional[Sequence[Mapping[str, Any]]] = None,
        batch_size: int = 8,
        max_new_tokens: int = PAPER_MAX_NEW_TOKENS,
    ) -> Dict[str, Any]:
        """Evaluate one checkpoint and retain identity-aligned per-example scores."""
        if len(questions) != len(gold_answers):
            raise ValueError("questions and gold_answers must have equal length")
        if any(not isinstance(answer, str) or not answer.strip() for answer in gold_answers):
            raise ValueError(
                "Paper-protocol evaluation requires one non-empty scalar gold answer "
                "per question"
            )
        if qca_reference_types is not None and qca_llm_records is not None:
            raise ValueError("Oracle-QCA and QCA-LLM overrides are mutually exclusive")
        normalized_qca_reference_types: Optional[List[str]] = None
        qca_override_protocol: Optional[str] = None
        validated_qca_llm_records: Optional[List[Mapping[str, Any]]] = None
        raw_override_types: Optional[Sequence[str]] = qca_reference_types
        if qca_llm_records is not None:
            if len(qca_llm_records) != len(questions):
                raise ValueError(
                    "qca_llm_records must be aligned one-to-one with questions"
                )
            validated_qca_llm_records = list(qca_llm_records)
            raw_override_types = []
            for index, (question, record) in enumerate(
                zip(questions, validated_qca_llm_records)
            ):
                if not isinstance(record, Mapping):
                    raise ValueError(f"qca_llm_records[{index}] must be an object")
                expected_prompt = _qca_llm_prompt(question)
                if (
                    record.get("protocol") != QCA_LLM_PROTOCOL
                    or record.get("question") != question
                    or record.get("prompt") != expected_prompt
                    or record.get("prompt_version") != QCA_LLM_PROMPT_VERSION
                    or record.get("prompt_template_sha256")
                    != QCA_LLM_PROMPT_SHA256
                    or record.get("prompt_sha256")
                    != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
                    or record.get("adapters_disabled") is not True
                    or record.get("decoding") != "greedy-eos-or-64"
                    or record.get("base_model") != self.model.decoder_model_name
                    or record.get("base_model_revision_resolved")
                    != str(
                        getattr(
                            self.model.config,
                            "decoder_model_resolved_revision",
                            "",
                        )
                    ).lower()
                ):
                    raise ValueError(
                        f"qca_llm_records[{index}] violates the prompt/protocol contract"
                    )
                parsed_type, rationale = _parse_qca_llm_output(
                    record.get("raw_output")
                )
                if (
                    record.get("parsed_type") != parsed_type
                    or record.get("rationale") != rationale
                ):
                    raise ValueError(
                        f"qca_llm_records[{index}] does not match its raw output"
                    )
                latency = record.get("latency_ms")
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(float(latency))
                    or float(latency) < 0.0
                ):
                    raise ValueError(
                        f"qca_llm_records[{index}].latency_ms must be finite and non-negative"
                    )
                raw_override_types.append(parsed_type)
            qca_override_protocol = QCA_LLM_PROTOCOL
        elif qca_reference_types is not None:
            if len(qca_reference_types) != len(questions):
                raise ValueError(
                    "qca_reference_types must be aligned one-to-one with questions"
                )
            qca_override_protocol = ORACLE_QCA_PROTOCOL

        if raw_override_types is not None:
            model_configuration = getattr(
                getattr(self.model, "config", None),
                "aria_rag_configuration",
                None,
            )
            model_compression_rate = getattr(self.model, "compr_rate", None)
            if (
                isinstance(model_compression_rate, bool)
                or not isinstance(model_compression_rate, int)
            ):
                model_compression_rate = 0
            _validate_oracle_qca_conditions(
                retrieval_mode=self.retrieval_mode,
                rag_configuration=model_configuration,
                compression_rate=model_compression_rate,
            )
            cfg = self.rag_config
            if (
                not self.use_rag_pipeline
                or self.no_compression
                or cfg is None
                or not all(
                    (
                        cfg.use_qca,
                        cfg.use_ahr,
                        cfg.use_igfr,
                        cfg.use_mads,
                        cfg.use_ccef,
                        cfg.use_cfrs,
                        cfg.use_acr,
                        cfg.use_mtfrl,
                    )
                )
                or cfg.acr_allocation_mode != "adaptive"
                or cfg.second_retrieval_mode != "memory_feedback"
                or cfg.compression_rate != 16
            ):
                raise ValueError(
                    "QCA label-only overrides require the complete full-ARIA "
                    "runtime at 16x"
                )
            try:
                normalized_qca_reference_types = [
                    QuestionType(value).value for value in raw_override_types
                ]
            except (TypeError, ValueError) as exc:
                allowed = ", ".join(value.value for value in QuestionType)
                raise ValueError(
                    f"qca_reference_types must contain only {allowed}"
                ) from exc
        if gold_doc_ids is not None:
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
        fixed_oracle_pool_records = (
            list(oracle_pool_records) if oracle_pool_records is not None else None
        )
        if fixed_oracle_pool_records is not None:
            if self.retrieval_mode != "oracle":
                raise ValueError("oracle_pool_records are valid only for Oracle retrieval")
            if len(fixed_oracle_pool_records) != len(questions):
                raise ValueError("oracle_pool_records must align one-to-one with questions")
            for row_index, record in enumerate(fixed_oracle_pool_records):
                if (
                    not isinstance(record, OraclePoolRecord)
                    or record.protocol != ORACLE_TOP100_PROTOCOL
                    or len(record.pool_indices) != 100
                    or len(record.pool_page_ids) != 100
                    or len(set(record.pool_page_ids)) != 100
                    or any(
                        index < 0 or index >= len(self.corpus_docs)
                        for index in record.pool_indices
                    )
                ):
                    raise ValueError(
                        f"oracle_pool_records[{row_index}] violates the shared top-100 contract"
                    )
        if self.use_rag_pipeline and documents is not None:
            raise ValueError(
                "Paper-protocol evaluation must retrieve from the full KILT corpus; "
                "pre-retrieved documents cannot be supplied"
            )
        if not self.use_rag_pipeline and self.retrieval_mode == "oracle":
            if fixed_oracle_pool_records is None:
                raise ValueError(
                    "CLaRa Oracle evaluation requires the shared fixed pool records"
                )
            if any(
                value is not None
                for value in (documents, clara_candidate_doc_ids, clara_candidate_page_ids)
            ):
                raise ValueError(
                    "CLaRa Oracle candidates are materialized only from the shared pool"
                )
            if self.corpus_ids is None or self.corpus_page_ids is None:
                raise RuntimeError("CLaRa Oracle corpus identities are unavailable")
            documents = [
                [self.corpus_docs[index] for index in record.pool_indices]
                for record in fixed_oracle_pool_records
            ]
            clara_candidate_doc_ids = [
                [self.corpus_ids[index] for index in record.pool_indices]
                for record in fixed_oracle_pool_records
            ]
            clara_candidate_page_ids = [
                [self.corpus_page_ids[index] for index in record.pool_indices]
                for record in fixed_oracle_pool_records
            ]
        if documents is not None and len(documents) != len(questions):
            raise ValueError("documents must be aligned one-to-one with questions")
        clara_metadata = (
            clara_candidate_doc_ids,
            clara_candidate_page_ids,
        )
        if any(value is not None for value in clara_metadata) and not all(
            value is not None for value in clara_metadata
        ):
            raise ValueError(
                "CLaRa evaluation requires candidate document IDs and page IDs together"
            )
        clara_recall_enabled = not self.use_rag_pipeline
        clara_retrieval_provenance: Optional[Dict[str, Any]] = None
        if clara_recall_enabled:
            if (
                documents is None
                or clara_candidate_doc_ids is None
                or clara_candidate_page_ids is None
                or gold_doc_ids is None
            ):
                raise ValueError(
                    "CLaRa paper evaluation requires mapped corpus candidates and "
                    "prepared corpus-level gold_doc_ids"
                )
            if self._corpus_id_to_index is None or self.corpus_page_ids is None:
                raise RuntimeError("CLaRa full-corpus identity mapping is unavailable")
            expected_candidate_count = (
                100
                if self.retrieval_mode == "oracle"
                else _REPOSITORY_BGE_CANDIDATE_COUNT
            )
            if not (
                len(clara_candidate_doc_ids)
                == len(clara_candidate_page_ids)
                == len(documents)
                == len(questions)
            ):
                raise ValueError("CLaRa candidate metadata must align with questions")
            for row_index, (row_documents, row_doc_ids, row_page_ids) in enumerate(
                zip(documents, clara_candidate_doc_ids, clara_candidate_page_ids)
            ):
                if not (
                    len(row_documents)
                    == len(row_doc_ids)
                    == len(row_page_ids)
                    == expected_candidate_count
                ):
                    raise ValueError(
                        f"CLaRa row {row_index} must align exactly "
                        f"{expected_candidate_count} candidates"
                    )
                for candidate_index, (document, document_id, page_id) in enumerate(
                    zip(row_documents, row_doc_ids, row_page_ids)
                ):
                    corpus_index = self._corpus_id_to_index.get(document_id)
                    if (
                        corpus_index is None
                        or self.corpus_docs[corpus_index].strip() != document.strip()
                        or self.corpus_page_ids[corpus_index] != page_id
                    ):
                        raise ValueError(
                            "CLaRa candidate lacks a reliable full-corpus identity at "
                            f"row {row_index}, candidate {candidate_index}"
                        )

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
                "protocol": (
                    ORACLE_TOP100_PROTOCOL
                    if self.retrieval_mode == "oracle"
                    else CLARA_EVALUATION_CANDIDATE_PROTOCOL
                ),
                "retrieval_mode": self.retrieval_mode,
                "candidate_source": (
                    "shared-oracle-bge-top100"
                    if self.retrieval_mode == "oracle"
                    else "retained-repository-bge-top20"
                ),
                "candidate_count": expected_candidate_count,
                "hard_selection_count": int(self.model.generation_top_k),
                "candidate_identity_join": (
                    "shared-oracle-corpus-index-v1"
                    if self.retrieval_mode == "oracle"
                    else "exact-corpus-text-unique-v1"
                ),
                "document_id_scope": "full-kilt-corpus",
                "page_id_scheme": "canonical-page-url-v1",
                "page_deduplicated_recall": True,
                "support_scope": "prepared-full-corpus-Q_sup",
                "example_count": len(questions),
                "candidate_document_order_sha256": _ordered_rows_sha256(
                    clara_candidate_doc_ids
                ),
                "candidate_page_order_sha256": _ordered_rows_sha256(
                    clara_candidate_page_ids
                ),
            }
        if example_ids is None:
            example_ids = list(range(len(questions)))
        if len(example_ids) != len(questions):
            raise ValueError("example_ids must be aligned one-to-one with questions")
        oracle_qca_provenance: Optional[Dict[str, Any]] = None
        qca_llm_provenance: Optional[Dict[str, Any]] = None
        if (
            normalized_qca_reference_types is not None
            and qca_override_protocol == ORACLE_QCA_PROTOCOL
        ):
            panel_hasher = hashlib.sha256()
            for example_id, question_type in zip(
                example_ids, normalized_qca_reference_types
            ):
                panel_hasher.update(
                    json.dumps(
                        [str(example_id), question_type],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                panel_hasher.update(b"\n")
            oracle_qca_provenance = {
                "protocol": ORACLE_QCA_PROTOCOL,
                "scope": "evaluation-only-labeled-subset",
                "retrieval_mode": "normal",
                "rag_configuration": "full",
                "compression_rate": 16,
                "labeled_example_count": len(normalized_qca_reference_types),
                "label_counts": dict(Counter(normalized_qca_reference_types)),
                "labeled_panel_sha256": panel_hasher.hexdigest(),
                "preserved_surface_fields": [
                    "confidence",
                    "matched_rules",
                    "hop_count",
                    "sub_questions",
                    "entity_count",
                ],
            }
        if validated_qca_llm_records is not None:
            revisions = {
                str(record.get("base_model_revision_resolved"))
                for record in validated_qca_llm_records
            }
            models = {
                str(record.get("base_model"))
                for record in validated_qca_llm_records
            }
            if models != {PAPER_QCA_LLM_MODEL} or len(revisions) != 1:
                raise ValueError(
                    "QCA-LLM records must share one exact Mistral base revision"
                )
            qca_llm_provenance = {
                "protocol": QCA_LLM_PROTOCOL,
                "endpoint": "full-benchmark-qa-label-only-override",
                "retrieval_mode": "normal",
                "rag_configuration": "full",
                "compression_rate": 16,
                "example_count": len(validated_qca_llm_records),
                "prompt_version": QCA_LLM_PROMPT_VERSION,
                "prompt_template_sha256": QCA_LLM_PROMPT_SHA256,
                "base_model": PAPER_QCA_LLM_MODEL,
                "base_model_revision_resolved": next(iter(revisions)),
                "adapters_disabled": True,
                "decoding": "greedy-eos-or-64",
                "mean_router_latency_ms": float(
                    np.mean(
                        [
                            float(record.get("latency_ms"))
                            for record in validated_qca_llm_records
                        ]
                    )
                ),
                "preserved_surface_fields": [
                    "confidence",
                    "matched_rules",
                    "hop_count",
                    "sub_questions",
                    "entity_count",
                ],
            }
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_new_tokens != PAPER_MAX_NEW_TOKENS:
            raise ValueError(
                "Paper-protocol evaluation requires max_new_tokens=64"
            )
        no_compression_protocol = (
            {
                "name": ARIA_NO_COMPRESSION_SCHEME,
                "checkpoint_training_configuration": "full",
                "retrieval_mode": "normal",
                "retrieval_stages": ["QCA", "AHR", "IGFR", "MADS", "CCEF"],
                "retrieval_rounds": 1,
                "selected_document_count": 5,
                "document_order": "first-pass CCEF order",
                "document_separator": "two-newlines",
                "memory_compression": False,
                "cfrs": False,
                "acr": False,
                "mtfrl": False,
                "context_policy": ARIA_NO_COMPRESSION_CONTEXT_POLICY,
                "protocol_context_ceiling": ARIA_NO_COMPRESSION_CONTEXT_CEILING,
                "effective_context_ceiling": self.no_compression_context_limit,
                "passage_truncation": "evidence-tail-only; system-and-question-preserved",
                "decoding": "greedy-one-beam-eos-or-64",
            }
            if self.no_compression
            else None
        )

        all_metrics: List[Dict[str, float]] = []
        retrieval_cutoffs = (1, 3, 5) if self.retrieval_mode == "oracle" else (5,)
        retrieval_recalls: Dict[int, List[float]] = {
            cutoff: [] for cutoff in retrieval_cutoffs
        }
        evidence_memory_tokens: List[int] = []
        direct_context_document_tokens: List[int] = []
        direct_context_prompt_tokens: List[int] = []
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
                        self.no_compression
                        or (
                            gold_doc_ids is not None
                            and self.retrieval_mode == "normal"
                        )
                    ),
                )
                if self.no_compression:
                    generation_kwargs["no_compression"] = True
                if normalized_qca_reference_types is not None:
                    generation_kwargs["qca_reference_types"] = (
                        normalized_qca_reference_types[
                            start : start + len(batch_questions)
                        ]
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
                    if fixed_oracle_pool_records is not None:
                        generation_kwargs["oracle_pool_records"] = (
                            fixed_oracle_pool_records[
                                start : start + len(batch_questions)
                            ]
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
            if (
                normalized_qca_reference_types is not None
                and batch_diagnostics is None
            ):
                raise RuntimeError(
                    "Oracle-QCA evaluation requires per-example QCA diagnostics"
                )

            batch_retrieved_ids: Optional[List[List[str]]] = None
            batch_retrieved_page_ids: Optional[List[List[str]]] = None
            batch_first_pass_corpus_indices: Optional[List[List[int]]] = None
            if self.no_compression:
                if not isinstance(retrieved_indices, torch.Tensor):
                    raise RuntimeError(
                        "ARIA-NoComp requires tensor-valued first-pass corpus indices"
                    )
                if (
                    retrieved_indices.ndim != 2
                    or retrieved_indices.shape
                    != (len(batch_questions), self.model.generation_top_k)
                    or retrieved_indices.dtype == torch.bool
                    or torch.is_floating_point(retrieved_indices)
                ):
                    raise RuntimeError(
                        "ARIA-NoComp first-pass indices must have integer shape "
                        "(B, top_k)"
                    )
                if self.corpus_ids is None:
                    raise RuntimeError("ARIA-NoComp stable corpus IDs are unavailable")
                batch_first_pass_corpus_indices = []
                for row_index, raw_row in enumerate(
                    retrieved_indices.detach().cpu().tolist()
                ):
                    if -1 in raw_row:
                        first_padding = raw_row.index(-1)
                        if any(value != -1 for value in raw_row[first_padding:]):
                            raise RuntimeError(
                                "ARIA-NoComp padded first-pass indices must use "
                                "trailing -1 values only"
                            )
                        raw_row = raw_row[:first_padding]
                    index_row = [int(value) for value in raw_row]
                    if (
                        not 1 <= len(index_row) <= self.model.generation_top_k
                        or len(index_row) != len(set(index_row))
                        or any(
                            value < 0 or value >= len(self.corpus_ids)
                            for value in index_row
                        )
                    ):
                        raise RuntimeError(
                            "ARIA-NoComp returned invalid first-pass corpus indices "
                            f"at batch row {row_index}"
                        )
                    batch_first_pass_corpus_indices.append(index_row)
            batch_oracle_pool_records: Optional[List[Any]] = None
            if self.retrieval_mode == "oracle":
                if clara_recall_enabled:
                    if fixed_oracle_pool_records is None:
                        raise RuntimeError("CLaRa Oracle pool records are unavailable")
                    batch_oracle_pool_records = fixed_oracle_pool_records[
                        start : start + len(batch_questions)
                    ]
                else:
                    getter = getattr(self.model, "get_oracle_pool_records", None)
                    if getter is None:
                        raise RuntimeError(
                            "Oracle model must expose its constructed pool records"
                        )
                    batch_oracle_pool_records = getter()
                if len(batch_oracle_pool_records) != len(batch_questions):
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
            if gold_doc_ids is not None and not clara_recall_enabled:
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
                    if len(index_row) != 5:
                        raise RuntimeError(
                            "Paper CCEF returns exactly five survivors; "
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
                gold = gold_answers[index]
                metrics = QAMetrics.compute_all(prediction, gold)
                all_metrics.append(metrics)
                prediction_record: Dict[str, Any] = {
                    "example_id": str(example_ids[index]),
                    "question": batch_questions[offset],
                    "prediction": str(prediction),
                    "gold_answer": gold,
                    **metrics,
                }
                if self.no_compression:
                    if batch_first_pass_corpus_indices is None:
                        raise RuntimeError(
                            "ARIA-NoComp first-pass corpus provenance is unavailable"
                        )
                    prediction_record["first_pass_corpus_indices"] = (
                        batch_first_pass_corpus_indices[offset]
                    )
                if batch_diagnostics is not None:
                    diagnostic = batch_diagnostics[offset]
                    if normalized_qca_reference_types is not None:
                        rule_type = str(
                            getattr(diagnostic, "rule_question_type", "")
                        )
                        overridden_type = str(
                            getattr(diagnostic, "oracle_question_type", "")
                        )
                        expected_type = normalized_qca_reference_types[index]
                        if rule_type not in {value.value for value in QuestionType}:
                            raise RuntimeError(
                                "Oracle-QCA diagnostics omitted the surface rule type"
                            )
                        if overridden_type != expected_type:
                            raise RuntimeError(
                                "QCA override diagnostics do not match the routed type"
                            )
                        prediction_record.update({
                            "qca_override_protocol": qca_override_protocol,
                            "qca_rule_type": rule_type,
                        })
                        if qca_override_protocol == ORACLE_QCA_PROTOCOL:
                            prediction_record["qca_oracle_type"] = overridden_type
                        elif qca_override_protocol == QCA_LLM_PROTOCOL:
                            if validated_qca_llm_records is None:
                                raise RuntimeError(
                                    "QCA-LLM route provenance is unavailable"
                                )
                            record = validated_qca_llm_records[index]
                            prediction_record.update(
                                {
                                    "qca_llm_type": overridden_type,
                                    "qca_llm_prompt": record["prompt"],
                                    "qca_llm_prompt_version": record[
                                        "prompt_version"
                                    ],
                                    "qca_llm_prompt_sha256": record[
                                        "prompt_sha256"
                                    ],
                                    "qca_llm_raw_output": record["raw_output"],
                                    "qca_llm_rationale": record["rationale"],
                                    "qca_llm_latency_ms": float(
                                        record["latency_ms"]
                                    ),
                                }
                            )
                    if self.no_compression:
                        document_tokens = int(
                            getattr(diagnostic, "direct_context_document_tokens", 0)
                        )
                        prompt_tokens = int(
                            getattr(diagnostic, "direct_context_prompt_tokens", 0)
                        )
                        context_ceiling = int(
                            getattr(diagnostic, "direct_context_ceiling", 0)
                        )
                        if (
                            document_tokens <= 0
                            or prompt_tokens <= 0
                            or context_ceiling != self.no_compression_context_limit
                        ):
                            raise RuntimeError(
                                "ARIA-NoComp diagnostics violate the direct-context "
                                "top-five/32k protocol"
                            )
                        direct_context_document_tokens.append(document_tokens)
                        direct_context_prompt_tokens.append(prompt_tokens)
                        prediction_record["retrieval_diagnostics"] = {
                            "final_document_count": int(
                                getattr(diagnostic, "final_candidates", 0)
                            ),
                            "second_round_candidate_count": 0,
                            "direct_context_document_tokens": document_tokens,
                            "direct_context_prompt_tokens": prompt_tokens,
                            "direct_context_ceiling": context_ceiling,
                        }
                    else:
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
                        if batch_oracle_pool_records is None or self.corpus_ids is None:
                            raise RuntimeError("Oracle pool provenance was not materialized")
                        pool_record = batch_oracle_pool_records[offset]
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
                predictions.append(prediction_record)

        if not all_metrics:
            empty_result = {
                "em": 0.0,
                "cem": 0.0,
                "f1": 0.0,
                "count": 0,
                "predictions": [],
            }
            if gold_doc_ids is not None:
                for cutoff in retrieval_cutoffs:
                    empty_result[f"recall_at_{cutoff}"] = 0.0
                empty_result["recall_at_5_support_count"] = 0
            if clara_retrieval_provenance is not None:
                empty_result["clara_retrieval_provenance"] = (
                    clara_retrieval_provenance
                )
            if no_compression_protocol is not None:
                empty_result["no_compression_protocol"] = no_compression_protocol
            if oracle_qca_provenance is not None:
                empty_result["oracle_qca_provenance"] = oracle_qca_provenance
            if qca_llm_provenance is not None:
                empty_result["qca_llm_provenance"] = qca_llm_provenance
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
        if self.no_compression:
            if not (
                len(direct_context_document_tokens)
                == len(direct_context_prompt_tokens)
                == len(all_metrics)
            ):
                raise RuntimeError(
                    "ARIA-NoComp context diagnostics must cover every example"
                )
            result["mean_direct_context_document_tokens"] = float(
                np.mean(direct_context_document_tokens)
            )
            result["mean_direct_context_prompt_tokens"] = float(
                np.mean(direct_context_prompt_tokens)
            )
            result["no_compression_protocol"] = no_compression_protocol
        if gold_doc_ids is not None:
            support_page_rows = gold_page_ids or []
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
        if oracle_qca_provenance is not None:
            result["oracle_qca_provenance"] = oracle_qca_provenance
        if qca_llm_provenance is not None:
            result["qca_llm_provenance"] = qca_llm_provenance
        return result

    def evaluate_multi_seed(
        self,
        questions: List[str],
        gold_answers: List[str],
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
    for context_metric in (
        "mean_direct_context_document_tokens",
        "mean_direct_context_prompt_tokens",
    ):
        context_presence = [context_metric in result for result in results]
        if any(context_presence) and not all(context_presence):
            raise ValueError(
                f"{context_metric} must be present for every checkpoint result "
                "or omitted from all"
            )
        if all(context_presence):
            metric_names.append(context_metric)
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
    no_compression_protocols = [
        result.get("no_compression_protocol") for result in results
    ]
    if any(value is not None for value in no_compression_protocols):
        if any(value is None for value in no_compression_protocols) or any(
            value != no_compression_protocols[0]
            for value in no_compression_protocols[1:]
        ):
            raise ValueError(
                "All ARIA-NoComp checkpoints must share exact direct-context provenance"
            )
        aggregated["no_compression_protocol"] = no_compression_protocols[0]
    oracle_qca_provenance = [
        result.get("oracle_qca_provenance") for result in results
    ]
    if any(value is not None for value in oracle_qca_provenance):
        if any(value is None for value in oracle_qca_provenance) or any(
            value != oracle_qca_provenance[0]
            for value in oracle_qca_provenance[1:]
        ):
            raise ValueError(
                "All Oracle-QCA checkpoints must share the same labeled panel"
            )
        aggregated["oracle_qca_provenance"] = oracle_qca_provenance[0]
    qca_llm_provenance = [
        result.get("qca_llm_provenance") for result in results
    ]
    if any(value is not None for value in qca_llm_provenance):
        if any(value is None for value in qca_llm_provenance) or any(
            value != qca_llm_provenance[0]
            for value in qca_llm_provenance[1:]
        ):
            raise ValueError(
                "All QCA-LLM checkpoints must reuse the same cached router records"
            )
        aggregated["qca_llm_provenance"] = qca_llm_provenance[0]
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
            canonical_content: Dict[str, Tuple[str, str]] = {}
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
                    saved_gold = prediction.get("gold_answer")
                    if not isinstance(saved_gold, str) or not saved_gold.strip():
                        raise ValueError(
                            f"{label} {dataset_name} prediction lacks scalar gold_answer"
                        )
                    signature = (
                        str(prediction.get("question")),
                        QAMetrics.normalize_answer(saved_gold),
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
            candidate_gold = candidate_prediction.get("gold_answer")
            baseline_gold = baseline_prediction.get("gold_answer")
            if (
                not isinstance(candidate_gold, str)
                or not candidate_gold.strip()
                or not isinstance(baseline_gold, str)
                or not baseline_gold.strip()
            ):
                raise ValueError("Saved prediction comparison requires scalar gold_answer")
            candidate_signature = (
                str(candidate_prediction.get("question")),
                QAMetrics.normalize_answer(candidate_gold),
            )
            baseline_signature = (
                str(baseline_prediction.get("question")),
                QAMetrics.normalize_answer(baseline_gold),
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


def _create_evaluation_rag_config(
    runtime_configuration: str,
    compression_rate: int,
    **overrides: Any,
) -> RAGPipelineConfig:
    """Build retrieval state for an evaluation-only runtime protocol."""
    configuration = (
        "forward_path_off"
        if runtime_configuration == ARIA_NO_COMPRESSION_CONFIGURATION
        else runtime_configuration
    )
    return create_paper_rag_config(
        configuration,
        compression_rate,
        **overrides,
    )


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
        or tuple(epoch_schedule) != PAPER_PHASE2_EPOCH_SEEDS
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
        "retrieval_straight_through_scheme": RETRIEVAL_STRAIGHT_THROUGH_SCHEME,
    }
    for key, expected in canonical_architecture.items():
        if getattr(config, key, None) != expected:
            raise ValueError(
                f"Checkpoint architecture {key!r} must be {expected!r}, "
                f"got {getattr(config, key, None)!r}"
            )
    expected_loss_weights = {
        "lambda_mse": 0.0 if checkpoint_configuration == "clara_baseline" else 0.10
    }
    if getattr(config, "aria_loss_weights", None) != expected_loss_weights:
        raise ValueError(
            "Checkpoint Phase-II objective metadata must be "
            f"{expected_loss_weights!r}"
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
                "Paper MTFRL uses W_BGE-derived initialization, not a low-rank factor"
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


def _evaluate_qca_llm_panel(
    args: argparse.Namespace,
    seed_checkpoints: Sequence[Tuple[Optional[int], str]],
    checkpoint_configs: Sequence[CLaRaConfig],
) -> None:
    """Evaluate only the fixed 1,000-example QCA classification endpoint."""
    if args.qca_llm_labels is None:
        raise ValueError("QCA-LLM panel mode requires --qca_llm_labels")
    base_models = {
        getattr(config, "decoder_model_name", None) for config in checkpoint_configs
    }
    revisions = {
        getattr(config, "decoder_model_resolved_revision", None)
        for config in checkpoint_configs
    }
    if base_models != {PAPER_QCA_LLM_MODEL} or len(revisions) != 1:
        raise ValueError(
            "QCA-LLM panel checkpoints must share one exact Mistral-7B base revision"
        )
    revision = next(iter(revisions))
    if not isinstance(revision, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}", revision
    ) is None:
        raise ValueError("QCA-LLM panel requires an exact resolved base revision")

    panel_rows: List[Dict[str, Any]] = []
    label_sources: Dict[str, Dict[str, Any]] = {}
    for dataset_name in ORACLE_QCA_PANEL_COUNTS:
        dataset, question_key, _ = load_eval_dataset(
            dataset_name,
            None,
            args.eval_data_path,
            require_clara_archive=False,
        )
        all_example_ids = _extract_example_ids(dataset, dataset_name)
        label_path = _format_artifact_path(
            args.qca_llm_labels,
            dataset=dataset_name,
            compression_rate=args.compression_rate,
        )
        labels = _load_oracle_qca_labels(label_path)
        indices, references = _oracle_qca_labeled_subset(
            all_example_ids,
            labels,
            dataset_name=dataset_name,
        )
        matched_ids = [all_example_ids[index] for index in indices]
        _validate_oracle_qca_paper_panel(dataset_name, matched_ids)
        matched_hasher = hashlib.sha256()
        for example_id in matched_ids:
            matched_hasher.update(str(example_id).encode("utf-8"))
            matched_hasher.update(b"\n")
        label_sources[dataset_name] = {
            "source_path": str(Path(label_path).expanduser().resolve()),
            "source_sha256": file_sha256(Path(label_path)),
            "source_label_count": len(labels),
            "matched_label_count": len(matched_ids),
            "matched_example_ids_sha256": matched_hasher.hexdigest(),
        }
        for index, example_id, reference_type in zip(
            indices, matched_ids, references
        ):
            panel_rows.append(
                {
                    "dataset": dataset_name,
                    "example_id": example_id,
                    "question": dataset[index][question_key],
                    "reference_type": QuestionType(reference_type).value,
                }
            )
    expected_total = sum(ORACLE_QCA_PANEL_COUNTS.values())
    if len(panel_rows) != expected_total:
        raise RuntimeError(
            f"QCA-LLM panel requires exactly {expected_total} examples"
        )

    training_seed, checkpoint_path = seed_checkpoints[0]
    _set_inference_seed(args.inference_seed)
    model = CLaRa.from_pretrained(
        checkpoint_path,
        strict_aria_artifacts=True,
        external_bge_artifact=args.bge_projection_path is not None,
        pure_inference=True,
    )
    if args.decoder_model is not None and args.decoder_model != model.decoder_model_name:
        raise ValueError(
            "--decoder_model must match the backbone recorded by the checkpoint"
        )
    protocol_fingerprint = _validate_checkpoint_protocol(
        model,
        checkpoint_path,
        training_seed,
        args.compression_rate,
        "full",
    )
    model = model.to(args.device)
    model.eval()
    records = _run_qca_llm_router(
        model,
        [row["question"] for row in panel_rows],
        batch_size=args.batch_size,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions = [record["parsed_type"] for record in records]
    references_all = [row["reference_type"] for row in panel_rows]
    per_benchmark: Dict[str, Dict[str, Any]] = {}
    for dataset_name, expected_count in ORACLE_QCA_PANEL_COUNTS.items():
        indices = [
            index
            for index, row in enumerate(panel_rows)
            if row["dataset"] == dataset_name
        ]
        references = [references_all[index] for index in indices]
        predicted = [predictions[index] for index in indices]
        if len(indices) != expected_count:
            raise RuntimeError("QCA-LLM benchmark panel count changed unexpectedly")
        per_benchmark[dataset_name] = {
            "count": expected_count,
            "weighted_f1": _qca_weighted_f1(references, predicted),
        }

    output_records = []
    for row, record in zip(panel_rows, records):
        output_records.append(
            {
                **row,
                **record,
                "correct": row["reference_type"] == record["parsed_type"],
            }
        )
    payload = {
        "metadata": {
            "protocol": QCA_LLM_PROTOCOL,
            "endpoint": "fixed-primary-1000-query-classification-panel",
            "paper_panel_counts": dict(ORACLE_QCA_PANEL_COUNTS),
            "paper_panel_total": expected_total,
            "prompt_version": QCA_LLM_PROMPT_VERSION,
            "prompt_template_sha256": QCA_LLM_PROMPT_SHA256,
            "base_model": PAPER_QCA_LLM_MODEL,
            "base_model_revision_resolved": revision.lower(),
            "adapters_disabled": True,
            "decoding": "greedy-eos-or-64",
            "checkpoint_used": str(Path(checkpoint_path).expanduser().resolve()),
            "training_seed": training_seed,
            "checkpoint_protocol": protocol_fingerprint,
            "label_sources": label_sources,
            "mean_router_latency_ms": float(
                np.mean([record["latency_ms"] for record in records])
            ),
        },
        "weighted_f1": _qca_weighted_f1(references_all, predictions),
        "per_benchmark": per_benchmark,
        "predictions": output_records,
    }
    output_path = os.path.join(args.output_dir, "qca_llm_panel.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(
            _json_safe(payload),
            output_file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    print(f"QCA-LLM panel results saved to {output_path}")


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
    parser.add_argument(
        "--oracle_qca_labels",
        type=str,
        default=None,
        help=(
            "Evaluation-only JSON mapping example_id to simple, multi_aspect, or "
            "multi_hop, or JSONL rows with example_id and question_type. Supports "
            "a {dataset} path template. Only explicitly labeled rows are evaluated."
        ),
    )
    parser.add_argument(
        "--qca_llm_mode",
        choices=["qa", "panel"],
        default=None,
        help=(
            "Evaluator-only zero-shot Mistral QCA router. 'qa' runs the full "
            "four-benchmark QA endpoint; 'panel' reports weighted F1 on the "
            "fixed 1,000-query primary annotation panel."
        ),
    )
    parser.add_argument(
        "--qca_llm_labels",
        type=str,
        default=None,
        help=(
            "Keyed primary QCA labels for --qca_llm_mode panel. Supports a "
            "{dataset} path template and is not used by the QA endpoint."
        ),
    )
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help=(
            "Scalar-answer DatasetDict created by `aria-data --stage eval`"
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
            "(nq.zip, hotpotqa.zip, musique.zip, 2wiki.zip). Required for matched "
            "CLaRa Normal evaluation only; archives are not bundled."
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
        choices=sorted(
            set(RAG_CONFIGURATION_SPECS) | {ARIA_NO_COMPRESSION_CONFIGURATION}
        ),
        default=None,
        help=(
            "Explicit training/runtime protocol. Paper coupling rows use fixed_* "
            "or forward_path_off on full checkpoints; remove_all_coupling is an "
            "additional independently retrained 108-token/static-D2 control. "
            "no_compression is the evaluator-only ARIA-NoComp diagnostic: it "
            "requires a full Phase-II checkpoint and Normal retrieval."
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
    if args.oracle_qca_labels is not None:
        try:
            _validate_oracle_qca_conditions(
                retrieval_mode=args.retrieval_mode,
                rag_configuration=expected_configuration,
                compression_rate=args.compression_rate,
                max_samples=args.max_samples,
                dataset=args.dataset,
            )
        except ValueError as exc:
            parser.error(str(exc))
    if args.qca_llm_mode is not None:
        if args.oracle_qca_labels is not None:
            parser.error("QCA-LLM and Oracle-QCA cannot be combined")
        try:
            _validate_qca_llm_conditions(
                retrieval_mode=args.retrieval_mode,
                rag_configuration=expected_configuration,
                compression_rate=args.compression_rate,
                dataset=args.dataset,
                max_samples=args.max_samples,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.qca_llm_mode == "panel" and args.qca_llm_labels is None:
            parser.error("--qca_llm_mode panel requires --qca_llm_labels")
        if args.qca_llm_mode == "qa" and args.qca_llm_labels is not None:
            parser.error("--qca_llm_labels is valid only in QCA-LLM panel mode")
        if args.baseline_results is not None and args.qca_llm_mode == "panel":
            parser.error("QCA-LLM panel mode does not accept --baseline_results")
    elif args.qca_llm_labels is not None:
        parser.error("--qca_llm_labels requires --qca_llm_mode panel")
    is_no_compression = (
        expected_configuration == ARIA_NO_COMPRESSION_CONFIGURATION
    )
    if is_no_compression and args.retrieval_mode != "normal":
        parser.error("--rag_configuration no_compression requires Normal retrieval")
    is_clara_baseline = expected_configuration == "clara_baseline"
    if is_clara_baseline:
        if args.retrieval_mode == "normal" and args.clara_archive_dir is None:
            parser.error(
                "Normal --rag_configuration clara_baseline requires --clara_archive_dir"
            )
        if args.retrieval_mode == "oracle" and args.clara_archive_dir is not None:
            parser.error("CLaRa Oracle uses the shared top-100 pool, not an archive")
    elif args.clara_archive_dir is not None:
        parser.error("--clara_archive_dir is valid only for matched CLaRa evaluation")
    if (
        args.qca_llm_mode != "panel"
        and (args.corpus_path is None or args.doc_embeddings is None)
    ):
        parser.error(
            "Paper evaluation requires --corpus_path and --doc_embeddings"
        )

    seed_checkpoints = _resolve_seed_checkpoints(parser, args)
    checkpoint_configs = [
        CLaRaConfig.from_pretrained(checkpoint_path)
        for _, checkpoint_path in seed_checkpoints
    ]
    first_checkpoint_config = checkpoint_configs[0]
    training_index_sha256 = getattr(
        first_checkpoint_config, "aria_training_retrieval_index_sha256", None
    )
    if not isinstance(training_index_sha256, str) or len(training_index_sha256) != 64:
        parser.error("checkpoint requires its Phase-II training BGE-index fingerprint")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.qca_llm_mode == "panel":
        _evaluate_qca_llm_panel(args, seed_checkpoints, checkpoint_configs)
        return

    datasets_to_eval = (
        ["nq", "hotpotqa", "musique", "2wikimultihopqa"]
        if args.dataset == "all"
        else [args.dataset]
    )
    all_results: Dict[str, Dict[str, Any]] = {}
    evaluation_retrieval_provenance: Dict[str, Dict[str, Any]] = {}
    oracle_qca_label_sources: Dict[str, Dict[str, Any]] = {}

    for dataset_name in datasets_to_eval:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'=' * 60}")

        dataset, question_key, answer_key = load_eval_dataset(
            dataset_name,
            args.max_samples,
            args.eval_data_path,
            require_clara_archive=(
                is_clara_baseline and args.retrieval_mode == "normal"
            ),
            clara_archive_dir=args.clara_archive_dir,
        )
        all_example_ids = _extract_example_ids(dataset, dataset_name)
        qca_reference_types: Optional[List[str]] = None
        if args.oracle_qca_labels is not None:
            label_path = _format_artifact_path(
                args.oracle_qca_labels,
                dataset=dataset_name,
                compression_rate=args.compression_rate,
            )
            qca_labels = _load_oracle_qca_labels(label_path)
            labeled_indices, qca_reference_types = _oracle_qca_labeled_subset(
                all_example_ids,
                qca_labels,
                dataset_name=dataset_name,
            )
            dataset = dataset.select(labeled_indices)
            example_ids = [all_example_ids[index] for index in labeled_indices]
            _validate_oracle_qca_paper_panel(dataset_name, example_ids)
            matched_id_hasher = hashlib.sha256()
            for example_id in example_ids:
                matched_id_hasher.update(str(example_id).encode("utf-8"))
                matched_id_hasher.update(b"\n")
            oracle_qca_label_sources[dataset_name] = {
                "protocol": ORACLE_QCA_PROTOCOL,
                "source_path": str(Path(label_path).resolve()),
                "source_sha256": file_sha256(Path(label_path)),
                "source_label_count": len(qca_labels),
                "matched_label_count": len(qca_reference_types),
                "matched_example_ids_sha256": matched_id_hasher.hexdigest(),
            }
        else:
            example_ids = all_example_ids
        questions = [item[question_key] for item in dataset]
        gold_answers = [_extract_gold_answer(item, answer_key) for item in dataset]
        gold_document_ids = _extract_gold_document_ids(dataset)
        if args.retrieval_mode == "oracle" and gold_document_ids is None:
            raise ValueError(
                "--retrieval_mode oracle requires prepared evaluation rows with "
                "corpus-level gold_doc_ids"
            )
        clara_documents: Optional[List[List[str]]] = None
        clara_candidate_doc_ids: Optional[List[List[str]]] = None
        clara_candidate_page_ids: Optional[List[List[str]]] = None
        if is_clara_baseline and args.retrieval_mode == "normal":
            clara_documents = _extract_clara_candidate_columns(dataset)
        print(f"Loaded {len(questions)} examples")

        corpus_docs: List[str] = []
        corpus_ids: List[str] = []
        corpus_urls: List[str] = []
        corpus_digest: Optional[str] = None
        doc_embeddings: Optional[torch.Tensor] = None
        bm25_index: Optional[_BM25Index] = None
        try:
            corpus = load_corpus(args.corpus_path)
        except Exception as exc:
            raise RuntimeError(
                f"Paper evaluation requires a loadable KILT corpus ({dataset_name})"
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
        loaded_embeddings = load_doc_embeddings(
            embeddings_path,
            len(corpus_docs),
            expected_ids=corpus_ids,
            expected_hashes=corpus_hashes,
            expected_page_ids=corpus_urls,
            return_index_sha256=True,
            return_encoder_spec=args.retrieval_mode == "oracle",
        )
        oracle_encoder_spec: Optional[Dict[str, Any]] = None
        if args.retrieval_mode == "oracle":
            doc_embeddings, evaluation_index_sha256, oracle_encoder_spec = (
                loaded_embeddings
            )
        else:
            doc_embeddings, evaluation_index_sha256 = loaded_embeddings
        if clara_documents is not None:
            (
                clara_candidate_doc_ids,
                clara_candidate_page_ids,
                _,
            ) = _map_clara_candidates_to_corpus(
                clara_documents,
                corpus_docs=corpus_docs,
                corpus_ids=corpus_ids,
                corpus_page_ids=corpus_urls,
            )
        shared_oracle_pool_records: Optional[List[OraclePoolRecord]] = None
        if args.retrieval_mode == "oracle":
            if gold_document_ids is None or oracle_encoder_spec is None:
                raise RuntimeError("Oracle pool inputs are unavailable")
            oracle_query_embeddings = _encode_oracle_queries(
                questions, oracle_encoder_spec
            )
            shared_oracle_pool_records = _build_shared_oracle_pool_records(
                oracle_query_embeddings,
                doc_embeddings,
                corpus_ids=corpus_ids,
                corpus_page_ids=corpus_urls,
                gold_doc_ids=gold_document_ids,
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
            "bge_model": PAPER_BGE_MODEL,
            "text_sha256_scheme": TEXT_SHA256_SCHEME,
            "mads_semantic_source": "shared_bge_document_embeddings",
        }
        if oracle_encoder_spec is not None:
            evaluation_retrieval_provenance[dataset_name] = {
                **evaluation_retrieval_provenance[dataset_name],
                "oracle_query_embedding_protocol": ORACLE_QUERY_EMBEDDING_PROTOCOL,
                "oracle_encoder_source": oracle_encoder_spec["source"],
                "oracle_encoder_revision": oracle_encoder_spec["revision"],
            }
        if qca_reference_types is not None:
            evaluation_retrieval_provenance[dataset_name].update(
                {
                    "qca_mode": ORACLE_QCA_PROTOCOL,
                    "qca_labeled_example_count": len(qca_reference_types),
                    "qca_label_source_sha256": oracle_qca_label_sources[
                        dataset_name
                    ]["source_sha256"],
                }
            )
        if args.qca_llm_mode == "qa":
            evaluation_retrieval_provenance[dataset_name].update(
                {
                    "qca_mode": QCA_LLM_PROTOCOL,
                    "qca_prompt_version": QCA_LLM_PROMPT_VERSION,
                    "qca_prompt_template_sha256": QCA_LLM_PROMPT_SHA256,
                    "qca_adapters_disabled": True,
                }
            )
        if args.retrieval_mode == "normal":
            _assert_normal_retrieval_is_not_training_index(
                first_checkpoint_config,
                evaluation_corpus_sha256=corpus_digest,
                evaluation_index_sha256=evaluation_index_sha256,
            )
        if not is_clara_baseline:
            # BM25 is immutable after build, so all independently trained
            # checkpoints for this benchmark can safely share one full index.
            bm25_index = _BM25Index().build(corpus_docs)

        rag_config = _create_evaluation_rag_config(
            expected_configuration,
            args.compression_rate,
        )

        checkpoint_results: List[Dict[str, Any]] = []
        checkpoint_times: List[float] = []
        protocol_fingerprints: List[Dict[str, Any]] = []
        qca_llm_records: Optional[List[Dict[str, Any]]] = None
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
            if args.qca_llm_mode == "qa" and qca_llm_records is None:
                qca_llm_records = _run_qca_llm_router(
                    model,
                    questions,
                    batch_size=args.batch_size,
                )
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
                no_compression=is_no_compression,
            )

            start_time = time.time()
            checkpoint_result = evaluator.evaluate(
                questions=questions,
                gold_answers=gold_answers,
                example_ids=example_ids,
                gold_doc_ids=gold_document_ids,
                documents=clara_documents,
                clara_candidate_doc_ids=clara_candidate_doc_ids,
                clara_candidate_page_ids=clara_candidate_page_ids,
                oracle_pool_records=shared_oracle_pool_records,
                qca_reference_types=qca_reference_types,
                qca_llm_records=qca_llm_records,
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
    if args.oracle_qca_labels is not None:
        result_prefix += "_oracle_qca"
    elif args.qca_llm_mode == "qa":
        result_prefix += "_qca_llm"
    compression_label = (
        f"cr1_sourcecr{args.compression_rate}"
        if is_no_compression
        else f"cr{args.compression_rate}"
    )
    output_path = os.path.join(
        args.output_dir,
        f"{result_prefix}_{args.dataset}_{compression_label}.json",
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
            "compression_rate": 1 if is_no_compression else args.compression_rate,
            "source_checkpoint_compression_rate": args.compression_rate,
            "training_seeds": [seed for seed, _ in seed_checkpoints],
            "checkpoints": [path for _, path in seed_checkpoints],
            "inference_seed": args.inference_seed,
            "normalization": "ARIA Appendix A.35",
            "answer_contract": EVALUATION_ANSWER_CONTRACT,
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
            "no_compression_protocol": (
                next(
                    (
                        result.get("no_compression_protocol")
                        for name, result in all_results.items()
                        if name != "avg"
                    ),
                    None,
                )
                if is_no_compression
                else None
            ),
            "evaluation_retrieval_provenance": evaluation_retrieval_provenance,
            "oracle_qca": (
                {
                    "protocol": ORACLE_QCA_PROTOCOL,
                    "scope": "explicitly-labeled-evaluation-subset-only",
                    "paper_panel_counts": dict(ORACLE_QCA_PANEL_COUNTS),
                    "paper_panel_total": sum(ORACLE_QCA_PANEL_COUNTS.values()),
                    "surface_fields_preserved": [
                        "confidence",
                        "matched_rules",
                        "hop_count",
                        "sub_questions",
                        "entity_count",
                    ],
                    "label_sources": oracle_qca_label_sources,
                }
                if args.oracle_qca_labels is not None
                else None
            ),
            "qca_llm": (
                {
                    "protocol": QCA_LLM_PROTOCOL,
                    "endpoint": "full-four-benchmark-qa",
                    "prompt_version": QCA_LLM_PROMPT_VERSION,
                    "prompt_template_sha256": QCA_LLM_PROMPT_SHA256,
                    "base_model": PAPER_QCA_LLM_MODEL,
                    "base_model_revision_resolved": getattr(
                        first_checkpoint_config,
                        "decoder_model_resolved_revision",
                        None,
                    ),
                    "adapters_disabled": True,
                    "decoding": "greedy-eos-or-64",
                    "surface_fields_preserved": [
                        "confidence",
                        "matched_rules",
                        "hop_count",
                        "sub_questions",
                        "entity_count",
                    ],
                }
                if args.qca_llm_mode == "qa"
                else None
            ),
            "clara_archive_sha256": (
                dict(_REPOSITORY_EVAL_ARCHIVE_SHA256)
                if is_clara_baseline and args.retrieval_mode == "normal"
                else None
            ),
            "oracle_protocol": (
                {
                    "name": ORACLE_TOP100_PROTOCOL,
                    "pool_size": 100,
                    "base_order": "BGE score descending, corpus index ascending on ties",
                    "query_embedding": ORACLE_QUERY_EMBEDDING_PROTOCOL,
                    "shared_between_aria_and_clara": True,
                    "page_deduplication": "retain first ranked occurrence of each canonical page URL",
                    "positive_insertion": "missing gold pages at tail in annotation order",
                    "eviction": "lowest-ranked non-gold pages first",
                    "candidate_acquisition_scope": "shared fixed top-100 pool",
                    "first_ranking_stage": (
                        "CLaRa trained-QR hard top-5 selector"
                        if is_clara_baseline
                        else "MADS then CCEF"
                    ),
                    "mtfrl_scope": (
                        None
                        if is_clara_baseline
                        else "same fixed top-100 pool"
                    ),
                    "reported_recall_scope": (
                        "CLaRa hard top-5 page order"
                        if is_clara_baseline
                        else "final page-deduplicated CFRS order at k=1,3,5"
                    ),
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
