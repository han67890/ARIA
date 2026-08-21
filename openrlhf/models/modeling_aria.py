#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
# Modified for ARIA in 2026.
# Copyright (C) 2026 Yiheng Han (ARIA modifications only).
#

import os
import torch
import time
import json
import hashlib
import zipfile
import re
from pathlib import Path
from contextlib import contextmanager

from torch import nn
from torch.nn import functional as F
from torch.nn.functional import gelu
from jinja2.exceptions import TemplateError
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PretrainedConfig,
    StoppingCriteria,
)
from huggingface_hub import hf_hub_download
from typing import List, Dict, Any, Mapping, Optional, Tuple, Sequence, Union, Callable
from openrlhf.utils.aria_provenance import (
    SOURCE_SNAPSHOT_SCHEME,
    build_source_snapshot_manifest,
    file_sha256,
    write_source_snapshot,
)

# ── RAG Enhancement: additional standard-library imports ──────────────────────
import math
import string
from enum import Enum
from dataclasses import dataclass, field
from collections import Counter, OrderedDict
import numpy as np

# Constants
IGNORE_INDEX = -100
ARIA_NUMERICAL_EPSILON = 1e-6
QR_INPUT_SCHEME = "native-tokenizer-final-token-v1"
MTFRL_INITIALIZATION_SCHEME = "w_bge_paired_gelu_identity_v1"
CFRS_RECONSTRUCTION_SCHEME = "teacher-forced-squared-probability-v1"
RETRIEVAL_STRAIGHT_THROUGH_SCHEME = "hard-forward-soft-permutation-v1"
CFRS_SOFT_PERMUTATION_TEMPERATURE = 0.1
ARIA_NO_COMPRESSION_CONFIGURATION = "no_compression"
ARIA_NO_COMPRESSION_SCHEME = "first-pass-top5-raw-direct-context-v1"
ARIA_NO_COMPRESSION_CONTEXT_POLICY = "fixed-32768-right-truncate-if-needed-v1"
ARIA_NO_COMPRESSION_CONTEXT_CEILING = 32_768
ARIA_NO_COMPRESSION_MAX_NEW_TOKENS = 64
ORACLE_TOP100_PROTOCOL = "bge-top100-page-dedup-tail-inject-annotation-order-v2"
CLARA_SELECTOR_SCHEME = "hard-forward-soft-backward-st-topk-v1"
CLARA_DOCUMENT_REPRESENTATION_SCHEME = "frozen-variable-memory-masked-mean-v2"
CLARA_PHASE2_OBJECTIVE = "answer-cross-entropy-only-v1"
CLARA_MEMORY_ALLOCATION_SCHEME = "max-one-floor-truncated-length-over-r-v1"
CLARA_EVALUATION_CANDIDATE_PROTOCOL = "retained-bge-top20-st-hard-top5-release-v1"
CLARA_ARCHIVE_DOCUMENT_ID_SCHEME = "sha256-exact-archive-text-v1"
CLARA_ARCHIVE_PAGE_ID_SCHEME = "sha256-casefold-collapsed-title-header-v1"
COUPLING_CONTROL_PROTOCOL = "matched-retraining-vs-fixed-forward-v1"
UNIFORM_BUDGET_ALLOCATION_SCHEME = (
    "score-independent-query-total-ratio-release-convention-v1"
)
STATIC_SECOND_QUERY_SCHEME = "original-qr-projected-by-frozen-w-bge-release-convention-v1"
FIXED_UNIFORM_ALLOCATION_SCHEME = "score-independent-rho-0.625-v1"
MATCHED_EVIDENCE_TOKEN_BUDGET = 108
CFRS_RECONSTRUCTION_PREFIX = (
    "Reconstruct the original passage from the memory tokens. Output only the "
    "reconstructed passage."
)


def _mtfrl_hidden_width(input_dim: int, output_dim: int) -> int:
    """Return the paper's H/2 hidden width for the feedback projection."""
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("MTFRL dimensions must be positive")
    if input_dim % 2:
        raise ValueError("the paper's MTFRL projection requires an even hidden size")
    return input_dim // 2


@contextmanager
def _base_decoder_only(model: nn.Module):
    """Disable LoRA adapters while preserving gradients to memory inputs."""
    singular = getattr(model, "disable_adapter", None)
    if callable(singular):
        context = singular()
        if hasattr(context, "__enter__") and hasattr(context, "__exit__"):
            with context:
                yield
            return
    disable = getattr(model, "disable_adapters", None)
    enable = getattr(model, "enable_adapters", None)
    if not callable(disable) or not callable(enable):
        raise RuntimeError(
            "CFRS requires an adapter API that can expose the frozen base decoder"
        )
    disable()
    try:
        yield
    finally:
        enable()


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Enhancement Modules
# Five-stage pipeline: QCA → AHR → IGFR → MADS → CCEF
# These classes are self-contained and injected into CLaRa via setup_rag_pipeline()
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Question Complexity Assessment (QCA) ────────────────────────────

class QuestionType(str, Enum):
    SIMPLE       = "simple"
    MULTI_ASPECT = "multi_aspect"
    MULTI_HOP    = "multi_hop"


@dataclass
class QCAResult:
    question:      str
    question_type: QuestionType
    confidence:    float
    hop_count:     int
    entity_count:  int
    sub_questions: List[str] = field(default_factory=list)
    matched_rules: Tuple[str, ...] = field(default_factory=tuple)
    reasoning:     str = ""


# Machine-readable weights/metadata for Appendix Tables A51-A52. Predicates are
# evaluated below because several combine phrase, NER spans, and dependency tests.
_QCA_RULES_PATH = Path(__file__).resolve().parents[1] / "configs" / "qca_rules.json"
with _QCA_RULES_PATH.open("r", encoding="utf-8") as _qca_rules_handle:
    _QCA_RULE_SPEC = json.load(_qca_rules_handle)
_QCA_RULE_WEIGHTS: Dict[str, float] = {
    rule["id"]: float(rule["weight"]) for rule in _QCA_RULE_SPEC["rules"]
}
_QCA_TOTAL_WEIGHT = sum(_QCA_RULE_WEIGHTS.values())
_QCA_ENTITY_TYPES = set(_QCA_RULE_SPEC["entity_types"])
# Appendix Tables A51--A52 distinguish the strongest explicit phrase families
# from the remaining (partly count-driven) rules.  Resolve cross-category
# conflicts by these tiers, not by a blanket "any H beats any A" rule.
_QCA_EXPLICIT_HOP_RULES = frozenset({"H02", "H03", "H04", "H05"})
_QCA_EXPLICIT_ASPECT_RULES = frozenset(
    {"A01", "A02", "A03", "A04", "A05", "A06"}
)
_QCA_BRIDGE_CONNECTIVES = ("that", "which", "whose", "where")
_QCA_COMPARATIVE_RE = re.compile(
    r"\b(?:same|different|difference|similar|compare|contrast|more|less|better|worse|"
    r"largest|smallest|oldest|youngest|highest|lowest|longest|shortest|tallest)\b",
    re.IGNORECASE,
)
_QCA_TRANSITIVE_FALLBACK = re.compile(
    r"\b(?:wrote|invented|founded|directed|discovered|created|made|won|played|"
    r"located|headquartered|produced|designed|built|published|married)\b",
    re.IGNORECASE,
)

_QCA_SPACY_NLP = None

def _qca_get_spacy():
    global _QCA_SPACY_NLP
    if _QCA_SPACY_NLP is None:
        try:
            import spacy
            _QCA_SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception as exc:
            raise RuntimeError(
                "ARIA QCA/IGFR/MADS requires spaCy en_core_web_sm; install it with "
                "`python -m spacy download en_core_web_sm`"
            ) from exc
    return _QCA_SPACY_NLP


def _qca_entities(text: str) -> set:
    nlp = _qca_get_spacy()
    if nlp:
        try:
            return {
                ent.text.casefold().strip()
                for ent in nlp(text).ents
                if ent.label_ in _QCA_ENTITY_TYPES and ent.text.strip()
            }
        except Exception:
            pass
    raise RuntimeError("spaCy NER returned no pipeline")


def _qca_entity_spans_surface(text: str) -> List[str]:
    """Return distinct QCA entity surfaces in their left-to-right order."""
    nlp = _qca_get_spacy()
    try:
        parsed = nlp(text)
    except Exception as exc:
        raise RuntimeError("spaCy NER returned no pipeline") from exc
    seen: set[str] = set()
    spans: List[str] = []
    for entity in parsed.ents:
        value = entity.text.strip()
        key = value.casefold()
        if entity.label_ in _QCA_ENTITY_TYPES and value and key not in seen:
            seen.add(key)
            spans.append(value)
    return spans


def _qca_entity_count(text: str) -> int:
    return len(_qca_entities(text))


def _qca_has_transitive_verb(text: str) -> bool:
    nlp = _qca_get_spacy()
    if nlp:
        try:
            doc = nlp(text)
            return any(
                tok.pos_ == "VERB" and any(child.dep_ in {"dobj", "obj"} for child in tok.children)
                for tok in doc
            )
        except Exception:
            pass
    raise RuntimeError("spaCy dependency parsing returned no pipeline")


def _qca_has_temporal_entity_verb(text: str, entities: set[str]) -> bool:
    """Implement H11's after/before ENTITY VERB constraint with spaCy POS."""
    parsed = _qca_get_spacy()(text)
    lowered = text.casefold()
    for entity in entities:
        match = re.search(
            rf"\b(?:after|before)\s+{re.escape(entity)}\b",
            lowered,
        )
        if match is None:
            continue
        following = next(
            (token for token in parsed if token.idx >= match.end()),
            None,
        )
        if following is not None and following.pos_ == "VERB":
            return True
    return False


def _qca_superlative_positions(text: str) -> List[int]:
    """Return spaCy-validated superlative token offsets for rule H12."""
    parsed = _qca_get_spacy()(text)
    positions: List[int] = []
    for index, token in enumerate(parsed):
        degrees = set(token.morph.get("Degree")) if hasattr(token, "morph") else set()
        if token.tag_ == "JJS" or "Sup" in degrees or token.lower_ in {"best", "worst"}:
            positions.append(token.idx)
            continue
        if token.lower_ not in {"most", "least"} or index + 1 >= len(parsed):
            continue
        following = parsed[index + 1]
        if following.pos_ in {"ADJ", "ADV"} or following.tag_ in {
            "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
        }:
            positions.append(token.idx)
    return positions


def _qca_has_superlative_entity_pattern(text: str, entities: set[str]) -> bool:
    """Implement H12's ``the SUPERLATIVE ... in the ENTITY`` pattern."""
    lowered = text.casefold()
    for position in _qca_superlative_positions(text):
        if re.search(r"\bthe\s*$", lowered[:position]) is None:
            continue
        suffix = lowered[position:]
        if any(
            re.search(rf"\bin\s+the\s+{re.escape(entity)}\b", suffix)
            for entity in entities
        ):
            return True
    return False


def _qca_has_plural_trigger(text: str) -> bool:
    """Implement A10 using the tokenizer's plural-noun POS tag."""
    parsed = _qca_get_spacy()(text)
    for index, token in enumerate(parsed[:-1]):
        if token.lower_ in {"multiple", "various", "several"}:
            return parsed[index + 1].tag_ in {"NNS", "NNPS"}
    return False


def _qca_has_main_plural_aspect(text: str) -> bool:
    """Implement A07 without restricting the paper's plural aspect noun."""
    parsed = _qca_get_spacy()(text)
    lowered = [token.lower_ for token in parsed]
    for adjective in ("main", "key", "primary"):
        prefix = ["what", "are", "the", adjective]
        for start in range(0, len(lowered) - len(prefix)):
            if lowered[start : start + len(prefix)] != prefix:
                continue
            following = parsed[start + len(prefix)]
            if following.tag_ in {"NNS", "NNPS"}:
                return True
    return False


def _qca_starts(text: str, *prefixes: str) -> bool:
    return any(re.match(rf"^{re.escape(prefix)}(?:\b|$)", text) for prefix in prefixes)


def _qca_contains(text: str, *phrases: str) -> bool:
    return any(phrase in text for phrase in phrases)


def _qca_match_rules(question: str, entities: set) -> Dict[str, bool]:
    """Evaluate the exact 38-rule specification in Tables A51-A52."""
    q = " ".join(question.casefold().split())
    n_ent = len(entities)
    bridge = any(re.search(rf"\b{word}\b", q) for word in _QCA_BRIDGE_CONNECTIVES)
    comparative = bool(_QCA_COMPARATIVE_RE.search(q))
    superlative = bool(_qca_superlative_positions(question))
    wh_start = bool(re.match(r"^(?:what|who|when|where)\b", q))
    factoid_verb = bool(re.search(r"\b(?:is|was|are|were)\b", q))
    entity_alternation = "|".join(
        re.escape(entity) for entity in sorted(entities, key=lambda value: (-len(value), value))
    )
    entity_pattern = rf"(?:{entity_alternation})" if entity_alternation else r"(?!)"

    matches: Dict[str, bool] = {
        # Multi-Hop H01-H14
        "H01": n_ent >= 3,
        "H02": _qca_contains(q, "in which country", "in which city", "in which state", "in which year"),
        "H03": _qca_contains(q, "born in the same", "headquartered in the", "located in the same"),
        "H04": n_ent >= 2 and _qca_contains(q, "who invented", "who founded", "who directed", "who wrote", "who discovered"),
        "H05": _qca_contains(q, "where was") and " born" in q
               or _qca_contains(q, "where did") and " die" in q
               or _qca_contains(q, "where is") and " headquartered" in q,
        "H06": n_ent == 2 and bridge,
        "H07": n_ent >= 2 and bool(
            re.search(rf"\bthe\s+{entity_pattern}\s+of\s+(?:the\s+)?{entity_pattern}\b", q)
        ),
        "H08": bool(re.search(r"\b(?:same\s+\w+\s+as|the same\b.+\bas)\b", q)),
        "H09": _qca_contains(q, "also known as", "formerly known as", "previously called"),
        "H10": bool(
            re.match(
                rf"^what is the\s+.+?\s+of the\s+{entity_pattern}\s+that\b",
                q,
            )
        ),
        "H11": _qca_has_temporal_entity_verb(question, entities),
        "H12": _qca_has_superlative_entity_pattern(question, entities),
        "H13": n_ent == 2 and wh_start and _qca_has_transitive_verb(question),
        "H14": _qca_contains(q, "through which", "via which", "by means of which"),
        # Multi-Aspect A01-A12
        "A01": _qca_starts(q, "compare", "what are the differences between", "contrast"),
        "A02": _qca_contains(q, "advantages and disadvantages", "pros and cons", "benefits and drawbacks"),
        "A03": n_ent >= 2 and (
            bool(re.search(r"\bboth\b.+\band\b", q)) or bool(re.search(r"\beither\b.+\bor\b", q))
        ),
        "A04": _qca_contains(q, "difference between", "similarities between", "relationship between"),
        "A05": _qca_contains(q, "list", "enumerate", "name all", "give examples of"),
        "A06": bool(
            re.search(r"\bhow does\b.+\bcompare to\b", q)
            or re.search(r"\bhow is\b.+\bdifferent from\b", q)
        ),
        "A07": _qca_has_main_plural_aspect(question),
        "A08": _qca_contains(q, "in terms of", "with respect to") or ("regarding" in q and " and " in q),
        "A09": n_ent >= 2 and _qca_contains(q, "as well as", "in addition to"),
        "A10": _qca_has_plural_trigger(question),
        "A11": _qca_contains(q, "what factors", "what aspects", "what characteristics"),
        "A12": (
            "overview of" in q
            or bool(re.search(r"\bdescribe the\s+.+?\s+of\b", q))
            or bool(re.search(r"\bexplain the\s+\w+.+\band the\s+\w+", q))
        ),
        # Simple S01-S12
        "S01": n_ent == 1 and _qca_starts(q, "what is", "who is", "when is"),
        "S02": n_ent == 1 and _qca_starts(q, "when was", "where was", "how many"),
        "S03": n_ent == 1 and _qca_starts(q, "who wrote", "who invented"),
        "S04": _qca_starts(q, "what year", "what date", "in what year", "how old is"),
        "S05": _qca_contains(q, "the capital of", "the population of", "the president of"),
        "S06": n_ent == 0 and factoid_verb,
        "S07": _qca_starts(q, "what is the name of the", "who is the current"),
        "S08": _qca_contains(q, "the birthplace of", "the nationality of", "the height of"),
        "S09": _qca_starts(q, "how tall", "how long", "how far", "how deep", "how wide"),
        "S10": _qca_starts(q, "what language", "what currency", "what color", "what type of"),
        "S11": n_ent == 1 and not bridge and not comparative and not superlative,
        "S12": False,  # filled after hop/aspect indicators are known
    }
    hop_or_aspect = any(matches[f"H{i:02d}"] for i in range(1, 15)) or any(
        matches[f"A{i:02d}"] for i in range(1, 13)
    )
    matches["S12"] = not hop_or_aspect
    return matches


def _qca_split_sub_questions(
    question: str, matched_hop_rules: Optional[Sequence[str]] = None
) -> List[str]:
    """Build the deterministic Appendix A.4 IGFR template sequence."""
    q = question.strip().rstrip("?")
    subs: List[str] = []
    rules = set(matched_hop_rules or ())
    entities = _qca_entity_spans_surface(q)

    # H02--H12/H14: solve the bridge-bearing clause first, then instantiate
    # the remaining predicate online from the documents retrieved so far.
    # Explicit relation/who-action rules take precedence over the generic
    # connective splitter.  In particular, sentence-initial H05 ``Where ...``
    # must not be consumed as an empty-prefix bridge split.
    explicit_template_rules = {
        "H03", "H04", "H05", "H07", "H08", "H09", "H11", "H12"
    }
    bridge_match = (
        None
        if rules.intersection(explicit_template_rules)
        else re.search(
            r"\b(in which country|in which city|in which state|in which year|"
            r"through which|via which|by means of which|that|which|whose|where)\b",
            q,
            re.IGNORECASE,
        )
    )
    if bridge_match:
        prefix = q[:bridge_match.start()].strip(" ,")
        bridge_clause = q[bridge_match.end():].strip(" ,")
        if bridge_clause:
            connective = bridge_match.group(1).casefold()
            if connective == "where" or connective.startswith("in which"):
                subs.append(f"Where {bridge_clause}?")
            else:
                subs.append(f"What {bridge_clause}?")
        if prefix:
            subs.append(f"{prefix} {{BRIDGE}}?")

    # H07 possessive nesting: resolve the inner ``... of ENTITY`` relation,
    # then substitute its answer into the outer predicate.
    if not subs and "H07" in rules:
        possessive = re.match(r"(?P<outer>.+?\bof)\s+(?P<inner>.+)", q, re.IGNORECASE)
        if possessive:
            inner = possessive.group("inner").strip()
            subs.extend([f"What is {inner}?", f"{possessive.group('outer')} {{BRIDGE}}?"])

    # Explicit H03--H05/H08--H12 who-action and relation templates are
    # evaluated before the H01/H13 count-only fallback.
    if not subs and entities and "H03" in rules:
        relation = "born" if "born in the same" in q.casefold() else "located"
        subs.extend(
            [f"Where was {entities[0]} {relation}?", q.replace(entities[0], "{BRIDGE}", 1) + "?"]
        )
    if not subs and entities and "H04" in rules:
        action_match = re.search(
            r"\bwho\s+(invented|founded|directed|wrote|discovered)\b",
            q,
            re.IGNORECASE,
        )
        if action_match:
            action = action_match.group(1)
            subs.extend(
                [f"Who {action} {entities[0]}?", q.replace(entities[0], "{BRIDGE}", 1) + "?"]
            )
    if not subs and entities and "H05" in rules:
        h05_patterns = (
            (r"^where was\s+(?P<subject>.+?)\s+born\b", "Where was {BRIDGE} born?"),
            (r"^where did\s+(?P<subject>.+?)\s+die\b", "Where did {BRIDGE} die?"),
            (
                r"^where is\s+(?P<subject>.+?)\s+headquartered\b",
                "Where is {BRIDGE} headquartered?",
            ),
        )
        for pattern, follow_up in h05_patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match is not None:
                subject_text = match.group("subject").strip(" ,")
                subject = next(
                    (
                        entity
                        for entity in entities
                        if entity.casefold() in subject_text.casefold()
                    ),
                    subject_text,
                )
                subs.extend([f"Who is {subject}?", follow_up])
                break
    if not subs and entities and "H08" in rules:
        subs.extend(
            [f"What is the relevant attribute of {entities[0]}?", q.replace(entities[0], "{BRIDGE}", 1) + "?"]
        )
    if not subs and entities and "H09" in rules:
        subs.extend(
            [f"What is {entities[0]} also known as?", q.replace(entities[0], "{BRIDGE}", 1) + "?"]
        )
    if not subs and entities and "H11" in rules:
        subs.extend(
            [f"What event involved {entities[0]}?", q.replace(entities[0], "{BRIDGE}", 1) + "?"]
        )
    if not subs and entities and "H12" in rules:
        subs.extend(
            [f"Where is {entities[0]}?", q.replace(entities[0], "{BRIDGE}", 1) + "?"]
        )

    # Only H01/H13 use the paper's first-two-NER surface-order fallback.
    if not subs and rules.intersection({"H01", "H13"}) and len(entities) >= 2:
        first, second = entities[:2]
        first_pattern = re.compile(re.escape(first), re.IGNORECASE)
        second_pattern = re.compile(re.escape(second), re.IGNORECASE)
        first_query = second_pattern.sub("", q, count=1)
        first_query = " ".join(first_query.replace("  ", " ").split()).strip(" ,")
        second_query = first_pattern.sub("{BRIDGE}", q, count=1).strip(" ,")
        if first_query:
            subs.append(first_query + "?")
        if second_query:
            subs.append(second_query + "?")

    # Preserve order while removing duplicates.
    unique: List[str] = []
    for sub in subs or [question]:
        if sub not in unique:
            unique.append(sub)
    return unique


class QuestionComplexityAssessor:
    """
    Stage 1 — QCA: 38 条加权表层规则将问题分类为 Simple / Multi-aspect / Multi-hop。

    论文公式: c(q) = Σ w_i·m_i(q) / Σ w_i
    其中 m_i(q) ∈ {0, 1} 表示规则 i 是否匹配，w_i 为规则权重。
    c(q) is a logged heuristic summary only; routing uses rule precedence.
    """

    def assess(self, question: str) -> QCAResult:
        q = question.strip()
        entities = _qca_entities(q)
        matches = _qca_match_rules(q, entities)
        fired = tuple(rule_id for rule_id in _QCA_RULE_WEIGHTS if matches[rule_id])
        hop_fired = [rule_id for rule_id in fired if rule_id.startswith("H")]
        aspect_fired = [rule_id for rule_id in fired if rule_id.startswith("A")]
        # The original routing contract gives every hop rule precedence over
        # aspect rules, but accepts a hop route only when at least two named
        # entities are present.  The entity guard prevents a lexical hop cue
        # by itself from fabricating a multi-document dependency.
        if hop_fired and len(entities) >= 2:
            qtype = QuestionType.MULTI_HOP
        elif aspect_fired:
            qtype = QuestionType.MULTI_ASPECT
        else:
            qtype = QuestionType.SIMPLE

        conf = sum(_QCA_RULE_WEIGHTS[rule_id] for rule_id in fired) / _QCA_TOTAL_WEIGHT
        if qtype == QuestionType.MULTI_HOP:
            hop = min(4, max(2, len(entities) - 1))
            sub_qs = _qca_split_sub_questions(q, hop_fired)
        elif qtype == QuestionType.MULTI_ASPECT:
            hop = 2
            sub_qs = [q]
        else:
            hop = 1
            sub_qs = [q]

        return QCAResult(
            question=q,
            question_type=qtype,
            confidence=float(conf),
            hop_count=hop,
            entity_count=len(entities),
            sub_questions=sub_qs,
            matched_rules=fired,
            reasoning=f"rule-based({qtype.value};rules={','.join(fired)};conf={conf:.4f})",
        )

    def assess_batch(self, questions: List[str]) -> List[QCAResult]:
        return [self.assess(q) for q in questions]


# ── Stage 2: Adaptive Hybrid Retrieval (AHR) ─────────────────────────────────

# 论文 Table 1 / Eq. 所在段落：类型条件化的 BM25/dense 插值权重
# Simple: (0.75, 0.25) — 侧重词汇匹配
# Multi-Aspect: (0.30, 0.70) — 侧重语义覆盖
# Multi-Hop: (0.25, 0.75) — 侧重语义桥接
_AHR_DEFAULT_WEIGHTS: Dict[QuestionType, Tuple[float, float]] = {
    QuestionType.SIMPLE:       (0.75, 0.25),
    QuestionType.MULTI_ASPECT: (0.30, 0.70),
    QuestionType.MULTI_HOP:    (0.25, 0.75),
}


def _ahr_get_weights(qca_result: QCAResult) -> Tuple[float, float]:
    """Interpolate continuously from balanced to type-conditioned retrieval.

    The paper says that ``c(q)`` drives AHR and that low-confidence routing
    falls back to ``(0.5, 0.5)``, but does not introduce a threshold.  A
    continuous interpolation is therefore the least-assumptive executable
    reading: c=0 is exactly balanced and c=1 is exactly the type default.
    """
    confidence = min(1.0, max(0.0, float(qca_result.confidence)))
    target_bm25, target_dense = _AHR_DEFAULT_WEIGHTS.get(
        qca_result.question_type, (0.5, 0.5)
    )
    bm25 = (1.0 - confidence) * 0.5 + confidence * target_bm25
    dense = (1.0 - confidence) * 0.5 + confidence * target_dense
    return bm25, dense


def _ahr_tokenize(text: str) -> List[str]:
    return re.sub(r"[" + string.punctuation + r"]", " ", text.lower()).split()


class _BM25Index:
    """轻量级原生 BM25 索引，无外部依赖。"""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs: List[str] = []
        self._toks: List[List[str]] = []
        self._idf:  Dict[str, float] = {}
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0

    def build(self, docs: List[str]) -> "_BM25Index":
        self._docs = docs
        self._toks = [_ahr_tokenize(d) for d in docs]
        self._doc_lens = [len(tokens) for tokens in self._toks]
        n = len(docs)
        self._avgdl = sum(len(t) for t in self._toks) / max(n, 1)
        df: Dict[str, int] = {}
        postings: Dict[str, List[Tuple[int, int]]] = {}
        for doc_index, toks in enumerate(self._toks):
            counts = Counter(toks)
            for term, frequency in counts.items():
                df[term] = df.get(term, 0) + 1
                postings.setdefault(term, []).append((doc_index, frequency))
        self._idf = {t: math.log((n - cnt + 0.5) / (cnt + 0.5) + 1) for t, cnt in df.items()}
        self._postings = postings
        self._toks = []  # postings/doc lengths are sufficient for search
        return self

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        q_toks = _ahr_tokenize(query)
        scores = np.zeros(len(self._docs))
        k1, b, avgdl = self.k1, self.b, self._avgdl
        for term in q_toks:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, tf in self._postings.get(term, ()):
                dl = self._doc_lens[i]
                scores[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
        candidate_count = min(max(int(top_k), 1), len(scores))
        all_indices = np.arange(len(scores))
        if candidate_count >= len(scores):
            top = all_indices[np.lexsort((all_indices, -scores))]
        else:
            candidate_positions = np.argpartition(-scores, candidate_count - 1)[
                :candidate_count
            ]
            cutoff = scores[candidate_positions].min()
            above = all_indices[scores > cutoff]
            tied = all_indices[scores == cutoff]
            selected = np.concatenate(
                (above, tied[: candidate_count - above.size])
            )
            top = selected[np.lexsort((selected, -scores[selected]))]
        return [(int(i), float(scores[i])) for i in top[:candidate_count]]


@dataclass
class _RetrievedDoc:
    doc_id:       str
    text:         str
    corpus_index: int = -1
    bm25_score:   float = 0.0
    dense_score:  float = 0.0
    hybrid_score: float = 0.0
    from_second_round: bool = False


@dataclass(frozen=True)
class OraclePoolRecord:
    """Auditable construction record for one fixed Oracle candidate pool."""

    base_indices: Tuple[int, ...]
    gold_indices: Tuple[int, ...]
    injected_indices: Tuple[int, ...]
    evicted_indices: Tuple[int, ...]
    pool_indices: Tuple[int, ...]
    pool_sha256: str
    base_page_ids: Tuple[str, ...] = ()
    gold_page_ids: Tuple[str, ...] = ()
    injected_page_ids: Tuple[str, ...] = ()
    evicted_page_ids: Tuple[str, ...] = ()
    pool_page_ids: Tuple[str, ...] = ()
    protocol: str = ORACLE_TOP100_PROTOCOL


def _page_deduplicate_ranked_indices(
    ranked_indices: Sequence[int],
    corpus_page_ids: Sequence[str],
    *,
    limit: Optional[int] = None,
) -> Tuple[int, ...]:
    """Retain the first (highest-ranked) corpus occurrence of each page ID."""
    if limit is not None and limit <= 0:
        raise ValueError("page-deduplication limit must be positive")
    result: List[int] = []
    seen_pages: set[str] = set()
    corpus_size = len(corpus_page_ids)
    for raw_index in ranked_indices:
        if isinstance(raw_index, bool):
            raise ValueError("ranked corpus indices must be integers")
        index = int(raw_index)
        if index < 0 or index >= corpus_size:
            raise ValueError("ranked corpus index is outside corpus_page_ids")
        page_id = corpus_page_ids[index]
        if not isinstance(page_id, str) or not page_id.strip():
            raise ValueError("corpus_page_ids must contain non-empty strings")
        if page_id in seen_pages:
            continue
        seen_pages.add(page_id)
        result.append(index)
        if limit is not None and len(result) == limit:
            break
    return tuple(result)


def _construct_oracle_top100_indices(
    base_indices: Sequence[int],
    gold_indices: Sequence[int],
    *,
    corpus_page_ids: Optional[Sequence[str]] = None,
    pool_size: int = 100,
) -> OraclePoolRecord:
    """Page-deduplicate BGE candidates and tail-inject missing gold pages.

    The highest-ranked passage occurrence represents each page. The BGE order
    is preserved for every retained page. Missing gold-page representatives are
    appended in annotation order, while the same number of lowest-ranked
    non-gold pages are evicted. MADS is therefore the first stage allowed to
    rank injected positives.
    """
    if pool_size <= 0:
        raise ValueError("Oracle pool_size must be positive")
    raw_base = tuple(int(index) for index in base_indices)
    raw_gold = tuple(int(index) for index in gold_indices)
    if not raw_gold:
        raise ValueError("Oracle gold indices must be non-empty")
    if any(index < 0 for index in (*raw_base, *raw_gold)):
        raise ValueError("Oracle corpus indices must be non-negative")
    if corpus_page_ids is None:
        largest_index = max((*raw_base, *raw_gold), default=-1)
        page_ids = tuple(str(index) for index in range(largest_index + 1))
    else:
        page_ids = tuple(corpus_page_ids)
    base = _page_deduplicate_ranked_indices(
        raw_base, page_ids, limit=pool_size
    )
    gold = _page_deduplicate_ranked_indices(raw_gold, page_ids)
    if len(base) != pool_size:
        raise ValueError(
            f"Oracle requires {pool_size} page-unique BGE candidates, got {len(base)}"
        )
    if len(gold) > pool_size:
        raise ValueError("Oracle cannot place more positive pages than its pool size")

    base_pages = tuple(page_ids[index] for index in base)
    gold_pages = tuple(page_ids[index] for index in gold)
    gold_page_set = set(gold_pages)
    base_page_set = set(base_pages)
    injected = tuple(
        index for index in gold if page_ids[index] not in base_page_set
    )
    eviction_worst_first = [
        index for index in reversed(base) if page_ids[index] not in gold_page_set
    ][: len(injected)]
    if len(eviction_worst_first) != len(injected):
        raise ValueError("Oracle pool has insufficient non-positive eviction slots")
    evicted_set = set(eviction_worst_first)
    # Record evictions in original BGE rank order; the selection itself is from
    # the tail, as made explicit by ``eviction_worst_first`` above.
    evicted = tuple(index for index in base if index in evicted_set)
    pool = tuple(index for index in base if index not in evicted_set) + injected
    pool_pages = tuple(page_ids[index] for index in pool)
    if (
        len(pool) != pool_size
        or len(set(pool_pages)) != pool_size
        or not gold_page_set.issubset(pool_pages)
    ):
        raise RuntimeError("Oracle pool construction violated its fixed-size contract")
    digest_payload = {
        "protocol": ORACLE_TOP100_PROTOCOL,
        "pool_size": pool_size,
        "pool_indices": list(pool),
        "pool_page_ids": list(pool_pages),
    }
    pool_sha256 = hashlib.sha256(
        json.dumps(
            digest_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return OraclePoolRecord(
        base_indices=base,
        gold_indices=gold,
        injected_indices=injected,
        evicted_indices=evicted,
        pool_indices=pool,
        pool_sha256=pool_sha256,
        base_page_ids=base_pages,
        gold_page_ids=gold_pages,
        injected_page_ids=tuple(page_ids[index] for index in injected),
        evicted_page_ids=tuple(page_ids[index] for index in evicted),
        pool_page_ids=pool_pages,
    )


def _merge_bounded_retrieval_pool(
    existing: Sequence[_RetrievedDoc],
    additions: Sequence[_RetrievedDoc],
    limit: int,
) -> List[_RetrievedDoc]:
    """Stable ID union capped to the paper's fixed secondary-pool size."""
    if limit <= 0:
        raise ValueError("retrieval pool limit must be positive")
    merged: Dict[str, _RetrievedDoc] = {doc.doc_id: doc for doc in existing}
    for doc in additions:
        current = merged.get(doc.doc_id)
        if current is None or doc.hybrid_score > current.hybrid_score:
            merged[doc.doc_id] = doc
    return sorted(
        merged.values(),
        key=lambda doc: (-doc.hybrid_score, doc.corpus_index, doc.doc_id),
    )[:limit]


def _chunked_inner_product_topk(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    top_k: int,
    chunk_size: int = 65_536,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact top-k inner-product search without materializing a B x corpus matrix."""
    if queries.dim() == 1:
        queries = queries.unsqueeze(0)
    if queries.dim() != 2 or corpus.dim() != 2:
        raise ValueError("queries and corpus must both be 2-D matrices")
    if queries.size(1) != corpus.size(1):
        raise ValueError("query and corpus embedding dimensions do not match")
    k = min(max(int(top_k), 1), corpus.size(0))
    best_values = torch.empty((queries.size(0), 0), dtype=torch.float32)
    best_indices = torch.empty((queries.size(0), 0), dtype=torch.long)

    def deterministic_topk(
        values: torch.Tensor,
        indices: torch.Tensor,
        keep: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select exact top-k, breaking only cutoff ties by smaller corpus ID."""
        if keep <= 0:
            return values[:, :0], indices[:, :0]
        # ``topk`` finds the exact score threshold in linear-selection time.  We
        # then inspect only scores above/at that threshold so a score tie cannot
        # inherit the backend-dependent ordering of torch.topk.
        cutoffs = torch.topk(values, k=keep, dim=1, largest=True, sorted=False).values.min(
            dim=1
        ).values
        selected_values: List[torch.Tensor] = []
        selected_indices: List[torch.Tensor] = []
        for row in range(values.size(0)):
            row_values = values[row]
            row_indices = indices[row]
            above_positions = torch.nonzero(
                row_values > cutoffs[row], as_tuple=False
            ).flatten()
            remaining = keep - above_positions.numel()
            tied_positions = torch.nonzero(
                row_values == cutoffs[row], as_tuple=False
            ).flatten()
            if remaining < 0 or tied_positions.numel() < remaining:
                raise RuntimeError("invalid dense top-k cutoff partition")
            if tied_positions.numel() > remaining:
                tied_indices = row_indices[tied_positions]
                lowest_tie_positions = torch.topk(
                    tied_indices,
                    k=remaining,
                    largest=False,
                    sorted=False,
                ).indices
                tied_positions = tied_positions[lowest_tie_positions]
            chosen_positions = torch.cat((above_positions, tied_positions[:remaining]))
            chosen_values = row_values[chosen_positions]
            chosen_indices = row_indices[chosen_positions]
            # Sort only the retained k values, not every value in the corpus
            # chunk: descending score, then ascending corpus index.
            by_index = torch.argsort(chosen_indices, stable=True)
            chosen_values = chosen_values[by_index]
            chosen_indices = chosen_indices[by_index]
            by_score = torch.argsort(chosen_values, descending=True, stable=True)
            selected_values.append(chosen_values[by_score])
            selected_indices.append(chosen_indices[by_score])
        return torch.stack(selected_values), torch.stack(selected_indices)

    for start in range(0, corpus.size(0), chunk_size):
        stop = min(start + chunk_size, corpus.size(0))
        scores = queries @ corpus[start:stop].T
        indices = torch.arange(start, stop, dtype=torch.long).expand(queries.size(0), -1)
        candidate_values = torch.cat((best_values, scores), dim=1)
        candidate_indices = torch.cat((best_indices, indices), dim=1)
        keep = min(k, candidate_values.size(1))
        best_values, best_indices = deterministic_topk(
            candidate_values,
            candidate_indices,
            keep,
        )
    return best_values, best_indices


def _chunked_inner_product_topk_unique_pages(
    queries: torch.Tensor,
    corpus: torch.Tensor,
    corpus_page_ids: Sequence[str],
    unique_page_count: int,
    chunk_size: int = 65_536,
) -> torch.Tensor:
    """Return exact dense ranks after retaining each page's first occurrence.

    The candidate depth grows deterministically until the globally ranked
    prefix contains the requested number of distinct pages. Because every
    omitted passage ranks below that prefix, the retained page representatives
    are the exact first ``unique_page_count`` pages in full-corpus BGE order.
    """
    if unique_page_count <= 0:
        raise ValueError("unique_page_count must be positive")
    if corpus.size(0) != len(corpus_page_ids):
        raise ValueError("corpus_page_ids must align one-to-one with corpus rows")
    if any(
        not isinstance(page_id, str) or not page_id.strip()
        for page_id in corpus_page_ids
    ):
        raise ValueError("corpus_page_ids must contain non-empty strings")
    candidate_count = min(max(unique_page_count, 1), corpus.size(0))
    while True:
        _, ranked_indices = _chunked_inner_product_topk(
            queries, corpus, candidate_count, chunk_size=chunk_size
        )
        page_unique_rows = [
            _page_deduplicate_ranked_indices(
                row.tolist(), corpus_page_ids, limit=unique_page_count
            )
            for row in ranked_indices
        ]
        if all(len(row) == unique_page_count for row in page_unique_rows):
            return torch.tensor(page_unique_rows, dtype=torch.long)
        if candidate_count == corpus.size(0):
            raise ValueError(
                f"corpus contains fewer than {unique_page_count} unique page IDs"
            )
        candidate_count = min(corpus.size(0), candidate_count * 2)


_DENSE_UNIT_NORM_CHECK_CHUNK_SIZE = 65_536
_DENSE_UNIT_NORM_ATOL = 8 * torch.finfo(torch.float32).eps


def _tensor_is_finite_in_chunks(
    tensor: torch.Tensor,
    *,
    max_chunk_elements: int = 4_194_304,
) -> bool:
    """Validate a large tensor without allocating a full-size boolean mask."""
    if max_chunk_elements <= 0:
        raise ValueError("max_chunk_elements must be positive")
    if tensor.dim() == 0:
        return bool(torch.isfinite(tensor).item())
    if tensor.numel() == 0:
        return True
    elements_per_row = max(int(tensor[0].numel()), 1)
    rows_per_chunk = max(max_chunk_elements // elements_per_row, 1)
    for start in range(0, tensor.size(0), rows_per_chunk):
        if not torch.isfinite(tensor[start : start + rows_per_chunk]).all().item():
            return False
    return True


def _can_reuse_normalized_dense_embeddings(
    embeddings: torch.Tensor,
    *,
    chunk_size: int = _DENSE_UNIT_NORM_CHECK_CHUNK_SIZE,
) -> bool:
    """Check the zero-copy corpus-embedding contract with bounded scratch space."""
    if (
        embeddings.layout != torch.strided
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.dim() != 2
        or not embeddings.is_contiguous()
    ):
        return False
    if chunk_size <= 0:
        raise ValueError("dense-embedding norm-check chunk_size must be positive")

    # BGE artifacts are normalized in float32.  Recomputing their norms can
    # differ from one by a few float32 ULPs, so accept only that rounding-sized
    # error.  Larger deviations retain the historical normalize-and-copy path.
    for start in range(0, embeddings.size(0), chunk_size):
        norms = torch.linalg.vector_norm(
            embeddings[start : start + chunk_size], dim=1
        )
        if not torch.isfinite(norms).all().item():
            return False
        if torch.any((norms - 1.0).abs() > _DENSE_UNIT_NORM_ATOL).item():
            return False
    return True


def _prepare_dense_corpus_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Reuse a validated BGE matrix, otherwise preserve normalized-copy behavior."""
    detached = embeddings.detach()
    if _can_reuse_normalized_dense_embeddings(detached):
        return detached
    return F.normalize(
        detached.float().cpu(), dim=-1, eps=ARIA_NUMERICAL_EPSILON
    ).contiguous()


class _AdaptiveHybridRetriever:
    """AHR: BM25 + 密集向量混合检索，权重由 QCA 结果动态决定。"""

    def __init__(self, bm25: _BM25Index, corpus_docs: List[str],
                 corpus_ids: List[str],
                 dense_embeddings: Optional[torch.Tensor] = None,
                 corpus_page_ids: Optional[List[str]] = None):
        self.bm25            = bm25
        self.corpus_docs     = corpus_docs
        self.corpus_ids      = corpus_ids
        self.corpus_page_ids = (
            list(corpus_page_ids) if corpus_page_ids is not None else list(corpus_ids)
        )
        self.dense_embeddings = (
            _prepare_dense_corpus_embeddings(dense_embeddings)
            if dense_embeddings is not None else None
        )
        # Prebuilt O(1) reverse index for deterministic document-ID lookup.
        self._id_to_idx: Dict[str, int] = {did: i for i, did in enumerate(corpus_ids)}
        if len(self._id_to_idx) != len(corpus_ids):
            raise ValueError("corpus_doc_ids must be unique")
        if len(self.corpus_page_ids) != len(corpus_docs):
            raise ValueError("corpus_page_ids must align one-to-one with corpus_docs")
        if any(
            not isinstance(page_id, str) or not page_id.strip()
            for page_id in self.corpus_page_ids
        ):
            raise ValueError("corpus_page_ids must contain non-empty strings")
        self.unique_page_count = len(set(self.corpus_page_ids))
        if dense_embeddings is not None and dense_embeddings.size(0) != len(corpus_docs):
            raise ValueError(
                "doc_embeddings first dimension must match corpus_docs: "
                f"{dense_embeddings.size(0)} != {len(corpus_docs)}"
            )

    @staticmethod
    def _normalize(vals: List[float]) -> List[float]:
        if not vals:
            return vals
        mn, mx = min(vals), max(vals)
        span = mx - mn
        return [0.5] * len(vals) if span <= 1e-6 else [(v - mn) / span for v in vals]

    def retrieve(self, query: str, qca_result: QCAResult,
                 query_emb: Optional[torch.Tensor] = None,
                 top_k: int = 20) -> List[_RetrievedDoc]:
        query_embeddings = (
            query_emb.detach().reshape(1, -1) if query_emb is not None else None
        )
        return self.retrieve_batch(
            [query],
            [qca_result],
            query_embeddings=query_embeddings,
            top_k=top_k,
        )[0]

    def retrieve_batch(
        self,
        queries: Sequence[str],
        qca_results: Sequence[QCAResult],
        query_embeddings: Optional[torch.Tensor] = None,
        top_k: int = 20,
    ) -> List[List[_RetrievedDoc]]:
        """Run one exact dense scan for a query batch, with scalar BM25 arms."""
        if not queries or len(queries) != len(qca_results):
            raise ValueError("queries and qca_results must be non-empty and aligned")
        arm_k = min(max(int(top_k), 1), len(self.corpus_docs))

        bm25_maps: List[Dict[str, float]] = []
        for query in queries:
            raw_bm25 = self.bm25.search(query, top_k=arm_k)
            normalized = self._normalize([score for _, score in raw_bm25])
            bm25_maps.append({
                self.corpus_ids[index]: score
                for (index, _), score in zip(raw_bm25, normalized)
            })

        dense_maps: List[Dict[str, float]] = [{} for _ in queries]
        if self.dense_embeddings is not None and query_embeddings is not None:
            dense_queries = query_embeddings.detach().float().cpu()
            if dense_queries.dim() == 1:
                dense_queries = dense_queries.unsqueeze(0)
            if dense_queries.dim() != 2 or dense_queries.size(0) != len(queries):
                raise ValueError("query_embeddings must have shape (batch, dense_dim)")
            if dense_queries.size(1) != self.dense_embeddings.size(1):
                raise ValueError(
                    "AHR dense-query dimension mismatch; pass W_BGE(QR(q)): "
                    f"query={dense_queries.size(1)}, corpus={self.dense_embeddings.size(1)}"
                )
            values, indices = _chunked_inner_product_topk(
                F.normalize(
                    dense_queries, dim=-1, eps=ARIA_NUMERICAL_EPSILON
                ),
                self.dense_embeddings,
                arm_k,
            )
            for row in range(len(queries)):
                raw_dense = [
                    (self.corpus_ids[index], float(score))
                    for score, index in zip(
                        values[row].tolist(), indices[row].tolist()
                    )
                ]
                normalized = self._normalize([score for _, score in raw_dense])
                dense_maps[row] = {
                    doc_id: score
                    for (doc_id, _), score in zip(raw_dense, normalized)
                }

        batch_results: List[List[_RetrievedDoc]] = []
        for qca_result, bm25_map, dense_map in zip(
            qca_results, bm25_maps, dense_maps
        ):
            bm25_weight, dense_weight = _ahr_get_weights(qca_result)
            all_ids = set(bm25_map) | set(dense_map)
            bm25_minimum = min(bm25_map.values()) if bm25_map else 0.5
            dense_minimum = min(dense_map.values()) if dense_map else 0.5
            hybrid = {
                doc_id: (
                    bm25_weight * bm25_map.get(doc_id, bm25_minimum)
                    + dense_weight * dense_map.get(doc_id, dense_minimum)
                )
                for doc_id in all_ids
            }
            top_ids = sorted(
                hybrid,
                key=lambda doc_id: (
                    -hybrid[doc_id], self._id_to_idx[doc_id], doc_id
                ),
            )[:top_k]
            batch_results.append([
                _RetrievedDoc(
                    doc_id=doc_id,
                    text=self.corpus_docs[self._id_to_idx[doc_id]],
                    corpus_index=self._id_to_idx[doc_id],
                    bm25_score=bm25_map.get(doc_id, bm25_minimum),
                    dense_score=dense_map.get(doc_id, dense_minimum),
                    hybrid_score=hybrid[doc_id],
                )
                for doc_id in top_ids
            ])
        return batch_results


# ── Stage 4: Multi-Agent Document Scoring (MADS) ─────────────────────────────

class _LexicalAgent:
    """MADS lexical agent: candidate-set TF-IDF cosine similarity."""
    def score(self, query: str, docs: List[str]) -> List[float]:
        q_toks = _ahr_tokenize(query)
        doc_toks = [_ahr_tokenize(d) for d in docs]
        if not docs:
            return []

        # The vectorizer is fitted jointly to the query and current candidate
        # pool, as specified in Appendix A.4.
        fitted_rows = [q_toks] + doc_toks
        df = Counter(term for toks in fitted_rows for term in set(toks))
        n_docs = len(fitted_rows)
        idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

        def tfidf(tokens: List[str]) -> Dict[str, float]:
            counts = Counter(tokens)
            return {term: float(count) * idf.get(term, 0.0) for term, count in counts.items()}

        q_vec = tfidf(q_toks)
        q_norm = math.sqrt(sum(value * value for value in q_vec.values()))
        scores: List[float] = []
        for tokens in doc_toks:
            d_vec = tfidf(tokens)
            d_norm = math.sqrt(sum(value * value for value in d_vec.values()))
            dot = sum(value * d_vec.get(term, 0.0) for term, value in q_vec.items())
            scores.append(dot / max(q_norm * d_norm, 1e-12))
        return scores


# Module-level cache avoids loading the frozen semantic expert repeatedly while
# keeping distinct Hugging Face revisions isolated.
_SEMANTIC_MODEL_CACHE: Dict[Tuple[str, Optional[str], str], Any] = {}


class _SemanticAgent:
    """Frozen BGE document encoder for corpus-external MADS documents."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        model_revision: Optional[str] = None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("MADS semantic model name must be non-empty")
        if model_revision is not None and (
            not isinstance(model_revision, str) or not model_revision.strip()
        ):
            raise ValueError("MADS semantic model revision must be non-empty when set")
        self.model_name = model_name
        self.model_revision = model_revision
        self.resolved_revision: Optional[str] = None
        self._tok = self._enc = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _lazy_load(self):
        if self._tok is not None:
            return
        cache_key = (self.model_name, self.model_revision, self._device)
        if cache_key in _SEMANTIC_MODEL_CACHE:
            self._tok, self._enc, self.resolved_revision = _SEMANTIC_MODEL_CACHE[cache_key]
            return
        try:
            from transformers import AutoTokenizer as AT, AutoModel as AM
            load_kwargs = (
                {"revision": self.model_revision}
                if self.model_revision is not None
                else {}
            )
            self._tok = AT.from_pretrained(self.model_name, **load_kwargs)
            self._enc = AM.from_pretrained(self.model_name, **load_kwargs).to(self._device).eval()
            commit_candidates = {
                value.lower()
                for value in (
                    getattr(getattr(self._enc, "config", None), "_commit_hash", None),
                    getattr(self._tok, "init_kwargs", {}).get("_commit_hash")
                    if isinstance(getattr(self._tok, "init_kwargs", None), Mapping)
                    else None,
                )
                if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value)
            }
            if len(commit_candidates) > 1:
                raise RuntimeError(
                    "MADS tokenizer and encoder resolved to different Hub commits: "
                    + ", ".join(sorted(commit_candidates))
                )
            self.resolved_revision = (
                next(iter(commit_candidates)) if commit_candidates else None
            )
            if self.model_revision is not None and re.fullmatch(
                r"[0-9a-fA-F]{40}", self.model_revision
            ):
                declared_commit = self.model_revision.lower()
                if (
                    self.resolved_revision is not None
                    and self.resolved_revision != declared_commit
                ):
                    raise RuntimeError(
                        f"Requested MADS commit {declared_commit} but loaded "
                        f"{self.resolved_revision}"
                    )
                # An exact caller-supplied commit is itself a resolved immutable
                # identifier even if an older Transformers config omits
                # `_commit_hash` after a successful load.
                self.resolved_revision = declared_commit
            _SEMANTIC_MODEL_CACHE[cache_key] = (
                self._tok,
                self._enc,
                self.resolved_revision,
            )
        except Exception as exc:
            revision = (
                f" at revision {self.model_revision!r}"
                if self.model_revision is not None
                else ""
            )
            raise RuntimeError(
                f"MADS semantic agent requires the frozen BGE checkpoint "
                f"{self.model_name!r}{revision}"
            ) from exc

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode passages exactly as BGE-large-en-v1.5 corpus artifacts do."""
        self._lazy_load()
        if self._enc is None:
            raise RuntimeError("MADS BGE document encoder is unavailable")
        embs = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            enc = self._tok(batch, return_tensors="pt", truncation=True,
                            max_length=768, padding=True).to(self._device)
            out = self._enc(**enc)
            # The official BGE v1.5 SentenceTransformer configuration uses
            # CLS pooling.  Keeping this fallback identical to the offline
            # corpus builder prevents external passages from entering a
            # different semantic space.
            embs.append(out.last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def score_projected(
        self, projected_query: torch.Tensor, docs: Sequence[str]
    ) -> List[float]:
        """Score BGE document vectors against the fixed W_BGE(QR) query."""
        if not docs:
            return []
        query = F.normalize(
            projected_query.detach().float().cpu().reshape(1, -1),
            dim=-1,
            eps=ARIA_NUMERICAL_EPSILON,
        )
        documents = torch.from_numpy(self._embed(list(docs))).float()
        documents = F.normalize(
            documents, dim=-1, eps=ARIA_NUMERICAL_EPSILON
        )
        if documents.size(1) != query.size(1):
            raise ValueError("MADS BGE query/document dimensions do not match")
        return (documents @ query.T).squeeze(1).tolist()

class _EntityAgent:
    """专家3: 命名实体覆盖率评分。"""
    def __init__(self, cache_max_entries: int = 50_000):
        if cache_max_entries <= 0:
            raise ValueError("entity cache size must be positive")
        self._nlp = None
        self._cache_max_entries = int(cache_max_entries)
        self._document_cache: "OrderedDict[str, Tuple[str, set]]" = OrderedDict()

    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = _qca_get_spacy()
        return self._nlp

    def _entities(self, text: str) -> set:
        nlp = self._get_nlp()
        if nlp:
            try:
                return {e.text.lower().strip() for e in nlp(text).ents}
            except Exception:
                pass
        raise RuntimeError("spaCy NER returned no pipeline")

    def _document_entities(
        self,
        docs: Sequence[str],
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[set]:
        """Batch missing spaCy parses and retain only a bounded LRU cache."""
        if doc_ids is not None and len(doc_ids) != len(docs):
            raise ValueError("doc_ids must align one-to-one with docs")
        keys = list(doc_ids) if doc_ids is not None else list(docs)
        results: List[Optional[set]] = [None] * len(docs)
        missing: "OrderedDict[str, Tuple[str, List[int]]]" = OrderedDict()
        for index, (key, text) in enumerate(zip(keys, docs)):
            cached = self._document_cache.get(key)
            if cached is not None and cached[0] == text:
                self._document_cache.move_to_end(key)
                results[index] = cached[1]
                continue
            if cached is not None:
                del self._document_cache[key]
            if key not in missing:
                missing[key] = (text, [])
            elif missing[key][0] != text:
                raise ValueError("one document ID cannot refer to multiple texts")
            missing[key][1].append(index)

        if missing:
            nlp = self._get_nlp()
            texts = [value[0] for value in missing.values()]
            try:
                parsed = list(nlp.pipe(texts, batch_size=128))
            except Exception:
                try:
                    parsed = [nlp(text) for text in texts]
                except Exception as exc:
                    raise RuntimeError("spaCy NER returned no pipeline") from exc
            if len(parsed) != len(texts):
                raise RuntimeError("spaCy NER returned an incomplete batch")
            for (key, (_, positions)), parsed_doc in zip(missing.items(), parsed):
                entities = {
                    entity.text.casefold().strip()
                    for entity in parsed_doc.ents
                    if entity.text.strip()
                }
                self._document_cache[key] = (missing[key][0], entities)
                self._document_cache.move_to_end(key)
                while len(self._document_cache) > self._cache_max_entries:
                    self._document_cache.popitem(last=False)
                for index in positions:
                    results[index] = entities

        if any(value is None for value in results):
            raise RuntimeError("spaCy NER did not return every requested document")
        return [value for value in results if value is not None]

    def score(
        self,
        query: str,
        docs: List[str],
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[float]:
        q_ents = self._entities(query)
        if not q_ents:
            return [0.0] * len(docs)
        document_entities = self._document_entities(docs, doc_ids=doc_ids)
        return [len(q_ents & entities) / len(q_ents) for entities in document_entities]


@dataclass
class _ScoredDoc:
    doc_id:        str
    text:          str
    corpus_index:  int = -1
    lex_score:     float = 0.0
    sem_score:     float = 0.0
    ent_score:     float = 0.0
    sem_raw:       float = 0.0
    sem_min:       float = 0.0
    sem_span:      float = 1.0
    mads_score:    float = 0.0
    fused_score:   float = 0.0
    confidence:    float = 0.0   # inter-agent agreement
    is_filtered:   bool  = False
    hybrid_score:  float = 0.0   # from AHR stage
    from_second_round: bool = False


# ── Stage 5: Confidence-Calibrated Evidence Fusion (CCEF) ────────────────────

class _CCEF:
    """
    MADS+CCEF 合并执行: 三专家打分 → 方差衡量共识度 → 折扣融合 → 低置信过滤。
    """
    def __init__(self,
                 weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
                 discount_alpha: float = 0.5,      # 论文 α=0.5 (Eq. 5)
                 filter_threshold: float = 0.30,
                 entity_cache_max_entries: int = 50_000,
                 semantic_model_name: str = "BAAI/bge-large-en-v1.5",
                 semantic_model_revision: Optional[str] = None):   # 论文 τ=0.30 (line 769)
        self.lex     = _LexicalAgent()
        self.sem     = _SemanticAgent(semantic_model_name, semantic_model_revision)
        self.ent     = _EntityAgent(cache_max_entries=entity_cache_max_entries)
        self.weights = weights
        self.alpha   = discount_alpha
        self.thresh  = filter_threshold

    @staticmethod
    def _minmax(values: Sequence[float], eps: float = 1e-6) -> List[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        span = hi - lo
        if span <= eps:
            return [0.5] * len(values)
        return ((arr - lo) / span).tolist()

    def score_and_fuse(
        self,
        query: str,
        retrieved: List[_RetrievedDoc],
        semantic_scores: Optional[Sequence[float]] = None,
    ) -> List[_ScoredDoc]:
        if not retrieved:
            return []
        texts = [d.text for d in retrieved]
        # The paper requires independent per-query min-max normalization for
        # every MADS agent before equal-weight fusion.
        raw_ls = self.lex.score(query, texts)
        if semantic_scores is None or len(semantic_scores) != len(retrieved):
            raise ValueError(
                "MADS semantic scoring requires W_BGE(QR) against BGE document vectors"
            )
        raw_ss = list(semantic_scores)
        raw_es = self.ent.score(
            query,
            texts,
            doc_ids=[document.doc_id for document in retrieved],
        )
        ls = self._minmax(raw_ls)
        ss = self._minmax(raw_ss)
        es = self._minmax(raw_es)
        sem_min = min(raw_ss) if raw_ss else 0.0
        sem_span = (max(raw_ss) - sem_min) if raw_ss else 1.0

        wl, ws, we = self.weights
        results = []
        for i, rd in enumerate(retrieved):
            arr = np.array([ls[i], ss[i], es[i]])
            weighted_mean = float(wl * ls[i] + ws * ss[i] + we * es[i])
            std  = float(arr.std())
            mean = float(arr.mean())
            agreement = min(max(1.0 - std / (mean + 1e-6), 0.0), 1.0)
            discount  = self.alpha + (1 - self.alpha) * agreement
            fused = weighted_mean * discount
            results.append(_ScoredDoc(
                doc_id=rd.doc_id, text=rd.text, corpus_index=rd.corpus_index,
                lex_score=ls[i], sem_score=ss[i], ent_score=es[i],
                sem_raw=float(raw_ss[i]), sem_min=float(sem_min), sem_span=float(sem_span),
                mads_score=weighted_mean,
                fused_score=fused, confidence=agreement,
                is_filtered=(fused < self.thresh),
                hybrid_score=rd.hybrid_score,
                from_second_round=rd.from_second_round,
            ))
        return results


# ── Stage 3: Iterative Gap-Filling Retrieval (IGFR) ──────────────────────────

# IGFR and QCA share the machine-readable entity inventory.
_IGFR_TARGET_CATS = set(_QCA_ENTITY_TYPES)


def _igfr_entities(text: str) -> set:
    nlp = _qca_get_spacy()
    if nlp:
        try:
            return {e.text.casefold().strip() for e in nlp(text).ents
                    if e.label_ in _IGFR_TARGET_CATS}
        except Exception:
            pass
    raise RuntimeError("spaCy NER returned no pipeline")


def _igfr_coverage(q_ents: set, doc_entity_sets: Sequence[set]) -> Tuple[float, set, set]:
    if not q_ents:
        return 1.0, set(), set()
    document_entities: set = set()
    for entities in doc_entity_sets:
        document_entities.update(entities)
    covered = q_ents & document_entities
    uncovered = q_ents - covered
    return len(covered) / max(len(q_ents), 1), covered, uncovered


def _igfr_build_gap_queries(question: str, uncovered: set,
                             sub_qs: Optional[List[str]] = None,
                             resolved_bridges: Optional[Sequence[str]] = None,
                             max_q: int = 5) -> List[str]:
    ctx = re.sub(r"^(who|what|where|when|why|how|which|whose|whom)\s+",
                 "", question.rstrip("?").strip(), flags=re.IGNORECASE)
    stable_uncovered = sorted(uncovered)
    queries: List[str] = []
    if sub_qs:
        for sq in sub_qs:
            if "{BRIDGE}" not in sq and sq not in queries:
                queries.append(sq)
        for bridge in resolved_bridges or ():
            for sq in sub_qs:
                if "{BRIDGE}" in sq:
                    instantiated = sq.replace("{BRIDGE}", bridge)
                    if instantiated not in queries:
                        queries.append(instantiated)
    for entity in stable_uncovered:
        gap_query = f"{entity} {ctx}"
        if gap_query not in queries:
            queries.append(gap_query)
    return queries[:max_q]


# ── RAG Pipeline Config + Orchestrator ───────────────────────────────────────

@dataclass
class RAGPipelineConfig:
    top_k:                   int   = 5
    igfr_gap_threshold:      float = 0.50    # 论文 γ=0.5: coverage < 50% 触发 gap-filling (line 738)
    igfr_max_iterations: Optional[int] = None  # None => min(2, ceil(128 / CR))
    igfr_max_gap_queries:    int   = 5
    igfr_top_k_per_gap:      int   = 200    # 论文: secondary pool 为 top-200
    ccef_discount_alpha:     float = 0.5     # 论文 α=0.5 (Eq. 5)
    ccef_filter_threshold:   float = 0.30    # 论文 τ=0.30 (line 769)
    numerical_epsilon:       float = 1e-6
    mads_weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    mads_semantic_model_name: str = "BAAI/bge-large-en-v1.5"
    mads_semantic_model_revision: Optional[str] = None
    compression_rate: Optional[int] = None
    # Stage switches used by the paper's component ablations.
    use_qca:                 bool  = True
    use_ahr:                 bool  = True
    use_igfr:                bool  = True
    use_mads:                bool  = True
    use_ccef:                bool  = True
    # ── AHR 候选池大小 ──────────────────────────────────────────────────
    ahr_candidate_pool:      int   = 4000   # 论文: AHR 返回 4000-candidate pool (line 732)
    verbose:                 bool  = False
    # Runtime-only bound; batching/caching does not alter entity scores.
    entity_cache_max_entries: int = 50_000
    # ── 创新点1: 压缩保真度重排序 ──────────────────────────────────────────
    use_cfrs:                bool  = True   # Compression Fidelity Reranking Signal
    cfrs_weight:             float = 0.3    # 融合到最终得分的权重
    # ── 创新点2: 自适应压缩率 ──────────────────────────────────────────────
    use_acr:                 bool  = True   # Adaptive Compression Rate
    # ``adaptive`` is Eq. (8). ``uniform_budget`` and ``static_query`` below
    # are deterministic release conventions for the paper's matched controls:
    # the paper fixes their aggregate budget/topology but does not uniquely
    # specify per-example integer rounding or the static query construction.
    acr_allocation_mode:     str   = "adaptive"
    uniform_evidence_token_budget: int = MATCHED_EVIDENCE_TOKEN_BUDGET
    uniform_constant_ratio:  float = 0.625
    acr_min_token_ratio:     float = 0.25   # 最低分文档保留的 memory token 比例
    acr_max_token_ratio:     float = 1.0    # 最高分文档保留的 memory token 比例
    acr_sigmoid_beta:        float = 10.0
    # Appendix A.26 analysis-only deployment mitigation; off for headline runs.
    use_128x_complexity_floor: bool = False
    acr_complexity_floor_128: float = 0.45
    # ── 创新点3: Memory Token 反馈检索循环 ─────────────────────────────────
    use_mtfrl:               bool  = True   # Memory Token Feedback Retrieval Loop
    second_retrieval_mode:   str   = "memory_feedback"
    mtfrl_second_top_k:      int   = 200    # 论文: D_2 第二轮检索返回 top-200 (line 834)

    def __post_init__(self) -> None:
        # Preserve the historical boolean constructor while making the actual
        # allocation/retrieval intervention explicit and serializable.
        if not self.use_acr and self.acr_allocation_mode == "adaptive":
            self.acr_allocation_mode = "full"
        if not self.use_mtfrl and self.second_retrieval_mode == "memory_feedback":
            self.second_retrieval_mode = "disabled"
        if self.acr_allocation_mode not in {
            "adaptive", "uniform_budget", "uniform_constant", "full"
        }:
            raise ValueError(
                "acr_allocation_mode must be adaptive, uniform_budget, "
                "uniform_constant, or full"
            )
        if self.second_retrieval_mode not in {
            "memory_feedback", "static_query", "disabled"
        }:
            raise ValueError(
                "second_retrieval_mode must be memory_feedback, static_query, "
                "or disabled"
            )
        if self.use_acr != (self.acr_allocation_mode == "adaptive"):
            raise ValueError(
                "use_acr must be true exactly for adaptive allocation; uniform "
                "and full-retention controls replace ACR"
            )
        if self.use_mtfrl != (self.second_retrieval_mode == "memory_feedback"):
            raise ValueError(
                "use_mtfrl must be true exactly for memory-feedback retrieval"
            )
        if self.uniform_evidence_token_budget <= 0:
            raise ValueError("uniform_evidence_token_budget must be positive")
        if not 0.0 < self.uniform_constant_ratio <= 1.0:
            raise ValueError("uniform_constant_ratio must lie in (0, 1]")
        if self.mtfrl_second_top_k != 200:
            raise ValueError("the paper's second retrieval requires D2 top-200")

    def resolved_igfr_iterations(self) -> int:
        if not self.use_igfr:
            return 0
        if self.igfr_max_iterations is not None:
            return max(0, int(self.igfr_max_iterations))
        if self.compression_rate:
            return min(2, math.ceil(128 / self.compression_rate))
        return 2


# One authoritative interpretation shared by training, inference and analysis.
# The legacy names ``remove_acr`` and ``remove_mtfrl`` remain accepted as
# aliases for the corresponding fixed-checkpoint interventions.
RAG_CONFIGURATION_SPECS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "remove_qca": {"use_qca": False},
    "remove_ahr": {"use_ahr": False},
    "remove_igfr": {"use_igfr": False},
    "remove_mads": {"use_mads": False},
    "remove_ccef": {"use_ccef": False},
    # Appendix A.31 matched, independently retrained controls (16x).
    "remove_cfrs": {"use_cfrs": False},
    "uniform_acr": {
        "use_acr": False,
        "acr_allocation_mode": "uniform_budget",
    },
    "static_second_retrieval": {
        "use_mtfrl": False,
        "second_retrieval_mode": "static_query",
    },
    "remove_all_coupling": {
        "use_cfrs": False,
        "use_acr": False,
        "acr_allocation_mode": "uniform_budget",
        "use_mtfrl": False,
        "second_retrieval_mode": "static_query",
    },
    # Appendix A.31 fixed-checkpoint forward-path interventions.
    "fixed_remove_cfrs": {"use_cfrs": False},
    "fixed_uniform_acr": {
        "use_acr": False,
        "acr_allocation_mode": "uniform_constant",
    },
    "fixed_remove_mtfrl": {
        "use_mtfrl": False,
        "second_retrieval_mode": "disabled",
    },
    "forward_path_off": {
        "use_cfrs": False,
        "use_acr": False,
        "acr_allocation_mode": "full",
        "use_mtfrl": False,
        "second_retrieval_mode": "disabled",
    },
    # Backward-compatible, unambiguous aliases for old CLI spellings.
    "remove_acr": {
        "use_acr": False,
        "acr_allocation_mode": "uniform_constant",
    },
    "remove_mtfrl": {
        "use_mtfrl": False,
        "second_retrieval_mode": "disabled",
    },
    "clara_baseline": {
        "use_qca": False,
        "use_ahr": False,
        "use_igfr": False,
        "use_mads": False,
        "use_ccef": False,
        "use_cfrs": False,
        "use_acr": False,
        "acr_allocation_mode": "full",
        "use_mtfrl": False,
        "second_retrieval_mode": "disabled",
    },
}

MATCHED_RETRAINING_CONFIGURATIONS = frozenset({
    "remove_cfrs",
    "uniform_acr",
    "static_second_retrieval",
    "remove_all_coupling",
})
FIXED_CHECKPOINT_CONFIGURATIONS = frozenset({
    "remove_qca",
    "remove_ahr",
    "remove_igfr",
    "remove_mads",
    "remove_ccef",
    "fixed_remove_cfrs",
    "fixed_uniform_acr",
    "fixed_remove_mtfrl",
    "forward_path_off",
    "remove_acr",
    "remove_mtfrl",
})


def required_checkpoint_configuration(runtime_configuration: str) -> str:
    """Resolve a runtime intervention to its immutable training label."""
    if runtime_configuration == ARIA_NO_COMPRESSION_CONFIGURATION:
        return "full"
    if runtime_configuration not in RAG_CONFIGURATION_SPECS:
        raise ValueError(f"unknown RAG configuration: {runtime_configuration}")
    return "full" if runtime_configuration in FIXED_CHECKPOINT_CONFIGURATIONS else runtime_configuration


def create_paper_rag_config(
    configuration: str,
    compression_rate: int,
    **overrides: Any,
) -> RAGPipelineConfig:
    """Build the paper protocol from one explicit, non-overloaded label."""
    if configuration not in RAG_CONFIGURATION_SPECS:
        raise ValueError(f"unknown RAG configuration: {configuration}")
    if compression_rate <= 0:
        raise ValueError("compression_rate must be positive")
    if configuration in MATCHED_RETRAINING_CONFIGURATIONS and compression_rate != 16:
        raise ValueError(
            "the budget/topology-matched retraining study is defined only at 16x"
        )
    values: Dict[str, Any] = {
        "top_k": 5,
        "compression_rate": compression_rate,
        "cfrs_weight": 0.3,
        "acr_min_token_ratio": 0.25,
        "acr_max_token_ratio": 1.0,
        "uniform_evidence_token_budget": MATCHED_EVIDENCE_TOKEN_BUDGET,
        "uniform_constant_ratio": 0.625,
        "mtfrl_second_top_k": 200,
        "igfr_gap_threshold": 0.50,
        "igfr_max_iterations": None,
        "ccef_discount_alpha": 0.5,
        "ccef_filter_threshold": 0.30,
        "mads_weights": (1 / 3, 1 / 3, 1 / 3),
    }
    values.update(RAG_CONFIGURATION_SPECS[configuration])
    values.update(overrides)
    return RAGPipelineConfig(**values)


@dataclass
class RAGDiagnostics:
    question_type:       str   = ""
    qca_confidence:      float = 0.0
    bm25_weight:         float = 0.5
    dense_weight:        float = 0.5
    igfr_iterations:     int   = 0
    initial_coverage:    float = 1.0
    final_coverage:      float = 1.0
    ccef_avg_confidence: float = 0.0
    ccef_filtered:       int   = 0
    initial_candidates:  int   = 0
    final_candidates:    int   = 0
    second_round_candidates: int = 0
    evidence_memory_tokens: int = 0
    direct_context_document_tokens: int = 0
    direct_context_prompt_tokens: int = 0
    direct_context_ceiling: int = 0
    latency_ms:          float = 0.0


class RAGEnhancementPipeline:
    """
    CLaRa RAG Enhancement Pipeline — QCA → AHR → IGFR → MADS → CCEF.

    创建方法:
        pipeline = RAGEnhancementPipeline.from_corpus(corpus_docs, config=config)

    调用方法:
        docs, diag = pipeline.retrieve("your question", query_emb=...)
    """

    def __init__(self, qca: QuestionComplexityAssessor,
                 ahr: _AdaptiveHybridRetriever,
                 ccef: _CCEF,
                 config: RAGPipelineConfig):
        self.qca    = qca
        self.ahr    = ahr
        self.ccef   = ccef
        self.config = config
        if config.entity_cache_max_entries <= 0:
            raise ValueError("entity_cache_max_entries must be positive")
        self._igfr_entity_cache: "OrderedDict[str, Tuple[str, set]]" = OrderedDict()

    @classmethod
    def from_corpus(cls, corpus_docs: List[str],
                    corpus_doc_ids: Optional[List[str]] = None,
                    doc_embeddings: Optional[torch.Tensor] = None,
                    config: Optional[RAGPipelineConfig] = None,
                    bm25_index: Optional[_BM25Index] = None,
                    corpus_page_ids: Optional[List[str]] = None) -> "RAGEnhancementPipeline":
        cfg = config or RAGPipelineConfig()
        ids = corpus_doc_ids or [str(i) for i in range(len(corpus_docs))]
        if bm25_index is None:
            bm25 = _BM25Index().build(corpus_docs)
        else:
            if bm25_index._docs is not corpus_docs:
                raise ValueError(
                    "a prebuilt BM25 index must reference the exact corpus_docs list"
                )
            bm25 = bm25_index
        qca  = QuestionComplexityAssessor()
        ahr  = _AdaptiveHybridRetriever(
            bm25,
            corpus_docs,
            ids,
            corpus_page_ids=corpus_page_ids,
            dense_embeddings=doc_embeddings,
        )
        ccef = _CCEF(weights=cfg.mads_weights,
                     discount_alpha=cfg.ccef_discount_alpha,
                     filter_threshold=cfg.ccef_filter_threshold,
                     entity_cache_max_entries=cfg.entity_cache_max_entries,
                     semantic_model_name=cfg.mads_semantic_model_name,
                     semantic_model_revision=cfg.mads_semantic_model_revision)
        return cls(qca=qca, ahr=ahr, ccef=ccef, config=cfg)

    @staticmethod
    def _as_retrieved(doc: _ScoredDoc) -> _RetrievedDoc:
        return _RetrievedDoc(
            doc_id=doc.doc_id,
            text=doc.text,
            corpus_index=doc.corpus_index,
            hybrid_score=doc.hybrid_score,
            from_second_round=doc.from_second_round,
        )

    def _document_entity_sets(self, documents: Sequence[_RetrievedDoc]) -> List[set]:
        """Return exact IGFR entities using batched spaCy and a bounded LRU."""
        results: List[Optional[set]] = [None] * len(documents)
        missing: "OrderedDict[str, Tuple[str, List[int]]]" = OrderedDict()
        for index, document in enumerate(documents):
            cached = self._igfr_entity_cache.get(document.doc_id)
            if cached is not None and cached[0] == document.text:
                self._igfr_entity_cache.move_to_end(document.doc_id)
                results[index] = cached[1]
                continue
            if cached is not None:
                del self._igfr_entity_cache[document.doc_id]
            if document.doc_id not in missing:
                missing[document.doc_id] = (document.text, [])
            elif missing[document.doc_id][0] != document.text:
                raise ValueError("one document ID cannot refer to multiple texts")
            missing[document.doc_id][1].append(index)

        if missing:
            nlp = _qca_get_spacy()
            texts = [value[0] for value in missing.values()]
            try:
                parsed = list(nlp.pipe(texts, batch_size=128))
            except Exception:
                try:
                    parsed = [nlp(text) for text in texts]
                except Exception as exc:
                    raise RuntimeError("spaCy NER returned no pipeline") from exc
            if len(parsed) != len(texts):
                raise RuntimeError("spaCy NER returned an incomplete batch")
            for (doc_id, (_, positions)), parsed_doc in zip(missing.items(), parsed):
                entities = {
                    entity.text.casefold().strip()
                    for entity in parsed_doc.ents
                    if entity.label_ in _IGFR_TARGET_CATS and entity.text.strip()
                }
                self._igfr_entity_cache[doc_id] = (missing[doc_id][0], entities)
                self._igfr_entity_cache.move_to_end(doc_id)
                while len(self._igfr_entity_cache) > self.config.entity_cache_max_entries:
                    self._igfr_entity_cache.popitem(last=False)
                for index in positions:
                    results[index] = entities

        if any(value is None for value in results):
            raise RuntimeError("spaCy NER did not return every requested document")
        return [value for value in results if value is not None]

    @staticmethod
    def _resolved_bridge_entities(
        entity_sets: Sequence[set], query_entities: set, max_entities: int
    ) -> List[str]:
        counts = Counter(
            entity
            for entities in entity_sets
            for entity in entities
            if entity not in query_entities
        )
        return [
            entity
            for entity, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            [:max_entities]
        ]

    def _mads_ccef(
        self,
        query: str,
        retrieved: List[_RetrievedDoc],
        top_k: int,
        query_emb: Optional[torch.Tensor] = None,
        diagnostics: Optional[RAGDiagnostics] = None,
    ) -> List[_ScoredDoc]:
        """Run Stages 4-5 exactly in MADS-top100 then CCEF-top5 order."""
        cfg = self.config
        if cfg.use_mads:
            # Appendix A.4 defines MADS semantic relevance in the same frozen
            # BGE document space used by AHR, with W_BGE(QR(q)) as the query.
            if query_emb is None:
                raise ValueError("MADS requires the fixed W_BGE(QR) query embedding")
            query_vector = F.normalize(
                query_emb.detach().float().cpu().reshape(1, -1),
                dim=-1,
                eps=cfg.numerical_epsilon,
            ).squeeze(0)
            semantic_scores: List[float] = [0.0] * len(retrieved)
            indexed_positions = [
                position
                for position, document in enumerate(retrieved)
                if (
                    self.ahr.dense_embeddings is not None
                    and 0 <= document.corpus_index < self.ahr.dense_embeddings.size(0)
                )
            ]
            if indexed_positions:
                rows = torch.tensor(
                    [retrieved[position].corpus_index for position in indexed_positions],
                    dtype=torch.long,
                )
                indexed_scores = (
                    self.ahr.dense_embeddings.index_select(0, rows) @ query_vector
                ).tolist()
                for position, score in zip(indexed_positions, indexed_scores):
                    semantic_scores[position] = float(score)
            indexed_set = set(indexed_positions)
            external_positions = [
                position for position in range(len(retrieved)) if position not in indexed_set
            ]
            if external_positions:
                external_scores = self.ccef.sem.score_projected(
                    query_vector,
                    [retrieved[position].text for position in external_positions],
                )
                for position, score in zip(external_positions, external_scores):
                    semantic_scores[position] = float(score)
            scored = self.ccef.score_and_fuse(
                query, retrieved, semantic_scores=semantic_scores
            )
        else:
            hybrid = _CCEF._minmax([doc.hybrid_score for doc in retrieved])
            scored = [
                _ScoredDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    corpus_index=doc.corpus_index,
                    mads_score=hybrid[i],
                    fused_score=hybrid[i],
                    confidence=1.0,
                    is_filtered=(hybrid[i] < cfg.ccef_filter_threshold),
                    hybrid_score=doc.hybrid_score,
                    from_second_round=doc.from_second_round,
                )
                for i, doc in enumerate(retrieved)
            ]

        # Stage 4 is based only on the equal-weight normalized agent average;
        # CCEF filtering must not influence which 100 documents enter Stage 5.
        mads_ranked = sorted(
            scored,
            key=lambda doc: (-doc.mads_score, doc.doc_id),
        )
        # MADS reranks candidate occurrences and takes the first 100 directly.
        # Page-ID deduplication belongs only to Oracle-pool construction and
        # reported Recall@k, not to the Normal/training retrieval pipeline.
        mads_top100 = mads_ranked[:100]

        if cfg.use_ccef:
            # Python's stable sort preserves incoming MADS order for exact ties.
            ccef_sorted = sorted(mads_top100, key=lambda doc: -doc.fused_score)
            survivors = [doc for doc in ccef_sorted if not doc.is_filtered]
            filtered = len(ccef_sorted) - len(survivors)
            if not survivors and ccef_sorted:
                # Algorithm 1 retains the first MADS-ordered maximizer.
                best_score = ccef_sorted[0].fused_score
                survivors = [
                    next(doc for doc in mads_top100 if doc.fused_score == best_score)
                ]
        else:
            survivors = sorted(
                mads_top100,
                key=lambda doc: (-doc.mads_score, doc.doc_id),
            )
            for doc in survivors:
                doc.fused_score = doc.mads_score
                doc.is_filtered = False
            filtered = 0

        selected = survivors[:top_k]
        if top_k == 5 and len(selected) != 5:
            raise RuntimeError(
                "CCEF thresholding retained fewer than the paper's fixed top-five set"
            )

        if diagnostics is not None:
            diagnostics.ccef_filtered += filtered
            diagnostics.ccef_avg_confidence = (
                float(np.mean([doc.confidence for doc in selected])) if selected else 0.0
            )
        return selected

    def retrieve_initial_batch(
        self,
        queries: Sequence[str],
        query_embeddings: Optional[torch.Tensor],
    ) -> Tuple[List[List[_RetrievedDoc]], List[QCAResult]]:
        """Batch QCA/AHR for the first 4,000-candidate retrieval pass."""
        if not queries:
            raise ValueError("initial retrieval batch must not be empty")
        if self.config.use_qca:
            qca_results = self.qca.assess_batch(list(queries))
        else:
            qca_results = [
                QCAResult(
                    question=query,
                    question_type=QuestionType.SIMPLE,
                    confidence=0.0,
                    hop_count=1,
                    entity_count=_qca_entity_count(query),
                    sub_questions=[query],
                    reasoning="Direct single-question routing",
                )
                for query in queries
            ]
        retrieval_qca = qca_results
        dense_queries = query_embeddings
        if not self.config.use_ahr:
            retrieval_qca = [
                QCAResult(
                    question=result.question,
                    question_type=QuestionType.SIMPLE,
                    confidence=1.0,
                    hop_count=1,
                    entity_count=result.entity_count,
                )
                for result in qca_results
            ]
            dense_queries = None
        retrieved = self.ahr.retrieve_batch(
            queries,
            retrieval_qca,
            query_embeddings=dense_queries,
            top_k=self.config.ahr_candidate_pool,
        )
        return retrieved, qca_results

    def retrieve_scored(
        self,
        query: str,
        query_emb: Optional[torch.Tensor] = None,
        override_top_k: Optional[int] = None,
        embed_subquery: Optional[Callable[[str], torch.Tensor]] = None,
        qca_result: Optional[QCAResult] = None,
        initial_retrieved: Optional[Sequence[_RetrievedDoc]] = None,
    ) -> Tuple[List[_ScoredDoc], QCAResult, RAGDiagnostics]:
        """Run Algorithm 1 through the first CCEF and retain s_fused metadata."""
        cfg   = self.config
        top_k = override_top_k or cfg.top_k
        diag  = RAGDiagnostics()
        t0    = time.perf_counter()

        # Stage 1: QCA
        if qca_result is not None:
            if qca_result.question != query:
                raise ValueError("precomputed QCA result belongs to a different query")
            qca_r = qca_result
        elif cfg.use_qca:
            qca_r = self.qca.assess(query)
        else:
            qca_r = QCAResult(
                question=query,
                question_type=QuestionType.SIMPLE,
                confidence=0.0,
                hop_count=1,
                entity_count=_qca_entity_count(query),
                sub_questions=[query],
                reasoning="Direct single-question routing",
            )
        diag.question_type  = qca_r.question_type.value
        diag.qca_confidence = qca_r.confidence
        bw, dw = _ahr_get_weights(qca_r)
        if not cfg.use_ahr:
            bw, dw = 1.0, 0.0
        diag.bm25_weight  = bw
        diag.dense_weight = dw

        # Stage 2: AHR (paper: over-retrieve to 4000-candidate pool)
        pool_size  = cfg.ahr_candidate_pool
        if not cfg.use_ahr:
            qca_for_retrieval = QCAResult(
                question=query,
                question_type=QuestionType.SIMPLE,
                confidence=1.0,
                hop_count=1,
                entity_count=qca_r.entity_count,
            )
        else:
            qca_for_retrieval = qca_r
        if initial_retrieved is None:
            retrieved = self.ahr.retrieve(
                query,
                qca_for_retrieval,
                query_emb=query_emb if cfg.use_ahr else None,
                top_k=pool_size,
            )
        else:
            retrieved = list(initial_retrieved)
            if not retrieved:
                raise ValueError("precomputed AHR pool must not be empty")
            if len(retrieved) > pool_size:
                raise ValueError("precomputed AHR pool exceeds ahr_candidate_pool")
        diag.initial_candidates = len(retrieved)

        # Stage 3: IGFR (only for Multi-Hop)
        igfr_iters = 0
        q_ents = _igfr_entities(query)
        entity_sets = self._document_entity_sets(retrieved)
        init_cov, _, uncovered = _igfr_coverage(q_ents, entity_sets)
        diag.initial_coverage = init_cov
        final_cov = init_cov

        # The -QCA ablation keeps IGFR enabled and routes every query through its
        # coverage gate, isolating QCA while holding the remaining stages fixed.
        run_igfr = cfg.use_igfr and (
            qca_r.question_type == QuestionType.MULTI_HOP or not cfg.use_qca
        )
        if run_igfr and q_ents:
            seen_ids: set[str] = {document.doc_id for document in retrieved}
            all_docs = list(retrieved)
            next_template = 0
            for _ in range(cfg.resolved_igfr_iterations()):
                # Appendix A.4: equality stops, and exactly the next unused
                # template is consumed on each iteration.
                if (
                    final_cov >= cfg.igfr_gap_threshold
                    or next_template >= len(qca_r.sub_questions)
                ):
                    break
                template = qca_r.sub_questions[next_template]
                next_template += 1
                gap_query = template
                if "{BRIDGE}" in template:
                    bridge_entities = self._resolved_bridge_entities(
                        self._document_entity_sets(all_docs),
                        q_ents,
                        1,
                    )
                    if not bridge_entities:
                        # The template is used but cannot be instantiated from
                        # D^(k-1); no retrieval query is fabricated in its place.
                        continue
                    gap_query = template.replace("{BRIDGE}", bridge_entities[0])

                gap_embedding = (
                    embed_subquery(gap_query)
                    if cfg.use_ahr and embed_subquery is not None
                    else None
                )
                gap_results = self.ahr.retrieve(
                    gap_query,
                    qca_r if cfg.use_ahr else qca_for_retrieval,
                    query_emb=gap_embedding,
                    top_k=cfg.igfr_top_k_per_gap,
                )
                # A single template contributes at most 200 previously unseen
                # corpus documents, in the retriever's stable order.
                new_documents: List[_RetrievedDoc] = []
                for document in gap_results:
                    if document.doc_id in seen_ids:
                        continue
                    seen_ids.add(document.doc_id)
                    new_documents.append(document)
                    if len(new_documents) == cfg.igfr_top_k_per_gap:
                        break
                all_docs.extend(new_documents)
                entity_sets = self._document_entity_sets(all_docs)
                final_cov, _, uncovered = _igfr_coverage(q_ents, entity_sets)
                igfr_iters += 1
            retrieved = all_docs

        diag.igfr_iterations = igfr_iters
        diag.final_coverage  = final_cov

        top_docs = self._mads_ccef(
            query, retrieved, top_k, query_emb=query_emb, diagnostics=diag
        )
        diag.final_candidates = len(top_docs)
        diag.latency_ms = (time.perf_counter() - t0) * 1000

        if cfg.verbose:
            print(f"[RAG] {diag.question_type}|bm25={bw:.2f}/dense={dw:.2f}|"
                  f"cov={final_cov:.0%}|igfr_iters={igfr_iters}|"
                  f"filtered={diag.ccef_filtered}|{diag.latency_ms:.0f}ms")

        return top_docs, qca_r, diag

    def rescore_union(
        self,
        query: str,
        first_round: List[_ScoredDoc],
        second_round: List[_RetrievedDoc],
        query_emb: Optional[torch.Tensor] = None,
        top_k: Optional[int] = None,
        diagnostics: Optional[RAGDiagnostics] = None,
    ) -> List[_ScoredDoc]:
        """Algorithm 1 lines 13-14: union, then rerun MADS and CCEF."""
        dedup: Dict[str, _RetrievedDoc] = {
            doc.doc_id: self._as_retrieved(doc) for doc in first_round
        }
        for doc in second_round:
            dedup.setdefault(doc.doc_id, doc)
        if diagnostics is not None:
            diagnostics.second_round_candidates = len(second_round)
        return self._mads_ccef(
            query,
            list(dedup.values()),
            top_k or self.config.top_k,
            query_emb=query_emb,
            diagnostics=diagnostics,
        )

    def retrieve(self, query: str,
                 query_emb: Optional[torch.Tensor] = None,
                 override_top_k: Optional[int] = None) -> Tuple[List[str], RAGDiagnostics]:
        """Backward-compatible text-only view of the first Algorithm-1 pass."""
        docs, _, diag = self.retrieve_scored(
            query, query_emb=query_emb, override_top_k=override_top_k
        )
        return [doc.text for doc in docs], diag

# ═══════════════════════════════════════════════════════════════════════════════
# 创新点 1: Compression Fidelity Reranking Signal (CFRS)
# 用冻结 base decoder 的 teacher-forced next-token 概率误差衡量压缩保真度。
# ═══════════════════════════════════════════════════════════════════════════════

class CompressionFidelityReranker:
    """Teacher-forced reconstruction fidelity with a differentiable score."""

    @staticmethod
    def squared_probability_error(
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ``||softmax(logits)-one_hot(target)||_2^2`` per row."""
        if logits.ndim < 2 or target_ids.shape != logits.shape[:-1]:
            raise ValueError("target_ids must align with logits except for vocabulary")
        if target_ids.dtype == torch.bool or torch.is_floating_point(target_ids):
            raise ValueError("target_ids must use an integer dtype")
        work = logits.float()
        log_z = torch.logsumexp(work, dim=-1)
        sum_squared_probability = torch.exp(
            torch.logsumexp(2.0 * work, dim=-1) - 2.0 * log_z
        )
        target_probability = torch.exp(
            work.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1) - log_z
        )
        error = sum_squared_probability - 2.0 * target_probability + 1.0
        if target_mask is None:
            return error.mean(dim=-1)
        if target_mask.shape != target_ids.shape:
            raise ValueError("target_mask must have the same shape as target_ids")
        mask = target_mask.to(device=error.device, dtype=torch.bool)
        counts = mask.sum(dim=-1)
        if (counts == 0).any():
            raise ValueError("every CFRS row requires at least one target token")
        return (error * mask).sum(dim=-1) / counts

    @staticmethod
    def rerank(original_scores: torch.Tensor,
               per_doc_mse: torch.Tensor,
               cfrs_weight: float = 0.3,
               eps: float = 1e-6,
               document_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        将保真度分数融合进原始检索分数。

        Args:
            original_scores: (B, N) — 来自 AHR/MADS 的原始相关性分数
            per_doc_mse:     (B, N) — 每篇文档的 squared-probability error
            cfrs_weight:     CFRS 分数的融合权重
        Returns:
            reranked_scores: (B, N)
        """
        B, N = original_scores.shape
        mse = per_doc_mse.view(B, N)
        if document_mask is None:
            valid = torch.ones_like(original_scores, dtype=torch.bool)
        else:
            valid = document_mask.to(original_scores.device, dtype=torch.bool)
            if valid.shape != original_scores.shape or (~valid).all(dim=1).any():
                raise ValueError("CFRS requires at least one valid document per row")

        # Stop gradients through the min/max normalization statistics, as in
        # the paper's CFRS derivative, while retaining d f_i / d error_i.
        error = mse
        error_stats = error.detach()
        error_min = error_stats.masked_fill(~valid, float("inf")).min(
            dim=1, keepdim=True
        ).values
        error_max = error_stats.masked_fill(~valid, float("-inf")).max(
            dim=1, keepdim=True
        ).values
        span = error_max - error_min
        fidelity = (error_max - error) / (span + eps)
        fidelity = fidelity.masked_fill(~valid, 0.0)

        reranked = (
            (1 - cfrs_weight) * original_scores
            + cfrs_weight * fidelity.to(original_scores.dtype)
        )
        return reranked.masked_fill(~valid, float("-inf"))


# ═══════════════════════════════════════════════════════════════════════════════
# 创新点 2: Adaptive Compression Rate (ACR)
# 根据文档相关性得分，动态分配 memory token 预算。
# 相关性高的文档 → 保留更多 memory token（少压缩）
# 相关性低的文档 → 保留更少 memory token（多压缩）
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveCompressionAllocator:
    """
    ACR: 在不改变模型结构的前提下，通过"软剪枝"实现自适应压缩率。

    实现方式:
        每篇文档的 memory token embedding 被一个 [0,1] 的掩码缩放。
        低相关性文档的后半段 memory token 被逐渐归零，
        等效于减少该文档对解码器的信息贡献，即"更高压缩率"。

    这样做的好处:
        - 不需要改变 tokenizer 或模型输入维度
        - 梯度可以正常回传（掩码是连续值，不是硬截断）
        - 计算开销极小（只是一次逐元素乘法）
    """

    def __init__(self, min_ratio: float = 0.25, max_ratio: float = 1.0,
                 beta: float = 10.0, eps: float = 1e-6):
        """
        Args:
            min_ratio: 最低相关性文档保留的 memory token 比例（0.25 = 保留 25%）
            max_ratio: 最高相关性文档保留的 memory token 比例（1.0 = 全部保留）
        """
        assert 0.0 < min_ratio <= max_ratio <= 1.0
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.beta = float(beta)
        self.eps = float(eps)

    def ratios_from_scores(
        self,
        relevance_scores: torch.Tensor,
        min_ratios: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eq. 8, normalized independently for every query's top-five set."""
        if relevance_scores.dim() == 1:
            relevance_scores = relevance_scores.unsqueeze(0)
        scores = relevance_scores.detach().float()
        score_min = scores.min(dim=1, keepdim=True).values
        score_max = scores.max(dim=1, keepdim=True).values
        span = score_max - score_min
        if min_ratios is None:
            floors = torch.full(
                (scores.size(0), 1), self.min_ratio,
                device=scores.device, dtype=scores.dtype,
            )
        else:
            floors = min_ratios.to(device=scores.device, dtype=scores.dtype).reshape(-1, 1)
            if floors.size(0) != scores.size(0):
                raise ValueError("one ACR minimum ratio is required per query")
        normalized = (scores - score_min) / (span + self.eps)
        return floors + normalized * (self.max_ratio - floors)

    @staticmethod
    def uniform_ratios_for_budget(
        base_token_counts: torch.Tensor,
        target_tokens: int,
    ) -> torch.Tensor:
        """Return a score-independent common ratio for one query.

        Appendix A.45 reports the aggregate 108-token constraint but does not
        define how integer gates are rounded for every variable-length example.
        The release convention divides the target by the sum of the five real
        K0 counts, clips at one, and applies the same ratio to every document.
        """
        counts = base_token_counts.reshape(-1).float()
        if counts.numel() == 0 or (counts <= 0).any():
            raise ValueError("uniform allocation requires positive real-document K0 counts")
        if target_tokens <= 0:
            raise ValueError("uniform allocation target must be positive")
        ratio = min(1.0, float(target_tokens) / float(counts.sum().item()))
        return torch.full_like(counts, ratio, dtype=torch.float32)

    def apply_ratios(
        self,
        doc_embeddings: torch.Tensor,
        retain_ratios: torch.Tensor,
        base_token_counts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the paper's sigmoid ACR mask and report hard MTFRL counts."""
        n_docs, n_tokens, _ = doc_embeddings.shape
        ratios = retain_ratios.reshape(-1).to(doc_embeddings.device, torch.float32)
        if ratios.numel() != n_docs:
            raise ValueError(f"ACR ratios/doc mismatch: {ratios.numel()} != {n_docs}")

        # Paper indexes t from 1 through floor(L/r).
        token_positions = torch.arange(
            1, n_tokens + 1, device=doc_embeddings.device, dtype=torch.float32
        )
        if base_token_counts is None:
            base_counts = torch.full_like(ratios, n_tokens)
        else:
            base_counts = base_token_counts.reshape(-1).to(
                device=doc_embeddings.device, dtype=torch.float32
            )
            if base_counts.numel() != n_docs:
                raise ValueError("one base memory-token count is required per document")
            if (base_counts <= 0).any() or (base_counts > n_tokens).any():
                raise ValueError("base memory-token counts must lie in [1, T]")
        boundary = ratios.unsqueeze(1) * base_counts.unsqueeze(1)
        soft_mask = torch.sigmoid(self.beta * (boundary - token_positions))
        valid_base = token_positions.unsqueeze(0) <= base_counts.unsqueeze(1)
        soft_mask = soft_mask * valid_base.to(soft_mask.dtype)

        # T_i is used only by MTFRL's hard prefix.  The decoder always consumes
        # the full differentiable sigmoid-masked sequence from Eq. (8).
        effective_counts = torch.floor(boundary.squeeze(1)).long().clamp(
            min=1, max=n_tokens
        )
        gated = doc_embeddings * soft_mask.to(doc_embeddings.dtype).unsqueeze(-1)
        return gated, soft_mask, effective_counts

    def apply(self,
              doc_embeddings: torch.Tensor,
              relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        对压缩后的 memory token embeddings 按相关性进行软剪枝。

        Args:
            doc_embeddings:   (N, n_mem_tokens, hidden) — 压缩后的 memory token
            relevance_scores: (N,) — 每篇文档的相关性得分，已归一化到 [0, 1]
        Returns:
            pruned_embeddings: (N, n_mem_tokens, hidden) — 剪枝后的 memory token
        """
        ratios = self.ratios_from_scores(relevance_scores).reshape(-1)
        return self.apply_ratios(doc_embeddings, ratios)[0]

    def allocate_and_apply(self,
                           doc_embeddings: torch.Tensor,
                           reranked_scores: torch.Tensor,
                           n_docs_per_question: int,
                           min_ratios: Optional[torch.Tensor] = None,
                           base_token_counts: Optional[torch.Tensor] = None,
                           return_metadata: bool = False
                           ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        批量入口: 对每道题的所有候选文档做 ACR。

        Args:
            doc_embeddings:      (B*N, T, H)
            reranked_scores:     (B, N) — CFRS 重排后的分数
            n_docs_per_question: N
        Returns:
            pruned: (B*N, T, H)
        """
        B = reranked_scores.shape[0]
        N = n_docs_per_question

        if reranked_scores.shape != (B, N):
            raise ValueError(
                f"expected ACR score shape {(B, N)}, got {tuple(reranked_scores.shape)}"
            )
        ratios = self.ratios_from_scores(reranked_scores, min_ratios=min_ratios)
        pruned, masks, counts = self.apply_ratios(
            doc_embeddings,
            ratios.reshape(B * N),
            base_token_counts=base_token_counts,
        )
        if return_metadata:
            return pruned, ratios, masks.view(B, N, -1), counts.view(B, N)
        return pruned


# ═══════════════════════════════════════════════════════════════════════════════
# 创新点 3: Memory Token Feedback Retrieval Loop (MTFRL)
# 用第一轮压缩后的 memory token 均值作为新的 query 向量，
# 对语料库发起第二轮密集检索，补充第一轮遗漏的相关文档。
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryTokenFeedbackRetriever:
    """
    MTFRL: Memory Token → 第二轮检索 → 合并去重。

    核心思想:
        第一轮检索: 用原始问题的 query_rep 检索 → 得到文档 D1
        压缩 D1:    把 D1 压缩成 memory token M
        第二轮检索: 用 mean(M) 作为新的查询向量，再次检索 → 得到文档 D2
        合并:       D1 ∪ D2 去重后重排，取 top-k 送入解码器

    为什么有效:
        M 包含了"模型读过 D1 之后的理解"，比原始问题 query_rep 更精准地
        表达了"还需要什么信息"，能找到第一轮因语义距离而漏掉的关键文档。
    """

    def __init__(self, corpus_embeddings: Optional[torch.Tensor] = None,
                 corpus_docs: Optional[List[str]] = None,
                 corpus_ids: Optional[List[str]] = None,
                 embeddings_are_normalized: bool = False):
        """
        Args:
            corpus_embeddings: (C, T, H) 或 (C, H) 预编码的语料库向量
                               如果是 (C, T, H) 则取 mean(T) 降维
            corpus_docs:       语料库文本列表
            corpus_ids:        语料库文档 ID 列表
        """
        self.corpus_docs = corpus_docs or []
        self.corpus_ids  = corpus_ids  or [str(i) for i in range(len(self.corpus_docs))]
        self._corpus_emb: Optional[torch.Tensor] = None   # (C, H)

        if corpus_embeddings is not None:
            if corpus_embeddings.dim() == 3:
                # (C, T, H) → 取 memory token 均值 → (C, H)
                self._corpus_emb = F.normalize(
                    corpus_embeddings.mean(dim=1).float().cpu(),
                    dim=-1,
                    eps=ARIA_NUMERICAL_EPSILON,
                ).contiguous()
            else:
                if embeddings_are_normalized:
                    if corpus_embeddings.device.type != "cpu" or corpus_embeddings.dtype != torch.float32:
                        raise ValueError(
                            "shared normalized corpus embeddings must be CPU float32"
                        )
                    self._corpus_emb = corpus_embeddings
                else:
                    self._corpus_emb = F.normalize(
                        corpus_embeddings.float().cpu(),
                        dim=-1,
                        eps=ARIA_NUMERICAL_EPSILON,
                    ).contiguous()

    def update_corpus_embeddings(self, emb: torch.Tensor):
        """在线更新语料库向量（例如每轮训练后重新编码）。"""
        pooled = emb.mean(dim=1).float() if emb.dim() == 3 else emb.float()
        self._corpus_emb = F.normalize(
            pooled.cpu(), dim=-1, eps=ARIA_NUMERICAL_EPSILON
        ).contiguous()

    @torch.no_grad()
    def second_round_retrieve(self,
                              feedback_queries: torch.Tensor,
                              already_retrieved_ids: Sequence[Sequence[str]],
                              top_k: int = 200,
                              allowed_corpus_indices: Optional[
                                  Sequence[Sequence[int]]
                              ] = None) -> List[List[_RetrievedDoc]]:
        """
        Search the full fixed dense index with projected MTFRL feedback queries.

        Args:
            feedback_queries: (B, 1024), output of the trained MTFRL MLP.
            already_retrieved_ids: D1 IDs, used only to validate batch alignment;
                Algorithm 1 applies duplicate removal to D1 union D2 afterwards.
            top_k: D2 size; Algorithm 1 fixes this to 200.
            allowed_corpus_indices: optional per-query fixed pool. Oracle mode
                supplies its same top-100 pool here; ``None`` retains Normal's
                full-corpus search.
        Returns:
            One list of corpus-backed retrieved documents per query.
        """
        if self._corpus_emb is None or len(self.corpus_docs) == 0:
            raise RuntimeError("MTFRL requires full-corpus dense embeddings")

        # Hard full-corpus search is non-differentiable in Algorithm 1. Keep the
        # fixed index on CPU instead of copying a KILT-scale matrix to every GPU
        # at every batch; the selected-doc straight-through term is added later.
        query_feedback = F.normalize(
            feedback_queries.detach().float().cpu(),
            dim=-1,
            eps=ARIA_NUMERICAL_EPSILON,
        )
        if query_feedback.dim() != 2:
            raise ValueError("feedback_queries must have shape (batch, embedding_dim)")
        if len(already_retrieved_ids) != query_feedback.size(0):
            raise ValueError("already_retrieved_ids must align with feedback queries")
        if top_k <= 0:
            raise ValueError("MTFRL top_k must be positive")

        # 语料库向量归一化
        corpus_n = self._corpus_emb  # (C, H), CPU and pre-normalized
        if query_feedback.size(-1) != corpus_n.size(-1):
            raise ValueError(
                "MTFRL/corpus embedding dimension mismatch: "
                f"{query_feedback.size(-1)} != {corpus_n.size(-1)}"
            )

        new_docs: List[List[_RetrievedDoc]] = []
        if allowed_corpus_indices is not None:
            if len(allowed_corpus_indices) != query_feedback.size(0):
                raise ValueError(
                    "allowed_corpus_indices must align with feedback queries"
                )
            corpus_size = len(self.corpus_docs)
            for batch_index, allowed_values in enumerate(allowed_corpus_indices):
                allowed = [int(index) for index in allowed_values]
                if not allowed or len(allowed) != len(set(allowed)):
                    raise ValueError(
                        "each MTFRL allowed pool must be non-empty and unique"
                    )
                if any(index < 0 or index >= corpus_size for index in allowed):
                    raise ValueError("MTFRL allowed pool contains an invalid corpus index")
                row_indices = torch.tensor(allowed, dtype=torch.long)
                row_scores = (
                    corpus_n.index_select(0, row_indices)
                    @ query_feedback[batch_index]
                ).tolist()
                ranked = sorted(
                    zip(row_scores, allowed), key=lambda pair: (-pair[0], pair[1])
                )
                batch_new: List[_RetrievedDoc] = []
                for score, corpus_index in ranked[:top_k]:
                    doc_id = self.corpus_ids[corpus_index]
                    batch_new.append(
                        _RetrievedDoc(
                            doc_id=doc_id,
                            text=self.corpus_docs[corpus_index],
                            corpus_index=corpus_index,
                            dense_score=float(score),
                            hybrid_score=float(score),
                            from_second_round=True,
                        )
                    )
                new_docs.append(batch_new)
            return new_docs

        search_k = min(len(self.corpus_docs), top_k)
        top_values, top_indices = _chunked_inner_product_topk(
            query_feedback, corpus_n, search_k
        )
        for b in range(query_feedback.size(0)):
            batch_new: List[_RetrievedDoc] = []
            for score, idx in zip(top_values[b].tolist(), top_indices[b].tolist()):
                doc_id = self.corpus_ids[idx]
                batch_new.append(_RetrievedDoc(
                    doc_id=doc_id,
                    text=self.corpus_docs[idx],
                    corpus_index=idx,
                    dense_score=float(score),
                    hybrid_score=float(score),
                    from_second_round=True,
                ))
            new_docs.append(batch_new)

        return new_docs

# ═══════════════════════════════════════════════════════════════════════════════
# End of RAG Enhancement Modules
# ═══════════════════════════════════════════════════════════════════════════════


class StopOnCriteria(StoppingCriteria):
    """Custom stopping criteria for generation."""

    def __init__(self, tokenizer, stop_strings: List[str] = None, stop_token_ids: List[int] = None):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings or []
        self.stop_token_ids = stop_token_ids or []
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        # Check if last token is in stop_token_ids
        last_token = input_ids[0, -1].item()
        if last_token in self.stop_token_ids:
            self.reason = f"stop_token_{last_token}"
            return True

        # Check if any stop_strings appear in generated text
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        for stop_str in self.stop_strings:
            if stop_str in text:
                self.reason = f"stop_string_{stop_str}"
                return True

        return False


class LlamaRMSNorm(nn.Module):
    """Llama-style RMS normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Converter(nn.Module):
    """Converter module for dimension transformation."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.rms_norm = LlamaRMSNorm(input_dim)
        self.dense_in = nn.Linear(input_dim, output_dim)
        self.dense_out = nn.Linear(output_dim, output_dim)

        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print parameter statistics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Converter trainable parameters: {trainable_params}, Total parameters: {total_params}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.rms_norm(embeddings)
        x = self.dense_in(embeddings)
        x = self.dense_out(gelu(x))
        return x.to(torch.float32)


class CLaRaConfig(PretrainedConfig):
    """Configuration class for CLaRa model."""

    model_type = "CLaRa"

    def __init__(self,
                 decoder_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 decoder_model_revision: Optional[str] = None,
                 decoder_model_resolved_revision: Optional[str] = None,
                 doc_max_length: int = 768,
                 query_max_length: int = 256,
                 stage1_input_max_length: int = 2048,
                 stage2_input_max_length: int = 1024,
                 stage1_target_max_length: int = 512,
                 stage2_target_max_length: int = 128,
                 quantization: str = 'no',
                 sep: bool = False,
                 compr_model_name: str = "google-bert/bert-base-uncased",
                 compr_rate: int = 16,
                 compr_n_layers: int = None,
                 compr_every_n_layer: int = None,
                 compr_base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 compr_rms_norm: bool = False,
                 compr_mlp_hidden_dim: int = 8096,
                 compr_use_mlp: bool = True,
                 compr_linear_type: str = "concat",
                 lora: bool = False,
                 lora_compressor: bool = False,
                 training_form: str = "both",
                 training_stage: str = "stage1",
                 generation_top_k: int = 1,
                 lora_r: int = 16,
                 lora_r_compressor: int = None,
                 lora_target_modules: Optional[Union[str, Sequence[str]]] = None,
                 aria_rag_configuration: Optional[str] = None,
                 load_adapters: bool = True,
                 kbtc_training: bool = False,
                 optimize_mem_tokens: bool = False,
                 different_mem_tokens: bool = False,
                 attn_implementation: str = None,
                 _attn_implementation_autoset: bool = True,
                 ae_mode: str = "token",
                 max_new_tokens: int = 64,
                 stage2_retrieval_top_n: int = 1,
                 qr_input_scheme: str = QR_INPUT_SCHEME,
                 mads_semantic_model_name: str = "BAAI/bge-large-en-v1.5",
                 mads_semantic_model_revision: Optional[str] = None,
                 mads_semantic_model_resolved_revision: Optional[str] = None,
                 mtfrl_initialization_scheme: str = MTFRL_INITIALIZATION_SCHEME,
                 mtfrl_initialization_rank: Optional[int] = None,
                 mtfrl_hidden_width: Optional[int] = None,
                 cfrs_reconstruction_scheme: str = CFRS_RECONSTRUCTION_SCHEME,
                 cfrs_reconstruction_chunk_tokens: int = 128,
                 retrieval_straight_through_scheme: str = RETRIEVAL_STRAIGHT_THROUGH_SCHEME,
                 load_pretrained_checkpoint: bool = False,
                 device_map=None,
                 auto_map: dict = {
                     "AutoConfig": "openrlhf.models.modeling_aria.CLaRaConfig",
                     "AutoModel": "openrlhf.models.modeling_aria.CLaRa"
                 },
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name
        self.decoder_model_revision = decoder_model_revision
        self.decoder_model_resolved_revision = decoder_model_resolved_revision
        self.doc_max_length = doc_max_length
        self.query_max_length = query_max_length
        self.stage1_input_max_length = stage1_input_max_length
        self.stage2_input_max_length = stage2_input_max_length
        self.stage1_target_max_length = stage1_target_max_length
        self.stage2_target_max_length = stage2_target_max_length
        self.quantization = quantization
        self.sep = sep

        self.compr_model_name = compr_model_name
        self.compr_rate = compr_rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        self.compr_linear_type = compr_linear_type

        self.lora = lora
        self.lora_compressor = lora_compressor
        self.training_form = training_form
        self.lora_r = lora_r
        self.lora_r_compressor = lora_r_compressor or lora_r
        if lora_target_modules == "all-linear":
            # PEFT treats this exact string as its architecture-independent
            # all-Linear/Conv1D selector.  A one-element list would instead be
            # interpreted as a literal module suffix and silently match none.
            self.lora_target_modules: Union[str, List[str]] = "all-linear"
        elif lora_target_modules is None:
            self.lora_target_modules = ["q_proj"]
        else:
            self.lora_target_modules = list(lora_target_modules)
        self.aria_rag_configuration = aria_rag_configuration
        self.load_adapters = load_adapters
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        self.kbtc_training = kbtc_training
        self.training_stage = training_stage
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self._attn_implementation_autoset = _attn_implementation_autoset
        self.ae_mode = ae_mode
        self.max_new_tokens = max_new_tokens
        self.auto_map = auto_map
        self.load_pretrained_checkpoint = load_pretrained_checkpoint

        self.generation_top_k = generation_top_k
        self.stage2_retrieval_top_n = stage2_retrieval_top_n
        if qr_input_scheme != QR_INPUT_SCHEME:
            raise ValueError(
                f"Unsupported QR input scheme {qr_input_scheme!r}; "
                f"expected {QR_INPUT_SCHEME!r}"
            )
        self.qr_input_scheme = qr_input_scheme
        self.mads_semantic_model_name = mads_semantic_model_name
        self.mads_semantic_model_revision = mads_semantic_model_revision
        self.mads_semantic_model_resolved_revision = (
            mads_semantic_model_resolved_revision
        )
        if mtfrl_initialization_scheme != MTFRL_INITIALIZATION_SCHEME:
            raise ValueError(
                "Unsupported MTFRL initialization scheme "
                f"{mtfrl_initialization_scheme!r}; expected "
                f"{MTFRL_INITIALIZATION_SCHEME!r}"
            )
        self.mtfrl_initialization_scheme = mtfrl_initialization_scheme
        self.mtfrl_initialization_rank = mtfrl_initialization_rank
        self.mtfrl_hidden_width = mtfrl_hidden_width
        if cfrs_reconstruction_scheme != CFRS_RECONSTRUCTION_SCHEME:
            raise ValueError(
                "Unsupported CFRS reconstruction scheme "
                f"{cfrs_reconstruction_scheme!r}; expected "
                f"{CFRS_RECONSTRUCTION_SCHEME!r}"
            )
        if (
            not isinstance(cfrs_reconstruction_chunk_tokens, int)
            or isinstance(cfrs_reconstruction_chunk_tokens, bool)
            or cfrs_reconstruction_chunk_tokens <= 0
        ):
            raise ValueError("cfrs_reconstruction_chunk_tokens must be a positive integer")
        if retrieval_straight_through_scheme != RETRIEVAL_STRAIGHT_THROUGH_SCHEME:
            raise ValueError(
                "Unsupported retrieval straight-through scheme "
                f"{retrieval_straight_through_scheme!r}; expected "
                f"{RETRIEVAL_STRAIGHT_THROUGH_SCHEME!r}"
            )
        self.cfrs_reconstruction_scheme = cfrs_reconstruction_scheme
        self.cfrs_reconstruction_chunk_tokens = int(
            cfrs_reconstruction_chunk_tokens
        )
        self.retrieval_straight_through_scheme = retrieval_straight_through_scheme

        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora


# Utility functions
def add_memory_tokens_to_inputs(input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               n_mem_tokens: int,
                               tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add memory tokens to input sequences."""
    assert len(tokenizer.mem_tokens) == n_mem_tokens

    mem_tokens = torch.stack([tokenizer.mem_token_ids_pt] * input_ids.size(0), 0)
    assert len(mem_tokens) == input_ids.size(0)
    assert len(mem_tokens[0]) == n_mem_tokens

    input_ids = torch.cat([input_ids, mem_tokens], dim=1)
    memory_attention = torch.ones(
        input_ids.size(0), n_mem_tokens,
        device=attention_mask.device, dtype=attention_mask.dtype,
    )
    attention_mask = torch.cat([attention_mask, memory_attention], dim=1)

    return input_ids, attention_mask


def _pack_variable_encoder_memory_rows(
    source_rows: Sequence[Sequence[int]],
    memory_counts: Sequence[int],
    memory_token_ids: Sequence[int],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append real per-document memory slots before right-padding the batch.

    Padding a source to the global document ceiling *before* appending memory
    creates an internal attention hole and gives short documents the positional
    indices of a 1,024-token document.  Pack each row first so memory follows its
    source immediately, then pad only at the end of the complete row.
    """
    if not source_rows or len(source_rows) != len(memory_counts):
        raise ValueError("source_rows and memory_counts must be non-empty and aligned")
    if not memory_token_ids:
        raise ValueError("memory_token_ids must not be empty")
    packed_rows: List[List[int]] = []
    for row_index, (source, count_value) in enumerate(zip(source_rows, memory_counts)):
        source_ids = [int(token_id) for token_id in source]
        count = int(count_value)
        if not source_ids:
            raise ValueError(f"encoder source row {row_index} is empty")
        if count < 1 or count > len(memory_token_ids):
            raise ValueError(
                f"encoder memory count at row {row_index} must be in "
                f"[1, {len(memory_token_ids)}], got {count}"
            )
        packed_rows.append(
            source_ids + [int(token_id) for token_id in memory_token_ids[:count]]
        )

    max_length = max(len(row) for row in packed_rows)
    input_ids = torch.full(
        (len(packed_rows), max_length), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(packed_rows):
        input_ids[row_index, :len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, :len(row)] = 1
    return input_ids, attention_mask


def _fixed_memory_prompt_max_length(
    document_count: int,
    memory_tokens_per_document: int,
    document_max_length: int,
) -> int:
    """Reserve enough decoder input space for every fixed memory slot."""
    if min(document_count, memory_tokens_per_document, document_max_length) <= 0:
        raise ValueError("fixed memory prompt dimensions must be positive")
    return max(
        2048,
        document_count * memory_tokens_per_document + document_max_length + 256,
    )


def _prune_padded_memory_slots(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    query_position_mask: torch.Tensor,
    base_counts: torch.Tensor,
    memory_token_ids: torch.Tensor,
    slots_per_document: int,
    pad_token_id: int,
    padding_side: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove only source-length padding from fixed Phase-II memory blocks.

    The collator cannot know which documents the integrated retriever will
    select, so it reserves ``slots_per_document`` tokens for every document.
    Once compression has produced ``base_counts=floor(L_i / r)``, slots after
    that per-document boundary are padding rather than ACR memory.  Delete
    those slots from the decoder sequence and repad the batch, while retaining
    every real base slot (including slots softly suppressed by ACR).
    """
    tensors = (input_ids, attention_mask, labels, query_position_mask)
    if input_ids.dim() != 2 or any(tensor.shape != input_ids.shape for tensor in tensors):
        raise ValueError("decoder ids, attention, labels, and query mask must share shape (B, L)")
    if base_counts.dim() != 2 or base_counts.size(0) != input_ids.size(0):
        raise ValueError("base_counts must have shape (B, N)")
    if slots_per_document <= 0:
        raise ValueError("slots_per_document must be positive")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")

    counts = base_counts.to(device=input_ids.device, dtype=torch.long)
    if ((counts < 0) | (counts > slots_per_document)).any():
        raise ValueError("each base memory count must be in [0, slots_per_document]")
    memory_ids = memory_token_ids.to(input_ids.device)
    expected_slots = base_counts.size(1) * slots_per_document
    retained_rows = []

    for row_index in range(input_ids.size(0)):
        valid = attention_mask[row_index].bool()
        memory_positions = (
            torch.isin(input_ids[row_index], memory_ids) & valid
        ).nonzero(as_tuple=True)[0]
        if memory_positions.numel() != expected_slots:
            raise ValueError(
                "fixed Phase-II prompt has an unexpected memory-slot count: "
                f"row {row_index} found {memory_positions.numel()}, expected {expected_slots}"
            )

        keep = valid.clone()
        for doc_index, count_value in enumerate(counts[row_index]):
            block_start = doc_index * slots_per_document
            block = memory_positions[block_start:block_start + slots_per_document]
            keep[block[int(count_value.item()):]] = False
        retained_rows.append(keep.nonzero(as_tuple=True)[0])

    max_length = max(indices.numel() for indices in retained_rows)
    pruned_ids = input_ids.new_full((input_ids.size(0), max_length), pad_token_id)
    pruned_attention = attention_mask.new_zeros((input_ids.size(0), max_length))
    pruned_labels = labels.new_full((input_ids.size(0), max_length), IGNORE_INDEX)
    pruned_query_mask = query_position_mask.new_zeros(
        (input_ids.size(0), max_length), dtype=torch.bool
    )

    for row_index, indices in enumerate(retained_rows):
        length = indices.numel()
        start = max_length - length if padding_side == "left" else 0
        target = slice(start, start + length)
        pruned_ids[row_index, target] = input_ids[row_index, indices]
        pruned_attention[row_index, target] = attention_mask[row_index, indices]
        pruned_labels[row_index, target] = labels[row_index, indices]
        pruned_query_mask[row_index, target] = query_position_mask[row_index, indices].bool()

    return pruned_ids, pruned_attention, pruned_labels, pruned_query_mask


def build_pos_mask(pos_index: List[List[int]], N: int, device: torch.device) -> torch.Tensor:
    """Build positive mask for retrieval training."""
    if isinstance(pos_index, (list, tuple)):
        B = len(pos_index)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b, idxs in enumerate(pos_index):
            if len(idxs) > 0:
                mask[b, torch.as_tensor(idxs, device=device, dtype=torch.long)] = True
        return mask
    else:  # tensor [B, M]
        B, M = pos_index.shape
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for m in range(M):
            col = pos_index[:, m]
            v = col >= 0
            if v.any():
                mask[v, col[v]] = True
        return mask


def differentiable_topk_top_1(logits: torch.Tensor, k: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implements differentiable top-1 selection using Gumbel-Softmax."""
    y = logits / temperature
    y_soft = F.softmax(y, dim=-1).float()

    # Hard one-hot version
    index = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)

    # Straight-through estimator
    z = y_hard + y_soft - y_soft.detach()
    z = z.unsqueeze(1).to(logits.dtype)

    return z, index


def differentiable_topk(logits: torch.Tensor, k: int, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable top-k selection."""
    B, N = logits.shape
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= N:
        raise ValueError(f"straight-through top-k requires 1 <= k <= {N}, got {k!r}")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("straight-through top-k temperature must be finite and positive")
    perturbed = logits / max(temperature, 1e-6)

    # Hard top-k indices
    topk_vals, topk_idx = perturbed.topk(k, dim=-1)
    K_hard = torch.zeros(B, k, N, device=logits.device, dtype=logits.dtype)
    K_hard.scatter_(2, topk_idx.unsqueeze(-1), 1.0)

    # Soft distributions for each slot
    K_soft = torch.zeros_like(K_hard)
    taken = torch.zeros(B, N, device=logits.device, dtype=logits.dtype)

    for j in range(k):
        mask = (1.0 - taken.detach())
        masked = perturbed + (mask + 1e-8).log()
        pj = F.softmax(masked, dim=-1).float()
        K_soft[:, j, :] = pj
        taken = torch.clamp(taken + K_hard[:, j, :], max=1.0)

    # Straight-through estimator
    W = K_hard + (K_soft - K_soft.detach())
    return W, topk_idx


def _clara_st_select_candidate_memory(
    query_representations: torch.Tensor,
    candidate_memory: torch.Tensor,
    k: int,
    *,
    candidate_memory_counts: Optional[torch.Tensor] = None,
    temperature: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply CLaRa's hard-forward/soft-backward document selector.

    Candidates are padded to one tensor width while ``candidate_memory_counts``
    records each real ``K_i=max(1,floor(L_i/r))`` allocation. The document
    score is cosine similarity between the QR state and its masked memory mean.
    ``differentiable_topk`` supplies one ST row per selected slot, so the exact
    same formula is retained when ``N == k``.
    """
    if query_representations.ndim != 2:
        raise ValueError("CLaRa QR representations must have shape (B, H)")
    if candidate_memory.ndim != 4:
        raise ValueError("CLaRa candidate memory must have shape (B, N, K0, H)")
    batch, candidate_count, memory_count, hidden_size = candidate_memory.shape
    if query_representations.shape != (batch, hidden_size):
        raise ValueError("CLaRa QR and candidate-memory batch/hidden dimensions must align")
    if candidate_count < k:
        raise ValueError(
            f"CLaRa requires at least k={k} candidates per row, got N={candidate_count}"
        )
    if memory_count < 1:
        raise ValueError("CLaRa candidate representations cannot be empty")
    if not torch.isfinite(query_representations).all() or not torch.isfinite(
        candidate_memory
    ).all():
        raise ValueError("CLaRa selector received NaN or infinite representations")

    if candidate_memory_counts is None:
        candidate_memory_counts = torch.full(
            (batch, candidate_count),
            memory_count,
            device=candidate_memory.device,
            dtype=torch.long,
        )
    if candidate_memory_counts.shape != (batch, candidate_count):
        raise ValueError("CLaRa requires one memory count per candidate")
    if candidate_memory_counts.dtype == torch.bool or torch.is_floating_point(
        candidate_memory_counts
    ):
        raise ValueError("CLaRa candidate memory counts must use an integer dtype")
    if torch.any(candidate_memory_counts < 1) or torch.any(
        candidate_memory_counts > memory_count
    ):
        raise ValueError("CLaRa candidate memory counts must lie in [1, padded K]")
    positions = torch.arange(memory_count, device=candidate_memory.device).view(
        1, 1, -1
    )
    memory_mask = positions < candidate_memory_counts.to(
        candidate_memory.device
    ).unsqueeze(-1)
    document_representations = (
        candidate_memory.float() * memory_mask.unsqueeze(-1)
    ).sum(dim=2) / candidate_memory_counts.to(
        candidate_memory.device, dtype=torch.float32
    ).unsqueeze(-1)
    scores = torch.einsum(
        "bh,bnh->bn",
        F.normalize(query_representations.float(), dim=-1),
        F.normalize(document_representations, dim=-1),
    )
    weights, topk_indices = differentiable_topk(
        scores, k, temperature=temperature
    )
    selected = torch.einsum(
        "bkn,bnth->bkth", weights.to(candidate_memory.dtype), candidate_memory
    )
    return selected, topk_indices, scores, weights


class CLaRa(PreTrainedModel):
    """CLaRa: Unified Retrieval-Augmented Generation Model."""

    config_class = CLaRaConfig

    def __init__(self, cfg: CLaRaConfig):
        super().__init__(cfg)
        self.decoder_model_name = cfg.decoder_model_name
        self.decoder = self._create_decoder(cfg)
        self.doc_max_length = cfg.doc_max_length
        self.query_max_length = getattr(cfg, "query_max_length", 256)
        self.stage1_input_max_length = getattr(cfg, "stage1_input_max_length", 2048)
        self.stage2_input_max_length = getattr(cfg, "stage2_input_max_length", 1024)
        self.stage1_target_max_length = getattr(cfg, "stage1_target_max_length", 512)
        self.stage2_target_max_length = getattr(cfg, "stage2_target_max_length", 128)

        print(f'Base decoder parameters: {self.decoder.num_parameters()}')

        # Model configuration
        self.compr_model_name = cfg.compr_model_name
        self.training_form = cfg.training_form
        self.lora = cfg.lora
        self.adapter_keys = []
        self.compr = None

        # Initialize LoRA adapters if needed
        if cfg.lora and not getattr(cfg, 'pure_inference', False):
            self._setup_lora_adapters(cfg)

        print(f'Model adapter keys: {self.adapter_keys}')

        # Initialize tokenizer and resize embeddings
        self.decoder_tokenizer = self._create_decoder_tokenizer(cfg)
        self._record_decoder_model_revision(cfg)
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self._configure_generation_config()

        # Model parameters
        self.generation_top_k = cfg.generation_top_k
        self.training_stage = cfg.training_stage
        self.stage2_retrieval_top_n = cfg.stage2_retrieval_top_n
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')

        self.n_mem_tokens = max(1, self.doc_max_length // self.compr_rate)
        self.hidden_size = self.decoder.config.hidden_size

        # Setup adapters and memory token optimization
        if self.lora:
            self._setup_adapter_training()
        else:
            print(f'Total trainable parameters: {self.num_parameters(only_trainable=True)}')

        self._prepare_mem_tokens_optimization()

        # RAG Enhancement Pipeline (可选; 通过 setup_rag_pipeline() 挂载)
        self.rag_pipeline: Optional[RAGEnhancementPipeline] = None
        self._rag_diagnostics: List[RAGDiagnostics] = []
        self._oracle_pool_records: List[OraclePoolRecord] = []

        # ── RAG 组件句柄：由 setup_rag_pipeline() 按配置挂载 ───────────────
        self._cfrs: Optional[CompressionFidelityReranker] = None
        self._acr:  Optional[AdaptiveCompressionAllocator] = None
        self._mtfrl: Optional[MemoryTokenFeedbackRetriever] = None
        self._rag_config: RAGPipelineConfig = RAGPipelineConfig()

        # ── BGE projection W_BGE: ℝ^{d_h} → ℝ^{1024} (fixed, non-trainable) ──────
        # Maps QR output into BGE-large-en-v1.5 dense-search space.
        # Fitted once with cosine loss and AdamW, then frozen during training.
        self._bge_projection: Optional[nn.Linear] = None
        self._bge_projection_metadata: Optional[Dict[str, Any]] = None

        # ── MTFRL projection head: 2-layer MLP (4096→2048→1024) ≈10.5M params ────
        # Maps aggregated memory token mean to retriever embedding space for
        # second-round dense retrieval.
        self._mtfrl_projection: Optional[nn.Sequential] = None

    # ── RAG Enhancement 公开接口 ──────────────────────────────────────────────

    def setup_rag_pipeline(self,
                           corpus_docs: List[str],
                           corpus_doc_ids: Optional[List[str]] = None,
                           doc_embeddings: Optional[torch.Tensor] = None,
                           rag_config: Optional[RAGPipelineConfig] = None,
                           initialize_missing_mtfrl: bool = False,
                           bm25_index: Optional[_BM25Index] = None,
                           corpus_page_ids: Optional[List[str]] = None) -> None:
        """
        挂载 RAG Enhancement Pipeline 到本 CLaRa 实例。

        挂载后，generate_from_questions 在未传入 documents 参数时，
        会自动走 QCA→AHR→IGFR→MADS→CCEF 五级管道，而不是原有的 URL 检索。

        Args:
            corpus_docs:    文档字符串列表（检索语料库）。
            corpus_doc_ids: 可选文档 ID，默认 "0","1",...
            corpus_page_ids: 与语料逐行对齐的规范化页面 ID；默认退化为文档 ID。
            doc_embeddings: 预计算的密集向量 (N, D) tensor，有则同时启用向量检索臂。
            rag_config:     RAGPipelineConfig；None 则使用默认参数。
        """
        cfg = rag_config or RAGPipelineConfig(top_k=self.stage2_retrieval_top_n or 5)
        recorded_mads_name = getattr(
            self.config,
            "mads_semantic_model_name",
            "BAAI/bge-large-en-v1.5",
        )
        recorded_mads_revision = getattr(
            self.config, "mads_semantic_model_resolved_revision", None
        ) or getattr(self.config, "mads_semantic_model_revision", None)
        if cfg.mads_semantic_model_name != recorded_mads_name:
            raise ValueError(
                "Runtime MADS semantic model must match checkpoint provenance: "
                f"{cfg.mads_semantic_model_name!r} != {recorded_mads_name!r}"
            )
        if cfg.mads_semantic_model_revision is None:
            cfg.mads_semantic_model_revision = recorded_mads_revision
        elif (
            recorded_mads_revision is not None
            and cfg.mads_semantic_model_revision != recorded_mads_revision
        ):
            raise ValueError(
                "Runtime MADS semantic revision must match checkpoint provenance: "
                f"{cfg.mads_semantic_model_revision!r} != {recorded_mads_revision!r}"
            )
        if not corpus_docs:
            raise ValueError("ARIA retrieval corpus must not be empty")
        if corpus_doc_ids is None:
            corpus_doc_ids = [str(index) for index in range(len(corpus_docs))]
        if len(corpus_doc_ids) != len(corpus_docs):
            raise ValueError("corpus_doc_ids must align one-to-one with corpus_docs")
        if len(set(corpus_doc_ids)) != len(corpus_doc_ids):
            raise ValueError("corpus_doc_ids must be unique")
        if corpus_page_ids is None:
            corpus_page_ids = list(corpus_doc_ids)
        if len(corpus_page_ids) != len(corpus_docs):
            raise ValueError("corpus_page_ids must align one-to-one with corpus_docs")
        if any(
            not isinstance(page_id, str) or not page_id.strip()
            for page_id in corpus_page_ids
        ):
            raise ValueError("corpus_page_ids must contain non-empty strings")
        if any(not isinstance(text, str) or not text.strip() for text in corpus_docs):
            raise ValueError("corpus_docs must contain only non-empty strings")
        if cfg.compression_rate is None:
            cfg.compression_rate = self.compr_rate
        if (
            cfg.use_ahr
            or cfg.use_mads
            or cfg.second_retrieval_mode != "disabled"
        ) and doc_embeddings is None:
            raise ValueError(
                "ARIA AHR/MADS/MTFRL requires the shared full-corpus BGE index"
            )
        if doc_embeddings is not None and (doc_embeddings.dim() != 2 or doc_embeddings.size(-1) != 1024):
            raise ValueError(
                "doc_embeddings must have shape (corpus_size, 1024) for BGE-large-en-v1.5"
            )
        if doc_embeddings is not None and doc_embeddings.size(0) != len(corpus_docs):
            raise ValueError("doc_embeddings rows must align one-to-one with corpus_docs")
        if doc_embeddings is not None and not _tensor_is_finite_in_chunks(doc_embeddings):
            raise ValueError("doc_embeddings contains NaN or infinite values")
        if cfg.use_ahr and self._bge_projection is None:
            raise RuntimeError("AHR requires the fitted, frozen W_BGE projection from Appendix A.1")
        if cfg.use_ahr:
            if (
                self._bge_projection.in_features != self.hidden_size
                or self._bge_projection.out_features != 1024
            ):
                raise ValueError("W_BGE must have shape hidden_size -> 1024")
            if any(parameter.requires_grad for parameter in self._bge_projection.parameters()):
                raise ValueError("W_BGE must be frozen before ARIA retrieval is attached")
            metadata = self._bge_projection_metadata or {}
            expected_metadata = {
                "base_model": self.decoder_model_name,
                "bge_model": "BAAI/bge-large-en-v1.5",
                "sample_count": 50_000,
                "epochs": 2,
                "batch_size": 128,
                "learning_rate": 5e-4,
                "query_max_length": 256,
                "text_sha256_scheme": "utf8-strip-v1",
                "qr_input_scheme": QR_INPUT_SCHEME,
            }
            decoder_revision = getattr(
                self.config, "decoder_model_resolved_revision", None
            )
            if decoder_revision is not None:
                expected_metadata["base_model_revision_resolved"] = decoder_revision
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"W_BGE metadata {key!r} must be {expected!r}, "
                        f"got {metadata.get(key)!r}"
                    )
            for key in (
                "query_sha256",
                "passage_id_sha256",
                "passage_text_sha256",
                "test_url_sha256",
            ):
                value = metadata.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"W_BGE metadata requires {key}")
            if not isinstance(metadata.get("seed"), int):
                raise ValueError("W_BGE metadata requires an integer fitting seed")
            expected_test_digest = getattr(
                self.config, "aria_test_url_sha256", None
            )
            if (
                expected_test_digest is not None
                and metadata["test_url_sha256"] != expected_test_digest
            ):
                raise ValueError(
                    "W_BGE and checkpoint use different official test URL sets"
                )
        if cfg.use_mtfrl and self._mtfrl_projection is None:
            if initialize_missing_mtfrl:
                self.setup_mtfrl_projection(initialize_from_bge=True)
            else:
                raise RuntimeError(
                    "ARIA MTFRL inference requires checkpoint projection weights "
                    "or explicit projection initialization"
                )
        if cfg.use_mtfrl:
            linear_layers = [
                module
                for module in self._mtfrl_projection.modules()
                if isinstance(module, nn.Linear)
            ]
            expected_hidden = _mtfrl_hidden_width(self.hidden_size, 1024)
            if (
                len(linear_layers) != 2
                or linear_layers[0].in_features != self.hidden_size
                or linear_layers[0].out_features != expected_hidden
                or linear_layers[1].in_features != expected_hidden
                or linear_layers[1].out_features != 1024
            ):
                raise ValueError(
                    "MTFRL projection must be H -> H/2 -> 1024"
                )
        self.rag_pipeline = RAGEnhancementPipeline.from_corpus(
            corpus_docs=corpus_docs,
            corpus_doc_ids=corpus_doc_ids,
            doc_embeddings=doc_embeddings,
            config=cfg,
            bm25_index=bm25_index,
            corpus_page_ids=corpus_page_ids,
        )

        # ── 创新点1: CFRS ──────────────────────────────────────────────────
        self._cfrs = CompressionFidelityReranker() if cfg.use_cfrs else None

        # ── 创新点2: ACR ───────────────────────────────────────────────────
        self._acr  = AdaptiveCompressionAllocator(
            min_ratio=cfg.acr_min_token_ratio,
            max_ratio=cfg.acr_max_token_ratio,
            beta=cfg.acr_sigmoid_beta,
            eps=cfg.numerical_epsilon,
        ) if cfg.acr_allocation_mode != "full" else None

        # ── 创新点3: MTFRL ─────────────────────────────────────────────────
        # Share the dense corpus index validated by setup_rag_pipeline with MTFRL.
        self._mtfrl = MemoryTokenFeedbackRetriever(
            # Share AHR's normalized CPU matrix instead of materializing a second
            # KILT-scale copy for the feedback retriever.
            corpus_embeddings=self.rag_pipeline.ahr.dense_embeddings,
            corpus_docs=corpus_docs,
            corpus_ids=corpus_doc_ids,
            embeddings_are_normalized=True,
        ) if cfg.second_retrieval_mode != "disabled" else None
        self._rag_config = cfg   # 保存配置供后续方法读取

        print(f"[CLaRa] RAG Enhancement Pipeline 已挂载 "
              f"(语料={len(corpus_docs)} 篇, top_k={cfg.top_k}, "
              f"CFRS={'on' if self._cfrs else 'off'}, "
              f"allocation={cfg.acr_allocation_mode}, "
              f"second_retrieval={cfg.second_retrieval_mode})")

    def get_rag_diagnostics(self) -> List[RAGDiagnostics]:
        """返回最近所有调用积累的 RAGDiagnostics 列表。"""
        return list(self._rag_diagnostics)

    def get_oracle_pool_records(self) -> List[OraclePoolRecord]:
        """Return Oracle pools created since the last diagnostic reset."""
        return list(self._oracle_pool_records)

    def clear_rag_diagnostics(self) -> None:
        """清空诊断记录。"""
        self._rag_diagnostics.clear()
        self._oracle_pool_records.clear()

    # ── BGE Projection & MTFRL Head ──────────────────────────────────────────

    def setup_bge_projection(self, bge_dim: int = 1024, freeze: bool = True) -> None:
        """
        初始化 BGE 投影矩阵 W_BGE: ℝ^{d_h} → ℝ^{bge_dim}。

        将 QR 输出的 query representation 映射到 BGE-large-en-v1.5 的密集搜索空间。
        该投影在训练时冻结（frozen），仅在拟合阶段被学习。

        Args:
            bge_dim: BGE embedding dimension (default 1024 for BGE-large-en-v1.5)
        """
        self._bge_projection = nn.Linear(
            self.hidden_size, bge_dim, bias=False
        )
        # It is trainable only during the dedicated alignment stage, then frozen.
        for p in self._bge_projection.parameters():
            p.requires_grad = not freeze
        print(f"[CLaRa] BGE projection initialized: {self.hidden_size} → {bge_dim}")

    def fit_bge_projection(
        self,
        query_texts: List[str],
        bge_embeddings: torch.Tensor,
        max_length: Optional[int] = None,
        epochs: int = 2,
        batch_size: int = 128,
        learning_rate: float = 5e-4,
        seed: int = 42,
        require_paper_sample_count: bool = True,
    ) -> List[float]:
        """
        用余弦距离损失和 AdamW 拟合 W_BGE，使 QR 输出在投影后匹配 BGE 嵌入。

        Args:
            query_texts: 用于拟合的查询文本列表
            bge_embeddings: 对应的 BGE 嵌入 (N, bge_dim)
            max_length: 最大输入长度
        """
        if max_length is None:
            max_length = self.query_max_length
        if max_length != 256:
            raise ValueError("The paper protocol fits W_BGE with query max length 256")
        if len(query_texts) != bge_embeddings.size(0):
            raise ValueError("query_texts and BGE passage embeddings must have equal length")
        if require_paper_sample_count and len(query_texts) != 50_000:
            raise ValueError(
                f"Appendix A.1 requires exactly 50,000 alignment pairs, got {len(query_texts)}"
            )
        if self._bge_projection is None:
            self.setup_bge_projection(bge_dim=bge_embeddings.size(-1), freeze=False)
        for parameter in self._bge_projection.parameters():
            parameter.requires_grad = True

        device = self.decoder.device
        self.eval()
        optimizer = torch.optim.AdamW(self._bge_projection.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        epoch_losses: List[float] = []

        for _ in range(epochs):
            order = torch.randperm(len(query_texts), generator=generator).tolist()
            running_loss = 0.0
            steps = 0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                batch = [query_texts[index] for index in indices]
                q_tok = self._prepare_query_inputs(batch, max_length=max_length)
                with torch.no_grad():
                    qr_out = self._compr_query_reasoner_stage2(
                        q_tok["input_ids"].to(device),
                        q_tok["attention_mask"].to(device),
                    ).float()
                targets = bge_embeddings[indices].to(device=device, dtype=torch.float32)
                projected = self._bge_projection(qr_out)
                loss = (1.0 - F.cosine_similarity(projected, targets, dim=-1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach())
                steps += 1
            epoch_losses.append(running_loss / max(steps, 1))

        for parameter in self._bge_projection.parameters():
            parameter.requires_grad = False
        self._bge_projection.eval()
        return epoch_losses

    def setup_mtfrl_projection(self, initialize_from_bge: bool = True) -> None:
        """
        Initialize the paper's H -> H/2 -> 1024 GELU feedback head.

        将聚合后的 memory token 均值映射到检索器嵌入空间，
        用于第二轮密集检索 (DenseSearch)。
        """
        hidden = self.hidden_size
        bge_dim = (
            self._bge_projection.out_features
            if self._bge_projection is not None
            else 1024
        )
        projection_hidden = _mtfrl_hidden_width(hidden, bge_dim)
        self._mtfrl_projection = nn.Sequential(
            nn.Linear(hidden, projection_hidden, bias=True),
            nn.GELU(),
            nn.Linear(projection_hidden, bge_dim, bias=True),
        ).to(self.decoder.device)
        configured_width = getattr(self.config, "mtfrl_hidden_width", None)
        if configured_width is not None and int(configured_width) != projection_hidden:
            raise ValueError(
                "Checkpoint MTFRL hidden width does not match H/2"
            )
        self.config.mtfrl_hidden_width = projection_hidden
        if initialize_from_bge:
            if self._bge_projection is None:
                raise RuntimeError("W_BGE is required to initialize P_fb")
            if (
                self._bge_projection.in_features != hidden
                or self._bge_projection.out_features != bge_dim
            ):
                raise ValueError("W_BGE dimensions do not match the MTFRL head")
            if projection_hidden < 2 * bge_dim:
                raise ValueError(
                    "W_BGE-derived GELU identity requires H/2 >= 2*BGE_dim"
                )
            first = self._mtfrl_projection[0]
            final = self._mtfrl_projection[2]
            with torch.no_grad():
                first.weight.zero_()
                first.bias.zero_()
                final.weight.zero_()
                final.bias.zero_()
                w_bge = self._bge_projection.weight.to(
                    device=first.weight.device, dtype=first.weight.dtype
                )
                # GELU(z)-GELU(-z)=z.  These paired channels therefore make
                # P_fb(x) exactly W_BGE(x) at initialization; wider heads
                # (e.g. Qwen's 2560 channel bottleneck) leave the tail zero.
                first.weight[:bge_dim].copy_(w_bge)
                first.weight[bge_dim : 2 * bge_dim].copy_(-w_bge)
                identity = torch.eye(
                    bge_dim, device=final.weight.device, dtype=final.weight.dtype
                )
                final.weight[:, :bge_dim].copy_(identity)
                final.weight[:, bge_dim : 2 * bge_dim].copy_(-identity)
            self.config.mtfrl_initialization_scheme = MTFRL_INITIALIZATION_SCHEME
            self.config.mtfrl_initialization_rank = None
        # 投影头使用全参数训练（非 LoRA）
        total = sum(p.numel() for p in self._mtfrl_projection.parameters())
        print(f"[CLaRa] MTFRL projection head initialized: "
              f"{hidden} → {projection_hidden} → {bge_dim} ({total/1e6:.1f}M params)")

    def _compute_mtfrl_feedback_query(
        self,
        memory_embeddings: torch.Tensor,
        effective_counts: Optional[torch.Tensor] = None,
        document_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从 memory token embeddings 计算 MTFRL 反馈查询向量。

        实现论文中的:
        q_{fb} = Normalize(1/N Σ_i 1/T_i Σ_t m_i^{(t)})

        Args:
            memory_embeddings: (B, N, T, H) 或 (B*N, T, H)
        Returns:
            q_fb: (B, 1024) — 归一化后的反馈查询向量
        """
        if memory_embeddings.dim() != 4:
            raise ValueError("MTFRL aggregation requires (batch, documents, tokens, hidden)")
        batch, n_docs, n_tokens, _ = memory_embeddings.shape
        if n_docs != 5:
            raise ValueError("the paper's MTFRL feedback query requires exactly five documents")
        if effective_counts is None or effective_counts.shape != (batch, n_docs):
            raise ValueError("MTFRL requires one hard T_i count per document")
        if effective_counts.dtype == torch.bool or torch.is_floating_point(effective_counts):
            raise ValueError("MTFRL T_i counts must use an integer dtype")
        counts = effective_counts.to(memory_embeddings.device, torch.long)
        if (counts < 1).any() or (counts > n_tokens).any():
            raise ValueError("MTFRL requires every hard prefix T_i to lie in [1, T]")
        positions = torch.arange(n_tokens, device=memory_embeddings.device).view(1, 1, -1)
        active = positions < counts.unsqueeze(-1)
        per_document = (
            memory_embeddings * active.to(memory_embeddings.dtype).unsqueeze(-1)
        ).sum(dim=2) / counts.to(memory_embeddings.dtype).unsqueeze(-1)
        if document_mask is None:
            valid_documents = torch.ones(
                batch, n_docs, device=memory_embeddings.device, dtype=torch.bool
            )
        else:
            valid_documents = document_mask.to(
                memory_embeddings.device, dtype=torch.bool
            )
            if (
                valid_documents.shape != (batch, n_docs)
                or not bool(valid_documents.all())
            ):
                raise ValueError("the paper's MTFRL path requires five real documents")
        q_memory = per_document.mean(dim=1)
        q_memory = F.normalize(
            q_memory.float(), dim=-1, eps=self._rag_config.numerical_epsilon
        )

        if self._mtfrl_projection is None:
            raise RuntimeError("MTFRL projection head has not been initialized or loaded")
        projection_device = self._mtfrl_projection[0].weight.device
        if projection_device != q_memory.device:
            raise RuntimeError(
                "MTFRL projection and memory tensors must be placed together before forward"
            )
        projected = self._mtfrl_projection(
            q_memory.to(self._mtfrl_projection[0].weight.dtype)
        )
        # Dense search uses cosine similarity, so normalize P_fb's output as a
        # retrieval representation as well; this does not change Eq. (MTFRL)'s
        # required input ordering above.
        return F.normalize(
            projected.float(), dim=-1, eps=self._rag_config.numerical_epsilon
        )

    def _compute_first_pass_feedback_query(
        self,
        first_pass: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool exactly the hard ``T_i`` prefix of five sigmoid-gated rows."""
        return self._compute_mtfrl_feedback_query(
            first_pass["memory"],
            effective_counts=first_pass["effective_counts"],
            document_mask=first_pass["document_mask"],
        )

    def _create_decoder(self, cfg: CLaRaConfig) -> AutoModelForCausalLM:
        """Create and configure the decoder model."""
        revision = (
            cfg.decoder_model_resolved_revision or cfg.decoder_model_revision
        )
        if not torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=False,
                device_map=cfg.device_map
            )

        if cfg.quantization == "no":
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
                device_map=cfg.device_map
            )
        elif cfg.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                revision=revision,
                quantization_config=quant_config,
                attn_implementation=cfg.attn_implementation,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=False,
                device_map=cfg.device_map
            )
        elif cfg.quantization == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype='bfloat16',
            )
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                revision=revision,
                quantization_config=quant_config,
                attn_implementation=cfg.attn_implementation,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=False,
                device_map=cfg.device_map
            )
        else:
            raise NotImplementedError(
                f"Quantization {cfg.quantization} requires a registered decoder implementation"
            )

    def _setup_lora_adapters(self, cfg: CLaRaConfig):
        """Setup LoRA adapters based on training stage."""
        peft_config = self._get_peft_config(lora_r=cfg.lora_r)

        if cfg.training_stage == "stage1" and cfg.load_adapters:
            print('Loading compressor encoder adapter for Phase I')
            self.decoder.add_adapter(peft_config, 'encoder_adapter')
            self.adapter_keys.append('encoder_adapter')
        elif cfg.training_stage == "stage2" and cfg.load_adapters:
            if 'encoder_adapter' not in self.adapter_keys:
                self.decoder.add_adapter(peft_config, 'encoder_adapter')
                self.adapter_keys.append('encoder_adapter')
            if 'decoder_adapter' not in self.adapter_keys:
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
            if 'query_reasoner_adapter' not in self.adapter_keys:
                self.decoder.add_adapter(peft_config, 'query_reasoner_adapter')
                self.adapter_keys.append('query_reasoner_adapter')
        elif cfg.training_stage == 'stage1_2':
            if not cfg.load_adapters:
                print('Loading decoder adapter for stage1_2')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
            elif cfg.load_adapters:
                print('Loading encoder and decoder adapter for stage1_2')
                self.decoder.add_adapter(peft_config, 'encoder_adapter')
                self.adapter_keys.append('encoder_adapter')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
        elif cfg.training_stage == 'stage2_reasoning':
            if not cfg.load_adapters:
                print('Loading decoder adapter for stage2_reasoning')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')

    def _setup_adapter_training(self):
        """Setup adapters for training."""
        for adapter_key in self.adapter_keys:
            self.decoder.set_adapter(adapter_key)
            print(f'Adapter {adapter_key} trainable parameters: {self.num_parameters(only_trainable=True)}')
        self._set_all_adapters()

    def _configure_generation_config(self):
        """Configure generation parameters."""
        self.decoder.generation_config.top_p = None
        self.decoder.generation_config.temperature = None
        self.decoder.generation_config.pad_token_id = self.decoder_tokenizer.pad_token_id

    @staticmethod
    def _create_decoder_tokenizer(cfg: CLaRaConfig) -> AutoTokenizer:
        """Create and configure the decoder tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.decoder_model_name,
            revision=(
                cfg.decoder_model_resolved_revision or cfg.decoder_model_revision
            ),
            use_fast=True,
            padding_side='left'
        )

        # Define special tokens
        n_mem_tokens = max(1, cfg.doc_max_length // cfg.compr_rate)
        existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])

        if cfg.different_mem_tokens:
            mem_tokens = [f'<MEM{i}>' for i in range(n_mem_tokens)]
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + mem_tokens + ['<AE>', '<ENC>', '<SEP>']
            })
            tokenizer.mem_tokens = mem_tokens
        else:
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + ['<MEM>', '<AE>', '<ENC>', '<SEP>']
            })
            tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens

        tokenizer.mem_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.mem_tokens]
        tokenizer.mem_token_ids_pt = torch.LongTensor(tokenizer.mem_token_ids)

        # Additional special tokens
        tokenizer.ae_token = '<AE>'
        tokenizer.ae_token_id = tokenizer.convert_tokens_to_ids('<AE>')
        tokenizer.enc_token = '<ENC>'
        tokenizer.sep_token = '<SEP>'
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids('<SEP>')

        # Handle model-specific tokens
        if tokenizer.bos_token is None and 'qwen' in cfg.decoder_model_name.lower():
            tokenizer.bos_token = tokenizer.special_tokens_map['additional_special_tokens'][0]
            tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

        if tokenizer.eos_token is None and "qwen" in cfg.decoder_model_name.lower():
            tokenizer.eos_token = tokenizer.special_tokens_map['additional_special_tokens'][1]
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # KBTC training tokens
        if cfg.kbtc_training:
            tokenizer.add_special_tokens({'additional_special_tokens': ['<KBTC>']})
            tokenizer.kbtc_token = '<KBTC>'
            tokenizer.kbtc_token_id = tokenizer.convert_tokens_to_ids('<KBTC>')

        # Set pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.bos_token_id

        print(f'Memory token count: {n_mem_tokens}')
        return tokenizer

    def _record_decoder_model_revision(self, cfg: CLaRaConfig) -> None:
        """Bind decoder and tokenizer to one exact Hub commit when available."""
        candidates = {
            value.lower()
            for value in (
                getattr(getattr(self.decoder, "config", None), "_commit_hash", None),
                getattr(self.decoder_tokenizer, "init_kwargs", {}).get("_commit_hash")
                if isinstance(
                    getattr(self.decoder_tokenizer, "init_kwargs", None), Mapping
                )
                else None,
            )
            if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value)
        }
        if len(candidates) > 1:
            raise RuntimeError(
                "Base decoder and tokenizer resolved to different Hub commits: "
                + ", ".join(sorted(candidates))
            )
        loaded_revision = next(iter(candidates)) if candidates else None
        declared_revision = cfg.decoder_model_revision
        if isinstance(declared_revision, str) and re.fullmatch(
            r"[0-9a-fA-F]{40}", declared_revision
        ):
            declared_commit = declared_revision.lower()
            if loaded_revision is not None and loaded_revision != declared_commit:
                raise RuntimeError(
                    f"Requested base-model commit {declared_commit} but loaded "
                    f"{loaded_revision}"
                )
            loaded_revision = declared_commit
        recorded_revision = cfg.decoder_model_resolved_revision
        if (
            recorded_revision is not None
            and loaded_revision is not None
            and str(recorded_revision).lower() != loaded_revision
        ):
            raise RuntimeError(
                "Loaded base-model revision does not match checkpoint provenance: "
                f"{loaded_revision} != {recorded_revision}"
            )
        if loaded_revision is None and isinstance(recorded_revision, str):
            loaded_revision = recorded_revision.lower()
        cfg.decoder_model_resolved_revision = loaded_revision
        self.decoder_model_revision = cfg.decoder_model_revision
        self.decoder_model_resolved_revision = loaded_revision

    def _get_peft_config(self, lora_r: int) -> LoraConfig:
        """Build the PEFT configuration."""
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=2*lora_r,
            target_modules=self.config.lora_target_modules,
            lora_dropout=0.1,
            bias="none",
        )

    def configure_clara_phase2_trainable_parameters(self) -> None:
        """Freeze CLaRa's compressor/base and expose only QR + generator LoRA.

        PEFT changes adapter activation while the model moves between the QR,
        compressor, and generator passes.  The optimizer contract is therefore
        established explicitly from parameter identity before AdamW is built;
        the compressor pass is additionally recomputed under ``no_grad``.
        """
        if self.training_stage != "stage2" or getattr(
            self.config, "aria_rag_configuration", None
        ) != "clara_baseline":
            raise ValueError("CLaRa Phase-II freezing is only valid for clara_baseline")
        required = {
            "encoder_adapter",
            "query_reasoner_adapter",
            "decoder_adapter",
        }
        if set(self.adapter_keys) != required:
            raise RuntimeError(
                "CLaRa Phase II requires exactly encoder, query-reasoner, and "
                f"decoder adapters; got {sorted(self.adapter_keys)}"
            )
        self.decoder.set_adapter(["query_reasoner_adapter", "decoder_adapter"])
        for name, parameter in self.named_parameters():
            parameter.requires_grad = (
                "query_reasoner_adapter" in name or "decoder_adapter" in name
            )
        trainable = [
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
        if not trainable or not all(
            "query_reasoner_adapter" in name or "decoder_adapter" in name
            for name in trainable
        ):
            raise RuntimeError("CLaRa trainable-parameter freezing failed closed")
        for adapter_name in ("query_reasoner_adapter", "decoder_adapter"):
            if not any(adapter_name in name for name in trainable):
                raise RuntimeError(f"CLaRa trainable adapter {adapter_name!r} is empty")

    def _prepare_mem_tokens_optimization(self):
        """Setup memory token optimization if enabled."""
        if self.config.optimize_mem_tokens and self.compr is None:
            # Enable gradients for input embeddings
            self.decoder.get_input_embeddings().weight.requires_grad = True

            # Apply hook to zero gradients except for memory tokens
            def hook(grad):
                mask = torch.zeros_like(grad)
                mask[self.decoder_tokenizer.mem_token_ids] = 1.0
                return grad * mask

            self.decoder.get_input_embeddings().weight.register_hook(hook)

    def _set_all_adapters(self):
        """Activate all adapters for training."""
        if len(self.adapter_keys) > 0:
            self.decoder.set_adapter(self.adapter_keys)

    # Core compression and generation methods
    def _extract_memory_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract variable per-document memory tokens into a padded batch."""
        token_mask = torch.isin(
            input_ids, self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        ) & attention_mask.bool()
        counts = token_mask.sum(dim=1).long()
        if (counts == 0).any():
            raise ValueError("every document/query must retain at least one memory token")
        max_tokens = self.n_mem_tokens
        padded = hidden_states.new_zeros(hidden_states.size(0), max_tokens, hidden_states.size(-1))
        ranks = token_mask.long().cumsum(dim=1) - 1
        batch_indices, sequence_indices = token_mask.nonzero(as_tuple=True)
        padded[batch_indices, ranks[batch_indices, sequence_indices]] = hidden_states[
            batch_indices, sequence_indices
        ]
        return padded, counts, token_mask

    def compress(self, enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress input documents."""
        if self.compr:
            return self.compr(enc_input_ids, enc_attention_mask)
        else:
            return self._compr_decoder(enc_input_ids, enc_attention_mask)

    def _compress_with_fidelity(self,
                                enc_input_ids: torch.Tensor,
                                enc_attention_mask: torch.Tensor,
                                return_alignment_context: bool = False,
                                ) -> Union[
                                    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                                    Tuple[
                                        torch.Tensor,
                                        torch.Tensor,
                                        Optional[torch.Tensor],
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                    ],
                                ]:
        """
        Run the compressor and optionally expose reconstruction targets.

        Returns:
            compressed_embs: (N, n_mem_tokens, hidden)
            mse_loss:        scalar — 原有的批量平均 MSE（用于训练 loss，保持不变）
            compatibility slot: always None
            reconstruction context (when requested): truncated source IDs,
                non-special content mask, and pre-ACR memory-token counts
        """
        if self.compr:
            # External-compressor path returns its standard compression outputs.
            compressed, mse_loss = self.compr(enc_input_ids, enc_attention_mask)
            if return_alignment_context:
                raise RuntimeError(
                    "CFRS reconstruction requires the integrated encoder-adapter compressor"
                )
            return compressed, mse_loss, None

        # Reuse the integrated compressor pass; CFRS then performs the old
        # contract's additional frozen-decoder reconstruction pass.
        assert enc_input_ids.size() == enc_attention_mask.size()

        if 'encoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('encoder_adapter')
        else:
            # Training stages without encoder_adapter use standard compression.
            compressed, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
            if return_alignment_context:
                raise RuntimeError("encoder_adapter is required for CFRS reconstruction")
            return compressed, mse_loss, None

        emb = self.decoder(
            input_ids=enc_input_ids,
            attention_mask=enc_attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]

        mask         = torch.isin(enc_input_ids,
                                  self.decoder_tokenizer.mem_token_ids_pt.to(enc_input_ids.device))
        attn         = enc_attention_mask.bool()
        mem_mask     = mask & attn
        non_mem_mask = (~mask) & attn
        special_ids = torch.as_tensor(
            self.decoder_tokenizer.all_special_ids,
            device=enc_input_ids.device,
            dtype=enc_input_ids.dtype,
        )
        special_mask = (
            torch.isin(enc_input_ids, special_ids)
            if special_ids.numel()
            else torch.zeros_like(enc_input_ids, dtype=torch.bool)
        )
        source_content_mask = non_mem_mask & ~special_mask
        mem_len     = mem_mask.sum(dim=1)
        non_mem_len = non_mem_mask.sum(dim=1)

        if (mem_len == 0).any():
            raise ValueError("Some samples have no memory tokens")
        if (non_mem_len == 0).any():
            raise ValueError("Some samples have no non-memory tokens")
        mem_mean     = (emb * mem_mask.unsqueeze(-1)).sum(dim=1) / mem_len.unsqueeze(-1)
        non_mem_mean = (emb * non_mem_mask.unsqueeze(-1)).sum(dim=1) / non_mem_len.unsqueeze(-1)

        # 批量平均 MSE（用于 training loss，与原始行为完全一致）
        mse_loss = F.mse_loss(non_mem_mean, mem_mean, reduction='mean')

        compressed_embs, memory_counts, _ = self._extract_memory_embeddings(
            emb, enc_input_ids, enc_attention_mask
        )
        if return_alignment_context:
            if (source_content_mask.sum(dim=1) == 0).any():
                raise ValueError(
                    "CFRS requires a non-special source position for every document"
                )
            return (
                compressed_embs,
                mse_loss,
                None,
                enc_input_ids.detach(),
                source_content_mask,
                memory_counts,
            )
        return compressed_embs, mse_loss, None

    def _compr_decoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use decoder as compressor."""
        assert input_ids.size() == attention_mask.size()

        if 'encoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('encoder_adapter')
        else:
            raise ValueError(f"encoder_adapter not in adapter_keys: {self.adapter_keys}")

        # Get embeddings from decoder
        emb = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]

        # Create mask for memory tokens
        mask = torch.isin(
            input_ids,
            self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        )

        # Calculate MSE loss between memory and non-memory regions
        attn = attention_mask.bool()
        mem_mask = mask & attn
        non_mem_mask = (~mask) & attn

        mem_len = mem_mask.sum(dim=1)
        non_mem_len = non_mem_mask.sum(dim=1)

        if (mem_len == 0).any():
            raise ValueError("Some samples have no memory tokens")
        if (non_mem_len == 0).any():
            raise ValueError("Some samples have no non-memory tokens")

        mem_sum = (emb * mem_mask.unsqueeze(-1)).sum(dim=1)
        non_mem_sum = (emb * non_mem_mask.unsqueeze(-1)).sum(dim=1)

        mem_mean = mem_sum / mem_len.unsqueeze(-1)
        non_mem_mean = non_mem_sum / non_mem_len.unsqueeze(-1)

        mse_loss = F.mse_loss(non_mem_mean, mem_mean, reduction='mean')

        compressed_embs, _, _ = self._extract_memory_embeddings(
            emb, input_ids, attention_mask
        )
        return compressed_embs, mse_loss

    def _compr_query_reasoner_stage2(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return QR(q): final-token, final-layer hidden state (B, d_h)."""
        assert input_ids.size() == attention_mask.size()

        if 'query_reasoner_adapter' in self.adapter_keys:
            self.decoder.set_adapter('query_reasoner_adapter')
        else:
            raise ValueError(f"query_reasoner_adapter not in adapter_keys: {self.adapter_keys}")

        emb = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]

        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        last_positions = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
        if (last_positions < 0).any():
            raise ValueError("QR received an all-padding query")
        batch_index = torch.arange(input_ids.size(0), device=input_ids.device)
        return emb[batch_index, last_positions]

    def _project_query_reps_to_bge(self, query_reps: torch.Tensor) -> torch.Tensor:
        if self._bge_projection is None:
            raise RuntimeError("fitted W_BGE projection is required for ARIA retrieval")
        projected = self._bge_projection.to(query_reps.device)(
            query_reps.to(self._bge_projection.weight.dtype)
        )
        return F.normalize(
            projected.float(), dim=-1, eps=self._rag_config.numerical_epsilon
        )

    def _encode_subquery_for_retrieval(self, query: str) -> torch.Tensor:
        encoded = self._prepare_query_inputs([query], max_length=self.query_max_length)
        query_rep = self._compr_query_reasoner_stage2(
            encoded["input_ids"].to(self.decoder.device),
            encoded["attention_mask"].to(self.decoder.device),
        )
        return self._project_query_reps_to_bge(query_rep)[0]

    def _differentiable_fused_scores(
        self,
        query_bge: torch.Tensor,
        evidence: Sequence[Sequence[_ScoredDoc]],
        feedback_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rebuild MADS/CCEF gradients without changing stored forward scores.

        Lexical/entity values and semantic min/max statistics are fixed scorer
        inputs recorded on each selected document.  Only the BGE cosine is
        recomputed from QR.  For documents admitted by second-round retrieval,
        an identity-forward feedback cosine supplies the otherwise missing
        DenseSearch derivative to P_fb and the first-pass compressor.
        """
        if query_bge.ndim != 2 or len(evidence) != query_bge.size(0):
            raise ValueError("differentiable MADS scores require an aligned batch")
        if not evidence or any(len(documents) != 5 for documents in evidence):
            raise ValueError("differentiable MADS scores require exactly five documents")
        if self.rag_pipeline is None or self.rag_pipeline.ahr.dense_embeddings is None:
            raise RuntimeError("differentiable MADS scores require the frozen BGE index")
        if feedback_query is not None and feedback_query.shape != query_bge.shape:
            raise ValueError("feedback and QR retrieval representations must align")

        dense_index = self.rag_pipeline.ahr.dense_embeddings
        device = query_bge.device
        doc_vectors = []
        for documents in evidence:
            indices = torch.tensor(
                [document.corpus_index for document in documents],
                device=dense_index.device,
                dtype=torch.long,
            )
            if (indices < 0).any() or (indices >= dense_index.size(0)).any():
                raise ValueError("MADS gradient path requires corpus-backed documents")
            doc_vectors.append(
                dense_index.index_select(0, indices).to(device=device, dtype=torch.float32)
            )
        documents_bge = F.normalize(
            torch.stack(doc_vectors),
            dim=-1,
            eps=self._rag_config.numerical_epsilon,
        )
        query = F.normalize(
            query_bge.float(), dim=-1, eps=self._rag_config.numerical_epsilon
        )
        semantic_raw = torch.einsum("bd,bnd->bn", query, documents_bge)

        def field(name: str) -> torch.Tensor:
            return torch.tensor(
                [[float(getattr(document, name)) for document in documents]
                 for documents in evidence],
                device=device,
                dtype=torch.float32,
            )

        semantic_min = field("sem_min")
        semantic_span = field("sem_span")
        semantic = torch.where(
            semantic_span > self._rag_config.numerical_epsilon,
            (semantic_raw - semantic_min) / semantic_span,
            torch.full_like(semantic_raw, 0.5),
        )
        agent_scores = torch.stack(
            (field("lex_score"), semantic, field("ent_score")), dim=-1
        )
        weights = torch.tensor(
            self._rag_config.mads_weights, device=device, dtype=torch.float32
        )
        weighted_mean = (agent_scores * weights).sum(dim=-1)
        agent_mean = agent_scores.mean(dim=-1)
        agent_std = agent_scores.std(dim=-1, unbiased=False)
        agreement = (
            1.0
            - agent_std / (agent_mean + self._rag_config.numerical_epsilon)
        ).clamp(0.0, 1.0)
        alpha = float(self._rag_config.ccef_discount_alpha)
        live_fused = weighted_mean * (alpha + (1.0 - alpha) * agreement)
        stored_fused = field("fused_score")
        scores = stored_fused + (live_fused - live_fused.detach())

        if feedback_query is not None:
            feedback = F.normalize(
                feedback_query.float(),
                dim=-1,
                eps=self._rag_config.numerical_epsilon,
            )
            feedback_cosine = torch.einsum("bd,bnd->bn", feedback, documents_bge)
            from_second = torch.tensor(
                [[document.from_second_round for document in documents]
                 for documents in evidence],
                device=device,
                dtype=torch.bool,
            )
            feedback_surrogate = feedback_cosine * from_second.to(feedback_cosine.dtype)
            scores = scores + (feedback_surrogate - feedback_surrogate.detach())
        return scores

    def _acr_min_ratios(
        self,
        qca_results: Sequence[QCAResult],
        device: torch.device,
    ) -> torch.Tensor:
        cfg = self._rag_config
        floors = []
        for result in qca_results:
            floor = cfg.acr_min_token_ratio
            if (
                cfg.use_128x_complexity_floor
                and self.compr_rate == 128
                and result.question_type in {QuestionType.MULTI_HOP, QuestionType.MULTI_ASPECT}
            ):
                floor = cfg.acr_complexity_floor_128
            floors.append(floor)
        return torch.tensor(floors, device=device, dtype=torch.float32)

    def _compute_cfrs_errors(
        self,
        memory_embeddings: torch.Tensor,
        memory_counts: torch.Tensor,
        source_input_ids: torch.Tensor,
        source_content_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the old CFRS teacher-forced reconstruction for every document."""
        if memory_embeddings.dim() != 4:
            raise ValueError("CFRS memory must have shape (B, N, T, H)")
        batch, n_docs, n_tokens, _ = memory_embeddings.shape
        if memory_counts.shape != (batch, n_docs):
            raise ValueError("CFRS requires one memory count per selected document")
        if source_input_ids.shape != source_content_mask.shape:
            raise ValueError("CFRS source IDs and content mask must align")
        if source_input_ids.size(0) != batch * n_docs:
            raise ValueError("CFRS requires one truncated source row per document")

        prefix_ids = self.decoder_tokenizer.encode(
            CFRS_RECONSTRUCTION_PREFIX,
            add_special_tokens=True,
        )
        if not prefix_ids:
            raise RuntimeError("CFRS reconstruction prefix tokenized to an empty sequence")
        memory_token_ids = self.decoder_tokenizer.mem_token_ids
        causal_lm = (
            self.decoder.get_base_model()
            if hasattr(self.decoder, "get_base_model")
            else self.decoder
        )
        transformer = getattr(causal_lm, "model", None)
        output_head = causal_lm.get_output_embeddings()
        if transformer is None or output_head is None:
            raise RuntimeError("CFRS requires a causal LM backbone and output head")
        chunk_tokens = int(self.config.cfrs_reconstruction_chunk_tokens)
        flat_memory = memory_embeddings.reshape(batch * n_docs, n_tokens, -1)
        flat_counts = memory_counts.reshape(-1)
        per_document: List[torch.Tensor] = []

        # Per-document execution bounds the vocabulary-logit working set while
        # preserving exact gold-prefix conditioning.  Base-decoder parameters
        # stay frozen, but gradients flow through the inserted memory states.
        with _base_decoder_only(self.decoder):
            for row_index in range(batch * n_docs):
                count = int(flat_counts[row_index].item())
                if count < 1 or count > n_tokens:
                    raise ValueError("CFRS memory counts must lie in [1, T]")
                target_ids = source_input_ids[row_index][
                    source_content_mask[row_index].bool()
                ].to(self.decoder.device)
                if target_ids.numel() == 0:
                    raise ValueError("CFRS requires at least one source target token")
                row_ids = torch.tensor(
                    prefix_ids
                    + [int(value) for value in memory_token_ids[:count]]
                    + target_ids.detach().cpu().tolist(),
                    device=self.decoder.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                inputs_embeds = self.decoder.get_input_embeddings()(row_ids).clone()
                prefix_length = len(prefix_ids)
                inputs_embeds[0, prefix_length : prefix_length + count] = flat_memory[
                    row_index, :count
                ].to(inputs_embeds.dtype)
                base_output = transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=torch.ones_like(row_ids),
                    use_cache=False,
                    return_dict=True,
                )
                prediction_start = prefix_length + count - 1
                prediction_states = base_output.last_hidden_state[
                    0, prediction_start : prediction_start + target_ids.numel()
                ]
                if prediction_states.size(0) != target_ids.numel():
                    raise RuntimeError("CFRS target predictions were truncated")
                error_sum = prediction_states.new_zeros((), dtype=torch.float32)
                for start in range(0, target_ids.numel(), chunk_tokens):
                    stop = min(start + chunk_tokens, target_ids.numel())
                    logits = output_head(prediction_states[start:stop])
                    chunk_error = CompressionFidelityReranker.squared_probability_error(
                        logits.unsqueeze(0), target_ids[start:stop].unsqueeze(0)
                    )[0]
                    error_sum = error_sum + chunk_error * (stop - start)
                per_document.append(error_sum / target_ids.numel())
        return torch.stack(per_document).view(batch, n_docs)

    def _compress_evidence(
        self,
        evidence: List[List[_ScoredDoc]],
        qca_results: Sequence[QCAResult],
        query_bge: torch.Tensor,
        feedback_query: Optional[torch.Tensor] = None,
        compute_cfrs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compress the paper's fixed five-document CCEF survivor set."""
        batch = len(evidence)
        if batch == 0:
            raise ValueError("empty evidence batch")
        max_docs = self.generation_top_k
        document_counts = [len(documents) for documents in evidence]
        if (
            len(qca_results) != batch
            or query_bge.size(0) != batch
            or max_docs != 5
            or any(count != 5 for count in document_counts)
        ):
            raise ValueError("each ARIA CCEF set must contain exactly five documents")

        flat_documents = [doc.text for docs in evidence for doc in docs]
        encoded = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        (
            compressed,
            _,
            _,
            source_input_ids,
            source_content_mask,
            base_counts,
        ) = self._compress_with_fidelity(
            encoded["input_ids"].to(self.decoder.device),
            encoded["attention_mask"].to(self.decoder.device),
            return_alignment_context=True,
        )
        n_tokens = compressed.size(1)
        device = compressed.device
        raw_memory = compressed.new_zeros(
            (batch, max_docs, n_tokens, self.hidden_size)
        )
        gated_memory = torch.zeros_like(raw_memory)
        fused_scores = torch.full(
            (batch, max_docs), float("-inf"), device=device, dtype=torch.float32
        )
        ratios = torch.zeros(batch, max_docs, device=device, dtype=torch.float32)
        gates = torch.zeros(batch, max_docs, n_tokens, device=device, dtype=torch.float32)
        effective_counts = torch.zeros(batch, max_docs, device=device, dtype=torch.long)
        padded_base_counts = torch.zeros_like(effective_counts)
        document_mask = torch.zeros(batch, max_docs, device=device, dtype=torch.bool)
        per_doc_mse = torch.zeros(batch, max_docs, device=device, dtype=torch.float32)
        min_ratios = self._acr_min_ratios(qca_results, device)
        offset = 0
        for row_index, documents in enumerate(evidence):
            count = len(documents)
            row_slice = slice(offset, offset + count)
            row_compressed = compressed[row_slice]
            row_base_counts = base_counts[row_slice]
            row_scores = torch.tensor(
                [[document.fused_score for document in documents]],
                device=device,
                dtype=torch.float32,
            ).detach()
            allocation_mode = self._rag_config.acr_allocation_mode
            if allocation_mode == "adaptive":
                if self._acr is None:
                    raise RuntimeError("adaptive allocation requires the ACR allocator")
                row_gated, row_ratios, row_gates, row_effective = (
                    self._acr.allocate_and_apply(
                        row_compressed,
                        row_scores,
                        count,
                        min_ratios=min_ratios[row_index : row_index + 1],
                        base_token_counts=row_base_counts,
                        return_metadata=True,
                    )
                )
            elif allocation_mode in {"uniform_budget", "uniform_constant"}:
                if self._acr is None:
                    raise RuntimeError("uniform allocation requires the gate allocator")
                if allocation_mode == "uniform_budget":
                    flat_ratios = self._acr.uniform_ratios_for_budget(
                        row_base_counts,
                        self._rag_config.uniform_evidence_token_budget,
                    ).to(device=device)
                else:
                    flat_ratios = torch.full(
                        (count,),
                        self._rag_config.uniform_constant_ratio,
                        device=device,
                        dtype=torch.float32,
                    )
                row_gated, row_gates_flat, row_effective_flat = (
                    self._acr.apply_ratios(
                        row_compressed,
                        flat_ratios,
                        base_token_counts=row_base_counts,
                    )
                )
                row_ratios = flat_ratios.view(1, count)
                row_gates = row_gates_flat.view(1, count, n_tokens)
                row_effective = row_effective_flat.view(1, count)
            elif allocation_mode == "full":
                row_ratios = torch.ones(1, count, device=device)
                row_effective = row_base_counts.view(1, count)
                positions = torch.arange(n_tokens, device=device).view(1, 1, -1)
                row_gates = (positions < row_effective.unsqueeze(-1)).float()
                row_gated = row_compressed
            else:  # guarded by RAGPipelineConfig.__post_init__
                raise RuntimeError(f"unsupported allocation mode: {allocation_mode}")
            raw_memory[row_index, :count] = row_compressed
            gated_memory[row_index, :count] = row_gated.view(
                count, n_tokens, self.hidden_size
            )
            fused_scores[row_index, :count] = row_scores[0]
            ratios[row_index, :count] = row_ratios[0]
            gates[row_index, :count] = row_gates[0]
            effective_counts[row_index, :count] = row_effective[0]
            padded_base_counts[row_index, :count] = row_base_counts
            document_mask[row_index, :count] = True
            # CFRS reconstructs from the actual ACR-gated memory.  Ratios are
            # detached retrieval decisions, so this remains gradient-orthogonal
            # to ACR while still training the current compressor states.
            row_memory = gated_memory[row_index : row_index + 1, :count]
            if compute_cfrs:
                per_doc_mse[row_index, :count] = self._compute_cfrs_errors(
                    row_memory,
                    row_base_counts.view(1, count),
                    source_input_ids[row_slice],
                    source_content_mask[row_slice],
                )[0]
            offset += count
        if offset != compressed.size(0):
            raise RuntimeError("variable evidence packing lost a document")

        return {
            # ACR is always the differentiable sigmoid mask in both training
            # and inference.  Hard T_i is reserved for MTFRL pooling only.
            "memory": gated_memory,
            "fused_scores": fused_scores,
            "per_doc_mse": per_doc_mse,
            "ratios": ratios,
            "gates": gates,
            "effective_counts": effective_counts,
            "base_counts": padded_base_counts,
            "context_counts": padded_base_counts,
            "document_mask": document_mask,
        }

    @staticmethod
    def _straight_through_cfrs_permutation(
        memory_embeddings: torch.Tensor,
        final_scores: torch.Tensor,
        document_mask: torch.Tensor,
        temperature: float = CFRS_SOFT_PERMUTATION_TEMPERATURE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply hard descending order with a soft score-gradient surrogate.

        The forward permutation is exactly ``argsort(final_scores)``.  During
        backward, a soft assignment to detached hard-rank score anchors carries
        QA gradients through ``s_final -> fidelity -> reconstruction error``.
        This resolves the paper's otherwise unspecified derivative through a
        discrete ordering operation without adding an auxiliary objective.
        """
        if memory_embeddings.ndim != 4 or final_scores.ndim != 2:
            raise ValueError("CFRS permutation expects memory (B,N,T,H) and scores (B,N)")
        batch, n_docs = memory_embeddings.shape[:2]
        if final_scores.shape != (batch, n_docs):
            raise ValueError("CFRS scores must align with memory document slots")
        valid = document_mask.to(device=final_scores.device, dtype=torch.bool)
        if valid.shape != final_scores.shape or not bool(valid.all()):
            raise ValueError("the paper CFRS permutation requires five real documents")
        if temperature <= 0 or not math.isfinite(float(temperature)):
            raise ValueError("CFRS permutation temperature must be finite and positive")
        if not torch.isfinite(final_scores).all():
            raise ValueError("CFRS permutation scores must be finite")

        hard_order = torch.argsort(
            final_scores.detach(), dim=-1, descending=True, stable=True
        )
        hard = F.one_hot(hard_order, num_classes=n_docs).to(final_scores.dtype)
        anchors = final_scores.detach().gather(1, hard_order)
        soft_logits = -(
            final_scores.unsqueeze(1) - anchors.unsqueeze(2)
        ).square() / float(temperature)
        soft = torch.softmax(soft_logits, dim=-1)
        permutation = hard + (soft - soft.detach())
        ordered = torch.einsum(
            "brn,bnth->brth", permutation.to(memory_embeddings.dtype), memory_embeddings
        )
        return ordered, hard_order

    def _resolve_no_compression_context_limit(self) -> int:
        """Return the old diagnostic's explicit 32k context ceiling."""
        return ARIA_NO_COMPRESSION_CONTEXT_CEILING

    def _generate_no_compression_context(
        self,
        questions: Sequence[str],
        evidence: Sequence[Sequence[_ScoredDoc]],
        max_new_tokens: int,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, int]:
        """Generate directly from the first-pass top-five raw passages."""
        if max_new_tokens <= 0 or max_new_tokens >= ARIA_NO_COMPRESSION_CONTEXT_CEILING:
            raise ValueError("ARIA-NoComp generation budget must fit the 32k context")
        if len(questions) == 0 or len(evidence) != len(questions):
            raise ValueError("ARIA-NoComp questions and evidence must align")
        if any(
            len(documents) != 5
            for documents in evidence
        ):
            raise ValueError("ARIA-NoComp requires exactly five first-pass documents")

        raw_contexts: List[str] = []
        document_token_counts: List[int] = []
        for documents in evidence:
            document_texts = [document.text for document in documents]
            if any(not isinstance(text, str) or not text.strip() for text in document_texts):
                raise ValueError("ARIA-NoComp requires non-empty raw passage text")
            # Two newlines are the only inter-document separator. No document
            # labels, memory tokens, compressor delimiters, or rewritten text
            # are introduced.
            raw_contexts.append("\n\n".join(document_texts))
            token_count = sum(
                len(
                    self.decoder_tokenizer.encode(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                    )
                )
                for text in document_texts
            )
            if token_count <= 0:
                raise ValueError("ARIA-NoComp raw passages must contain decoder tokens")
            document_token_counts.append(token_count)

        prompts = [
            self._blend_standard_prompt(context, question, None)
            for question, context in zip(questions, raw_contexts)
        ]
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise RuntimeError("ARIA-NoComp failed to build a direct-context QA prompt")
        context_limit = self._resolve_no_compression_context_limit()
        prompt_limit = context_limit - max_new_tokens
        decoder_inputs = self.decoder_tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            truncation=False,
        )
        # Work on token IDs, not passage strings.  The reported ~2,950-token
        # path is unchanged; only an out-of-contract overflow is right-trimmed
        # to preserve the old diagnostic's explicit 32k ceiling.
        dec_input_ids = decoder_inputs["input_ids"][:, :prompt_limit].to(
            self.decoder.device
        )
        dec_attention_mask = decoder_inputs["attention_mask"][:, :prompt_limit].to(
            self.decoder.device
        )
        prompt_lengths = dec_attention_mask.sum(dim=1).long()
        if "decoder_adapter" not in self.adapter_keys:
            raise ValueError("ARIA-NoComp requires the Phase-II decoder_adapter")
        self.decoder.set_adapter("decoder_adapter")
        output_ids = self.decoder.generate(
            input_ids=dec_input_ids,
            attention_mask=dec_attention_mask,
            do_sample=False,
            top_p=None,
            temperature=None,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
        )
        prompt_width = dec_input_ids.size(1)
        if output_ids.ndim != 2 or output_ids.size(1) < prompt_width:
            raise RuntimeError("ARIA-NoComp decoder returned malformed token IDs")
        decoded = self.decoder_tokenizer.batch_decode(
            output_ids[:, prompt_width:], skip_special_tokens=True
        )
        return (
            decoded,
            prompt_lengths,
            torch.tensor(
                document_token_counts,
                device=prompt_lengths.device,
                dtype=torch.long,
            ),
            context_limit,
        )

    def generate_from_questions(
        self,
        questions: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.5,
        documents: Optional[List[List[str]]] = None,
        stage2_mips: bool = False,
        stage2_retrieval_top_n: Optional[int] = None,
        time_count: bool = False,
        return_first_pass_indices: bool = False,
        oracle_gold_indices: Optional[Sequence[Sequence[int]]] = None,
        no_compression: bool = False,
    ) -> Tuple[List[str], torch.Tensor]:
        """Run the full two-round ARIA inference algorithm from Appendix A.1.

        By default the returned indices identify the final documents consumed by
        the generator.  Appendix A.29 defines Normal Recall@5 over the first-pass
        retrieval pipeline, so evaluators can request those pre-MTFRL indices
        explicitly without changing the default inference API. Oracle mode
        supplies corpus-level positive indices and replaces AHR/IGFR with one
        deterministic, page-ID-deduplicated BGE top-100 pool whose scope is
        preserved through MTFRL.
        """
        del temperature  # Paper evaluation is deterministic/greedy.
        if stage2_mips:
            raise ValueError(
                "ARIA Algorithm 1 inference requires stage2_mips=False and "
                "the integrated retrieval pipeline"
            )
        if no_compression:
            if documents is not None or oracle_gold_indices is not None:
                raise ValueError(
                    "ARIA-NoComp supports only Normal full-corpus retrieval"
                )
            if self.generation_top_k != 5:
                raise ValueError("ARIA-NoComp requires the paper's top-5 decoder ceiling")
            if not all(
                (
                    self._rag_config.use_qca,
                    self._rag_config.use_ahr,
                    self._rag_config.use_igfr,
                    self._rag_config.use_mads,
                    self._rag_config.use_ccef,
                )
            ):
                raise ValueError(
                    "ARIA-NoComp requires all five first-pass retrieval stages"
                )
            if (
                self._rag_config.use_cfrs
                or self._rag_config.acr_allocation_mode != "full"
                or self._rag_config.second_retrieval_mode != "disabled"
            ):
                raise ValueError(
                    "ARIA-NoComp runtime must disable CFRS, ACR, and MTFRL"
                )
        if "query_reasoner_adapter" not in self.adapter_keys:
            raise ValueError("query_reasoner_adapter is required")
        if self.rag_pipeline is None:
            raise RuntimeError(
                "setup_rag_pipeline() with corpus BGE embeddings must be called before ARIA inference"
            )
        if self._bge_projection is None:
            raise RuntimeError("ARIA retrieval requires a fitted W_BGE projection")
        if oracle_gold_indices is not None:
            if documents is not None:
                raise ValueError("Oracle fixed pools cannot be combined with documents")
            if len(oracle_gold_indices) != len(questions):
                raise ValueError(
                    "oracle_gold_indices must align one-to-one with questions"
                )
            corpus_size = len(self.rag_pipeline.ahr.corpus_docs)
            if self.rag_pipeline.ahr.unique_page_count < 100:
                raise ValueError(
                    "Oracle top-100 requires a corpus with at least 100 unique pages"
                )
            for row_index, raw_indices in enumerate(oracle_gold_indices):
                values = [int(index) for index in raw_indices]
                if not values or len(values) != len(set(values)):
                    raise ValueError(
                        f"oracle_gold_indices[{row_index}] must be non-empty and unique"
                    )
                if any(index < 0 or index >= corpus_size for index in values):
                    raise ValueError(
                        f"oracle_gold_indices[{row_index}] contains an invalid corpus index"
                    )

        self.eval()
        query_time = compress_time = generate_time = 0.0
        overall_start = time.perf_counter()
        requested_top_k = stage2_retrieval_top_n or self._rag_config.top_k
        if requested_top_k != self.generation_top_k or requested_top_k != 5:
            raise ValueError(
                "the paper requires decoder document slots and retrieval top-k to equal five"
            )

        with torch.no_grad():
            query_start = time.perf_counter()
            query_inputs = self._prepare_query_inputs(
                list(questions), max_length=self.query_max_length
            )
            query_reps = self._compr_query_reasoner_stage2(
                query_inputs["input_ids"].to(self.decoder.device),
                query_inputs["attention_mask"].to(self.decoder.device),
            )
            query_bge = self._project_query_reps_to_bge(query_reps)

            initial_evidence: List[List[_ScoredDoc]] = []
            qca_results: List[QCAResult] = []
            diagnostics: List[RAGDiagnostics] = []
            oracle_records: Optional[List[OraclePoolRecord]] = None
            if oracle_gold_indices is not None:
                dense_corpus = self.rag_pipeline.ahr.dense_embeddings
                if dense_corpus is None:
                    raise RuntimeError("Oracle top-100 requires the fixed BGE corpus index")
                base_index_rows = _chunked_inner_product_topk_unique_pages(
                    F.normalize(
                        query_bge.detach().float().cpu(),
                        dim=-1,
                        eps=self._rag_config.numerical_epsilon,
                    ),
                    dense_corpus,
                    self.rag_pipeline.ahr.corpus_page_ids,
                    100,
                )
                oracle_records = [
                    _construct_oracle_top100_indices(
                        base_index_rows[row].tolist(),
                        oracle_gold_indices[row],
                        corpus_page_ids=self.rag_pipeline.ahr.corpus_page_ids,
                    )
                    for row in range(len(questions))
                ]
                if self._rag_config.use_qca:
                    initial_qca = self.rag_pipeline.qca.assess_batch(questions)
                else:
                    initial_qca = [
                        QCAResult(
                            question=question,
                            question_type=QuestionType.SIMPLE,
                            confidence=0.0,
                            hop_count=1,
                            entity_count=_qca_entity_count(question),
                            sub_questions=[question],
                            reasoning="QCA disabled by the checkpoint ablation",
                        )
                        for question in questions
                    ]
                initial_pools = []
                normalized_queries = F.normalize(
                    query_bge.detach().float().cpu(),
                    dim=-1,
                    eps=self._rag_config.numerical_epsilon,
                )
                for row, record in enumerate(oracle_records):
                    pool_rows = torch.tensor(record.pool_indices, dtype=torch.long)
                    pool_scores = (
                        dense_corpus.index_select(0, pool_rows)
                        @ normalized_queries[row]
                    ).tolist()
                    initial_pools.append(
                        [
                            _RetrievedDoc(
                                doc_id=self.rag_pipeline.ahr.corpus_ids[corpus_index],
                                text=self.rag_pipeline.ahr.corpus_docs[corpus_index],
                                corpus_index=corpus_index,
                                dense_score=float(score),
                                hybrid_score=float(score),
                            )
                            for corpus_index, score in zip(
                                record.pool_indices, pool_scores
                            )
                        ]
                    )
                self._oracle_pool_records.extend(oracle_records)
            elif documents is None:
                initial_pools, initial_qca = self.rag_pipeline.retrieve_initial_batch(
                    questions, query_bge
                )
            else:
                initial_pools = initial_qca = None
            for batch_index, question in enumerate(questions):
                if oracle_records is not None:
                    qca_result = initial_qca[batch_index]
                    diag = RAGDiagnostics(
                        question_type=qca_result.question_type.value,
                        qca_confidence=qca_result.confidence,
                        initial_candidates=100,
                    )
                    # Oracle replaces candidate acquisition only. MADS is the
                    # first ranking stage and CCEF selects D1 from the fixed pool.
                    scored = self.rag_pipeline._mads_ccef(
                        question,
                        initial_pools[batch_index],
                        requested_top_k,
                        query_emb=query_bge[batch_index],
                        diagnostics=diag,
                    )
                elif documents is None:
                    scored, qca_result, diag = self.rag_pipeline.retrieve_scored(
                        question,
                        query_emb=query_bge[batch_index],
                        override_top_k=requested_top_k,
                        embed_subquery=self._encode_subquery_for_retrieval,
                        qca_result=initial_qca[batch_index],
                        initial_retrieved=initial_pools[batch_index],
                    )
                else:
                    qca_result = self.rag_pipeline.qca.assess(question)
                    supplied = [
                        _RetrievedDoc(
                            doc_id=f"external:{batch_index}:{doc_index}",
                            text=text,
                            corpus_index=-1,
                        )
                        for doc_index, text in enumerate(documents[batch_index])
                    ]
                    diag = RAGDiagnostics(
                        question_type=qca_result.question_type.value,
                        qca_confidence=qca_result.confidence,
                        initial_candidates=len(supplied),
                    )
                    scored = self.rag_pipeline._mads_ccef(
                        question,
                        supplied,
                        requested_top_k,
                        query_emb=query_bge[batch_index],
                        diagnostics=diag,
                    )

                if len(scored) != requested_top_k:
                    raise RuntimeError(
                        "first CCEF retained fewer than the paper's five documents"
                    )
                initial_evidence.append(scored)
                qca_results.append(qca_result)
                diagnostics.append(diag)
            first_pass_indices = torch.full(
                (len(questions), requested_top_k),
                -1,
                device=self.decoder.device,
                dtype=torch.long,
            )
            for row_index, documents_for_question in enumerate(initial_evidence):
                first_pass_indices[row_index, : len(documents_for_question)] = (
                    torch.tensor(
                        [document.corpus_index for document in documents_for_question],
                        device=self.decoder.device,
                        dtype=torch.long,
                    )
                )
            query_time = time.perf_counter() - query_start

            if no_compression:
                generation_start = time.perf_counter()
                (
                    decoded,
                    prompt_lengths,
                    document_token_counts,
                    context_limit,
                ) = self._generate_no_compression_context(
                    questions,
                    initial_evidence,
                    max_new_tokens,
                )
                generate_time = time.perf_counter() - generation_start
                for row_index, diag in enumerate(diagnostics):
                    diag.final_candidates = len(initial_evidence[row_index])
                    diag.second_round_candidates = 0
                    diag.evidence_memory_tokens = 0
                    diag.direct_context_document_tokens = int(
                        document_token_counts[row_index].item()
                    )
                    diag.direct_context_prompt_tokens = int(
                        prompt_lengths[row_index].item()
                    )
                    diag.direct_context_ceiling = context_limit
                elapsed_ms = (time.perf_counter() - overall_start) * 1000.0
                for diag in diagnostics:
                    diag.latency_ms = elapsed_ms / max(len(diagnostics), 1)
                    self._rag_diagnostics.append(diag)
                if len(self._rag_diagnostics) > 1000:
                    self._rag_diagnostics = self._rag_diagnostics[-500:]
                if time_count:
                    total = query_time + generate_time
                    return (
                        decoded,
                        first_pass_indices,
                        0.0,
                        query_time,
                        generate_time,
                        total,
                    )
                return decoded, first_pass_indices

            # Algorithm 1 lines 9-10: first ACR and compression on first CCEF D1.
            compression_start = time.perf_counter()
            first_pass = self._compress_evidence(
                initial_evidence,
                qca_results,
                query_bge,
                compute_cfrs=self._cfrs is not None and self._mtfrl is None,
            )

            final_evidence = initial_evidence
            if self._mtfrl is not None:
                # Lines 11-14: the full path uses memory feedback.  Matched
                # no-MTFRL controls preserve D2=200 and union->MADS->CCEF with
                # the release's static original-QR/W_BGE query convention.
                if self._rag_config.second_retrieval_mode == "memory_feedback":
                    second_query = self._compute_first_pass_feedback_query(first_pass)
                elif self._rag_config.second_retrieval_mode == "static_query":
                    second_query = query_bge
                else:
                    raise RuntimeError("mounted second retriever has disabled mode")
                second_round = self._mtfrl.second_round_retrieve(
                    feedback_queries=second_query,
                    already_retrieved_ids=[
                        [doc.doc_id for doc in docs] for docs in initial_evidence
                    ],
                    top_k=self._rag_config.mtfrl_second_top_k,
                    allowed_corpus_indices=(
                        [record.pool_indices for record in oracle_records]
                        if oracle_records is not None
                        else None
                    ),
                )
                final_evidence = []
                for batch_index, question in enumerate(questions):
                    rescored = self.rag_pipeline.rescore_union(
                        question,
                        initial_evidence[batch_index],
                        second_round[batch_index],
                        query_emb=query_bge[batch_index],
                        top_k=requested_top_k,
                        diagnostics=diagnostics[batch_index],
                    )
                    if len(rescored) != requested_top_k:
                        raise RuntimeError(
                            "second CCEF retained fewer than the paper's five documents"
                        )
                    final_evidence.append(rescored)

                # Lines 15-16: recompute ACR from final s_fused and recompress.
                final_pass = self._compress_evidence(
                    final_evidence, qca_results, query_bge,
                    feedback_query=(
                        second_query
                        if self._rag_config.second_retrieval_mode == "memory_feedback"
                        else None
                    ),
                    compute_cfrs=self._cfrs is not None,
                )
            else:
                final_pass = first_pass

            # Lines 17-18: CFRS only reorders the final five; ACR consumed the
            # pre-CFRS s_fused above and was detached inside _compress_evidence.
            if self._cfrs is not None:
                final_scores = CompressionFidelityReranker.rerank(
                    final_pass["fused_scores"],
                    final_pass["per_doc_mse"],
                    cfrs_weight=self._rag_config.cfrs_weight,
                    eps=self._rag_config.numerical_epsilon,
                    document_mask=final_pass["document_mask"],
                )
            else:
                final_scores = final_pass["fused_scores"]
            order = final_scores.argsort(dim=-1, descending=True, stable=True)
            batch, n_docs, n_tokens, hidden = final_pass["memory"].shape
            memory_index = order.unsqueeze(-1).unsqueeze(-1).expand(
                batch, n_docs, n_tokens, hidden
            )
            ordered_memory = final_pass["memory"].gather(1, memory_index)
            ordered_counts = final_pass["context_counts"].gather(1, order)
            ordered_effective_counts = final_pass["effective_counts"].gather(
                1, order
            )
            for row_index, diag in enumerate(diagnostics):
                diag.evidence_memory_tokens = int(
                    ordered_effective_counts[row_index].sum().item()
                )
                diag.final_candidates = int(
                    final_pass["document_mask"][row_index].sum().item()
                )
            corpus_indices = torch.full(
                (batch, n_docs), -1, device=order.device, dtype=torch.long
            )
            for row_index, documents_for_question in enumerate(final_evidence):
                corpus_indices[row_index, : len(documents_for_question)] = torch.tensor(
                    [document.corpus_index for document in documents_for_question],
                    device=order.device,
                    dtype=torch.long,
                )
            topk_idx = corpus_indices.gather(1, order)
            if oracle_records is not None:
                for row, record in enumerate(oracle_records):
                    returned = {index for index in topk_idx[row].tolist() if index >= 0}
                    if not returned.issubset(record.pool_indices):
                        raise RuntimeError(
                            "Oracle final evidence escaped its fixed top-100 pool"
                        )
            compress_time = time.perf_counter() - compression_start

            # The generator consumes every sigmoid-gated memory position.  Only
            # MTFRL realizes T_i as a hard prefix.
            generation_start = time.perf_counter()
            instructions = [
                self._blend_prompt_and_selected_memory_tokens(
                    query=question,
                    memory_counts=ordered_counts[index].tolist(),
                )[1]
                for index, question in enumerate(questions)
            ]
            decoder_inputs = self.decoder_tokenizer(
                instructions,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False,
                truncation=False,
            )
            prompt_lengths = decoder_inputs["attention_mask"].sum(dim=1)
            if torch.any(prompt_lengths > self.stage2_input_max_length):
                row = int(
                    torch.nonzero(
                        prompt_lengths > self.stage2_input_max_length,
                        as_tuple=False,
                    )[0].item()
                )
                raise ValueError(
                    "Phase-II inference prompt exceeds the paper's 1024-token "
                    "input ceiling; refusing to right-truncate the question: "
                    f"row {row} has {int(prompt_lengths[row].item())} tokens"
                )
            dec_input_ids = decoder_inputs["input_ids"].to(self.decoder.device)
            dec_attention_mask = decoder_inputs["attention_mask"].to(self.decoder.device)
            inputs_embeds = self._replace_variable_memory_embeddings(
                ordered_memory, ordered_counts, dec_input_ids
            )

            if "decoder_adapter" not in self.adapter_keys:
                raise ValueError("generator decoder_adapter is required for Phase II inference")
            self.decoder.set_adapter("decoder_adapter")
            output_ids = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=dec_attention_mask,
                do_sample=False,
                top_p=None,
                temperature=None,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                eos_token_id=self.decoder_tokenizer.eos_token_id,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
            )
            decoded = self.decoder_tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            generate_time = time.perf_counter() - generation_start

        elapsed_ms = (time.perf_counter() - overall_start) * 1000.0
        for diag in diagnostics:
            diag.latency_ms = elapsed_ms / max(len(diagnostics), 1)
            self._rag_diagnostics.append(diag)
        if len(self._rag_diagnostics) > 1000:
            self._rag_diagnostics = self._rag_diagnostics[-500:]

        if time_count:
            total = query_time + compress_time + generate_time
            reported_indices = first_pass_indices if return_first_pass_indices else topk_idx
            return decoded, reported_indices, compress_time, query_time, generate_time, total
        reported_indices = first_pass_indices if return_first_pass_indices else topk_idx
        return decoded, reported_indices

    def generate_from_paraphrase(self, questions: list[str], documents: list[list[str]], max_new_tokens: int = 64) -> list[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: list of string
        documents: list of list of strings (they should all be of equal length: the nb of doc for each question)
        """
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])
        flat_documents = sum(documents, [])

        model_input = {}

        # Creating encoder inputs:
        input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)
        memory_token_counts = input_encoder.get("memory_token_counts")
        if memory_token_counts is None:
            raise ValueError("paraphrase generation requires real memory-token counts")
        model_input["memory_token_counts"] = memory_token_counts.to(device).view(
            len(questions), self.generation_top_k
        )

        # Creating decoder inputs
        instr = [
            self._blend_prompt_and_memory_tokens(
                query="",
                stage="stage1",
                paraphrase_loss=True,
                memory_counts=model_input["memory_token_counts"][index].tolist(),
            )
            for index, _ in enumerate(questions)
        ]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=1024)
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)

        # Generation
        return self._generate(model_input, max_new_tokens=max_new_tokens)


    @torch.inference_mode()
    def generate_from_text(self,
                          questions: List[str],
                          documents: List[List[str]],
                          max_new_tokens: int = 64,
                          return_selected_indices: bool = False,
                          ) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
        """Generate answers from documents via compression then decoding."""
        if getattr(self.config, "aria_rag_configuration", None) == "clara_baseline":
            return self._generate_clara_from_candidates(
                questions,
                documents,
                max_new_tokens=max_new_tokens,
                return_selected_indices=return_selected_indices,
            )
        if return_selected_indices:
            raise ValueError(
                "return_selected_indices is defined only for matched CLaRa evaluation"
            )
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all(len(context) == len(documents[0]) for context in documents)

        flat_documents = sum(documents, [])

        # Create encoder inputs
        input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        device = self.decoder.device
        enc_input_ids = input_encoder['input_ids'].to(device)
        enc_attention_mask = input_encoder['attention_mask'].to(device)
        memory_token_counts = input_encoder.get("memory_token_counts")
        if memory_token_counts is None:
            raise ValueError("CLaRa generation requires real memory-token counts")
        memory_token_counts = memory_token_counts.to(device).view(
            len(questions), self.generation_top_k
        )

        # Create decoder inputs
        instructions = [
            self._blend_prompt_and_memory_tokens(
                query=question,
                stage="stage1_2",
                memory_counts=memory_token_counts[index].tolist(),
            )
            for index, question in enumerate(questions)
        ]
        decoder_max_length = _fixed_memory_prompt_max_length(
            self.generation_top_k,
            self.n_mem_tokens,
            self.doc_max_length,
        )
        inp_dec = self.decoder_tokenizer(
            instructions,
            return_tensors='pt',
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=decoder_max_length,
        )
        dec_input_ids = inp_dec['input_ids'].to(device)
        dec_attention_mask = inp_dec['attention_mask'].to(device)
        memory_ids = self.decoder_tokenizer.mem_token_ids_pt.to(dec_input_ids.device)
        expected_slots = memory_token_counts.sum(dim=1)
        actual_slots = torch.isin(dec_input_ids, memory_ids).sum(dim=1)
        if not torch.equal(actual_slots, expected_slots.to(actual_slots.device)):
            raise ValueError(
                "CLaRa decoder prompt truncated memory placeholders: "
                f"expected {expected_slots.tolist()}, got {actual_slots.tolist()}"
            )

        # Generate
        return self._generate({
            'enc_input_ids': enc_input_ids,
            'enc_attention_mask': enc_attention_mask,
            'dec_input_ids': dec_input_ids,
            'dec_attention_mask': dec_attention_mask,
            'memory_token_counts': memory_token_counts,
        }, max_new_tokens=max_new_tokens)

    def _generate_clara_from_candidates(
        self,
        questions: List[str],
        documents: List[List[str]],
        *,
        max_new_tokens: int,
        return_selected_indices: bool = False,
    ) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
        """Run the matched CLaRa selector/generator on a ranked candidate pool."""
        if len(questions) != len(documents) or not questions:
            raise ValueError("CLaRa questions and candidate pools must be non-empty and aligned")
        candidate_counts = {len(row) for row in documents}
        if len(candidate_counts) != 1:
            raise ValueError("CLaRa inference requires equal candidate counts in a batch")
        candidate_count = candidate_counts.pop()
        selected_count = self.generation_top_k
        if candidate_count < selected_count:
            raise ValueError(
                f"CLaRa requires N >= k candidates, got N={candidate_count}, k={selected_count}"
            )
        if any(
            not isinstance(document, str) or not document.strip()
            for row in documents
            for document in row
        ):
            raise ValueError("CLaRa inference candidates must be non-empty strings")
        if self.rag_pipeline is not None:
            raise RuntimeError("Matched CLaRa inference cannot use the ARIA RAG pipeline")
        if set(self.adapter_keys) != {
            "encoder_adapter",
            "query_reasoner_adapter",
            "decoder_adapter",
        }:
            raise RuntimeError("Matched CLaRa inference requires its exact three-adapter set")

        flat_documents = [document for row in documents for document in row]
        encoded = self._prepare_clara_encoder_inputs(flat_documents)
        compressed, _ = self.compress(
            encoded["input_ids"].to(self.decoder.device),
            encoded["attention_mask"].to(self.decoder.device),
        )
        candidate_memory = compressed.detach().view(
            len(questions), candidate_count, self.n_mem_tokens, self.hidden_size
        )
        candidate_memory_counts = encoded["memory_token_counts"].to(
            self.decoder.device
        ).view(len(questions), candidate_count)
        query_inputs = self._prepare_query_inputs(
            questions, max_length=self.query_max_length
        )
        query_representations = self._compr_query_reasoner_stage2(
            query_inputs["input_ids"].to(self.decoder.device),
            query_inputs["attention_mask"].to(self.decoder.device),
        )
        selected, topk_idx, _, _ = _clara_st_select_candidate_memory(
            query_representations,
            candidate_memory,
            selected_count,
            candidate_memory_counts=candidate_memory_counts,
        )
        selected_counts = candidate_memory_counts.gather(1, topk_idx)
        instructions = [
            self._blend_prompt_and_selected_memory_tokens(
                query=question,
                memory_counts=selected_counts[row].tolist(),
            )[1]
            for row, question in enumerate(questions)
        ]
        decoder_inputs = self.decoder_tokenizer(
            instructions,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            truncation=False,
        )
        prompt_lengths = decoder_inputs["attention_mask"].sum(dim=1)
        if torch.any(prompt_lengths > self.stage2_input_max_length):
            row = int(
                torch.nonzero(
                    prompt_lengths > self.stage2_input_max_length,
                    as_tuple=False,
                )[0].item()
            )
            raise ValueError(
                "CLaRa decoder prompt exceeds the Phase-II 1024-token input "
                f"ceiling at row {row}: {int(prompt_lengths[row].item())}"
            )
        dec_input_ids = decoder_inputs["input_ids"].to(self.decoder.device)
        dec_attention_mask = decoder_inputs["attention_mask"].to(self.decoder.device)
        inputs_embeds = self._replace_variable_memory_embeddings(
            selected, selected_counts, dec_input_ids
        )
        self.decoder.set_adapter("decoder_adapter")
        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            do_sample=False,
            top_p=None,
            temperature=None,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
        )
        decoded = self.decoder_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        if return_selected_indices:
            return decoded, topk_idx
        return decoded

    def generate_from_compressed_documents_and_questions(self,
                                                        questions: List[str],
                                                        compressed_documents: torch.Tensor,
                                                        max_new_tokens: int = 64) -> List[str]:
        """Generate answers from compressed documents."""
        self.generation_top_k = compressed_documents.size(0) // len(questions)
        assert compressed_documents.size(0) % self.generation_top_k == 0

        # Create decoder inputs
        instructions = [self._blend_prompt_and_memory_tokens(query=q, stage="stage1_2") for q in questions]
        decoder_max_length = _fixed_memory_prompt_max_length(
            self.generation_top_k,
            self.n_mem_tokens,
            self.doc_max_length,
        )
        inp_dec = self.decoder_tokenizer(
            instructions,
            return_tensors='pt',
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=decoder_max_length,
        )
        device = self.decoder.device
        dec_input_ids = inp_dec['input_ids'].to(device)
        dec_attention_mask = inp_dec['attention_mask'].to(device)
        memory_ids = self.decoder_tokenizer.mem_token_ids_pt.to(dec_input_ids.device)
        expected_slots = self.generation_top_k * self.n_mem_tokens
        actual_slots = torch.isin(dec_input_ids, memory_ids).sum(dim=1)
        if not torch.all(actual_slots == expected_slots):
            raise ValueError(
                "CLaRa decoder prompt truncated memory placeholders: "
                f"expected {expected_slots}, got {actual_slots.tolist()}"
            )

        # Create input decoder embeddings from prompt + compressed documents
        inputs_embeds = self._replace_emb(compressed_documents, dec_input_ids)

        # Activate decoder generator
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            max_new_tokens=max_new_tokens
        )

        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def compress_documents(self, documents: List[str]) -> torch.Tensor:
        """Compress a list of documents."""
        input_encoder = self._prepare_encoder_inputs(documents, max_length=self.doc_max_length)
        enc_input_ids = input_encoder['input_ids'].to(self.decoder.device)
        attention_mask = input_encoder['attention_mask'].to(self.decoder.device)
        return self.compress(enc_input_ids=enc_input_ids, enc_attention_mask=attention_mask)

    # Helper methods
    def _prepare_query_inputs(
        self,
        questions: List[str],
        max_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize QR inputs independently of compressor memory tokens.

        ``W_BGE`` is fitted once and reused at every compression ratio. Query
        representations therefore cannot depend on the CR-specific ``<MEMi>``
        inventory or on the compressor-only ``<ENC>`` token. The QR consumes
        the final attended native-tokenizer token.
        """
        if not questions or any(
            not isinstance(question, str) or not question.strip()
            for question in questions
        ):
            raise ValueError("QR requires a non-empty list of question strings")
        return self.decoder_tokenizer(
            questions,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

    def _prepare_encoder_inputs(self, texts: List[str], max_length: int, q_texts: List[str] = None) -> Dict[str, torch.Tensor]:
        """Create inputs for the encoder."""
        if q_texts is not None:
            assert len(texts) == len(q_texts)

        if self.compr is None:
            return self._prepare_encoder_inputs_to_decoder(texts, max_length, q_texts)
        else:
            return self.compr.prepare_inputs(texts, max_length, q_texts)

    def _prepare_clara_encoder_inputs(
        self, texts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Build CLaRa inputs with ``K_i=max(1,floor(L_i/r))`` per document."""
        if self.compr is not None:
            raise RuntimeError(
                "Matched CLaRa requires the integrated, Phase-I LoRA compressor"
            )
        prepared = self._prepare_encoder_inputs_to_decoder(texts, self.doc_max_length)
        counts = prepared.get("memory_token_counts")
        if counts is None or counts.numel() != len(texts):
            raise RuntimeError("CLaRa compressor lost its per-document memory allocation")
        if torch.any(counts < 1) or torch.any(counts > self.n_mem_tokens):
            raise RuntimeError("CLaRa per-document memory allocation is out of range")
        return prepared

    def _prepare_encoder_inputs_to_decoder(self, texts: List[str], max_length: int, q_texts: List[str] = None) -> Dict[str, torch.Tensor]:
        """Truncate passage content first, then add fixed compressor delimiters."""
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("compressor passages must be non-empty strings")
        if q_texts is not None:
            if len(texts) != len(q_texts) or any(
                not isinstance(query, str) or not query.strip() for query in q_texts
            ):
                raise ValueError("query-aware compressor inputs must align and be non-empty")
        passage_ids = self.decoder_tokenizer(
            texts,
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        source_lengths = torch.tensor(
            [len(row) for row in passage_ids], dtype=torch.long
        )
        if (source_lengths < 1).any() or (source_lengths > max_length).any():
            raise ValueError("passage truncation must retain between 1 and max_length tokens")

        prefix_ids = [
            self.decoder_tokenizer.convert_tokens_to_ids(
                self.decoder_tokenizer.enc_token
            )
        ]
        if self.decoder_tokenizer.bos_token_id is not None:
            prefix_ids.append(self.decoder_tokenizer.bos_token_id)
        suffix_ids = (
            [self.decoder_tokenizer.eos_token_id]
            if self.decoder_tokenizer.eos_token_id is not None
            else []
        )
        if any(token_id is None or token_id < 0 for token_id in prefix_ids):
            raise ValueError("compressor delimiter token IDs are invalid")
        source_rows: List[List[int]] = []
        if q_texts is None:
            source_rows = [
                prefix_ids + list(document_ids) + suffix_ids
                for document_ids in passage_ids
            ]
        else:
            query_ids = self.decoder_tokenizer(
                q_texts,
                padding=False,
                max_length=self.query_max_length,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            query_marker = self.decoder_tokenizer.encode(
                "\nQuery:\n", add_special_tokens=False
            )
            document_marker = self.decoder_tokenizer.encode(
                "\nDocument:\n", add_special_tokens=False
            )
            source_rows = [
                prefix_ids
                + query_marker
                + list(query_row)
                + document_marker
                + list(document_row)
                + suffix_ids
                for query_row, document_row in zip(query_ids, passage_ids)
            ]

        num_mem_tokens = max(1, self.doc_max_length // self.compr_rate)
        assert num_mem_tokens == len(self.decoder_tokenizer.mem_tokens)

        # Paper budgets are per document: floor(L_i / r), not a fixed allocation
        # derived from the padding ceiling. Memory tokens immediately follow the
        # valid source; padding is added only after each complete source+memory row.
        memory_token_counts = torch.div(
            source_lengths, self.compr_rate, rounding_mode="floor"
        ).clamp(min=1, max=num_mem_tokens).long()
        input_ids, attention_mask = _pack_variable_encoder_memory_rows(
            source_rows,
            memory_token_counts.tolist(),
            self.decoder_tokenizer.mem_token_ids,
            self.decoder_tokenizer.pad_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "memory_token_counts": memory_token_counts,
        }

    def _replace_emb(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Replace memory tokens in decoder input with compressed embeddings."""
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)
        return self._replace_embeddings(compressed_embs, dec_input_ids, indices)

    def _replace_emb_stage2(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Replace memory tokens for stage 2."""
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)
        return self._replace_embeddings(compressed_embs, dec_input_ids, indices)

    def _replace_embeddings(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor, indices: range) -> torch.Tensor:
        """Replace memory tokens with compressed embeddings."""
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if self.sep else 0)

        # Get first memory token indices
        first_mem_token_indices = torch.argmax(
            (dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1
        )
        batch_size = inputs_embeds.size(0)

        # Replace with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j - indices[i]) * slot_len
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size()
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]

        return inputs_embeds

    def _replace_variable_memory_embeddings(
        self,
        compressed_embs: torch.Tensor,
        memory_counts: torch.Tensor,
        dec_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Insert variable-count memory blocks into an exactly matched prompt."""
        if compressed_embs.dim() != 4:
            raise ValueError("compressed_embs must have shape (B, N, T, H)")
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        memory_ids = self.decoder_tokenizer.mem_token_ids_pt.to(dec_input_ids.device)
        for batch_index in range(compressed_embs.size(0)):
            positions = torch.isin(dec_input_ids[batch_index], memory_ids).nonzero(as_tuple=True)[0]
            expected = int(memory_counts[batch_index].sum().item())
            if positions.numel() != expected:
                raise ValueError(
                    "variable ACR prompt requires the configured memory-slot count: "
                    f"found {positions.numel()} memory slots, expected {expected}"
                )
            offset = 0
            for doc_index, count_value in enumerate(memory_counts[batch_index]):
                count = int(count_value.item())
                target = positions[offset:offset + count]
                inputs_embeds[batch_index, target] = compressed_embs[
                    batch_index, doc_index, :count
                ]
                offset += count
        return inputs_embeds

    def _blend_prompt_and_memory_tokens(
        self,
        query: str,
        answer: str = None,
        qa_loss: bool = False,
        paraphrase_loss: bool = False,
        stage: str = "stage1",
        memory_counts: Optional[Sequence[int]] = None,
    ) -> Tuple[int, str]:
        """Blend prompt with memory tokens for different training stages."""
        if memory_counts is None:
            memory_counts = [len(self.decoder_tokenizer.mem_tokens)] * self.generation_top_k
        if len(memory_counts) != self.generation_top_k:
            raise ValueError("one memory-token count is required per document")
        docs = "".join(
            "".join(self.decoder_tokenizer.mem_tokens[:int(count)])
            + self.decoder_tokenizer.sep_token
            for count in memory_counts
        )
        if any(
            int(count) < 1 or int(count) > len(self.decoder_tokenizer.mem_tokens)
            for count in memory_counts
        ):
            raise ValueError("memory-token counts must fit the configured compressor budget")

        if stage == "stage1":
            if qa_loss:
                return self._blend_qa_prompt(docs, query, answer)
            elif paraphrase_loss:
                return self._blend_paraphrase_prompt(docs, query, answer)
        elif stage == "stage1_2":
            if not query:
                return self._blend_paraphrase_prompt(docs, "", answer)
            return self._blend_standard_prompt(docs, query, answer)

        raise ValueError(f"Unknown stage: {stage}")

    def _blend_qa_prompt(self, docs: str, query: List[str], answer: List[str]) -> Tuple[int, str]:
        """Create QA prompt for stage 1."""
        prompt_system = 'You are a helpful assistant. Given a document, your task is to generate some single questions to cover all key information of the document and answer them sequentially.'
        prompt_user = f"Background:\n{docs}"

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(r':\ ', ': ')}]

        qa_lines = [f"Question: {q}\nAnswer: {a}" for q, a in zip(query, answer)]
        query_answer = "\n".join(qa_lines)
        assistant_prompt = [{"role": "assistant", "content": query_answer}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                messages = [{"role": "user", "content": sys_prompt[0]['content'] + '\n' + user_prompt[0]['content']}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                # Handle response for unsupported system role
                messages_with_answer = messages + assistant_prompt
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e

        return prompt_len, response

    def _blend_paraphrase_prompt(
        self, docs: str, instruction: str, answer: Optional[str]
    ) -> Union[str, Tuple[int, str]]:
        """Create Phase-I paraphrase reconstruction from memory tokens only.

        ``instruction`` remains in the signature because all four Phase-I data
        families share one collator, but it is deliberately not serialized:
        the target may not receive an alternate semantic path around F(d).
        """
        del instruction
        prompt_system = (
            "You are a helpful assistant. Reconstruct the background from its "
            "memory tokens and output only the paraphrase."
        )
        prompt_user = f"Background:\n{docs}"

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(r':\ ', ': ')}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            if answer is None:
                return prompt

            assistant_prompt = [{"role": "assistant", "content": answer}]
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(r':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if answer is None:
                    return prompt
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e

        return prompt_len, response

    def _blend_standard_prompt(
        self, docs: str, query: str, answer: Optional[str]
    ) -> Union[str, Tuple[int, str]]:
        """Create standard prompt for stage 1_2."""
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        prompt_user = f"Background:\n{docs}\n\nQuestion:{query}"

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(r':\ ', ': ')}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            if answer is None:
                return prompt

            assistant_prompt = [{"role": "assistant", "content": answer}]
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(r':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if answer is None:
                    return prompt
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e

        return prompt_len, response

    def _blend_prompt_and_selected_memory_tokens(
        self,
        query: str,
        answer: str = None,
        memory_counts: Optional[Sequence[int]] = None,
    ) -> Tuple[int, str]:
        """Create prompt for stage 2 with selected memory tokens."""
        if memory_counts is None:
            memory_counts = [len(self.decoder_tokenizer.mem_tokens)] * self.generation_top_k
        if len(memory_counts) != self.generation_top_k:
            raise ValueError("one memory-token count is required for each decoder document slot")
        if any(
            int(count) <= 0 or int(count) > len(self.decoder_tokenizer.mem_tokens)
            for count in memory_counts
        ):
            raise ValueError("ARIA requires exactly five non-empty evidence memory blocks")
        docs = "".join(
            "".join(self.decoder_tokenizer.mem_tokens[:int(count)])
            + self.decoder_tokenizer.sep_token
            for count in memory_counts
            if int(count) > 0
        )

        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.'
        prompt_user = f"Background:\n{docs}\n\nQuestion:{query}"

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(r':\ ', ': ')}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))

            if answer is not None:
                assistant_prompt = [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    sys_prompt + user_prompt + assistant_prompt,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False
                )
            else:
                response = prompt

        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(r':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]

                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))

                if answer is not None:
                    messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                    response = self.decoder_tokenizer.apply_chat_template(
                        messages_with_answer,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False
                    )
                else:
                    response = prompt
            else:
                raise e

        return prompt_len, response

    def _prepare_stage2_supervised_decoder_inputs(
        self,
        questions: Sequence[str],
        answers: Sequence[Any],
        memory_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the fixed five-document Phase-II QA sequences."""
        if (
            len(questions) != len(answers)
            or memory_counts.shape != (len(questions), self.generation_top_k)
        ):
            raise ValueError("Phase-II questions, answers, and memory counts must align")
        rows: List[List[int]] = []
        row_offsets: List[List[Tuple[int, int]]] = []
        prompt_lengths: List[int] = []
        rendered: List[str] = []
        for row_index, (question, raw_answer) in enumerate(zip(questions, answers)):
            if isinstance(raw_answer, str):
                answer = raw_answer
            elif (
                isinstance(raw_answer, (list, tuple))
                and len(raw_answer) == 1
                and isinstance(raw_answer[0], str)
            ):
                answer = raw_answer[0]
            else:
                raise ValueError(
                    f"Phase-II answer row {row_index} must be one non-empty string"
                )
            if not answer.strip():
                raise ValueError(f"Phase-II answer row {row_index} is empty")
            prompt_length, response = self._blend_prompt_and_selected_memory_tokens(
                query=question,
                answer=answer,
                memory_counts=memory_counts[row_index].detach().cpu().tolist(),
            )
            encoded = self.decoder_tokenizer(
                response,
                add_special_tokens=False,
                truncation=False,
                return_offsets_mapping=True,
            )
            token_ids = list(encoded["input_ids"])
            offsets = [tuple(value) for value in encoded["offset_mapping"]]
            if not 0 < prompt_length < len(token_ids):
                raise ValueError("Phase-II prompt/answer boundary is invalid")
            if prompt_length > self.stage2_input_max_length:
                raise ValueError(
                    "Phase-II training prompt exceeds the paper's 1024-token "
                    "input ceiling; refusing to truncate any part of the "
                    f"question: row {row_index} has {prompt_length} prompt tokens"
                )
            prompt_end = prompt_length
            target_end = min(
                len(token_ids), prompt_length + self.stage2_target_max_length
            )
            if target_end <= prompt_length:
                raise ValueError("Phase-II target was truncated to zero tokens")
            rows.append(token_ids[:prompt_end] + token_ids[prompt_length:target_end])
            row_offsets.append(offsets[:prompt_end] + offsets[prompt_length:target_end])
            prompt_lengths.append(prompt_end)
            rendered.append(response)

        width = max(len(row) for row in rows)
        pad_id = self.decoder_tokenizer.pad_token_id
        input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        query_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        left_padding = self.decoder_tokenizer.padding_side == "left"
        for row_index, row in enumerate(rows):
            start = width - len(row) if left_padding else 0
            end = start + len(row)
            input_ids[row_index, start:end] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, start:end] = 1
            target_start = start + prompt_lengths[row_index]
            labels[row_index, target_start:end] = input_ids[row_index, target_start:end]
            marker = f"Question:{questions[row_index]}"
            marker_start = rendered[row_index].find(marker)
            if marker_start < 0:
                raise ValueError("Phase-II prompt omitted its exact question marker")
            query_start = marker_start + len("Question:")
            query_end = query_start + len(questions[row_index])
            for local_index, (char_start, char_end) in enumerate(
                row_offsets[row_index]
            ):
                if (
                    local_index < prompt_lengths[row_index]
                    and char_end > char_start
                    and char_start < query_end
                    and char_end > query_start
                ):
                    query_mask[row_index, start + local_index] = True
            if not query_mask[row_index].any():
                raise ValueError("Phase-II question was truncated from its decoder input")
        return input_ids, attention_mask, labels, query_mask

    # ── Reasoning Support ────────────────────────────────────────────────────

    def _blend_prompt_and_selected_memory_tokens_for_reasoning(
        self, query: str, answer=None
    ) -> Tuple[int, str]:
        """
        为 stage2_reasoning 阶段构建 prompt，支持多轮思考路径。

        将 memory token 标记与 query/thinking_path 融合为完整的解码器输入。
        thinking_path 是包含 <information>, <think>, <answer>, <search> 标签的
        结构化推理路径。

        Args:
            query: 问题文本
            answer: 完整的 reasoning 路径（包含 XML 标签的结构化文本）
                   或 Dict[str, str] 的 thinking_path 字典

        Returns:
            (prompt_length, full_response_text)
        """
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        docs = mem_tokens_str * self.generation_top_k

        prompt_system = (
            'You are a helpful assistant. Your task is to extract relevant '
            'information from provided documents and answer questions step by step. '
            'Use <information>...</information> tags to reference document content, '
            '<think>...</think> tags for your reasoning, and '
            '<answer>...</answer> tags for the final answer.'
        )
        prompt_user = f"Background:\n{docs}\n\nQuestion:{query}"

        # 如果是 thinking_path 字典，将其序列化为结构化文本
        if isinstance(answer, dict):
            parts = []
            for key, value in sorted(answer.items()):
                parts.append(value)
            answer_text = "\n".join(parts)
        elif isinstance(answer, str):
            answer_text = answer
        else:
            answer_text = None

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(r':\ ', ': ')}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))

            if answer_text is not None:
                assistant_prompt = [{"role": "assistant", "content": answer_text}]
                response = self.decoder_tokenizer.apply_chat_template(
                    sys_prompt + user_prompt + assistant_prompt,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            else:
                response = prompt
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(r':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                if answer_text is not None:
                    messages_with_answer = messages + [{"role": "assistant", "content": answer_text}]
                    response = self.decoder_tokenizer.apply_chat_template(
                        messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                    )
                else:
                    response = prompt
            else:
                raise e

        return prompt_len, response

    def generate_from_reasoning(
        self,
        questions: List[str],
        max_new_tokens: int = 1024,
        answers: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        使用推理路径（reasoning traces）生成答案。

        用于 stage2_reasoning 阶段：模型先生成结构化的推理路径
        （包含 <information>, <think>, <search>, <answer> 标签），
        然后从推理路径中提取最终答案。

        Args:
            questions: 问题列表
            max_new_tokens: 最大生成 token 数
            answers: 可选的参考答案列表（仅用于 prompt 构建，不强制模型匹配）
            save_dir: 可选，保存生成结果的目录

        Returns:
            (predictions, reasoning_traces) — 预测答案列表和推理路径列表
        """
        self.eval()
        device = self.decoder.device

        with torch.no_grad():
            # 准备 prompt
            prompts = []
            for q in questions:
                _, prompt_text = self._blend_prompt_and_selected_memory_tokens_for_reasoning(
                    query=q, answer=None  # 推理时不需要 answer
                )
                prompts.append(prompt_text)

            # Tokenize prompts
            inp_dec = self.decoder_tokenizer(
                prompts,
                return_tensors='pt',
                padding="longest",
                add_special_tokens=False,
                truncation=True,
                max_length=2048,
            )
            dec_input_ids = inp_dec['input_ids'].to(device)
            dec_attention_mask = inp_dec['attention_mask'].to(device)

            # 使用 decoder adapter 生成
            if 'decoder_adapter' in self.adapter_keys:
                self.decoder.set_adapter('decoder_adapter')

            output_ids = self.decoder.generate(
                inputs=dec_input_ids,
                attention_mask=dec_attention_mask,
                do_sample=False,
                top_p=None,
                temperature=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
            )

            # 解码推理路径
            reasoning_traces = self.decoder_tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )

            # 从推理路径中提取 <answer>...</answer> 标签内的内容
            predictions = []
            for trace in reasoning_traces:
                ans_match = re.search(
                    r"<answer>(.*?)</answer>", trace, re.DOTALL | re.IGNORECASE
                )
                if ans_match:
                    predictions.append(ans_match.group(1).strip())
                else:
                    # For tag-free responses, use the final non-empty text segment.
                    lines = [line.strip() for line in trace.split('\n') if line.strip()]
                    predictions.append(lines[-1] if lines else trace.strip())

            # 恢复所有 adapter
            if len(self.adapter_keys) > 0:
                self.decoder.set_adapter(self.adapter_keys)

            # 可选：保存生成结果
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "reasoning_outputs.json")
                results = [
                    {"question": q, "prediction": p, "reasoning": r}
                    for q, p, r in zip(questions, predictions, reasoning_traces)
                ]
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        return predictions, reasoning_traces

    # Model saving and loading methods
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save only the LoRA adapters and their configurations."""
        if self.lora:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Save LoRA adapter weights
            torch.save(
                self._get_all_adapters_state_dict(),
                os.path.join(save_directory, "adapters.pth")
            )

            # Save first and last layers of decoder
            torch.save(
                self._get_decoder_first_and_last_layer_state_dict(),
                os.path.join(save_directory, "decoder_first_last_layers.pth")
            )

            # ── Save MTFRL projection head ──────────────────────────────────
            if self._mtfrl_projection is not None:
                torch.save(
                    self._mtfrl_projection.state_dict(),
                    os.path.join(save_directory, "mtfrl_projection.pth")
                )

            # ── Save BGE projection (if fitted) ─────────────────────────────
            if self._bge_projection is not None:
                torch.save(
                    {
                        "state_dict": self._bge_projection.state_dict(),
                        **(self._bge_projection_metadata or {}),
                    },
                    os.path.join(save_directory, "bge_projection.pth")
                )

            # Persist the exact MADS commit after its first lazy load.  Do not
            # fabricate a revision for checkpoints saved before MADS runs.
            if self.rag_pipeline is not None:
                semantic_agent = self.rag_pipeline.ccef.sem
                self.config.mads_semantic_model_name = semantic_agent.model_name
                if (
                    getattr(self.config, "mads_semantic_model_revision", None) is None
                    and getattr(
                        self.config,
                        "mads_semantic_model_resolved_revision",
                        None,
                    ) is None
                ):
                    # Preserve an original tag/branch declaration across a
                    # load-and-resave cycle; only fill it for a newly supplied
                    # runtime pin that the checkpoint did not already record.
                    self.config.mads_semantic_model_revision = (
                        semantic_agent.model_revision
                    )
                self.config.mads_semantic_model_resolved_revision = (
                    semantic_agent.resolved_revision
                )

            source_manifest = getattr(
                self, "_aria_source_snapshot_manifest", None
            )
            if source_manifest is None:
                source_manifest = build_source_snapshot_manifest()
                self._aria_source_snapshot_manifest = source_manifest
            self.config.aria_source_snapshot_scheme = source_manifest["scheme"]
            self.config.aria_source_git_commit = source_manifest["git_commit"]
            self.config.aria_source_git_dirty = source_manifest["git_dirty"]
            self.config.aria_source_tree_sha256 = source_manifest[
                "source_tree_sha256"
            ]
            self.config.aria_source_file_count = source_manifest[
                "source_file_count"
            ]
            with open(
                os.path.join(save_directory, "aria_source_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    source_manifest,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            write_source_snapshot(
                Path(save_directory) / "aria_source_snapshot.zip",
                source_manifest,
            )

            # Save configuration
            self.config.save_pretrained(save_directory)
            artifact_names = [
                "config.json",
                "adapters.pth",
                "decoder_first_last_layers.pth",
                "aria_source_manifest.json",
                "aria_source_snapshot.zip",
            ]
            for optional_name in ("bge_projection.pth", "mtfrl_projection.pth"):
                if os.path.isfile(os.path.join(save_directory, optional_name)):
                    artifact_names.append(optional_name)
            artifacts: Dict[str, Dict[str, Any]] = {}
            for name in sorted(artifact_names):
                path = Path(save_directory) / name
                artifacts[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            checkpoint_manifest = {
                "format": "aria-checkpoint-v2",
                "training_stage": self.config.training_stage,
                "rag_configuration": getattr(
                    self.config, "aria_rag_configuration", None
                ),
                "source_tree_sha256": source_manifest["source_tree_sha256"],
                "source_git_commit": source_manifest["git_commit"],
                "source_git_dirty": source_manifest["git_dirty"],
                "artifacts": artifacts,
            }
            with open(
                os.path.join(save_directory, "aria_checkpoint_manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    checkpoint_manifest,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
        else:
            super().save_pretrained(save_directory, **kwargs)

    def _get_all_adapters_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return the state dicts of all adapters."""
        return {
            key: {k: v.cpu() for k, v in self.decoder.get_adapter_state_dict(key).items()}
            for key in self.adapter_keys
        }

    def _get_decoder_first_and_last_layer_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get first and last layers that change when adding tokens."""
        out = {}
        for k, v in self.decoder.named_parameters():
            if 'lm_head.weight' in k or 'embed_tokens.weight' in k:
                out[k] = v.cpu()
        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Load model from pretrained checkpoint."""
        strict_aria_artifacts = bool(kwargs.pop("strict_aria_artifacts", False))
        external_bge_artifact = bool(kwargs.pop("external_bge_artifact", False))
        strict_source_training_stage = kwargs.pop("strict_source_training_stage", None)
        initialize_query_reasoner_adapter = bool(
            kwargs.pop("initialize_query_reasoner_adapter", True)
        )

        def resolve_checkpoint_file(filename: str) -> str:
            local_path = os.path.join(pretrained_model_name_or_path, filename)
            if os.path.isfile(local_path):
                return local_path
            try:
                return hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=filename,
                )
            except Exception:
                return local_path

        # Strict validation reads protocol fields from serialized config before
        # CLaRaConfig defaults can make an old checkpoint look newly compliant.
        source_config_dict: Optional[Dict[str, Any]] = None
        if strict_source_training_stage is not None or strict_aria_artifacts:
            source_config_dict, _ = CLaRaConfig.get_config_dict(
                pretrained_model_name_or_path
            )
            if "training_stage" not in source_config_dict:
                raise ValueError(
                    "Strict source checkpoint config requires serialized training_stage"
                )
        if strict_aria_artifacts and source_config_dict is not None:
            serialized_stage = source_config_dict["training_stage"]
            if serialized_stage == "stage2":
                required_protocol = {
                    "cfrs_reconstruction_scheme": CFRS_RECONSTRUCTION_SCHEME,
                    "retrieval_straight_through_scheme": (
                        RETRIEVAL_STRAIGHT_THROUGH_SCHEME
                    ),
                    "mtfrl_initialization_scheme": MTFRL_INITIALIZATION_SCHEME,
                }
                for key, expected in required_protocol.items():
                    if key not in source_config_dict:
                        raise ValueError(
                            f"Strict checkpoint config must serialize {key!r}; "
                            "defaults cannot establish historical provenance"
                        )
                    if source_config_dict[key] != expected:
                        raise ValueError(
                            f"Strict checkpoint config {key!r} must be {expected!r}"
                        )
                rag_configuration = source_config_dict.get(
                    "aria_rag_configuration"
                )
                expected_loss_weights = {
                    "lambda_mse": 0.0 if rag_configuration == "clara_baseline" else 0.10
                }
                if source_config_dict.get("aria_loss_weights") != expected_loss_weights:
                    raise ValueError(
                        "Strict checkpoint config has inconsistent Phase-II "
                        f"objective metadata; expected {expected_loss_weights!r}"
                    )
                if rag_configuration not in {
                    "remove_mtfrl",
                    "static_second_retrieval",
                    "remove_all_coupling",
                    "clara_baseline",
                }:
                    if source_config_dict.get("mtfrl_initialization_rank") is not None:
                        raise ValueError("W_BGE-initialized MTFRL must not serialize an SVD rank")
                    expected_width = _mtfrl_hidden_width(
                        int(source_config_dict.get("hidden_size", 4096)), 1024
                    )
                    if source_config_dict.get("mtfrl_hidden_width") != expected_width:
                        raise ValueError(
                            "Strict MTFRL checkpoint must serialize its exact hidden width"
                        )
            checkpoint_manifest_path = resolve_checkpoint_file(
                "aria_checkpoint_manifest.json"
            )
            if not os.path.isfile(checkpoint_manifest_path):
                raise FileNotFoundError(
                    "Strict ARIA loading requires aria_checkpoint_manifest.json"
                )
            with open(checkpoint_manifest_path, "r", encoding="utf-8") as handle:
                checkpoint_manifest = json.load(handle)
            if (
                not isinstance(checkpoint_manifest, Mapping)
                or checkpoint_manifest.get("format") != "aria-checkpoint-v2"
                or checkpoint_manifest.get("training_stage")
                != source_config_dict["training_stage"]
            ):
                raise ValueError("ARIA checkpoint manifest header is invalid")
            artifact_records = checkpoint_manifest.get("artifacts")
            if not isinstance(artifact_records, Mapping):
                raise ValueError("ARIA checkpoint manifest requires artifact hashes")
            required_artifacts = {
                "config.json",
                "adapters.pth",
                "decoder_first_last_layers.pth",
                "aria_source_manifest.json",
                "aria_source_snapshot.zip",
            }
            rag_configuration = source_config_dict.get("aria_rag_configuration")
            if source_config_dict["training_stage"] == "stage2":
                if not external_bge_artifact and rag_configuration != "clara_baseline":
                    required_artifacts.add("bge_projection.pth")
                if rag_configuration not in {
                    "remove_mtfrl",
                    "static_second_retrieval",
                    "remove_all_coupling",
                    "clara_baseline",
                }:
                    required_artifacts.add("mtfrl_projection.pth")
            missing_artifacts = sorted(required_artifacts - set(artifact_records))
            if missing_artifacts:
                raise ValueError(
                    "ARIA checkpoint manifest omits required artifacts: "
                    + ", ".join(missing_artifacts)
                )
            for name, record in artifact_records.items():
                if Path(name).name != name or not isinstance(record, Mapping):
                    raise ValueError("ARIA checkpoint manifest contains an unsafe artifact name")
                expected_digest = record.get("sha256")
                expected_bytes = record.get("bytes")
                if (
                    not isinstance(expected_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
                    or isinstance(expected_bytes, bool)
                    or not isinstance(expected_bytes, int)
                    or expected_bytes < 0
                ):
                    raise ValueError(f"ARIA checkpoint artifact record is invalid: {name}")
                artifact_path = Path(resolve_checkpoint_file(name))
                if (
                    not artifact_path.is_file()
                    or artifact_path.stat().st_size != expected_bytes
                    or file_sha256(artifact_path) != expected_digest
                ):
                    raise ValueError(f"ARIA checkpoint artifact hash mismatch: {name}")

            source_manifest_path = Path(
                resolve_checkpoint_file("aria_source_manifest.json")
            )
            with source_manifest_path.open("r", encoding="utf-8") as handle:
                source_manifest = json.load(handle)
            if (
                source_manifest.get("scheme") != SOURCE_SNAPSHOT_SCHEME
                or source_manifest.get("source_tree_sha256")
                != source_config_dict.get("aria_source_tree_sha256")
                or source_manifest.get("source_tree_sha256")
                != checkpoint_manifest.get("source_tree_sha256")
                or source_manifest.get("git_commit")
                != source_config_dict.get("aria_source_git_commit")
            ):
                raise ValueError("ARIA source snapshot provenance is inconsistent")
            source_files = source_manifest.get("files")
            if not isinstance(source_files, Mapping) or not source_files:
                raise ValueError("ARIA source manifest requires exact file hashes")
            snapshot_path = resolve_checkpoint_file("aria_source_snapshot.zip")
            with zipfile.ZipFile(snapshot_path) as snapshot:
                if sorted(snapshot.namelist()) != sorted(source_files):
                    raise ValueError("ARIA source snapshot file inventory mismatch")
                for name, expected_digest in source_files.items():
                    if hashlib.sha256(snapshot.read(name)).hexdigest() != expected_digest:
                        raise ValueError(
                            f"ARIA source snapshot content mismatch: {name}"
                        )
        config = CLaRaConfig.from_pretrained(pretrained_model_name_or_path)
        if (
            strict_source_training_stage is not None
            and config.training_stage != strict_source_training_stage
        ):
            raise ValueError(
                f"Checkpoint training_stage is {config.training_stage!r}, expected "
                f"{strict_source_training_stage!r}"
            )

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        map_location = torch.device("cpu")

        if config.lora:
            # Delay adapter construction
            config.load_adapters = False
            if 'device_map' in kwargs:
                config.device_map = kwargs['device_map']

            # Initialize model
            print(f"Initializing model from trained checkpoint: {config}")
            model = cls(config)

            # Load first and last layers
            first_and_last_layers_path = resolve_checkpoint_file(
                "decoder_first_last_layers.pth"
            )

            if os.path.exists(first_and_last_layers_path):
                first_and_last_decoder_state_dict = torch.load(
                    first_and_last_layers_path, map_location=map_location, weights_only=True
                )
                if not isinstance(first_and_last_decoder_state_dict, Mapping) or not first_and_last_decoder_state_dict:
                    raise ValueError("decoder_first_last_layers.pth must be a non-empty mapping")
                if any(
                    isinstance(value, torch.Tensor)
                    and not _tensor_is_finite_in_chunks(value)
                    for value in first_and_last_decoder_state_dict.values()
                ):
                    raise ValueError("decoder first/last layers contain NaN or infinity")
                expected_decoder_keys = set(
                    model._get_decoder_first_and_last_layer_state_dict()
                )
                if set(first_and_last_decoder_state_dict) != expected_decoder_keys:
                    raise ValueError(
                        "decoder_first_last_layers.pth does not contain the exact "
                        "embedding/lm_head key set"
                    )
                for key, value in first_and_last_decoder_state_dict.items():
                    if value.shape != model.decoder.state_dict()[key].shape:
                        raise ValueError(
                            f"Decoder artifact shape mismatch for {key}: "
                            f"{tuple(value.shape)} != "
                            f"{tuple(model.decoder.state_dict()[key].shape)}"
                        )
                model.decoder.load_state_dict(first_and_last_decoder_state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"LoRA checkpoint loading requires decoder_first_last_layers.pth: "
                    f"{first_and_last_layers_path}"
                )

            peft_config = model._get_peft_config(lora_r=config.lora_r)

            # Load LoRA adapters
            adapters_path = resolve_checkpoint_file("adapters.pth")

            if os.path.exists(adapters_path):
                adapters_state_dict = torch.load(adapters_path, map_location=map_location, weights_only=True)
                if not isinstance(adapters_state_dict, Mapping) or not adapters_state_dict:
                    raise ValueError("adapters.pth must be a non-empty mapping")
                if any(
                    not isinstance(adapter_state, Mapping)
                    or not adapter_state
                    or any(
                        isinstance(value, torch.Tensor)
                        and not _tensor_is_finite_in_chunks(value)
                        for value in adapter_state.values()
                    )
                    for adapter_state in adapters_state_dict.values()
                ):
                    raise ValueError("adapter states are empty, malformed, or non-finite")
                model._load_adapters_from_state_dict(
                    adapters_state_dict,
                    peft_config,
                    config,
                    strict_aria_artifacts=(
                        strict_aria_artifacts
                        and strict_source_training_stage != "stage1"
                    ),
                    initialize_query_reasoner_adapter=initialize_query_reasoner_adapter,
                )
            else:
                raise FileNotFoundError(
                    f"LoRA checkpoint loading requires adapters.pth: {adapters_path}"
                )

            # ── Load BGE projection ─────────────────────────────────────────
            bge_path = resolve_checkpoint_file("bge_projection.pth")
            if strict_source_training_stage == "stage1" and os.path.exists(bge_path):
                raise ValueError("Strict Phase-I checkpoint must not bundle W_BGE")
            if os.path.exists(bge_path):
                try:
                    model.setup_bge_projection(freeze=True)
                    bge_state = torch.load(bge_path, map_location=map_location, weights_only=True)
                    if isinstance(bge_state, dict) and isinstance(
                        bge_state.get("state_dict"), dict
                    ):
                        model._bge_projection_metadata = {
                            key: value
                            for key, value in bge_state.items()
                            if key != "state_dict"
                        }
                        bge_state = bge_state["state_dict"]
                    model._bge_projection.load_state_dict(bge_state, strict=True)
                    print("BGE projection loaded")
                except Exception as e:
                    raise RuntimeError(f"Invalid bundled BGE projection: {bge_path}") from e
            elif strict_aria_artifacts and strict_source_training_stage != "stage1" and not external_bge_artifact and getattr(
                config, "aria_rag_configuration", None
            ) != "clara_baseline":
                raise FileNotFoundError(
                    f"Strict ARIA loading requires bge_projection.pth: {bge_path}"
                )

            # ── Load MTFRL projection head after W_BGE ──────────────────────
            mtfrl_path = resolve_checkpoint_file("mtfrl_projection.pth")
            if strict_source_training_stage == "stage1" and os.path.exists(mtfrl_path):
                raise ValueError("Strict Phase-I checkpoint must not bundle MTFRL weights")
            rag_configuration = getattr(config, "aria_rag_configuration", None)
            mtfrl_required = rag_configuration not in {
                "remove_mtfrl",
                "static_second_retrieval",
                "remove_all_coupling",
                "clara_baseline",
            }
            if os.path.exists(mtfrl_path):
                try:
                    model.setup_mtfrl_projection(initialize_from_bge=False)
                    mtfrl_state = torch.load(mtfrl_path, map_location=map_location, weights_only=True)
                    if not isinstance(mtfrl_state, Mapping) or not mtfrl_state:
                        raise ValueError("MTFRL state must be a non-empty mapping")
                    if any(
                        isinstance(value, torch.Tensor)
                        and not _tensor_is_finite_in_chunks(value)
                        for value in mtfrl_state.values()
                    ):
                        raise ValueError("MTFRL state contains NaN or infinity")
                    model._mtfrl_projection.load_state_dict(mtfrl_state, strict=True)
                    print("MTFRL projection head loaded")
                except Exception as e:
                    raise RuntimeError(f"Invalid bundled MTFRL projection: {mtfrl_path}") from e
            elif (
                strict_aria_artifacts
                and strict_source_training_stage != "stage1"
                and mtfrl_required
            ):
                raise FileNotFoundError(
                    f"Strict ARIA loading requires mtfrl_projection.pth: {mtfrl_path}"
                )

            if strict_aria_artifacts:
                if strict_source_training_stage == "stage1":
                    expected_adapters = {"encoder_adapter"}
                    if set(adapters_state_dict) != expected_adapters:
                        raise ValueError(
                            "Strict Phase-I checkpoint must contain only encoder_adapter"
                        )
                else:
                    expected_adapters = {
                        "encoder_adapter",
                        "query_reasoner_adapter",
                        "decoder_adapter",
                    }
                    if set(adapters_state_dict) != expected_adapters:
                        raise ValueError(
                            "Strict Phase-II checkpoint adapter set mismatch: "
                            f"saved={sorted(adapters_state_dict)}, "
                            f"expected={sorted(expected_adapters)}"
                        )
                for adapter_name in expected_adapters:
                    saved_state = adapters_state_dict.get(adapter_name)
                    loaded_state = model.decoder.get_adapter_state_dict(adapter_name)
                    if not isinstance(saved_state, Mapping) or set(saved_state) != set(loaded_state):
                        raise ValueError(
                            f"Adapter {adapter_name!r} does not contain its exact key set"
                        )
                    for key, loaded_value in loaded_state.items():
                        saved_value = saved_state[key]
                        if (
                            not isinstance(saved_value, torch.Tensor)
                            or saved_value.shape != loaded_value.shape
                        ):
                            raise ValueError(
                                f"Adapter {adapter_name!r} shape mismatch for {key!r}"
                            )

            model._set_all_adapters()
            config.load_adapters = True
            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    def _load_adapters_from_state_dict(
        self,
        adapters_state_dict: Dict,
        peft_config: LoraConfig,
        config: CLaRaConfig,
        strict_aria_artifacts: bool = False,
        initialize_query_reasoner_adapter: bool = True,
    ):
        """Load adapters from state dict based on training stage."""
        if not getattr(config, 'pure_inference', False):
            for key, val in adapters_state_dict.items():
                # Skip certain adapters based on training stage
                if config.training_stage == 'stage1' and key == 'query_reasoner_adapter':
                    continue
                elif config.training_stage == 'stage1_2' and key in ['query_reasoner_adapter', 'decoder_adapter']:
                    continue
                elif config.training_stage == 'stage2_reasoning' and key == 'decoder_adapter':
                    continue

                self._load_adapter_from_state_dict(
                    peft_config=peft_config,
                    adapter_name=key,
                    adapter_state_dict=val
                )
        else:
            # Load all adapters for pure inference
            for key, val in adapters_state_dict.items():
                self._load_adapter_from_state_dict(
                    peft_config=peft_config,
                    adapter_name=key,
                    adapter_state_dict=val
                )

        # Handle special cases for stage 2 training
        if (
            config.training_stage == 'stage2'
            and initialize_query_reasoner_adapter
            and 'query_reasoner_adapter' not in adapters_state_dict
        ):
            if strict_aria_artifacts:
                raise ValueError(
                    "Strict Phase-II checkpoint requires query_reasoner_adapter"
                )
            self._copy_phase1_adapter(
                adapters_state_dict, peft_config, "query_reasoner_adapter"
            )
        if config.training_stage == 'stage2' and 'decoder_adapter' not in self.adapter_keys:
            if strict_aria_artifacts:
                raise ValueError("Strict Phase-II checkpoint requires decoder_adapter")
            if getattr(config, "aria_rag_configuration", None) == "clara_baseline":
                self._copy_phase1_adapter(
                    adapters_state_dict, peft_config, "decoder_adapter"
                )
            else:
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
                print('Initialized generator decoder_adapter for Phase II')

    def _load_adapter_from_state_dict(self, peft_config: LoraConfig, adapter_name: str, adapter_state_dict: Dict):
        """Create adapter from state dict."""
        print(f'Loading checkpoint adapter: {adapter_name}')
        self.decoder.load_adapter(
            peft_config=peft_config,
            adapter_name=adapter_name,
            adapter_state_dict=adapter_state_dict
        )
        self.adapter_keys.append(adapter_name)

    def _copy_phase1_adapter(
        self,
        adapters_state_dict: Dict,
        peft_config: LoraConfig,
        target_adapter: str,
    ) -> None:
        """Copy the Phase-I compressor LoRA exactly into a Phase-II adapter."""
        if target_adapter in self.adapter_keys:
            return
        encoder_state = adapters_state_dict.get("encoder_adapter")
        if not isinstance(encoder_state, Mapping) or not encoder_state:
            raise ValueError(
                f"Phase II requires the Phase-I encoder_adapter to initialize {target_adapter}"
            )
        self._load_adapter_from_state_dict(
            peft_config=peft_config,
            adapter_name=target_adapter,
            adapter_state_dict=encoder_state,
        )
        loaded_state = self.decoder.get_adapter_state_dict(target_adapter)
        if set(loaded_state) != set(encoder_state):
            raise RuntimeError(f"{target_adapter} copy changed the Phase-I adapter key set")
        for key, value in loaded_state.items():
            if not torch.equal(
                value.detach().cpu(), encoder_state[key].detach().cpu()
            ):
                raise RuntimeError(
                    f"{target_adapter} copy is not exact for adapter tensor {key!r}"
                )
        print(f"Copied encoder_adapter exactly into {target_adapter}")

    def _handle_query_reasoner_adapter_loading(self, adapters_state_dict: Dict, peft_config: LoraConfig):
        """Backward-compatible wrapper for Phase-II ARIA checkpoint loading."""
        self._copy_phase1_adapter(
            adapters_state_dict, peft_config, "query_reasoner_adapter"
        )

    # Forward pass methods
    def forward(self,
                batch: Dict = None,
                questions: List[str] = None,
                documents: List[List[str]] = None,
                answers: List[str] = None,
                original_answer_gen_api: str = None,
                stage2_mips: bool = False,
                stage2_retrieval_top_n: int = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through the validated ARIA batch interface.

        Args:
            batch: Validated preprocessed ARIA batch dictionary
            questions: Compatibility parameter; training uses ``batch``
            documents: Compatibility parameter; training uses ``batch``
            answers: Compatibility parameter; training uses ``batch``
            original_answer_gen_api: Compatibility API parameter
            stage2_mips: Whether to use MIPS for stage2
            stage2_retrieval_top_n: Top-n for stage2 retrieval

        Returns:
            Tuple of (loss, additional_outputs_dict)
        """
        if batch is None:
            raise ValueError(
                "ARIA paper-protocol training requires the validated batch interface"
            )
        return self._forward_batch(batch, stage2_mips, stage2_retrieval_top_n)

    def _forward_batch(self, batch: Dict, stage2_mips: bool, stage2_retrieval_top_n: int) -> Tuple[torch.Tensor, Dict]:
        """Handle batch-based forward pass."""
        stage = batch.get("stage", None)

        if stage == "stage1":
            return self._forward_stage1_batch(batch)
        elif stage == "stage2":
            return self._forward_stage2_batch(batch, stage2_mips, stage2_retrieval_top_n)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _forward_stage1_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 1 training."""
        # Move tensors to device
        enc_input_ids = batch["enc_input_ids"].to(self.decoder.device)
        enc_attention_mask = batch["enc_attention_mask"].to(self.decoder.device)
        dec_input_ids = batch["dec_input_ids"].to(self.decoder.device)
        dec_attention_mask = batch["dec_attention_mask"].to(self.decoder.device)
        labels = batch["labels"].to(self.decoder.device)
        memory_token_counts = batch["memory_token_counts"].to(self.decoder.device)

        out = self._forward_stage_1(
            enc_input_ids=enc_input_ids,
            enc_attention_mask=enc_attention_mask,
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
            labels=labels,
            memory_token_counts=memory_token_counts,
        )
        return out["loss"], {
            "logits": out["logits"],
            "mse_loss": out["mse_loss"],
        }

    def _compute_qa_lmse(
        self,
        hidden_states: torch.Tensor,
        dec_input_ids: torch.Tensor,
        dec_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        query_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. 4 on memory vs. query+answer positions of the same QA forward."""
        memory_mask = torch.isin(
            dec_input_ids,
            self.decoder_tokenizer.mem_token_ids_pt.to(dec_input_ids.device),
        ) & dec_attention_mask.bool()
        special_ids = torch.tensor(
            self.decoder_tokenizer.all_special_ids,
            device=dec_input_ids.device,
            dtype=dec_input_ids.dtype,
        )
        structural_mask = torch.isin(dec_input_ids, special_ids)
        answer_mask = (
            (labels != IGNORE_INDEX)
            & dec_attention_mask.bool()
            & ~memory_mask
            & ~structural_mask
        )
        query_mask = query_position_mask.to(dec_input_ids.device, dtype=torch.bool)
        if query_mask.shape != dec_input_ids.shape:
            raise ValueError("query_position_mask must have the same shape as dec_input_ids")
        # The collator derives this mask from fast-tokenizer character offsets.
        # Constrain it to the prompt so an answer that repeats the question can
        # never be mistaken for the query representation in Eq. (4).
        query_mask &= labels.eq(IGNORE_INDEX)
        query_mask &= dec_attention_mask.bool()
        query_mask &= ~memory_mask
        query_mask &= ~structural_mask
        non_memory_mask = (query_mask | answer_mask) & dec_attention_mask.bool() & ~memory_mask
        if (memory_mask.sum(dim=1) == 0).any() or (non_memory_mask.sum(dim=1) == 0).any():
            raise ValueError("LMSE requires both memory and query/answer positions in every sample")
        memory_mean = (
            hidden_states * memory_mask.unsqueeze(-1)
        ).sum(dim=1) / memory_mask.sum(dim=1, keepdim=True)
        non_memory_mean = (
            hidden_states * non_memory_mask.unsqueeze(-1)
        ).sum(dim=1) / non_memory_mask.sum(dim=1, keepdim=True)
        # The original objective is the squared L2 norm, with no 1/d_h factor.
        return ((memory_mean.float() - non_memory_mean.float()) ** 2).sum(dim=-1).mean()

    def _forward_stage2_batch(
        self,
        batch: Dict,
        stage2_mips: bool,
        stage2_retrieval_top_n: Optional[int],
    ) -> Tuple[torch.Tensor, Dict]:
        """Paper Phase II: QA plus unnormalized hidden-state squared L2."""
        if getattr(self.config, "aria_rag_configuration", None) == "clara_baseline":
            return self._forward_clara_baseline_batch(batch)
        if stage2_mips:
            raise ValueError(
                "ARIA Phase II requires stage2_mips=False and the integrated retrieval pipeline"
            )
        if self.rag_pipeline is None:
            raise RuntimeError(
                "Phase II requires setup_rag_pipeline() with the full corpus and BGE embeddings"
            )
        if self._rag_config.use_mtfrl and (
            self._mtfrl is None or self._mtfrl_projection is None
        ):
            raise RuntimeError("Enabled MTFRL requires its retrieval index and projection head")
        questions = list(batch["questions"])
        batch_size = len(questions)
        n_docs = self.generation_top_k
        if n_docs != 5:
            raise ValueError("the paper's Phase-II generator requires exactly five documents")
        if stage2_retrieval_top_n not in (None, n_docs):
            raise ValueError("paper CCEF uses exactly five retained documents")

        query_reps = self._compr_query_reasoner_stage2(
            batch["query_input_ids"].to(self.decoder.device),
            batch["query_attention_mask"].to(self.decoder.device),
        )
        query_bge = self._project_query_reps_to_bge(query_reps)

        initial_evidence: List[List[_ScoredDoc]] = []
        qca_results: List[QCAResult] = []
        initial_pools, initial_qca = self.rag_pipeline.retrieve_initial_batch(
            questions, query_bge
        )
        for batch_index, question in enumerate(questions):
            scored, qca_result, _ = self.rag_pipeline.retrieve_scored(
                question,
                query_emb=query_bge[batch_index],
                override_top_k=n_docs,
                embed_subquery=lambda text: self._encode_subquery_for_retrieval(text).detach(),
                qca_result=initial_qca[batch_index],
                initial_retrieved=initial_pools[batch_index],
            )
            if len(scored) != n_docs:
                raise RuntimeError(
                    "CCEF retained fewer than the paper's five documents after thresholding"
                )
            initial_evidence.append(scored)
            qca_results.append(qca_result)

        # First CCEF -> detached ACR -> compressor -> MTFRL.
        first_pass = self._compress_evidence(
            initial_evidence,
            qca_results,
            query_bge,
            compute_cfrs=self._cfrs is not None and self._mtfrl is None,
        )
        feedback: Optional[torch.Tensor] = None
        if self._mtfrl is not None:
            if self._rag_config.second_retrieval_mode == "memory_feedback":
                feedback = self._compute_first_pass_feedback_query(first_pass)
                second_query = feedback
            elif self._rag_config.second_retrieval_mode == "static_query":
                # Release convention for the topology-matched control: reuse
                # the original QR representation after frozen W_BGE projection.
                # ``feedback`` remains None because this control uses QR directly.
                second_query = query_bge
            else:
                raise RuntimeError("mounted second retriever has disabled mode")
            second_round = self._mtfrl.second_round_retrieve(
                feedback_queries=second_query,
                already_retrieved_ids=[
                    [doc.doc_id for doc in docs] for docs in initial_evidence
                ],
                top_k=self._rag_config.mtfrl_second_top_k,
            )

            # D1 union D2 -> second MADS/CCEF -> recomputed ACR/compression.
            final_evidence: List[List[_ScoredDoc]] = []
            for batch_index, question in enumerate(questions):
                rescored = self.rag_pipeline.rescore_union(
                    question,
                    initial_evidence[batch_index],
                    second_round[batch_index],
                    query_emb=query_bge[batch_index],
                    top_k=n_docs,
                )
                if len(rescored) != n_docs:
                    raise RuntimeError(
                        "second-round CCEF retained fewer than the paper's five documents"
                    )
                final_evidence.append(rescored)
            final_pass = self._compress_evidence(
                final_evidence,
                qca_results,
                query_bge,
                feedback_query=feedback,
                compute_cfrs=self._cfrs is not None,
            )
        else:
            final_evidence = initial_evidence
            final_pass = first_pass

        differentiable_fused = self._differentiable_fused_scores(
            query_bge, final_evidence, feedback
        )
        # CFRS is a differentiable ordering path, not an auxiliary loss term.
        final_scores = CompressionFidelityReranker.rerank(
            differentiable_fused,
            final_pass["per_doc_mse"],
            cfrs_weight=self._rag_config.cfrs_weight,
            eps=self._rag_config.numerical_epsilon,
            document_mask=final_pass["document_mask"],
        ) if self._cfrs is not None else differentiable_fused
        selected, hard_order = self._straight_through_cfrs_permutation(
            final_pass["memory"],
            final_scores,
            final_pass["document_mask"],
        )
        ordered_context_counts = final_pass["context_counts"].gather(
            1, hard_order
        )

        (
            dec_input_ids,
            dec_attention_mask,
            labels,
            query_position_mask,
        ) = self._prepare_stage2_supervised_decoder_inputs(
            questions,
            batch["answers"],
            ordered_context_counts,
        )
        dec_input_ids = dec_input_ids.to(self.decoder.device)
        dec_attention_mask = dec_attention_mask.to(self.decoder.device)
        labels = labels.to(self.decoder.device)
        query_position_mask = query_position_mask.to(self.decoder.device)
        inputs_embeds = self._replace_variable_memory_embeddings(
            selected, ordered_context_counts, dec_input_ids
        )
        self.decoder.set_adapter("decoder_adapter")
        dec_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        lmse_loss = self._compute_qa_lmse(
            dec_out.hidden_states[-1],
            dec_input_ids,
            dec_attention_mask,
            labels,
            query_position_mask,
        )

        corpus_indices = torch.full(
            (batch_size, n_docs), -1, device=hard_order.device, dtype=torch.long
        )
        for row_index, documents_for_question in enumerate(final_evidence):
            corpus_indices[row_index, : len(documents_for_question)] = torch.tensor(
                [document.corpus_index for document in documents_for_question],
                device=hard_order.device,
                dtype=torch.long,
            )
        topk_idx = corpus_indices.gather(1, hard_order)
        hard_order_rows = hard_order.detach().cpu().tolist()
        topk_doc_ids = [
            [
                final_evidence[batch_index][doc_index].doc_id
                for doc_index in order
                if doc_index < len(final_evidence[batch_index])
            ]
            for batch_index, order in enumerate(hard_order_rows)
        ]
        valid_float = final_pass["document_mask"].to(final_pass["ratios"].dtype)
        cfrs_error_metric = (
            final_pass["per_doc_mse"].detach() * valid_float
        ).sum() / valid_float.sum()
        acr_ratio_metric = (
            final_pass["ratios"].detach() * valid_float
        ).sum() / valid_float.sum()
        self._set_all_adapters()
        return dec_out.loss, {
            "logits": dec_out.logits,
            "topk_idx": topk_idx,
            "topk_doc_ids": topk_doc_ids,
            "mse_loss": lmse_loss,
            "cfrs_error": cfrs_error_metric,
            "acr_ratio": acr_ratio_metric,
        }

    def _forward_clara_baseline_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Appendix-A.37 CLaRa Phase II, isolated from the ARIA pipeline."""
        documents = [list(row) for row in batch["docs"]]
        batch_size = len(documents)
        selected_count = self.generation_top_k
        candidate_counts = {len(row) for row in documents}
        if batch_size == 0 or len(candidate_counts) != 1:
            raise ValueError(
                "CLaRa baseline requires one non-empty, fixed-size candidate pool per row"
            )
        candidate_count = candidate_counts.pop()
        if candidate_count < selected_count:
            raise ValueError(
                "CLaRa candidate count must be at least generation_top_k: "
                f"N={candidate_count}, k={selected_count}"
            )
        if any(
            not isinstance(document, str) or not document.strip()
            for row in documents
            for document in row
        ):
            raise ValueError("CLaRa candidates must be non-empty document strings")
        if self.rag_pipeline is not None:
            raise RuntimeError("Matched CLaRa must not mount the ARIA retrieval pipeline")
        flat_documents = [document for row in documents for document in row]
        encoder_inputs = self._prepare_clara_encoder_inputs(flat_documents)
        candidate_memory_counts = encoder_inputs["memory_token_counts"].to(
            self.decoder.device
        ).view(batch_size, candidate_count)

        # The Phase-I compressor and document representations are fixed.  Eval
        # mode also disables LoRA dropout, making online no-grad recomputation
        # exactly the deterministic equivalent of an offline representation.
        decoder_was_training = self.decoder.training
        self.decoder.eval()
        try:
            with torch.no_grad():
                compressed, _ = self.compress(
                    encoder_inputs["input_ids"].to(self.decoder.device),
                    encoder_inputs["attention_mask"].to(self.decoder.device),
                )
                compressed = compressed.detach()
        finally:
            self.decoder.train(decoder_was_training)
        candidate_memory = compressed.view(
            batch_size,
            candidate_count,
            compressed.size(1),
            compressed.size(2),
        )
        if candidate_memory.size(2) != self.n_mem_tokens:
            raise RuntimeError("CLaRa padded document representation width is inconsistent")

        query_representations = self._compr_query_reasoner_stage2(
            batch["query_input_ids"].to(self.decoder.device),
            batch["query_attention_mask"].to(self.decoder.device),
        )
        selected, topk_idx, selector_scores, selector_weights = (
            _clara_st_select_candidate_memory(
                query_representations,
                candidate_memory,
                selected_count,
                candidate_memory_counts=candidate_memory_counts,
            )
        )
        selected_counts = candidate_memory_counts.gather(1, topk_idx)
        (
            dec_input_ids,
            dec_attention_mask,
            labels,
            _,
        ) = self._prepare_stage2_supervised_decoder_inputs(
            batch["questions"], batch["answers"], selected_counts
        )
        dec_input_ids = dec_input_ids.to(self.decoder.device)
        dec_attention_mask = dec_attention_mask.to(self.decoder.device)
        labels = labels.to(self.decoder.device)
        inputs_embeds = self._replace_variable_memory_embeddings(
            selected,
            selected_counts,
            dec_input_ids,
        )
        self.decoder.set_adapter("decoder_adapter")
        dec_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=labels,
        )
        self.decoder.set_adapter(["query_reasoner_adapter", "decoder_adapter"])
        zero = dec_out.loss * 0.0
        return dec_out.loss, {
            "logits": dec_out.logits,
            "topk_idx": topk_idx,
            "selector_scores": selector_scores,
            "selector_weights": selector_weights,
            "mse_loss": zero,
        }

    def _forward_stage2_pretrain_batch(
        self,
        batch: Dict,
        stage2_mips: bool,
        stage2_retrieval_top_n: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compatibility entry point that directs training to ARIA Phase II."""
        raise RuntimeError("ARIA Phase-II training requires _forward_stage2_batch()")

    def _forward_stage2_reasoning_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 2 reasoning training."""
        enc_input_ids = batch["enc_input_ids"].to(self.decoder.device)
        enc_attention_mask = batch["enc_attention_mask"].to(self.decoder.device)
        dec_input_ids = batch["dec_input_ids"].to(self.decoder.device)
        dec_attention_mask = batch["dec_attention_mask"].to(self.decoder.device)
        labels = batch["labels"].to(self.decoder.device)

        if sum(batch["docs_num"]) != 0:
            with torch.no_grad():
                selected, mse_loss = self.compress(enc_input_ids, enc_attention_mask)
                indices = batch["docs_num"]
                inputs_embeds = self._replace_reasoning_embeddings(selected, dec_input_ids, indices)
        else:
            inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
            # Keep the metric as a scalar tensor for trainer aggregation.
            mse_loss = torch.zeros((), device=self.decoder.device, dtype=torch.float32)

        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        dec_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=labels,
        )

        # Restore the complete registered adapter set after decoder execution.
        if len(self.adapter_keys) > 0:
            self.decoder.set_adapter(self.adapter_keys)
        return dec_out.loss, {"logits": dec_out.logits, "mse_loss": mse_loss}

    def _forward_stage_1(self,
                        enc_input_ids: torch.LongTensor = None,
                        enc_attention_mask: torch.LongTensor = None,
                        dec_input_ids: torch.LongTensor = None,
                        dec_attention_mask: torch.LongTensor = None,
                        labels: torch.LongTensor = None,
                        memory_token_counts: Optional[torch.Tensor] = None,
                        ) -> Dict[str, torch.Tensor]:
        """Stage 1 forward pass for document compression and QA."""
        assert enc_input_ids.size() == enc_attention_mask.size()

        # Flatten 3D inputs to 2D if needed
        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)

        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k

        # Compress documents
        compressed_embs, mse_loss = self.compress(enc_input_ids, enc_attention_mask)

        # Replace only each document's real floor(L/r) slots.  The compressor
        # output is padded to the configured ceiling for batching, but those
        # padded rows are not decoder context.
        if memory_token_counts is None:
            raise ValueError("Phase-I requires real per-document memory-token counts")
        counts = memory_token_counts.reshape(dec_input_ids.size(0), self.generation_top_k)
        inputs_embeds = self._replace_variable_memory_embeddings(
            compressed_embs.view(
                dec_input_ids.size(0),
                self.generation_top_k,
                compressed_embs.size(1),
                compressed_embs.size(2),
            ),
            counts,
            dec_input_ids,
        )

        # Detach if compressor-only training
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds = inputs_embeds.detach()

        # Phase I trains only the compressor encoder adapter. The generator is
        # the frozen backbone and must not accidentally keep encoder_adapter
        # active from the compression pass.
        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')
            decoder_outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=dec_attention_mask,
                labels=labels,
            )
        else:
            # Transformers' integrated PEFT API exposes plural enable/disable
            # methods (the singular context manager exists only on PeftModel).
            # Restore in ``finally`` so failed forwards cannot leave LoRA off.
            self.decoder.disable_adapters()
            try:
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=dec_attention_mask,
                    labels=labels,
                )
            finally:
                self.decoder.enable_adapters()

        # Restore exactly the adapters registered for the active training stage.
        if len(self.adapter_keys) > 0:
            self.decoder.set_adapter(self.adapter_keys)

        return {
            "loss": decoder_outputs.loss,
            "logits": decoder_outputs.logits,
            "mse_loss": mse_loss,
        }

    def _replace_reasoning_embeddings(self,
                                    compressed_embs: torch.Tensor,
                                    dec_input_ids: torch.LongTensor,
                                    docs_per_example: List[int]) -> torch.Tensor:
        """Replace memory slots with compressed embeddings for reasoning."""
        device = dec_input_ids.device
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)

        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if getattr(self, "sep", False) else 0)

        if not isinstance(docs_per_example, torch.Tensor):
            docs_per_example = torch.tensor(docs_per_example, device=device, dtype=torch.long)
        else:
            docs_per_example = docs_per_example.to(device=device, dtype=torch.long)

        offsets = torch.zeros(docs_per_example.size(0) + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(docs_per_example, dim=0)
        total_docs = int(offsets[-1].item())
        assert total_docs == compressed_embs.size(0)

        mem_id = self.decoder_tokenizer.mem_token_ids[0]
        B, L, H = inputs_embeds.size()

        for i in range(B):
            # Find first memory token position
            mem_pos = (dec_input_ids[i] == mem_id).nonzero(as_tuple=True)[0]
            if mem_pos.numel() == 0:
                continue
            first_mem_idx = int(mem_pos[0].item())

            n_docs_i = int(docs_per_example[i].item())
            base = int(offsets[i].item())

            needed_len = first_mem_idx + n_docs_i * slot_len
            assert needed_len <= L

            for local_j in range(n_docs_i):
                global_j = base + local_j
                start_idx = first_mem_idx + local_j * slot_len
                target_slice = inputs_embeds[i, start_idx:start_idx + num_embs, :]
                src = compressed_embs[global_j]
                assert target_slice.size() == src.size()
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = src

        return inputs_embeds

    def _generate(self, model_input: Dict[str, torch.Tensor], max_new_tokens: int = 64,
                 return_doc_embeddings: bool = False) -> List[str]:
        """Generate text from model inputs."""
        enc_input_ids = model_input['enc_input_ids']
        enc_attention_mask = model_input['enc_attention_mask']
        dec_input_ids = model_input['dec_input_ids']
        dec_attention_mask = model_input['dec_attention_mask']

        assert enc_input_ids.size() == enc_attention_mask.size()
        batch_size = dec_input_ids.size(0)
        top_k = self.generation_top_k
        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)

        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k

        device = self.decoder.device
        compressed_embs, _ = self.compress(
            enc_input_ids.to(device), enc_attention_mask.to(device)
        )
        memory_token_counts = model_input.get("memory_token_counts")
        if memory_token_counts is None:
            inputs_embeds = self._replace_emb(
                compressed_embs, dec_input_ids.to(device)
            )
        else:
            counts = memory_token_counts.to(device=device, dtype=torch.long).view(
                batch_size, top_k
            )
            inputs_embeds = self._replace_variable_memory_embeddings(
                compressed_embs.view(
                    batch_size,
                    top_k,
                    compressed_embs.size(1),
                    compressed_embs.size(2),
                ),
                counts,
                dec_input_ids.to(device),
            )

        if 'decoder_adapter' in self.adapter_keys:
            self.decoder.set_adapter('decoder_adapter')

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask.to(device),
            do_sample=False,
            top_p=None,
            temperature=None,
            num_beams=1,
            eos_token_id=self.decoder_tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.decoder_tokenizer.pad_token_id,
        )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if return_doc_embeddings:
            compressed_embs = compressed_embs.view(batch_size, top_k, compressed_embs.size(1), compressed_embs.size(2))
            return decoded, compressed_embs
        return decoded
