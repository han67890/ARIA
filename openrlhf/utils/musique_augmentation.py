"""Generic deterministic MuSiQue prerequisite-prefix utility.

This optional utility enumerates annotated prerequisite prefixes and unresolved
subsequent-hop frontiers without exposing the final answer. It is not the
builder for the paper's historical 70,845-row derived artifact; paper data
preparation consumes and verifies the artifact-owner-supplied source directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


MUSIQUE_PARTIAL_CHAIN_PROTOCOL = "generic-prefix-frontier-v1"


def _required_text(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class MuSiQueHop:
    question: str
    answer: str


@dataclass(frozen=True)
class MuSiQueParent:
    parent_id: str
    question: str
    answer: Any
    hops: Tuple[MuSiQueHop, ...]
    gold_answers: Tuple[str, ...]


@dataclass(frozen=True)
class PartialChainState:
    parent_id: str
    hop_count: int
    kind: str
    known_hop_indices: Tuple[int, ...]
    frontier_hop_index: Optional[int]
    mandatory_prefix: bool
    state_id: str


def musique_partial_state_id(
    parent_id: str,
    hop_count: int,
    kind: str,
    known_hop_indices: Sequence[int],
    frontier_hop_index: Optional[int],
) -> str:
    """Return the content-addressed ID used by builders and validators."""
    payload = {
        "protocol": MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
        "parent_id": str(parent_id),
        "hop_count": int(hop_count),
        "kind": str(kind),
        "known_hop_indices": [int(index) for index in known_hop_indices],
        "frontier_hop_index": (
            None if frontier_hop_index is None else int(frontier_hop_index)
        ),
    }
    return "musique-partial:" + _canonical_sha256(payload)


def _normalize_parent(
    row: Mapping[str, Any],
    *,
    row_index: int,
    parent_id_key: str,
    question_key: str,
    answer_key: str,
    decomposition_key: str,
) -> MuSiQueParent:
    location = f"MuSiQue original row {row_index}"
    if parent_id_key not in row:
        raise ValueError(f"{location} requires {parent_id_key!r}")
    raw_parent_id = row[parent_id_key]
    if raw_parent_id is None or isinstance(raw_parent_id, bool):
        raise ValueError(f"{location}.id must be a stable string or integer")
    parent_id = _required_text(str(raw_parent_id), location=f"{location}.id")
    question = _required_text(row.get(question_key), location=f"{location}.question")
    if answer_key not in row:
        raise ValueError(f"{location} requires {answer_key!r}")
    answer = row[answer_key]
    if isinstance(answer, str):
        _required_text(answer, location=f"{location}.answer")
    elif not isinstance(answer, Sequence) or isinstance(answer, (bytes, bytearray)):
        raise ValueError(f"{location}.answer must be a string or non-empty answer list")
    elif not answer:
        raise ValueError(f"{location}.answer must not be empty")

    decomposition = row.get(decomposition_key)
    if not isinstance(decomposition, Sequence) or isinstance(
        decomposition, (str, bytes, bytearray)
    ):
        raise ValueError(f"{location}.{decomposition_key} must be an ordered hop list")
    if len(decomposition) not in {2, 3, 4}:
        raise ValueError(f"{location} must contain exactly 2, 3, or 4 hops")
    hops: List[MuSiQueHop] = []
    for hop_index, raw_hop in enumerate(decomposition):
        if not isinstance(raw_hop, Mapping):
            raise ValueError(f"{location} hop {hop_index} must be an object")
        hops.append(
            MuSiQueHop(
                question=_required_text(
                    raw_hop.get("question"),
                    location=f"{location}.hop[{hop_index}].question",
                ),
                answer=_required_text(
                    raw_hop.get("answer"),
                    location=f"{location}.hop[{hop_index}].answer",
                ),
            )
        )

    raw_golds = row.get("gold_answers", answer)
    if isinstance(raw_golds, str):
        gold_answers = (_required_text(raw_golds, location=f"{location}.gold_answers"),)
    elif isinstance(raw_golds, Sequence) and not isinstance(
        raw_golds, (bytes, bytearray)
    ):
        gold_answers = tuple(
            _required_text(value, location=f"{location}.gold_answers")
            for value in raw_golds
        )
    else:
        raise ValueError(f"{location}.gold_answers must be a string or list")
    if not gold_answers:
        raise ValueError(f"{location}.gold_answers must not be empty")
    return MuSiQueParent(
        parent_id=parent_id,
        question=question,
        answer=answer,
        hops=tuple(hops),
        gold_answers=tuple(dict.fromkeys(gold_answers)),
    )


def _enumerate_parent_states(parent: MuSiQueParent) -> List[PartialChainState]:
    """Enumerate strict prerequisite prefixes and their later frontiers."""
    states: List[PartialChainState] = []
    for prefix_length in range(1, len(parent.hops)):
        known = tuple(range(prefix_length))
        state_id = musique_partial_state_id(
            parent.parent_id,
            len(parent.hops),
            "evidence_prefix",
            known,
            None,
        )
        states.append(
            PartialChainState(
                parent_id=parent.parent_id,
                hop_count=len(parent.hops),
                kind="evidence_prefix",
                known_hop_indices=known,
                frontier_hop_index=None,
                mandatory_prefix=True,
                state_id=state_id,
            )
        )
        # A frontier exposes only a hop question and the literal <MISSING>
        # marker. Its answer, especially the final-hop answer, never enters the
        # prompt or the known evidence set.
        for frontier in range(prefix_length, len(parent.hops)):
            frontier_id = musique_partial_state_id(
                parent.parent_id,
                len(parent.hops),
                "missing_frontier",
                known,
                frontier,
            )
            states.append(
                PartialChainState(
                    parent_id=parent.parent_id,
                    hop_count=len(parent.hops),
                    kind="missing_frontier",
                    known_hop_indices=known,
                    frontier_hop_index=frontier,
                    mandatory_prefix=False,
                    state_id=frontier_id,
                )
            )
    return states


def select_musique_partial_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
    expected_original_count: Optional[int] = None,
    expected_subquestion_count: Optional[int] = None,
    parent_id_key: str = "source_row_id",
    question_key: str = "question",
    answer_key: str = "answer",
    decomposition_key: str = "question_decomposition",
) -> Tuple[List[PartialChainState], Dict[str, Any], Dict[str, MuSiQueParent]]:
    """Select an exact, order-independent, parent-balanced set of states."""
    if target_count <= 0:
        raise ValueError("MuSiQue partial-chain target_count must be positive")
    parents = [
        _normalize_parent(
            row,
            row_index=index,
            parent_id_key=parent_id_key,
            question_key=question_key,
            answer_key=answer_key,
            decomposition_key=decomposition_key,
        )
        for index, row in enumerate(rows)
    ]
    by_id = {parent.parent_id: parent for parent in parents}
    if len(by_id) != len(parents):
        raise ValueError("MuSiQue original parent IDs must be unique")
    if expected_original_count is not None and len(parents) != expected_original_count:
        raise ValueError(
            "MuSiQue original count mismatch: "
            f"expected {expected_original_count:,}, got {len(parents):,}"
        )
    subquestion_count = sum(len(parent.hops) for parent in parents)
    if (
        expected_subquestion_count is not None
        and subquestion_count != expected_subquestion_count
    ):
        raise ValueError(
            "MuSiQue subquestion count mismatch: "
            f"expected {expected_subquestion_count:,}, got {subquestion_count:,}"
        )

    parent_order = sorted(
        by_id,
        key=lambda parent_id: _canonical_sha256(
            {"protocol": MUSIQUE_PARTIAL_CHAIN_PROTOCOL, "parent_id": parent_id}
        ),
    )
    mandatory: List[PartialChainState] = []
    optional_by_parent: Dict[str, List[PartialChainState]] = {}
    capacity = 0
    for parent_id in parent_order:
        states = _enumerate_parent_states(by_id[parent_id])
        capacity += len(states)
        mandatory.extend(state for state in states if state.mandatory_prefix)
        optional_by_parent[parent_id] = sorted(
            (state for state in states if not state.mandatory_prefix),
            key=lambda state: state.state_id,
        )
    if len(mandatory) > target_count:
        raise ValueError(
            "Target is smaller than the mandatory k-1 prefix partition: "
            f"{target_count} < {len(mandatory)}"
        )
    if capacity < target_count:
        raise ValueError(
            f"Legal partial-state capacity {capacity:,} is below target {target_count:,}"
        )

    selected = list(mandatory)
    needed = target_count - len(selected)
    local_ordinal = 0
    while needed:
        progressed = False
        for parent_id in parent_order:
            candidates = optional_by_parent[parent_id]
            if local_ordinal >= len(candidates):
                continue
            selected.append(candidates[local_ordinal])
            needed -= 1
            progressed = True
            if needed == 0:
                break
        if not progressed:
            raise RuntimeError("MuSiQue state selection exhausted before its target")
        local_ordinal += 1

    selected.sort(key=lambda state: state.state_id)
    state_ids = [state.state_id for state in selected]
    if len(state_ids) != target_count or len(set(state_ids)) != target_count:
        raise RuntimeError("MuSiQue selected state IDs are not exact and unique")
    manifest = {
        "protocol": MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
        "target_count": target_count,
        "original_count": len(parents),
        "subquestion_count": subquestion_count,
        "mandatory_prefix_count": len(mandatory),
        "optional_selected_count": target_count - len(mandatory),
        "legal_state_capacity": capacity,
        "selection": "all_prefixes_plus_parent_round_robin_frontier_state_sha256",
        "selected_state_ids_sha256": hashlib.sha256(
            ("\n".join(state_ids) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return selected, manifest, by_id


def render_musique_partial_prompt(
    parent: MuSiQueParent, state: PartialChainState
) -> str:
    if state.parent_id != parent.parent_id or state.hop_count != len(parent.hops):
        raise ValueError("Partial state does not belong to its parent decomposition")
    if any(
        index < 0 or index >= len(parent.hops) - 1
        for index in state.known_hop_indices
    ):
        raise ValueError("Known evidence may contain prerequisite hops only")
    if state.known_hop_indices != tuple(range(len(state.known_hop_indices))):
        raise ValueError("Known evidence must be a strict prerequisite prefix")
    if state.frontier_hop_index is not None and (
        state.frontier_hop_index < 0
        or state.frontier_hop_index >= len(parent.hops)
        or state.frontier_hop_index < len(state.known_hop_indices)
    ):
        raise ValueError("Partial frontier must be one unknown valid hop")

    lines = [
        f"Original multi-hop question: {parent.question}",
        "Known chain evidence:",
    ]
    for hop_index in state.known_hop_indices:
        hop = parent.hops[hop_index]
        lines.extend(
            [
                f"Hop {hop_index + 1} question: {hop.question}",
                f"Hop {hop_index + 1} answer: {hop.answer}",
            ]
        )
    if state.frontier_hop_index is not None:
        frontier = parent.hops[state.frontier_hop_index]
        lines.extend(
            [
                "Unresolved frontier:",
                f"Hop {state.frontier_hop_index + 1} question: {frontier.question}",
                f"Hop {state.frontier_hop_index + 1} answer: <MISSING>",
            ]
        )
    lines.append(
        "Task: complete the missing reasoning chain and answer the original question."
    )
    return "\n".join(lines)


def build_musique_partial_rows(
    rows: Sequence[Mapping[str, Any]],
    **selection_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build partial-query rows that must receive fresh retrieval candidates."""
    states, manifest, parents = select_musique_partial_states(
        rows, **selection_kwargs
    )
    output: List[Dict[str, Any]] = []
    for state in states:
        parent = parents[state.parent_id]
        prompt = render_musique_partial_prompt(parent, state)
        output.append(
            {
                "source_row_id": state.state_id,
                "question": prompt,
                "answer": parent.answer,
                "gold_answers": list(parent.gold_answers),
                "data_type": "qa",
                "benchmark": "musique",
                "augmentation_type": "partial_chain",
                "augmentation_parent_id": parent.parent_id,
                "construction_method": MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
                "answer_preserved": True,
                "partial_state_id": state.state_id,
                "partial_state_protocol": MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
                "partial_state_kind": state.kind,
                "partial_hop_count": state.hop_count,
                "partial_known_hop_indices": list(state.known_hop_indices),
                "partial_frontier_hop_index": (
                    -1
                    if state.frontier_hop_index is None
                    else state.frontier_hop_index
                ),
                "partial_mandatory_prefix": state.mandatory_prefix,
                "partial_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "partial_selection_sha256": manifest[
                    "selected_state_ids_sha256"
                ],
                # Candidate documents from the parent are invalid after the
                # query changes. A separate BGE top-5 retrieval/materialization
                # step must replace this marker before Phase-II normalization.
                "needs_candidate_retrieval": True,
            }
        )
    return output, manifest


def validate_musique_partial_metadata(row: Mapping[str, Any]) -> None:
    """Validate state identity after fresh candidates have been attached."""
    protocol = row.get("partial_state_protocol")
    if protocol != MUSIQUE_PARTIAL_CHAIN_PROTOCOL:
        raise ValueError(
            f"partial_state_protocol must be {MUSIQUE_PARTIAL_CHAIN_PROTOCOL!r}"
        )
    state_id = row.get("partial_state_id")
    if not isinstance(state_id, str) or re.fullmatch(
        r"musique-partial:[0-9a-f]{64}", state_id
    ) is None:
        raise ValueError("partial_state_id must be a canonical MuSiQue SHA-256 ID")
    known = row.get("partial_known_hop_indices")
    if (
        not isinstance(known, Sequence)
        or isinstance(known, (str, bytes, bytearray))
        or not known
        or any(isinstance(index, bool) or not isinstance(index, int) for index in known)
        or list(known) != sorted(set(known))
    ):
        raise ValueError("partial_known_hop_indices must be sorted unique integers")
    hop_count = row.get("partial_hop_count")
    if isinstance(hop_count, bool) or hop_count not in {2, 3, 4}:
        raise ValueError("partial_hop_count must be 2, 3, or 4")
    if any(index < 0 or index >= hop_count - 1 for index in known):
        raise ValueError("partial known evidence cannot include the final-hop answer")
    if list(known) != list(range(len(known))):
        raise ValueError("partial known evidence must be a strict prerequisite prefix")
    kind = row.get("partial_state_kind")
    frontier_raw = row.get("partial_frontier_hop_index")
    frontier = None if frontier_raw == -1 else frontier_raw
    if kind not in {"evidence_prefix", "missing_frontier"}:
        raise ValueError("partial_state_kind is invalid")
    if kind == "evidence_prefix" and frontier is not None:
        raise ValueError("evidence_prefix states cannot have a frontier")
    if kind == "missing_frontier" and (
        isinstance(frontier, bool)
        or not isinstance(frontier, int)
        or frontier < 0
        or frontier >= hop_count
        or frontier < len(known)
    ):
        raise ValueError("missing_frontier states require one unknown hop")
    expected_id = musique_partial_state_id(
        str(row.get("augmentation_parent_id", "")),
        hop_count,
        kind,
        known,
        frontier,
    )
    if state_id != expected_id:
        raise ValueError("partial_state_id does not match its canonical state payload")
