import hashlib
import json
import zipfile
from dataclasses import asdict

import pytest
from datasets import Dataset

import openrlhf.cli.aria_data as aria_data_module
from openrlhf.cli.aria_data import (
    EVALUATION_COUNTS,
    MUSIQUE_AUGMENTATION_COUNTS,
    MUSIQUE_DERIVED_MANIFEST_PROTOCOL,
    PAPER_PHASE2_EPOCH_SEEDS,
    PHASE1_TOTAL,
    PHASE2_SAMPLES_PER_EPOCH,
    Phase2FieldMap,
    _validate_musique_derived_manifest,
    _validate_phase2_epoch_seeds,
    build_parser,
    _normalize_phase2_row,
    canonicalize_page_url,
)
from openrlhf.utils.aria_provenance import (
    SOURCE_SNAPSHOT_SCHEME,
    build_source_snapshot_manifest,
    corpus_id,
    corpus_page_url,
    corpus_text,
    text_sha256,
    write_source_snapshot,
)
from openrlhf.utils.musique_augmentation import (
    MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
    build_musique_partial_rows,
    validate_musique_partial_metadata,
)


def test_paper_counts_are_fixed():
    assert PHASE1_TOTAL == 7_808_465
    assert PHASE2_SAMPLES_PER_EPOCH == 38_400
    assert EVALUATION_COUNTS == {
        "nq": 6_489,
        "hotpotqa": 7_384,
        "musique": 2_417,
        "2wikimultihopqa": 12_576,
    }
    assert MUSIQUE_AUGMENTATION_COUNTS == {
        "original": 19_938,
        "subquestion": 52_107,
        "partial_chain": 70_845,
        "entity_variant": 25_855,
    }
    assert PAPER_PHASE2_EPOCH_SEEDS == (42, 123, 456, 789, 2024)


def test_phase2_epoch_seed_schedule_is_the_fixed_paper_protocol():
    assert _validate_phase2_epoch_seeds([42, 123, 456, 789, 2024]) == (
        42,
        123,
        456,
        789,
        2024,
    )
    with pytest.raises(ValueError, match="requires epoch seed schedule"):
        _validate_phase2_epoch_seeds([1, 2, 3, 4, 5])
    args = build_parser().parse_args(["--stage", "phase2"])
    assert tuple(args.epoch_seeds) == PAPER_PHASE2_EPOCH_SEEDS


def _musique_parent(parent_id, hop_count):
    hops = [
        {"question": f"subquestion {parent_id}-{index}", "answer": f"a{index}"}
        for index in range(hop_count - 1)
    ]
    hops.append(
        {"question": f"final subquestion {parent_id}", "answer": "FINAL_SECRET"}
    )
    return {
        "source_row_id": parent_id,
        "question": f"final question {parent_id}",
        "answer": "FINAL_SECRET",
        "gold_answers": ["FINAL_SECRET"],
        "question_decomposition": hops,
    }


def test_generic_musique_prefix_utility_is_unique_balanced_and_order_independent():
    parents = [
        _musique_parent("two", 2),
        _musique_parent("three", 3),
        _musique_parent("four", 4),
    ]
    kwargs = dict(
        target_count=14,
        expected_original_count=None,
        expected_subquestion_count=None,
    )
    rows, manifest = build_musique_partial_rows(parents, **kwargs)
    reversed_rows, reversed_manifest = build_musique_partial_rows(
        list(reversed(parents)), **kwargs
    )

    ids = [row["partial_state_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 14
    assert ids == [row["partial_state_id"] for row in reversed_rows]
    assert manifest["selected_state_ids_sha256"] == reversed_manifest[
        "selected_state_ids_sha256"
    ]
    # Every k-1 prefix is mandatory: 1 + 2 + 3 = 6.
    assert manifest["mandatory_prefix_count"] == 6
    assert sum(row["partial_mandatory_prefix"] for row in rows) == 6
    assert all(row["needs_candidate_retrieval"] is True for row in rows)
    assert all(
        row["partial_known_hop_indices"]
        == list(range(len(row["partial_known_hop_indices"])))
        for row in rows
    )
    assert all(
        row["partial_frontier_hop_index"] == -1
        or row["partial_frontier_hop_index"]
        >= len(row["partial_known_hop_indices"])
        for row in rows
    )
    # The final-hop answer is a training target only, never part of the prompt.
    assert all("FINAL_SECRET" not in row["question"] for row in rows)


def test_musique_partial_metadata_rejects_final_hop_as_known_evidence():
    row = {
        "augmentation_parent_id": "p",
        "partial_state_protocol": MUSIQUE_PARTIAL_CHAIN_PROTOCOL,
        "partial_state_id": "musique-partial:" + "0" * 64,
        "partial_state_kind": "evidence_prefix",
        "partial_hop_count": 3,
        "partial_known_hop_indices": [2],
        "partial_frontier_hop_index": -1,
    }
    with pytest.raises(ValueError, match="final-hop answer"):
        validate_musique_partial_metadata(row)


def test_historical_musique_partial_row_does_not_require_generated_state_metadata():
    rows, _ = build_musique_partial_rows(
        [_musique_parent("parent", 2)],
        target_count=1,
        expected_original_count=None,
        expected_subquestion_count=None,
    )
    row = rows[0]
    row.update(
        docs=[f"document {index}" for index in range(5)],
        page_url=[
            f"https://en.wikipedia.org/wiki/Document_{index}"
            for index in range(5)
        ],
        doc_ids=[f"doc-{index}" for index in range(5)],
        pos_index=[0],
        gold_doc_ids=["doc-0"],
    )
    normalized = _normalize_phase2_row(
        row,
        0,
        benchmark="musique",
        field_map=Phase2FieldMap(),
        test_urls=set(),
    )
    assert normalized["augmentation_type"] == "partial_chain"
    assert normalized["augmentation_parent_id"] == "parent"
    assert normalized["construction_method"] == MUSIQUE_PARTIAL_CHAIN_PROTOCOL
    assert "partial_state_id" not in normalized


def test_historical_musique_source_requires_matching_external_content_manifest(
    tmp_path, monkeypatch
):
    decomposition = [
        {"question": "bridge?", "answer": "bridge-answer"},
        {"question": "final?", "answer": "final-answer"},
    ]
    rows = [
        {
            "source_row_id": "p",
            "augmentation_type": "original",
            "augmentation_parent_id": "",
            "question": "original?",
            "answer": "final-answer",
            "question_decomposition": decomposition,
        },
        {
            "source_row_id": "s",
            "augmentation_type": "subquestion",
            "augmentation_parent_id": "p",
            "question": "bridge?",
            "answer": "bridge-answer",
            "question_decomposition": decomposition,
        },
        {
            "source_row_id": "c",
            "augmentation_type": "partial_chain",
            "augmentation_parent_id": "p",
            "question": "partial?",
            "answer": "final-answer",
            "question_decomposition": decomposition,
        },
        {
            "source_row_id": "e",
            "augmentation_type": "entity_variant",
            "augmentation_parent_id": "p",
            "question": "variant?",
            "answer": "final-answer",
            "question_decomposition": decomposition,
        },
    ]
    counts = {
        "original": 1,
        "subquestion": 1,
        "partial_chain": 1,
        "entity_variant": 1,
    }
    monkeypatch.setattr(aria_data_module, "MUSIQUE_AUGMENTATION_COUNTS", counts)
    monkeypatch.setitem(aria_data_module.PHASE2_POOL_COUNTS, "musique", 4)

    def canonical(value):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    source_hasher = hashlib.sha256()
    family_hashers = {family: hashlib.sha256() for family in counts}
    parent_hasher = hashlib.sha256()
    for row in rows:
        encoded = canonical(row).encode("utf-8") + b"\n"
        source_hasher.update(encoded)
        family_hashers[row["augmentation_type"]].update(encoded)
        parent_id = row.get("augmentation_parent_id", "")
        parent_hasher.update(
            canonical(
                [
                    row["augmentation_type"],
                    row["source_row_id"],
                    parent_id,
                    row["answer"],
                    row["question_decomposition"],
                ]
            ).encode("utf-8")
        )
        parent_hasher.update(b"\n")
    dataset = Dataset.from_list(rows)
    manifest = {
        "protocol": MUSIQUE_DERIVED_MANIFEST_PROTOCOL,
        "total_count": 4,
        "family_counts": counts,
        "source_columns": list(dataset.column_names),
        "field_map": asdict(Phase2FieldMap()),
        "source_content_sha256": source_hasher.hexdigest(),
        "family_content_sha256": {
            family: hasher.hexdigest()
            for family, hasher in family_hashers.items()
        },
        "parent_link_sha256": parent_hasher.hexdigest(),
    }
    path = tmp_path / "musique-derived.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = _validate_musique_derived_manifest(
        dataset, Phase2FieldMap(), str(path)
    )
    assert summary["source_content_sha256"] == source_hasher.hexdigest()
    assert summary["family_counts"] == counts

    changed_rows = [dict(row) for row in rows]
    changed_rows[2]["answer"] = "changed"
    with pytest.raises(ValueError, match="changes its parent answer"):
        _validate_musique_derived_manifest(
            Dataset.from_list(changed_rows), Phase2FieldMap(), str(path)
        )


def test_page_url_canonicalization():
    assert canonicalize_page_url(
        "HTTPS://EN.WIKIPEDIA.ORG/wiki/Ada_Lovelace/", location="test"
    ) == "https://en.wikipedia.org/wiki/Ada_Lovelace"


def test_shared_corpus_alias_precedence_and_exact_text_hash():
    row = {
        "passage": "  first\n line\twith  spaces  ",
        "content": "must not win",
        "document_id": "doc-7",
        "wikipedia_url": "HTTPS://EN.WIKIPEDIA.ORG//wiki/Test/#fragment",
    }
    text = "first\n line\twith  spaces"
    assert corpus_text(row, location="test") == text
    assert corpus_id(row, location="test") == "doc-7"
    assert corpus_page_url(row, location="test") == (
        "https://en.wikipedia.org/wiki/Test"
    )
    assert text_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_source_snapshot_captures_dirty_bytes_without_requiring_git(tmp_path):
    source_file = tmp_path / "openrlhf" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("torch==2.3.1\n", encoding="utf-8")

    manifest = build_source_snapshot_manifest(tmp_path)
    assert manifest["scheme"] == SOURCE_SNAPSHOT_SCHEME
    assert manifest["source_file_count"] == 2
    snapshot = tmp_path / "snapshot.zip"
    write_source_snapshot(snapshot, manifest, tmp_path)
    with zipfile.ZipFile(snapshot) as archive:
        assert sorted(archive.namelist()) == [
            "openrlhf/module.py",
            "requirements.txt",
        ]

    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_source_snapshot_manifest(tmp_path)
    assert changed["source_tree_sha256"] != manifest["source_tree_sha256"]


def test_evaluation_preserves_gold_documents_beyond_existing_top5_candidates():
    row = {
        "question": "Which document is relevant?",
        "answer": "document six",
        "source_row_id": "eval-1",
        "docs": [f"document {index}" for index in range(7)],
        "page_url": [
            f"https://en.wikipedia.org/wiki/Document_{index}" for index in range(7)
        ],
        "doc_ids": [f"doc-{index}" for index in range(7)],
        "pos_index": [6],
        # The complete annotation is independent of the ranked candidates and
        # may therefore include a positive absent from ``docs`` altogether.
        "gold_doc_ids": ["doc-6", "corpus-positive-not-in-candidates"],
    }

    multi_answer = dict(row)
    multi_answer["answer"] = ["document six", "the sixth document"]
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _normalize_phase2_row(
            multi_answer,
            0,
            benchmark="nq",
            field_map=Phase2FieldMap(),
            test_urls=set(),
            require_musique_augmentation=False,
            evaluation_mode=True,
        )

    normalized = _normalize_phase2_row(
        row,
        0,
        benchmark="nq",
        field_map=Phase2FieldMap(),
        test_urls=set(),
        require_musique_augmentation=False,
        evaluation_mode=True,
    )

    assert normalized["doc_ids"] == [f"doc-{index}" for index in range(5)]
    assert normalized["pos_index"] == []
    assert normalized["gold_doc_ids"] == [
        "doc-6",
        "corpus-positive-not-in-candidates",
    ]
    assert normalized["answer"] == "document six"
    assert "gold_answers" not in normalized


def test_candidate_preparation_keeps_first_occurrence_per_page_id():
    page_urls = [
        "https://en.wikipedia.org/wiki/Shared",
        "https://en.wikipedia.org/wiki/Shared",
        *[
            f"https://en.wikipedia.org/wiki/Document_{index}"
            for index in range(2, 7)
        ],
    ]
    row = {
        "question": "Which shared page is relevant?",
        "answer": "shared",
        "source_row_id": "page-dedup-1",
        "docs": [f"document {index}" for index in range(7)],
        "page_url": page_urls,
        "doc_ids": [f"doc-{index}" for index in range(7)],
        # The lower-ranked passage marks the retained first page occurrence as
        # positive, while the complete corpus annotation keeps its original ID.
        "pos_index": [1],
        "gold_doc_ids": ["doc-1"],
    }

    normalized = _normalize_phase2_row(
        row,
        0,
        benchmark="nq",
        field_map=Phase2FieldMap(),
        test_urls=set(),
        require_musique_augmentation=False,
        evaluation_mode=True,
    )

    assert normalized["doc_ids"] == ["doc-0", "doc-2", "doc-3", "doc-4", "doc-5"]
    assert normalized["pos_index"] == [0]
    assert normalized["gold_doc_ids"] == ["doc-1"]
    assert normalized["duplicate_candidate_urls_removed"] == 1


def test_phase2_training_preserves_supports_outside_top5_and_allows_empty_support():
    row = {
        "question": "Which document is relevant?",
        "answer": "document six",
        "source_row_id": "train-1",
        "docs": [f"document {index}" for index in range(7)],
        "page_url": [
            f"https://en.wikipedia.org/wiki/Document_{index}" for index in range(7)
        ],
        "doc_ids": [f"doc-{index}" for index in range(7)],
        "pos_index": [6],
        # The second support is absent from the candidate archive entirely.
        "gold_doc_ids": ["doc-6", "corpus-positive-not-in-candidates"],
    }
    normalized = _normalize_phase2_row(
        row,
        0,
        benchmark="nq",
        field_map=Phase2FieldMap(),
        test_urls=set(),
        require_musique_augmentation=False,
    )
    assert normalized["doc_ids"] == [f"doc-{index}" for index in range(5)]
    assert normalized["pos_index"] == []
    assert normalized["gold_doc_ids"] == [
        "doc-6",
        "corpus-positive-not-in-candidates",
    ]

    row["pos_index"] = []
    row["gold_doc_ids"] = []
    normalized = _normalize_phase2_row(
        row,
        1,
        benchmark="nq",
        field_map=Phase2FieldMap(),
        test_urls=set(),
        require_musique_augmentation=False,
    )
    assert normalized["gold_doc_ids"] == []
    assert normalized["pos_index"] == []


@pytest.mark.parametrize("evaluation_mode", [False, True])
def test_phase2_requires_explicit_complete_corpus_gold_ids(evaluation_mode):
    row = {
        "question": "Which document is relevant?",
        "answer": "document six",
        "source_row_id": "eval-1",
        "docs": [f"document {index}" for index in range(7)],
        "page_url": [
            f"https://en.wikipedia.org/wiki/Document_{index}" for index in range(7)
        ],
        "doc_ids": [f"doc-{index}" for index in range(7)],
        "pos_index": [6],
    }
    with pytest.raises(ValueError, match="requires field 'gold_doc_ids'"):
        _normalize_phase2_row(
            row,
            0,
            benchmark="nq",
            field_map=Phase2FieldMap(),
            test_urls=set(),
            require_musique_augmentation=False,
            evaluation_mode=evaluation_mode,
        )

    row["gold_doc_ids"] = ["a-different-positive"]
    with pytest.raises(ValueError, match="omits candidate positives"):
        _normalize_phase2_row(
            row,
            0,
            benchmark="nq",
            field_map=Phase2FieldMap(),
            test_urls=set(),
            require_musique_augmentation=False,
            evaluation_mode=evaluation_mode,
        )
