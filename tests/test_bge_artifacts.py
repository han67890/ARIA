import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from openrlhf.cli.build_bge_artifacts import (
    BGE_DIMENSION,
    BGE_MODEL,
    ENCODER_DIRECTORY_SHA256_SCHEME,
    build_alignment_target_artifact,
    build_parser,
    build_corpus_artifact,
    canonical_directory_sha256,
    canonical_float32_index_sha256,
    resolve_encoder_provenance,
)
from openrlhf.utils.aria_provenance import TEXT_SHA256_SCHEME, text_sha256


class _FakeEncoder:
    def __init__(self):
        self.batches = []

    def encode(self, texts, **kwargs):
        self.batches.append((list(texts), dict(kwargs)))
        rows = []
        for text in texts:
            # Deliberately return non-normalized but deterministic vectors so
            # the builder's own validation/normalization is exercised.
            seed = sum(text.encode("utf-8")) % BGE_DIMENSION
            row = torch.zeros(BGE_DIMENSION, dtype=torch.float64)
            row[seed] = 3.0
            row[(seed + 1) % BGE_DIMENSION] = 4.0
            rows.append(row)
        return torch.stack(rows)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _hub_provenance(*, declared="paper-pin", explicit=True):
    encoder = SimpleNamespace(config=SimpleNamespace(_commit_hash=_COMMIT))
    return resolve_encoder_provenance(
        BGE_MODEL,
        encoder,
        declared_revision=declared,
        revision_was_explicit=explicit,
    )


def test_build_corpus_artifact_matches_loader_contract_and_refuses_overwrite(tmp_path):
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "id": "d-1",
                    "text": "  Alpha  passage  ",
                    "page_url": "HTTPS://Example.COM/wiki/Alpha/",
                },
                {
                    "document_id": "d-2",
                    "passage": "Beta\npassage",
                    "url": "https://example.com/wiki/Beta",
                },
                {
                    "passage_id": "d-3",
                    "content": "Gamma",
                    "wikipedia_url": "https://example.com/wiki/Gamma",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "corpus.pt"
    encoder = _FakeEncoder()

    returned = build_corpus_artifact(
        source,
        output,
        encoder,
        batch_size=2,
        encoder_provenance=_hub_provenance(),
    )
    artifact = _load(output)

    assert [len(batch[0]) for batch in encoder.batches] == [2, 1]
    assert artifact["document_ids"] == ["d-1", "d-2", "d-3"]
    assert artifact["text_sha256"] == [
        text_sha256("Alpha  passage"),
        text_sha256("Beta\npassage"),
        text_sha256("Gamma"),
    ]
    assert artifact["page_urls"][0] == "https://example.com/wiki/Alpha"
    assert artifact["bge_model"] == BGE_MODEL
    assert artifact["artifact_format"] == "aria-bge-artifact-v2"
    assert artifact["encoder_source_kind"] == "huggingface-hub"
    assert artifact["encoder_revision_declared"] == "paper-pin"
    assert artifact["encoder_revision_resolved"] == _COMMIT
    assert artifact["encoder_revision_was_explicit"] is True
    assert artifact["text_sha256_scheme"] == TEXT_SHA256_SCHEME
    assert artifact["doc_embeddings"].dtype == torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(artifact["doc_embeddings"], dim=1),
        torch.ones(3),
    )
    assert artifact["index_sha256"] == canonical_float32_index_sha256(
        artifact["doc_embeddings"]
    )
    assert returned["index_sha256"] == artifact["index_sha256"]
    assert artifact["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_corpus_artifact(source, output, _FakeEncoder(), batch_size=2)


def test_build_alignment_targets_records_ids_hashes_and_provenance(tmp_path):
    source = tmp_path / "alignment.json"
    source.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "question": "Who is Alpha?",
                        "passage": " Alpha answer ",
                        "passage_id": "p-1",
                        "page_url": "https://example.org/wiki/Alpha",
                    },
                    {
                        "query": "Who is Beta?",
                        "document": "Beta answer",
                        "doc_id": "p-2",
                        "url": "https://example.org/wiki/Beta/",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "targets.pth"
    local_model = tmp_path / "local-bge"
    local_model.mkdir()
    (local_model / "config.json").write_text('{"model_type":"bert"}', encoding="utf-8")
    provenance = resolve_encoder_provenance(
        str(local_model),
        _FakeEncoder(),
        declared_revision=None,
        revision_was_explicit=False,
    )

    build_alignment_target_artifact(
        source,
        output,
        _FakeEncoder(),
        batch_size=1,
        expected_rows=2,
        encoder_source=str(local_model),
        encoder_provenance=provenance,
    )
    artifact = _load(output)

    assert artifact["artifact_kind"] == "alignment-targets"
    assert artifact["target_embeddings"].shape == (2, BGE_DIMENSION)
    assert artifact["document_ids"] == artifact["passage_ids"] == ["p-1", "p-2"]
    assert artifact["text_sha256"] == [
        text_sha256("Alpha answer"),
        text_sha256("Beta answer"),
    ]
    query_hashes = [text_sha256("Who is Alpha?"), text_sha256("Who is Beta?")]
    assert artifact["query_text_sha256"] == query_hashes
    assert artifact["query_sha256"] == hashlib.sha256(
        "\n".join(query_hashes).encode("utf-8")
    ).hexdigest()
    assert artifact["encoder_source"] == str(local_model.resolve())
    assert artifact["encoder_source_kind"] == "local-directory"
    assert artifact["encoder_revision_declared"] is None
    assert artifact["encoder_revision_resolved"] is None
    assert artifact["encoder_source_sha256"] == canonical_directory_sha256(local_model)
    assert artifact["encoder_source_sha256_scheme"] == ENCODER_DIRECTORY_SHA256_SCHEME
    assert artifact["row_count"] == 2
    assert artifact["index_sha256"] == canonical_float32_index_sha256(
        artifact["target_embeddings"]
    )

    with pytest.raises(ValueError, match="exactly 3 rows"):
        build_alignment_target_artifact(
            source,
            tmp_path / "wrong-count.pt",
            _FakeEncoder(),
            expected_rows=3,
        )


def test_revision_provenance_rejects_unresolved_ambiguous_and_false_commit():
    with pytest.raises(RuntimeError, match="found none"):
        resolve_encoder_provenance(
            BGE_MODEL,
            SimpleNamespace(),
            declared_revision="main",
            revision_was_explicit=False,
        )

    other_commit = "f" * 40
    ambiguous = SimpleNamespace(
        config=SimpleNamespace(_commit_hash=_COMMIT),
        tokenizer=SimpleNamespace(init_kwargs={"_commit_hash": other_commit}),
    )
    with pytest.raises(RuntimeError, match=_COMMIT):
        resolve_encoder_provenance(
            BGE_MODEL,
            ambiguous,
            declared_revision="main",
            revision_was_explicit=False,
        )

    with pytest.raises(RuntimeError, match="Requested BGE commit"):
        resolve_encoder_provenance(
            BGE_MODEL,
            SimpleNamespace(config=SimpleNamespace(_commit_hash=_COMMIT)),
            declared_revision=other_commit,
            revision_was_explicit=True,
        )


def test_local_encoder_digest_changes_with_bytes_and_rejects_revision(tmp_path):
    model_dir = tmp_path / "encoder"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"first")
    first = canonical_directory_sha256(model_dir)
    weights.write_bytes(b"second")
    assert canonical_directory_sha256(model_dir) != first

    with pytest.raises(ValueError, match="cannot be used with a local"):
        resolve_encoder_provenance(
            str(model_dir),
            SimpleNamespace(),
            declared_revision="main",
            revision_was_explicit=True,
        )


def test_builder_exposes_revision_pin_and_rejects_metadata_namespace_collision(tmp_path):
    args = build_parser().parse_args(
        [
            "corpus",
            "--input",
            "corpus.jsonl",
            "--output",
            "corpus.pt",
            "--revision",
            _COMMIT,
        ]
    )
    assert args.revision == _COMMIT

    source = tmp_path / "corpus.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "d1",
                "text": "passage",
                "page_url": "https://example.org/wiki/Page",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    invalid = {**_hub_provenance(), "bge_model": "collision"}
    with pytest.raises(ValueError, match="unexpected keys"):
        build_corpus_artifact(
            source,
            tmp_path / "bad.pt",
            _FakeEncoder(),
            encoder_provenance=invalid,
        )
