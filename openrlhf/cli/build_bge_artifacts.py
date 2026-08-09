#!/usr/bin/env python3
"""Build row-aligned BGE artifacts used by the ARIA pipeline.

The command intentionally keeps embedding construction separate from training
and evaluation.  It accepts only the paper's fixed BGE model name or a local
SentenceTransformer directory, normalizes every vector, and emits the exact
metadata contract enforced by the existing artifact loaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from openrlhf.utils.aria_provenance import (
    CORPUS_TEXT_FIELDS,
    CORPUS_URL_FIELDS,
    TEXT_SHA256_SCHEME,
    canonical_page_url,
    corpus_id,
    corpus_page_url,
    corpus_text,
    text_sha256,
)


BGE_MODEL = "BAAI/bge-large-en-v1.5"
BGE_DIMENSION = 1024
INDEX_HASH_CHUNK_ROWS = 16_384
ENCODER_DIRECTORY_SHA256_SCHEME = "relative-path-size-content-v1"
_HF_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HF_SNAPSHOT_RE = re.compile(r"(?:^|[/\\])snapshots[/\\]([0-9a-fA-F]{40})(?:[/\\]|$)")
_QUERY_TEXT_FIELDS = ("query", "question", "text")
_ALIGNMENT_ID_FIELDS = ("passage_id", "doc_id", "document_id", "id")
_ENCODER_PROVENANCE_KEYS = {
    "encoder_source",
    "encoder_source_kind",
    "encoder_revision_declared",
    "encoder_revision_resolved",
    "encoder_revision_was_explicit",
    "encoder_source_sha256",
    "encoder_source_sha256_scheme",
}


def canonical_float32_index_sha256(embeddings: torch.Tensor) -> str:
    """Return the digest used by both ARIA embedding artifact loaders."""
    canonical = embeddings.detach().cpu().to(dtype=torch.float32).contiguous()
    if canonical.ndim != 2:
        raise ValueError(f"Embedding index must be rank 2, got {tuple(canonical.shape)}")
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {"shape": list(canonical.shape), "dtype": "float32"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for start in range(0, canonical.shape[0], INDEX_HASH_CHUNK_ROWS):
        hasher.update(
            canonical[start : start + INDEX_HASH_CHUNK_ROWS]
            .contiguous()
            .numpy()
            .tobytes()
        )
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def canonical_directory_sha256(path: str | os.PathLike[str]) -> str:
    """Fingerprint every regular file in a local model directory.

    Local directories do not have a Hugging Face revision by definition.  A
    canonical tree digest therefore provides the immutable identifier needed
    to reproduce which bytes were actually loaded without inventing a commit.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Encoder source is not a local directory: {path}")
    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"Encoder directory is empty: {root}")
    hasher = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(candidate.stat().st_size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def _resolved_commit_candidates(encoder: Any) -> set[str]:
    """Collect exact Hub commit hashes exposed by SentenceTransformer modules."""
    objects: list[Any] = [encoder]
    modules = getattr(encoder, "_modules", None)
    if isinstance(modules, Mapping):
        objects.extend(modules.values())
    try:
        objects.extend(list(encoder.modules()))
    except (AttributeError, TypeError):
        pass

    candidates: set[str] = set()
    seen: set[int] = set()
    while objects:
        current = objects.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        values: list[Any] = []
        if isinstance(current, Mapping):
            values.extend(current.get(key) for key in ("_commit_hash", "commit_hash"))
        else:
            values.extend(
                getattr(current, key, None)
                for key in ("_commit_hash", "commit_hash", "name_or_path", "_name_or_path")
            )
            for attribute in ("config", "auto_model", "tokenizer", "model"):
                child = getattr(current, attribute, None)
                if child is not None:
                    objects.append(child)
            init_kwargs = getattr(current, "init_kwargs", None)
            if isinstance(init_kwargs, Mapping):
                objects.append(init_kwargs)

        for value in values:
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if _HF_COMMIT_RE.fullmatch(stripped):
                candidates.add(stripped.lower())
                continue
            snapshot_match = _HF_SNAPSHOT_RE.search(stripped)
            if snapshot_match is not None:
                candidates.add(snapshot_match.group(1).lower())
    return candidates


def resolve_encoder_provenance(
    source: str,
    encoder: Any,
    *,
    declared_revision: Optional[str],
    revision_was_explicit: bool,
) -> dict[str, Any]:
    """Return verifiable provenance for one loaded BGE encoder.

    Hub sources must expose one exact resolved commit.  Local directories use a
    content digest and deliberately leave the revision fields null.
    """
    local_path = Path(source).expanduser()
    if local_path.is_dir():
        if declared_revision is not None or revision_was_explicit:
            raise ValueError("--revision cannot be used with a local --model directory")
        return {
            "encoder_source": str(local_path.resolve()),
            "encoder_source_kind": "local-directory",
            "encoder_revision_declared": None,
            "encoder_revision_resolved": None,
            "encoder_revision_was_explicit": False,
            "encoder_source_sha256": canonical_directory_sha256(local_path),
            "encoder_source_sha256_scheme": ENCODER_DIRECTORY_SHA256_SCHEME,
        }

    if source != BGE_MODEL:
        raise ValueError(f"Unsupported remote encoder source: {source}")
    declared = (declared_revision or "main").strip()
    if not declared:
        raise ValueError("BGE revision must be a non-empty branch, tag, or commit")
    candidates = _resolved_commit_candidates(encoder)
    if len(candidates) != 1:
        detail = "none" if not candidates else ", ".join(sorted(candidates))
        raise RuntimeError(
            "Could not prove one resolved Hugging Face commit for the BGE encoder; "
            f"found {detail}. Refusing to publish an unverifiable artifact."
        )
    resolved = next(iter(candidates))
    if _HF_COMMIT_RE.fullmatch(declared) and declared.lower() != resolved:
        raise RuntimeError(
            f"Requested BGE commit {declared.lower()} but loaded {resolved}"
        )
    return {
        "encoder_source": source,
        "encoder_source_kind": "huggingface-hub",
        "encoder_revision_declared": declared,
        "encoder_revision_resolved": resolved,
        "encoder_revision_was_explicit": bool(revision_was_explicit),
        "encoder_source_sha256": None,
        "encoder_source_sha256_scheme": None,
    }


def _encoder_metadata(
    encoder_source: str,
    provenance: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the isolated encoder namespace before merging artifact metadata."""
    if provenance is None:
        return {
            "encoder_source": encoder_source,
            "encoder_source_kind": "unverified",
            "encoder_revision_declared": None,
            "encoder_revision_resolved": None,
            "encoder_revision_was_explicit": False,
            "encoder_source_sha256": None,
            "encoder_source_sha256_scheme": None,
        }
    metadata = dict(provenance)
    if set(metadata) != _ENCODER_PROVENANCE_KEYS:
        missing = sorted(_ENCODER_PROVENANCE_KEYS - set(metadata))
        extra = sorted(set(metadata) - _ENCODER_PROVENANCE_KEYS)
        raise ValueError(
            f"Encoder provenance has missing keys {missing} and unexpected keys {extra}"
        )
    expected_source = (
        str(Path(encoder_source).expanduser().resolve())
        if Path(encoder_source).expanduser().is_dir()
        else encoder_source
    )
    if metadata["encoder_source"] != expected_source:
        raise ValueError(
            "Encoder provenance source does not match the loaded encoder: "
            f"{metadata['encoder_source']!r} != {expected_source!r}"
        )
    if not isinstance(metadata["encoder_revision_was_explicit"], bool):
        raise ValueError("Encoder revision explicitness must be boolean")
    kind = metadata["encoder_source_kind"]
    if kind == "huggingface-hub":
        declared = metadata["encoder_revision_declared"]
        resolved = metadata["encoder_revision_resolved"]
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError("Hub encoder provenance requires its declared revision")
        if not isinstance(resolved, str) or not _HF_COMMIT_RE.fullmatch(resolved):
            raise ValueError("Hub encoder provenance requires an exact resolved commit")
        if metadata["encoder_source_sha256"] is not None:
            raise ValueError("Hub encoder provenance must not claim a local directory digest")
    elif kind == "local-directory":
        digest = metadata["encoder_source_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Local encoder provenance requires a SHA-256 tree digest")
        if metadata["encoder_source_sha256_scheme"] != ENCODER_DIRECTORY_SHA256_SCHEME:
            raise ValueError("Local encoder provenance has an unsupported digest scheme")
        if (
            metadata["encoder_revision_declared"] is not None
            or metadata["encoder_revision_resolved"] is not None
            or metadata["encoder_revision_was_explicit"] is not False
        ):
            raise ValueError("Local encoder provenance must not invent a Hub revision")
        current_digest = canonical_directory_sha256(metadata["encoder_source"])
        if current_digest != digest:
            raise RuntimeError(
                "Local encoder directory changed after it was loaded; refusing to "
                "publish stale provenance"
            )
    else:
        raise ValueError(f"Unsupported verified encoder source kind: {kind!r}")
    return metadata


def _first_nonempty_string(
    record: Mapping[str, Any], fields: Sequence[str], *, location: str, name: str
) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"{location} requires a non-empty {name} in one of {tuple(fields)}")


def _read_json_records(
    path: Path,
    *,
    container_keys: Sequence[str],
    record_fields: Sequence[str],
) -> list[tuple[Optional[str], Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input artifact does not exist: {path}")
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"Input must be a local .json or .jsonl artifact: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[tuple[Optional[str], Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append((None, json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not records:
            raise ValueError(f"Input artifact is empty: {path}")
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in container_keys:
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, list):
        records = [(None, value) for value in payload]
    elif isinstance(payload, dict):
        if any(field in payload for field in record_fields):
            records = [(None, payload)]
        else:
            records = [(str(key), value) for key, value in payload.items()]
    else:
        raise ValueError(f"JSON root in {path} must contain an array or object")
    if not records:
        raise ValueError(f"Input artifact is empty: {path}")
    return records


def load_corpus_rows(
    path: str | os.PathLike[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load the same strict corpus schema consumed by ARIA train/eval."""
    source = Path(path)
    records = _read_json_records(
        source,
        container_keys=("documents", "corpus", "passages", "data"),
        record_fields=CORPUS_TEXT_FIELDS,
    )
    texts: list[str] = []
    document_ids: list[str] = []
    page_urls: list[str] = []
    for index, (_, raw_record) in enumerate(records):
        location = f"corpus row {index}"
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"{location} must be an object with text, ID, and page_url")
        text = corpus_text(raw_record, location=location)
        texts.append(text)
        document_ids.append(corpus_id(raw_record, location=location))
        page_urls.append(corpus_page_url(raw_record, location=location))
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Corpus document IDs must be unique")
    return texts, document_ids, page_urls, [text_sha256(text) for text in texts]


def load_alignment_rows(
    path: str | os.PathLike[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Load paper W_BGE pairs without importing the training stack."""
    source = Path(path)
    records = _read_json_records(
        source,
        container_keys=("pairs", "queries", "questions", "data"),
        record_fields=_QUERY_TEXT_FIELDS,
    )
    queries: list[str] = []
    passages: list[str] = []
    passage_ids: list[str] = []
    page_urls: list[str] = []
    for index, (mapping_id, raw_record) in enumerate(records):
        location = f"alignment row {index}"
        if not isinstance(raw_record, Mapping):
            raise ValueError(
                f"{location} must contain question, passage, passage ID, and page_url"
            )
        query = _first_nonempty_string(
            raw_record,
            _QUERY_TEXT_FIELDS,
            location=location,
            name="question",
        )
        # This deliberately excludes the ambiguous `text` field, matching
        # train_sft.load_alignment_pairs where `text` is reserved for a query.
        passage = _first_nonempty_string(
            raw_record,
            tuple(field for field in CORPUS_TEXT_FIELDS if field != "text"),
            location=location,
            name="passage",
        )
        passage_id = next(
            (
                str(raw_record[field])
                for field in _ALIGNMENT_ID_FIELDS
                if raw_record.get(field) is not None and str(raw_record[field]).strip()
            ),
            mapping_id,
        )
        if not passage_id:
            raise ValueError(f"{location} requires a stable passage ID")
        raw_url = next(
            (raw_record[field] for field in CORPUS_URL_FIELDS if raw_record.get(field)),
            None,
        )
        queries.append(query)
        passages.append(passage)
        passage_ids.append(str(passage_id))
        page_urls.append(canonical_page_url(raw_url, location=f"{location}.page_url"))
    return queries, passages, passage_ids, page_urls, [text_sha256(x) for x in passages]


def encode_normalized(
    texts: Sequence[str],
    encoder: Any,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Encode in bounded batches and independently enforce unit L2 norm."""
    if not texts:
        raise ValueError("At least one text is required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Allocate the final CPU matrix once. This avoids a second corpus-sized
    # allocation in torch.cat when building the multi-million-row KILT index.
    embeddings = torch.empty((len(texts), BGE_DIMENSION), dtype=torch.float32)
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        values = encoder.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        elif not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        values = values.detach().cpu().to(dtype=torch.float32)
        if values.ndim != 2 or values.shape != (len(batch_texts), BGE_DIMENSION):
            raise ValueError(
                f"{BGE_MODEL} must return ({len(batch_texts)}, {BGE_DIMENSION}), "
                f"got {tuple(values.shape)}"
            )
        if not torch.isfinite(values).all().item():
            raise ValueError("BGE encoder returned NaN or infinite values")
        norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        if torch.any(norms <= 0).item():
            raise ValueError("BGE encoder returned a zero-length embedding")
        embeddings[start : start + len(batch_texts)].copy_(values / norms)
    return embeddings.contiguous()


def atomic_torch_save(payload: Mapping[str, Any], output_path: str | os.PathLike[str]) -> None:
    """Publish a complete artifact atomically, never replacing an existing path."""
    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("BGE artifact output must use a .pt or .pth suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Linking a finished temporary file is atomic and fails if another
        # process won the destination name between the preflight and publish.
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _source_metadata(
    source: Path,
    *,
    kind: str,
    row_count: int,
    verified_encoder: bool,
) -> dict[str, Any]:
    return {
        "artifact_format": (
            "aria-bge-artifact-v2" if verified_encoder else "aria-bge-artifact-v1"
        ),
        "artifact_kind": kind,
        "source_path": str(source.resolve()),
        "source_sha256": _file_sha256(source),
        "row_count": row_count,
        "bge_model": BGE_MODEL,
        "text_sha256_scheme": TEXT_SHA256_SCHEME,
        "embedding_normalization": "l2-unit-v1",
        "embedding_dtype": "float32",
        "embedding_dimension": BGE_DIMENSION,
    }


def build_corpus_artifact(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    encoder: Any,
    *,
    batch_size: int = 64,
    encoder_source: str = BGE_MODEL,
    encoder_provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    source = Path(input_path)
    texts, document_ids, page_urls, text_hashes = load_corpus_rows(source)
    embeddings = encode_normalized(texts, encoder, batch_size=batch_size)
    artifact = {
        **_source_metadata(
            source,
            kind="corpus-index",
            row_count=len(texts),
            verified_encoder=encoder_provenance is not None,
        ),
        **_encoder_metadata(encoder_source, encoder_provenance),
        "doc_embeddings": embeddings,
        "document_ids": document_ids,
        "text_sha256": text_hashes,
        "page_urls": page_urls,
        "document_id_sha256": _sequence_sha256(document_ids),
        "document_text_sha256": _sequence_sha256(text_hashes),
        "page_url_sha256": _sequence_sha256(page_urls),
        "index_sha256": canonical_float32_index_sha256(embeddings),
    }
    atomic_torch_save(artifact, output_path)
    return artifact


def build_alignment_target_artifact(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    encoder: Any,
    *,
    batch_size: int = 64,
    expected_rows: Optional[int] = 50_000,
    encoder_source: str = BGE_MODEL,
    encoder_provenance: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    source = Path(input_path)
    queries, passages, passage_ids, page_urls, passage_hashes = load_alignment_rows(source)
    if expected_rows is not None and len(queries) != expected_rows:
        raise ValueError(
            f"Alignment artifact requires exactly {expected_rows:,} rows, got {len(queries):,}"
        )
    embeddings = encode_normalized(passages, encoder, batch_size=batch_size)
    query_hashes = [text_sha256(query) for query in queries]
    artifact = {
        **_source_metadata(
            source,
            kind="alignment-targets",
            row_count=len(queries),
            verified_encoder=encoder_provenance is not None,
        ),
        **_encoder_metadata(encoder_source, encoder_provenance),
        "target_embeddings": embeddings,
        # load_bge_embeddings uses the corpus field names for both artifact kinds.
        "document_ids": passage_ids,
        "passage_ids": passage_ids,
        "text_sha256": passage_hashes,
        "page_urls": page_urls,
        "query_text_sha256": query_hashes,
        "query_sha256": _sequence_sha256(query_hashes),
        "passage_id_sha256": _sequence_sha256(passage_ids),
        "passage_text_sha256": _sequence_sha256(passage_hashes),
        "page_url_sha256": _sequence_sha256(page_urls),
        "index_sha256": canonical_float32_index_sha256(embeddings),
    }
    atomic_torch_save(artifact, output_path)
    return artifact


def _resolve_encoder_source(value: str) -> str:
    if value == BGE_MODEL:
        return value
    local_path = Path(value).expanduser()
    if not local_path.is_dir():
        raise ValueError(
            f"--model must be {BGE_MODEL!r} or an existing local model directory: {value}"
        )
    return str(local_path.resolve())


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_encoder(
    source: str,
    device: str,
    *,
    revision: Optional[str],
    revision_was_explicit: bool,
) -> tuple[Any, dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE construction requires the retrieval extra: pip install 'aria-rag[retrieval]'"
        ) from exc
    if Path(source).is_dir():
        if revision is not None or revision_was_explicit:
            raise ValueError("--revision cannot be used with a local --model directory")
        encoder = SentenceTransformer(source, device=device)
    else:
        encoder = SentenceTransformer(source, device=device, revision=revision or "main")
    encoder.eval()
    provenance = resolve_encoder_provenance(
        source,
        encoder,
        declared_revision=revision,
        revision_was_explicit=revision_was_explicit,
    )
    return encoder, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="artifact_kind", required=True)
    for name, help_text in (
        ("corpus", "Build the row-aligned KILT corpus BGE index"),
        ("alignment-targets", "Build passage targets for W_BGE's alignment pairs"),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("--input", required=True, help="Local .json/.jsonl source")
        subparser.add_argument("--output", required=True, help="New .pt/.pth artifact")
        subparser.add_argument(
            "--model",
            default=BGE_MODEL,
            help=f"{BGE_MODEL} or an existing local SentenceTransformer directory",
        )
        subparser.add_argument(
            "--revision",
            default=None,
            help=(
                "Hugging Face branch, tag, or exact commit for --model. "
                "The resolved commit is always recorded; local directories use a tree digest."
            ),
        )
        subparser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
        subparser.add_argument("--batch-size", type=int, default=64)
    subparsers.choices["alignment-targets"].add_argument(
        "--expected-rows",
        type=int,
        default=50_000,
        help="Expected alignment count (paper default: 50000)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    source = _resolve_encoder_source(args.model)
    if Path(source).is_dir():
        model_directory = Path(source).resolve()
        output = Path(args.output).expanduser().resolve()
        if output == model_directory or model_directory in output.parents:
            raise ValueError(
                "--output cannot be inside the local --model directory because it "
                "would invalidate the recorded encoder tree digest"
            )
    device = _resolve_device(args.device)
    encoder, encoder_provenance = _load_encoder(
        source,
        device,
        revision=args.revision,
        revision_was_explicit=args.revision is not None,
    )
    if args.artifact_kind == "corpus":
        artifact = build_corpus_artifact(
            args.input,
            args.output,
            encoder,
            batch_size=args.batch_size,
            encoder_source=source,
            encoder_provenance=encoder_provenance,
        )
    else:
        artifact = build_alignment_target_artifact(
            args.input,
            args.output,
            encoder,
            batch_size=args.batch_size,
            expected_rows=args.expected_rows,
            encoder_source=source,
            encoder_provenance=encoder_provenance,
        )
    print(
        f"Saved {artifact['artifact_kind']} with {artifact['row_count']:,} rows "
        f"to {args.output}; index_sha256={artifact['index_sha256']}"
    )


if __name__ == "__main__":
    main()
