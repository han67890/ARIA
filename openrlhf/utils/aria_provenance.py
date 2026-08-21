"""Shared corpus provenance rules for ARIA training and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


CORPUS_TEXT_FIELDS: Sequence[str] = (
    "text",
    "passage",
    "content",
    "document",
)
CORPUS_ID_FIELDS: Sequence[str] = (
    "id",
    "doc_id",
    "document_id",
    "passage_id",
    "_id",
)
CORPUS_URL_FIELDS: Sequence[str] = (
    "page_url",
    "url",
    "wikipedia_url",
)
TEXT_SHA256_SCHEME = "utf8-strip-v1"
CORPUS_SHA256_SCHEME = "ordered-id-text-url-jsonl-v1"
SOURCE_SNAPSHOT_SCHEME = "aria-source-snapshot-v1"
EVALUATION_GOLD_DOCUMENT_CONTRACT = {
    "field": "gold_doc_ids",
    "identifier_scope": "full_kilt_corpus",
    "positive_scope": "all_annotated_positives_independent_of_candidates",
    "source": "explicit_input_field",
}
EVALUATION_ANSWER_CONTRACT = {
    "field": "answer",
    "container": "non_empty_string",
    "scope": "single_benchmark_reference",
    "source": "explicit_input_field",
    "metric_reduction": "single_reference",
}


def first_nonempty_string(
    record: Mapping[str, Any],
    fields: Sequence[str],
    *,
    location: str,
    value_name: str,
) -> str:
    """Return the first non-empty string in the declared schema order."""
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"{location} has no non-empty string {value_name}; "
        f"expected one of {tuple(fields)}"
    )


def canonical_page_url(value: Any, *, location: str) -> str:
    """Canonicalize one absolute page URL identically everywhere."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty absolute page URL")
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"{location} must be an absolute http(s) URL")
    normalized_path = re.sub(r"/{2,}", "/", parts.path)
    if normalized_path != "/":
        normalized_path = normalized_path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), normalized_path, parts.query, "")
    )


def corpus_text(record: Mapping[str, Any], *, location: str) -> str:
    """Extract the exact outer-stripped text fed to retrieval/compression."""
    return first_nonempty_string(
        record,
        CORPUS_TEXT_FIELDS,
        location=location,
        value_name="text",
    )


def corpus_id(record: Mapping[str, Any], *, location: str) -> str:
    """Extract a stable corpus ID with the shared field precedence."""
    return first_nonempty_string(
        record,
        CORPUS_ID_FIELDS,
        location=location,
        value_name="document ID",
    )


def corpus_page_url(record: Mapping[str, Any], *, location: str) -> str:
    """Extract and canonicalize a page URL with the shared field precedence."""
    raw_url = first_nonempty_string(
        record,
        CORPUS_URL_FIELDS,
        location=location,
        value_name="page URL",
    )
    return canonical_page_url(raw_url, location=f"{location}.page_url")


def text_sha256(text: str) -> str:
    """Hash outer-stripped text while preserving every internal whitespace byte."""
    if not isinstance(text, str):
        raise TypeError("text_sha256 requires a string")
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def corpus_sha256(
    document_ids: Sequence[str],
    text_hashes: Sequence[str],
    page_urls: Sequence[str],
) -> str:
    """Fingerprint one ordered retrieval corpus and its exact provenance.

    This digest identifies a corpus, not an experiment-wide corpus role.  The
    caller must label it explicitly as either the page-URL-deduplicated
    *training* corpus or the full-KILT *evaluation/inference* corpus.  Keeping
    the hash algorithm shared prevents those two independent artifacts from
    being accidentally compared using subtly different canonicalization.
    """
    if not (len(document_ids) == len(text_hashes) == len(page_urls)):
        raise ValueError("Corpus fingerprint fields are not aligned")
    hasher = hashlib.sha256()
    for document_id, text_hash, page_url in zip(
        document_ids, text_hashes, page_urls
    ):
        hasher.update(
            json.dumps(
                [document_id, text_hash, page_url],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _aria_source_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in (
        "openrlhf/**/*.py",
        "openrlhf/configs/**/*",
        "scripts/*.sh",
        "tests/**/*.py",
    ):
        files.update(
            path
            for path in root.glob(pattern)
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".toml", ".sh"}
        )
    for name in ("pyproject.toml", "requirements.txt"):
        path = root / name
        if path.is_file():
            files.add(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_source_snapshot_manifest(root: Path | None = None) -> dict[str, Any]:
    """Fingerprint the exact runnable source tree, including dirty changes."""
    source_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    files = _aria_source_files(source_root)
    file_hashes = {
        path.relative_to(source_root).as_posix(): file_sha256(path)
        for path in files
    }
    tree_hasher = hashlib.sha256()
    for relative_path, digest in file_hashes.items():
        tree_hasher.update(
            json.dumps(
                [relative_path, digest], separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        tree_hasher.update(b"\n")

    git_commit = None
    git_dirty = None
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        candidate = commit_result.stdout.strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}", candidate):
            git_commit = candidate
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "openrlhf",
                "scripts",
                "tests",
                "pyproject.toml",
                "requirements.txt",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        git_dirty = bool(status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # An installed wheel may not have Git metadata; its byte-exact tree
        # digest and embedded snapshot still establish reproducibility.
        pass
    return {
        "scheme": SOURCE_SNAPSHOT_SCHEME,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "source_tree_sha256": tree_hasher.hexdigest(),
        "source_file_count": len(file_hashes),
        "files": file_hashes,
    }


def write_source_snapshot(
    destination: Path,
    manifest: Mapping[str, Any],
    root: Path | None = None,
) -> None:
    """Write a deterministic ZIP containing every file named by the manifest."""
    source_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("source snapshot manifest requires file hashes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for relative_path in sorted(files):
            path = source_root / relative_path
            if not path.is_file() or file_sha256(path) != files[relative_path]:
                raise ValueError(
                    f"source file changed while checkpointing: {relative_path}"
                )
            info = zipfile.ZipInfo(relative_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
