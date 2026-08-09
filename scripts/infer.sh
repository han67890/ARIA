#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
CORPUS_PATH="${CORPUS_PATH:?set CORPUS_PATH}"
DOC_EMBEDDINGS="${DOC_EMBEDDINGS:?set DOC_EMBEDDINGS}"

python -m openrlhf.cli.infer_aria \
  --model_path "${MODEL_PATH}" \
  --corpus_path "${CORPUS_PATH}" \
  --doc_embeddings "${DOC_EMBEDDINGS}" \
  --compression_rate "${COMPRESSION_RATE:-16}" \
  "$@"
