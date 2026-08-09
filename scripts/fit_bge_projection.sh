#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-}"
QUERIES_PATH="${BGE_FIT_QUERIES_PATH:?set BGE_FIT_QUERIES_PATH to the 50k query artifact}"
TARGETS_PATH="${BGE_FIT_EMBEDDINGS_PATH:?set BGE_FIT_EMBEDDINGS_PATH to its aligned BGE targets}"
TEST_URL_FILE="${TEST_URL_FILE:?set TEST_URL_FILE to official test page URLs}"
OUTPUT_PATH="${BGE_PROJECTION_PATH:-${ROOT_DIR}/artifacts/w_bge.pt}"
SEED="${SEED:-42}"
REVISION_ARGS=()
if [[ -n "${BASE_MODEL_REVISION}" ]]; then
  REVISION_ARGS+=(--pretrain_revision "${BASE_MODEL_REVISION}")
fi

python -m openrlhf.cli.train_sft \
  --pretrain "${BASE_MODEL}" \
  "${REVISION_ARGS[@]}" \
  --stage stage2 \
  --passage_max_length 768 \
  --query_max_length 256 \
  --input_max_length 1024 \
  --target_max_length 128 \
  --compress_rate 16 \
  --seed "${SEED}" \
  --fit_bge_projection_only \
  --bge_fit_queries_path "${QUERIES_PATH}" \
  --bge_fit_embeddings_path "${TARGETS_PATH}" \
  --test_url_file "${TEST_URL_FILE}" \
  --bge_projection_save_path "${OUTPUT_PATH}"
