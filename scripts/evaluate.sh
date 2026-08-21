#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CR="${1:-16}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:?set EVAL_DATA_PATH to the scalar-answer aria-data evaluation artifact}"
RAG_CONFIGURATION="${RAG_CONFIGURATION:-full}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-normal}"
MODEL_TEMPLATE="${MODEL_PATH_TEMPLATE:-}"
if [[ -z "${MODEL_TEMPLATE}" ]]; then
  if [[ "${RAG_CONFIGURATION}" == "clara_baseline" ]]; then
    MODEL_TEMPLATE="${ROOT_DIR}/checkpoints/clara_seed{seed}_cr{cr}"
  else
    MODEL_TEMPLATE="${ROOT_DIR}/checkpoints/aria_phase2_full_seed{seed}_cr{cr}"
  fi
fi
DOC_EMBEDDINGS="${DOC_EMBEDDINGS:?set DOC_EMBEDDINGS; a dataset token may be used in the path}"
CORPUS_PATH="${CORPUS_PATH:?set CORPUS_PATH to the full KILT corpus artifact}"
EXTRA_ARGS=(
  --eval_data_path "${EVAL_DATA_PATH}"
  --corpus_path "${CORPUS_PATH}"
  --doc_embeddings "${DOC_EMBEDDINGS}"
)
if [[ "${RAG_CONFIGURATION}" == "clara_baseline" && "${RETRIEVAL_MODE}" == "normal" ]]; then
  CLARA_ARCHIVE_DIR="${CLARA_ARCHIVE_DIR:?set CLARA_ARCHIVE_DIR to the external four-ZIP CLaRa archive directory}"
  EXTRA_ARGS+=(--clara_archive_dir "${CLARA_ARCHIVE_DIR}")
fi
if [[ -n "${BASELINE_RESULTS:-}" ]]; then
  EXTRA_ARGS+=(--baseline_results "${BASELINE_RESULTS}")
fi
python -m openrlhf.cli.evaluate_aria \
  --model_path_template "${MODEL_TEMPLATE}" \
  --seeds 42 123 456 789 2024 \
  --dataset all \
  --retrieval_mode "${RETRIEVAL_MODE}" \
  --rag_configuration "${RAG_CONFIGURATION}" \
  --compression_rate "${CR}" \
  --output_dir "${EVAL_OUTPUT_DIR:-${ROOT_DIR}/eval_results/${RETRIEVAL_MODE}_cr${CR}}" \
  --device "${DEVICE:-cuda}" \
  "${EXTRA_ARGS[@]}"
