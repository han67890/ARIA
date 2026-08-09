#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CR="${1:-16}"
PHASE1_CHECKPOINT="${2:?usage: scripts/train_phase2.sh CR PHASE1_CHECKPOINT [SEED]}"
SEED="${3:-42}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-}"
DATASET="${PHASE2_DATASET:-${ROOT_DIR}/data/aria/phase2}"
NUM_GPUS="${NUM_GPUS:-8}"
RAG_CONFIGURATION="${RAG_CONFIGURATION:-full}"
OUTPUT_DIR="${PHASE2_OUTPUT_DIR:-${ROOT_DIR}/checkpoints/aria_phase2_${RAG_CONFIGURATION}_seed${SEED}_cr${CR}}"

CORPUS_PATH="${CORPUS_PATH:?set CORPUS_PATH to the de-duplicated fixed corpus}"
TEST_URL_FILE="${TEST_URL_FILE:?set TEST_URL_FILE to official test page URLs}"
if [[ "${RAG_CONFIGURATION}" == "clara_baseline" ]]; then
  CORPUS_EMBEDDINGS_PATH="${CORPUS_EMBEDDINGS_PATH:-}"
  BGE_PROJECTION_PATH="${BGE_PROJECTION_PATH:-}"
  LAMBDA_MSE=0
  LAMBDA_CFRS=0
  LAMBDA_QR=0
  LAMBDA_MTFRL=0
else
  CORPUS_EMBEDDINGS_PATH="${CORPUS_EMBEDDINGS_PATH:?set CORPUS_EMBEDDINGS_PATH to aligned BGE vectors}"
  BGE_PROJECTION_PATH="${BGE_PROJECTION_PATH:?set BGE_PROJECTION_PATH to fitted W_BGE}"
  LAMBDA_MSE=0.1
  LAMBDA_CFRS=0.1
  LAMBDA_QR=0.05
  LAMBDA_MTFRL=0.05
fi

case "${RAG_CONFIGURATION}" in
  remove_cfrs)
    LAMBDA_CFRS=0
    ;;
  static_second_retrieval)
    LAMBDA_MTFRL=0
    ;;
  remove_all_coupling)
    LAMBDA_CFRS=0
    LAMBDA_MTFRL=0
    ;;
  uniform_acr)
    ;;
  forward_path_off|fixed_remove_cfrs|fixed_uniform_acr|fixed_remove_mtfrl|remove_acr|remove_mtfrl)
    echo "${RAG_CONFIGURATION} is fixed-checkpoint inference-only; train full" >&2
    exit 2
    ;;
esac
if [[ "${RAG_CONFIGURATION}" == "remove_cfrs" \
   || "${RAG_CONFIGURATION}" == "uniform_acr" \
   || "${RAG_CONFIGURATION}" == "static_second_retrieval" \
   || "${RAG_CONFIGURATION}" == "remove_all_coupling" ]]; then
  if [[ "${CR}" != "16" ]]; then
    echo "Matched coupling retraining is defined only at 16x" >&2
    exit 2
  fi
fi

if [[ "${BASE_MODEL,,}" == *qwen* ]]; then
  LR="${LEARNING_RATE:-1.6e-4}"
  GLOBAL_BATCH=16
else
  LR="${LEARNING_RATE:-2e-4}"
  GLOBAL_BATCH=32
fi
if (( NUM_GPUS <= 0 || GLOBAL_BATCH % NUM_GPUS != 0 )); then
  echo "NUM_GPUS=${NUM_GPUS} must divide the paper effective batch ${GLOBAL_BATCH}" >&2
  exit 2
fi
MICRO_BATCH="${MICRO_BATCH_SIZE:-$((GLOBAL_BATCH / NUM_GPUS))}"
if (( NUM_GPUS * MICRO_BATCH != GLOBAL_BATCH )); then
  echo "Phase II forbids gradient accumulation: NUM_GPUS * MICRO_BATCH_SIZE must equal ${GLOBAL_BATCH}" >&2
  exit 2
fi
REVISION_ARGS=()
if [[ -n "${BASE_MODEL_REVISION}" ]]; then
  REVISION_ARGS+=(--pretrain_revision "${BASE_MODEL_REVISION}")
fi
torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m openrlhf.cli.train_sft \
  --pretrain "${BASE_MODEL}" \
  "${REVISION_ARGS[@]}" \
  --pretrain_checkpoint "${PHASE1_CHECKPOINT}" \
  --stage stage2 \
  --dataset "${DATASET}" \
  --corpus_path "${CORPUS_PATH}" \
  --corpus_embeddings_path "${CORPUS_EMBEDDINGS_PATH}" \
  --bge_projection_path "${BGE_PROJECTION_PATH}" \
  --test_url_file "${TEST_URL_FILE}" \
  --passage_max_length 768 \
  --query_max_length 256 \
  --input_max_length 1024 \
  --target_max_length 128 \
  --compress_rate "${CR}" \
  --generation_top_k 5 \
  --stage2_retrieval_top_n 5 \
  --lambda_mse "${LAMBDA_MSE}" \
  --lambda_cfrs "${LAMBDA_CFRS}" \
  --lambda_qr "${LAMBDA_QR}" \
  --lambda_mtfrl "${LAMBDA_MTFRL}" \
  --rag_configuration "${RAG_CONFIGURATION}" \
  --max_epochs 5 \
  --learning_rate "${LR}" \
  --lr_warmup_steps 500 \
  --lr_scheduler cosine \
  --adam_betas 0.9 0.95 \
  --adam_eps 1e-8 \
  --l2 0 \
  --train_batch_size "${GLOBAL_BATCH}" \
  --micro_train_batch_size "${MICRO_BATCH}" \
  --zero_stage 2 \
  --bf16 \
  --flash_attn \
  --gradient_checkpointing \
  --seed "${SEED}" \
  --save_path "${OUTPUT_DIR}" \
  --ckpt_path "${OUTPUT_DIR}/deepspeed" \
  --save_steps -2 \
  --eval_steps -1 \
  --logging_steps 10
