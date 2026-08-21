#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CR="${1:-16}"
SEED="${2:-42}"
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-}"
DATASET="${PHASE1_DATASET:-${ROOT_DIR}/data/aria/phase1}"
RAG_CONFIGURATION="${RAG_CONFIGURATION:-full}"
if [[ "${RAG_CONFIGURATION}" != "full" && "${RAG_CONFIGURATION}" != "clara_baseline" ]]; then
  echo "Phase I RAG_CONFIGURATION must be full or clara_baseline" >&2
  exit 2
fi
if [[ "${RAG_CONFIGURATION}" == "full" ]]; then
  DEFAULT_OUTPUT_DIR="${ROOT_DIR}/checkpoints/aria_phase1_seed${SEED}_cr${CR}"
else
  DEFAULT_OUTPUT_DIR="${ROOT_DIR}/checkpoints/clara_phase1_seed${SEED}_cr${CR}"
fi
OUTPUT_DIR="${PHASE1_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
NUM_GPUS="${NUM_GPUS:-8}"
TEST_URL_FILE="${TEST_URL_FILE:?set TEST_URL_FILE to official test page URLs}"

if [[ "${BASE_MODEL,,}" == *qwen* ]]; then
  LR="${LEARNING_RATE:-1.6e-4}"
  GLOBAL_BATCH=16
  GRAD_ACCUMULATION=2
else
  LR="${LEARNING_RATE:-2e-4}"
  GLOBAL_BATCH=32
  GRAD_ACCUMULATION=1
fi
if (( NUM_GPUS <= 0 || GLOBAL_BATCH % (NUM_GPUS * GRAD_ACCUMULATION) != 0 )); then
  echo "NUM_GPUS*accumulation must divide paper batch ${GLOBAL_BATCH}" >&2
  exit 2
fi
MICRO_BATCH="${MICRO_BATCH_SIZE:-$((GLOBAL_BATCH / NUM_GPUS / GRAD_ACCUMULATION))}"
if (( NUM_GPUS * MICRO_BATCH * GRAD_ACCUMULATION != GLOBAL_BATCH )); then
  echo "NUM_GPUS * MICRO_BATCH_SIZE * accumulation must equal ${GLOBAL_BATCH}" >&2
  exit 2
fi
REVISION_ARGS=()
if [[ -n "${BASE_MODEL_REVISION}" ]]; then
  REVISION_ARGS+=(--pretrain_revision "${BASE_MODEL_REVISION}")
fi

torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m openrlhf.cli.train_sft \
  --pretrain "${BASE_MODEL}" \
  "${REVISION_ARGS[@]}" \
  --stage stage1 \
  --dataset "${DATASET}" \
  --test_url_file "${TEST_URL_FILE}" \
  --passage_max_length 768 \
  --query_max_length 256 \
  --input_max_length 2048 \
  --target_max_length 512 \
  --compress_rate "${CR}" \
  --generation_top_k 1 \
  --rag_configuration "${RAG_CONFIGURATION}" \
  --max_epochs 3 \
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
