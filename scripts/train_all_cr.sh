#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMPRESSION_RATES=(4 16 32 64 128)
TRAINING_SEEDS=(42 123 456 789 2024)

for seed in "${TRAINING_SEEDS[@]}"; do
  for cr in "${COMPRESSION_RATES[@]}"; do
    phase1_checkpoint="${ROOT_DIR}/checkpoints/aria_phase1_seed${seed}_cr${cr}"
    bash scripts/train_phase1.sh "${cr}" "${seed}"
    bash scripts/train_phase2.sh "${cr}" "${phase1_checkpoint}" "${seed}"
  done
done
