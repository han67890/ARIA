#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=()
if [[ -n "${CLARA_ARCHIVE_DIR:-}" ]]; then
  EXTRA_ARGS+=(--clara_archive_dir "${CLARA_ARCHIVE_DIR}")
fi
exec python -m openrlhf.cli.ablation_aria "${EXTRA_ARGS[@]}" "$@"
