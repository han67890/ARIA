#!/usr/bin/env bash
set -euo pipefail

# This decomposition uses fixed-checkpoint Forward-path-off on each aligned
# full checkpoint (rho=1, no D2). It is not the separately retrained,
# budget/topology-matched remove_all_coupling control.
EXTRA_ARGS=()
if [[ -n "${CLARA_ARCHIVE_DIR:-}" ]]; then
  EXTRA_ARGS+=(--clara_archive_dir "${CLARA_ARCHIVE_DIR}")
fi
exec python -m openrlhf.cli.counterfactual_decomposition "${EXTRA_ARGS[@]}" "$@"
