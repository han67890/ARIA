# Changelog

All notable repository, API, artifact-schema, and paper-protocol changes are
recorded here. The project uses [Semantic Versioning](https://semver.org/) for
its public Python and command-line surface. Research-protocol changes are
called out even when they remain backward-compatible at the API level.

## 0.2.0 - 2026-08-21

### Changed

- Added the canonical public repository URL and author affiliation to citation
  metadata and documentation.
- Narrowed lightweight package/CI support to Python 3.10--3.11 after the Python
  3.12 test job failed; the full paper-training environment remains Python 3.10.
- Corrected package licensing to the mixed SPDX expression
  `AML AND Apache-2.0 AND CC-BY-NC-4.0`, retained upstream notices, and scoped
  Apache-2.0 solely to original ARIA additions owned by their copyright holder.
- Made paper answer evaluation fail closed: every row now requires an explicit
  benchmark-provided `gold_answers` list and a matching evaluation-manifest
  contract. External scalar-answer ZIPs are candidate-only CLaRa artifacts.
- Removed all four 224 MiB CLaRa candidate ZIPs from the source tree. Matched
  CLaRa now requires an explicit `--clara_archive_dir`; all four files, raw ZIP
  digests, required members, candidate fingerprints, and exact question joins
  are validated fail-closed.
- Synchronized the full method contract with the camera-ready manuscript:
  Phase II now optimizes `L_QA + 0.10 L_MSE`; selected-document cosine
  straight-through factors carry that objective to QR and `P_fb` while keeping
  retrieval hard in the forward pass. Likelihood terms now use an exact
  data-parallel global target-token mean rather than an equal mean of rank-local
  means.
- Replaced the legacy frozen-decoder probability proxy with the paper's
  per-document memory/non-memory hidden-mean CFRS score.
- Made the ACR singleton/all-tied conventions and exact non-tied maximum
  explicit, added independently selectable soft and hard-ST training gates,
  and preserved hard thresholding at inference.
- Normalized MTFRL memory summaries before projection and its BGE query after
  projection, and fixed QCA conflict resolution with an explicit precedence
  chain.
- Added the evaluator-only `no_compression` protocol for the paper's
  ARIA-NoComp diagnostic: it requires full Phase-II checkpoints and Normal
  retrieval, preserves up to five first-pass CCEF passages without truncation,
  and records `cr1_sourcecr*` context and retrieval provenance in JSON output.
- Documented the four-category Phase-I objective, two-term Phase-II objective,
  distinct passage/query/input/target/evaluation limits, and the executable
  gradient contract.
- Removed the obsolete MiniLM MADS console entry from package metadata.
- Separated the 16x budget/topology-matched coupling retraining controls from
  fixed-checkpoint forward interventions. `remove_all_coupling` now denotes
  the independently retrained 108-token/static-D2 control, while
  `forward_path_off` denotes rho=1/no-D2 inference on a full checkpoint.

## 0.1.0

### Added

- Two-phase ARIA training, inference, evaluation, data-contract, ablation, and
  counterfactual entry points.
- CFRS, ACR, and MTFRL paper-protocol paths with provenance-bearing artifacts.
- CPU unit tests, GitHub Actions CI, community health files, and reproducibility
  documentation.
