# Changelog

All notable repository, API, artifact-schema, and paper-protocol changes are
recorded here. The project uses [Semantic Versioning](https://semver.org/) for
its public Python and command-line surface. Research-protocol changes are
called out even when they remain backward-compatible at the API level.

## Unreleased

### Changed

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
- Synchronized the public method contract with the current manuscript: MADS
  reuses frozen BGE document vectors, CFRS uses the frozen-decoder
  teacher-forced squared-probability proxy, and `P_fb` uses Xavier-uniform
  initialization rather than an SVD-derived initialization.
- Documented the full four-category Phase-I objective, all five Phase-II loss
  terms, and the paper's distinct passage/query/input/target/evaluation limits.
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
