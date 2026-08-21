# Changelog

All notable repository, API, artifact-schema, and paper-protocol changes are
recorded here. The project uses [Semantic Versioning](https://semver.org/) for
its public Python and command-line surface. Research-protocol changes are
called out even when they remain backward-compatible at the API level.

## 0.2.0 - 2026-08-21

### Changed

- Aligned the implementation and public contract with the method described in
  the accepted manuscript.
- Restored QCA's Multi-Hop requirement (a hop rule plus at least two entities),
  hop-over-aspect conflict handling, and confidence-conditioned AHR fallback
  toward balanced BM25/dense weights.
- Restored the literal epsilon-normalized ACR rates and the submission's
  differentiable sigmoid soft mask. Removed the later hard-ST experiment and
  its checkpoint/CLI protocol from the paper contract.
- Restored four-family, memory-only held-out-target conditioning in Phase I and the
  exact two-term Phase-II objective `L_QA + 0.10 L_MSE`, where `L_MSE` is the
  example mean of the unnormalized squared L2 hidden-state distance.
- Restored CFRS as the differentiable conditional-reconstruction fidelity
  path and MTFRL as a fixed-five-document hard-prefix average initialized from
  the pre-fitted `W_BGE` map.
- Restored the fixed top-five CCEF contract and removed later method-only
  checkpoint requirements, including physical-global likelihood-reduction,
  selected-document cosine-ST, hidden-mean CFRS, and hard-gate metadata.
- Kept ARIA-NoComp as the submission's fixed-checkpoint top-five direct-context
  diagnostic, with a 64-token reserve and question-preserving evidence-tail
  truncation under its 32,768-token ceiling.
- Documented release conventions separately from manuscript requirements:
  continuous confidence interpolation and fail-fast five-document validation
  affect only otherwise unspecified edges.
- Added the canonical public repository URL and author affiliation to citation
  metadata and documentation.
- Narrowed lightweight package/CI support to Python 3.10--3.11 after the Python
  3.12 test job failed; the full paper-training environment remains Python 3.10.
- Corrected package licensing to the mixed SPDX expression
  `AML AND Apache-2.0 AND CC-BY-NC-4.0`, retained upstream notices, and scoped
  Apache-2.0 solely to original ARIA additions owned by their copyright holder.
- Restored the official scalar-answer evaluation contract: every prepared row
  supplies one `answer`, and paper EM, CEM, F1, and training-time validation CEM
  use that same reference. The fixed Phase-II epoch-view schedule is recorded as
  `(42, 123, 456, 789, 2024)` independently of the training-run seed.
- Added full-corpus CLaRa Normal Recall, shared ARIA/CLaRa Oracle top-100 pools,
  exact-panel Oracle-QCA, and adapter-free zero-shot QCA-LLM QA/panel endpoints;
  all save their candidate, label, prompt, and identity provenance.
- Removed all four 224 MiB CLaRa candidate ZIPs from the source tree. Matched
  CLaRa now requires an explicit `--clara_archive_dir`; all four files, raw ZIP
  digests, required members, candidate fingerprints, and exact question joins
  are validated fail-closed.
- Removed the obsolete MiniLM MADS console entry from package metadata.
- The paper coupling table uses the `fixed_*` and `forward_path_off`
  interventions on full checkpoints. Independently retrained 16x
  budget/topology variants remain available as additional release controls.
- Bound the complete 168,745-row historical MuSiQue derived source to an
  external four-family manifest with recomputed content and parent-link
  digests; raw MuSiQue is no longer presented as sufficient to rebuild it.

## 0.1.0

### Added

- Two-phase ARIA training, inference, evaluation, data-contract, ablation, and
  counterfactual entry points.
- CFRS, ACR, and MTFRL paper-protocol paths with provenance-bearing artifacts.
- CPU unit tests, GitHub Actions CI, community health files, and reproducibility
  documentation.
