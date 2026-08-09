# Public release checklist

Complete these owner-controlled checks before publishing a Git repository or
package release. Passing the source-tree tests alone does not sanitize Git
history or grant additional distribution rights.

## Repository ownership and history

- Create a recoverable backup before rewriting any Git history.
- Audit every object reachable from every branch and tag for model/data
  archives, generated reports, credentials, private paths, and large build
  products. Deleting a file in the current worktree does not remove it from
  history.
- Publish from a deliberately clean history (for example, an owner-approved
  filtered history or a new release repository) and review the resulting
  object list and repository size before pushing it.
- Run a credential scanner over both the worktree and the complete release
  history. Rotate any real credential that was ever committed; rewriting
  history is not a substitute for rotation.
- Set the canonical repository URL, then replace the manuscript/CITATION
  placeholders and add the archival DOI when one exists.

## Rights and third-party artifacts

- Confirm that the release owner is authorized to apply Apache-2.0 to the
  original ARIA additions identified in `NOTICE`. The inherited CLaRa license
  has SPDX identifier `AML` but is not marked OSI-approved; PISCO-derived
  material identified upstream remains CC BY-NC 4.0. Do not present the mixed
  distribution as single-license or unrestricted commercial software.
- Preserve `LICENSE`, `NOTICE`, `ACKNOWLEDGEMENTS`, and inherited source
  headers. Recheck the terms for datasets, corpora, models, checkpoints, and
  generated artifacts; do not add them merely because `.gitignore` names
  their formats.
- Confirm the four CLaRa candidate ZIPs are external: none may appear in the
  worktree, package artifacts, or any history intended for publication. Test
  matched CLaRa with an explicit `--clara_archive_dir` and the pinned digests
  in `docs/evaluation.md`.
- Confirm that the bundled `ARIA.pdf` is byte-identical to the release
  manuscript and that citation metadata describes that version.

## Release verification

From a clean checkout of the exact candidate commit, run:

```bash
ruff check --select E9,F63,F7,F82 .
pytest -m 'not integration'
bash -n scripts/*.sh
python -m build
python -m twine check dist/*
```

Then verify the wheel and source archive contents, install the wheel in an
empty environment, exercise every console entry point with `--help`, and run
the external-artifact integration matrix on the supported Linux/CUDA stack.
Record release checksums, dependency lock/provenance, resolved model revisions,
and the paper-protocol artifact manifests before signing and tagging the
release.
