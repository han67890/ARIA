# Contributing

Contributions that strengthen ARIA's implementation, protocol tooling,
documentation, and tests are welcome.

## Development setup

Use Python 3.10--3.11 for source-level checks. The full paper-training stack
targets Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The lightweight editable install is suitable for CPU tests. Full training and
integration tests require the Linux/CUDA environment in
[Getting started](docs/getting_started.md) plus separately licensed artifacts.

## Change requirements

Before opening a pull request:

1. Preserve alignment with the ARIA method and experimental protocol.
2. If a method equation, default, data schema, or artifact format changes,
   update the relevant guide and [reproducibility checklist](docs/reproducibility.md).
3. Add or update focused tests for deterministic logic and bug fixes.
4. Run:

   ```bash
   ruff check --select E9,F63,F7,F82 .
   pytest -m 'not integration'
   bash -n scripts/*.sh
   python -m build
   python -m twine check dist/*
   ```

5. Mark tests that require external models, datasets, network access, GPUs, or
   multi-process launchers with `@pytest.mark.integration`.
6. Do not commit checkpoints, private datasets, credentials, logs, or generated
   evaluation results.
7. Preserve copyright and license notices in files derived from Apple CLaRa or
   OpenRLHF.

## Pull requests

Keep pull requests focused. Describe the paper component or protocol stage,
compatibility impact, tests run, and any external artifacts required for full
verification. Do not include private paths, access tokens, or unpublished data
in logs and examples. All CI checks must pass, and user-facing behavior needs
corresponding documentation.

For security-sensitive reports, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue. Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

By submitting a contribution, you confirm that you have the right to submit it
under the repository's applicable license terms. Unless explicitly stated
otherwise in writing, new original contributions are submitted under
Apache-2.0; this does not alter licenses that already apply to inherited or
third-party material.
