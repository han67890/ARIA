# Security Policy

## Supported versions

| Version | Security updates |
|---|---|
| 0.1.x | Supported |
| Earlier snapshots | Not supported |

Only the latest commit on the default branch and the latest tagged patch
release receive security fixes.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue, discussion, pull
request, or benchmark log. Use GitHub's private vulnerability reporting
feature for this repository. If it is not enabled, contact the repository owner
privately through the hosting platform and request a secure reporting channel.
Do not send reports to Apple or upstream projects unless the issue is confirmed
to affect their unmodified code.

Include the affected version/commit, impact, minimal reproduction, and any
suggested mitigation. Remove credentials, private datasets, model tokens, and
personal data. Maintainers will coordinate disclosure after triage.

## Artifact safety

This research code loads third-party model checkpoints, datasets, pickle-like
PyTorch artifacts, and optional remote-code models. Only load artifacts from
sources you trust. Verify published hashes before loading, prefer formats that
support safe deserialization, review any `trust_remote_code` dependency, and
run untrusted artifacts in an isolated environment without credentials.
