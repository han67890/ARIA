# Reproducibility checklist

This checklist targets the submitted ARIA method (`ARIA_old.tex`). Version
0.2.0 aligns the repository with that contract.

## Submission requirements

| Item | Submission-compatible behavior |
|---|---|
| QCA | Multi-Hop requires hop plus two entities; Multi-Aspect requires aspect and no hop |
| AHR | type weights; confidence drives fallback toward `(0.5, 0.5)` |
| CCEF | fixed top-five survivor contract |
| ACR | literal `range + 1e-6` normalization; sigmoid soft mask |
| Phase I | four target families; held-out target conditioned on memory only |
| Phase II | `L_QA + 0.10 L_MSE` only |
| `L_MSE` | example mean of squared L2; no `1/d_h` normalization |
| CFRS | differentiable conditional-reconstruction fidelity path |
| MTFRL | hard `T_i` prefixes from five documents; one top-200 round |
| `P_fb` initialization | derived from fitted `W_BGE` |
| ARIA-NoComp | fixed checkpoint, first-pass top-five direct context |
| Oracle-QCA | external keyed labels; label-only override under Normal/full/16x |
| QCA-LLM | adapter-free zero-shot Mistral label-only override; full QA and exact 1,000-query endpoints |
| Answer metrics | one prepared scalar `answer` shared by EM, CEM, and F1 |

## Optimization record

The submission reports:

| Setting | Value |
|---|---:|
| Compression ratios | `4, 16, 32, 64, 128` |
| Phase-I / Phase-II epochs | 3 / 5 |
| Mistral/Llama learning rate | `2e-4` |
| Qwen learning rate | `1.6e-4` |
| Warmup | 500 steps |
| Mistral/Llama effective batch | 32 |
| Qwen effective batch | 16 with `2x` accumulation |
| LoRA rank | 16 |
| MTFRL second-round pool | 200 |

Record exact code, model revisions, corpus/index hashes, dataset manifests,
seeds, and hardware for every rerun. Repository-level validation metadata may
also record only manuscript-stated method fields as checkpoint compatibility
requirements.

## Implementation conventions

The manuscript leaves a few edge cases open. Version 0.2.0 continuously
interpolates AHR weights between the balanced and type-conditioned endpoints
and fails fast when CCEF/MTFRL cannot supply five real documents. Tests label
these as release conventions. Neither changes the normal five-document paper
path.
ARIA-NoComp reserves 64 generation tokens under its 32,768-token ceiling and
truncates only the evidence tail when a direct-context prompt exceeds the
remaining budget.

## Checks

```bash
ruff check --select E9,F63,F7,F82 .
pytest -m 'not integration'
bash -n scripts/*.sh
```

A five-seed mean requires five independently trained checkpoints. Save the
command, environment, checkpoint path, artifact hashes, per-example outputs,
and summary for each run.
