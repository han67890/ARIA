# Reproducibility checklist

Use this checklist to distinguish a source-level smoke test from a result that
follows the complete ARIA paper protocol. The repository does not bundle the
full training pools, KILT corpus/index, model weights, or all trained
checkpoints.

## 1. Pin code and external resources

- Record the ARIA Git commit and whether the worktree is dirty.
- Pin the decoder and tokenizer to the same resolved model revision.
- Pin `BAAI/bge-large-en-v1.5` and record the resolved revision used to build
  every document/alignment embedding artifact.
- Record the SHA-256 and manifest schema for the corpus, BGE index,
  `W_BGE`, prepared datasets, and checkpoints.
- For matched CLaRa, keep the four candidate ZIPs outside Git and record the
  byte digests validated through `--clara_archive_dir`; a partial archive set
  is rejected even for a single-benchmark run.
- Keep the Phase-II page-URL-deduplicated training corpus/index separate from
  the complete KILT corpus/index used for Normal evaluation.

The canonical source repository is
<https://github.com/han67890/ARIA>. The manuscript and repository do not state
the historical Hub commit IDs used for the reported runs. A reproducer must
obtain those values from the release owner; this repository does not guess them.

## 2. Verify the method contract

| Item | Paper protocol |
|---|---|
| Ratios | independently trained `4, 16, 32, 64, 128` |
| Runs | `42, 123, 456, 789, 2024` in the supplied launch matrix |
| LoRA | `q_proj`, rank 16, alpha 32, dropout 0.10, no bias |
| Phase I | four-category 7,808,465-example conditional-generation mixture |
| Phase II loss | `QA + .10 MSE + .10 CFRS + .05 QR + .05 MTFRL` |
| MADS semantic axis | normalized `W_BGE q_rep` vs. frozen BGE document vector |
| CFRS | frozen-decoder teacher-forced next-token squared-probability proxy |
| `P_fb` | two-layer GELU, Xavier-uniform weights, zero biases |
| Retrieval | one MTFRL second round, 200 dense candidates |
| Decoding | greedy, one beam, EOS or 64 generated tokens |

In particular, a MiniLM MADS encoder, latent cross-attention CFRS objective, or
SVD-initialized feedback projection is a different method and must not be
reported as the paper configuration.

## 3. Verify lengths and optimization

| Setting | Value |
|---|---:|
| Passage / query maximum | 768 / 256 tokens |
| Phase-I input / target maximum | 2,048 / 512 tokens |
| Phase-II input / target maximum | 1,024 / 128 tokens |
| Evaluation generation maximum | 64 tokens |
| Phase-I epochs / effective batch / LR | 3 / 128 / `1e-4` (Mistral/Llama) |
| Phase-II epochs / effective batch / LR | 5 / 32 / `2e-4` (Mistral/Llama) |
| Qwen effective batch / Phase-II LR | 16 / `1.6e-4` |
| Phase-II warmup | 500 steps |
| Optimizer | AdamW, cosine decay, betas `(0.9, 0.95)`, eps `1e-8`, no weight decay |
| Precision / max gradient norm | bfloat16 / 1.0 |

Phase-I warmup is 3% of steps. The Mistral/Llama Phase-I effective batch is
128; Qwen uses 16 in both phases.

## 4. Verify data views

- Phase I has exact category counts `2,000,000 / 2,000,000 / 1,966,291 /
  1,842,174` after the documented test-page URL removal.
- Phase II uses pools of NQ 58,622, HotpotQA 90,185, augmented MuSiQue
  168,745, and 2WikiMultiHopQA 167,454.
- Each of five epochs draws 9,600 examples per benchmark without replacement:
  38,400 rows per epoch and 192,000 example views in total.
- The four evaluation split sizes are 6,489 / 7,384 / 2,417 / 12,576.
- Every evaluation row carries an explicit, non-empty list of
  benchmark-provided `gold_answers`; scalar answers from the external candidate
  archives are not a paper-metric substitute.
- Matched-CLaRa rows are joined to the external pinned top-20 candidate archives
  only by the complete official count and exact question order/text.
- Stable document/page IDs and explicit support IDs are used for retrieval
  metrics; candidate-relative positions are not treated as corpus-level gold
  annotations.

See [data.md](data.md) for the schemas and hash rules.

## 5. Run checks and record results

```bash
ruff check --select E9,F63,F7,F82 .
pytest -m 'not integration'
bash -n scripts/*.sh
```

Then run each external-artifact integration job and save its exact command,
environment, checkpoint path, manifest hashes, per-example predictions, and
summary JSON. A five-run mean requires five independently trained checkpoints;
repeated evaluation of one checkpoint is not a five-seed result.

The main paper endpoint is 44.50 average token-level F1 at nominal 16x Normal
retrieval with Mistral-7B over NQ, HotpotQA, MuSiQue, and 2WikiMultiHopQA.
Substring CEM is secondary. Normal first-pass Recall and Oracle top-100
post-reranking Recall are different protocols and should be labeled separately.
