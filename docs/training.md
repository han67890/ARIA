# Training the submission method

Version 0.2.0 restores compatibility with the training method stated in
`ARIA_old.tex`.

## Fit `W_BGE`

Fit the frozen hidden-size-to-1024 alignment map before the two training
phases. Preserve the artifact because the submission uses it for dense query
alignment and to initialize the MTFRL feedback projection.

```bash
export BGE_FIT_QUERIES_PATH=/data/kilt_alignment_queries.jsonl
export BGE_FIT_EMBEDDINGS_PATH=/data/kilt_alignment_targets.pt
export BGE_PROJECTION_PATH=$PWD/artifacts/w_bge.pt
bash scripts/fit_bge_projection.sh
```

## Phase I: memory-only conditional reconstruction

```bash
bash scripts/train_phase1.sh 16 42
```

Phase I trains the compressor adapter for three epochs over all four target
families: SimpleQA, ComplexQA, Paraphrase, and Entity-Augmented. The frozen
generator reconstructs each held-out target from compressed memory alone.
Dataset instructions and category labels remain in manifests for provenance,
but are not decoder-conditioning text in the submission objective.

## Phase II: two-term objective

```bash
export CORPUS_PATH=/data/kilt_corpus_no_test_overlap.jsonl
export CORPUS_EMBEDDINGS_PATH=/data/kilt_bge_aligned.pt
export BGE_PROJECTION_PATH=$PWD/artifacts/w_bge.pt
bash scripts/train_phase2.sh 16 checkpoints/aria_phase1_seed42_cr16 42
```

Phase II executes both retrieval rounds and minimizes

```text
L_QA + 0.10 * L_MSE
```

The QA hidden-state term is an unnormalized squared L2 distance across hidden
coordinates, averaged over examples. It is not a coordinate mean. CFRS stays
in the forward graph as the compressor-fidelity path intended by the
submission; it is not an extra weighted term in the Phase-II scalar objective.

ACR applies the differentiable sigmoid soft mask to generator memory states.
MTFRL uses the hard `T_i` prefixes of exactly five first-pass documents, and
its two-layer projection is initialized from the fitted `W_BGE` artifact.

## Submission-reported optimization settings

The submitted reproducibility table reports rank-16 LoRA, three Phase-I
epochs, five Phase-II epochs, learning rate `2e-4` for Mistral/Llama
(`1.6e-4` for Qwen), 500 warmup steps, effective batch 32 for Mistral/Llama,
and effective batch 16 for Qwen. The submission explicitly describes Qwen's
effective batch as using `2x` gradient accumulation.

## Direct requirements versus release conventions

Direct submission requirements are tested for the QCA routing rule, AHR
fallback, ACR epsilon formula and soft gate, four-family memory-only Phase I, the two-term
Phase-II loss, unnormalized MSE, differentiable CFRS intent, five-document hard
prefix MTFRL, `W_BGE`-derived feedback initialization, and fixed top-five CCEF.

Release conventions cover only unspecified edges: continuous AHR confidence
interpolation, fail-fast five-document shape validation, and a one-token MTFRL
prefix floor when `floor(rho_i*K_i)` is zero. Normal five-document execution
follows the submission equations directly.

## Evaluation-only NoComp

ARIA-NoComp reuses a trained Phase-II checkpoint without updating parameters.
It concatenates the first-pass top-five raw documents and bypasses compression
and the second round.
