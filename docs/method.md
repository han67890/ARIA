# Paper method contract

This guide is a compact implementation checklist for the current `ARIA.tex`.
The manuscript remains authoritative for derivations, analyses, and results.

## End-to-end order

```text
QR(q) -> QCA -> AHR -> optional IGFR -> MADS(top 100) -> CCEF(1..5)
      -> first ACR/compression -> one MTFRL search(top 200)
      -> MADS(top 100) -> CCEF(1..5) -> final ACR/compression
      -> CFRS ordering -> greedy generator
```

The second MADS/CCEF pass reranks the document-ID-deduplicated union of
first-pass survivors and MTFRL candidates. MADS stably sorts candidate passage
occurrences by descending score (with document ID as the exact-tie key) and
takes the first 100 directly; the Normal/training path does not deduplicate
those candidates by page ID. Page-ID deduplication is confined to Oracle-pool
construction and Recall@k accounting. ACR is recomputed on the final survivor set.
CFRS changes order, not set membership. Retrieval, sorting, index lookup, and
top-k decisions are hard in the forward pass; selected-document
straight-through factors supply the training gradient described below.

## Query representation and first-pass retrieval

The Query Reasoner (QR) is the frozen language-model base plus a Phase-II
rank-16 `q_proj` LoRA. `q_rep` is its final-token hidden state for the native
tokenizer encoding of the query. The fixed, pre-fitted map `W_BGE` projects it
to the 1,024-dimensional BGE space.

QCA applies the 38 declarative rules in
`openrlhf/configs/qca_rules.json` to assign Simple, Multi-Aspect, or Multi-Hop
and to instantiate IGFR templates. Its confidence is logged but does not route
the model. QCA uses surface rules rather than `q_rep`. Conflicting matches use
one deterministic precedence chain: explicit Multi-Hop phrases H02--H05,
explicit Multi-Aspect phrases A01--A06, the remaining Multi-Hop rules, the
remaining Multi-Aspect rules, then Simple. S12 is the fallback.

AHR retrieves 4,000 BM25 and 4,000 frozen-BGE candidates, forms their
deduplicated union, min--max-normalizes each channel over that union, and keeps
the best 4,000 hybrid scores. Missing-channel entries receive that channel's
minimum; an all-tied channel receives `0.5`. BM25/BGE weights are:

| QCA type | BM25 | Dense BGE |
|---|---:|---:|
| Simple | 0.75 | 0.25 |
| Multi-Aspect | 0.30 | 0.70 |
| Multi-Hop | 0.25 | 0.75 |

IGFR runs only for Multi-Hop queries with at least one extracted query entity
and coverage below `gamma=0.5`. Each iteration uses the next unused QCA
template, adds at most 200 candidates, and stops when coverage reaches the
threshold or templates are exhausted. The maximum iteration count is 2 for
nominal ratios 4--64 and 1 for ratio 128.

## MADS and CCEF

MADS computes three raw axes for every current candidate:

1. word-unigram TF--IDF cosine, fitted to the query and current pool;
2. cosine between normalized `W_BGE q_rep` and the frozen
   BGE-large-en-v1.5 document vector;
3. spaCy `en_core_web_sm` named-entity coverage, or zero for no query entity.

Each axis is independently min--max-normalized per query, with `0.5` for an
all-tied axis. The equal-weight mean is the MADS score. Sorting is stable and
uses corpus document ID for exact score ties; the best 100 continue. A
separate MiniLM semantic encoder or index is not part of MADS.

For normalized axis scores `s_j`, CCEF uses:

```text
agreement = clip(1 - population_std(s_j) / (mean(s_j) + 1e-6), 0, 1)
fused = mean(s_j) * (0.5 + 0.5 * agreement)
```

It keeps at most five passages with `fused >= 0.30`, preserving MADS order for
exact fused-score ties. If none passes, it keeps the first passage attaining
the maximum fused score. Downstream set size is therefore 1--5.

## ACR

For truncated passage length `L_i` and nominal ratio `r`, the pre-ACR memory
length is:

```text
K0_i = max(1, floor(L_i / r))
```

For a CCEF survivor set, fused scores are min--max-normalized and linearly
mapped to `rho_i` in `[0.25, 1.0]`. A singleton receives `1.0`; a
multi-document all-tied set receives the midpoint `0.625`; otherwise exact
min--max normalization makes the highest score `1.0`. These rates are
detached. At memory position `t=1..K0_i`, the main training configuration applies
`g_it = sigmoid(10 * (rho_i*K0_i - t))` to the raw state. Inference retains raw
states with `g_it >= 0.5`, falling back to the maximum-gate position if
necessary. The independently trained hard-gate analysis uses the same binary
mask in its forward pass and the sigmoid derivative in its backward pass
(`--acr_training_gate hard_st`). Removed memory is not redistributed.

## MTFRL

During training, MTFRL divides each document's gated-state sum by its gate
mass, then gives every first-pass document equal weight. At inference it means
the hard-retained raw states within each document and then across documents.
The pooled state is L2-normalized before `P_fb`; its projected output is also
normalized for cosine retrieval. `P_fb` is a trainable two-layer GELU map with
Xavier-uniform weights and zero biases:

| Backbone | Dimensions |
|---|---|
| Mistral-7B / Llama-3-8B | 4096 -> 2048 -> 1024 |
| Qwen-2.5-14B | 5120 -> 2560 -> 1024 |

The feedback query performs exactly one top-200 search over the frozen BGE
index. It is initialized independently of `W_BGE`.

## CFRS and generation

For every final survivor, CFRS compares the mean of its effective memory states
with the mean of its valid non-memory compressor states. Its error is the
coordinate mean squared distance between those vectors. A detached reverse
min--max transform converts the errors to fidelity scores; if the error range
is at most `1e-6`, every fidelity score is `0.5`. Final ordering uses
`0.70 * CCEF + 0.30 * fidelity` with stable CCEF tie order. CFRS changes only
the final document order and is distinct from the QA-pass alignment MSE.

The generator consumes the CFRS-ordered final memories. Evaluation decoding is
greedy, one beam, no sampling, and stops at EOS or 64 generated tokens.

## ARIA-NoComp diagnostic

ARIA-NoComp is an inference-only diagnostic over a full Phase-II checkpoint;
it is not a trainable RAG configuration. Under Normal retrieval it reuses the
first QCA -> AHR -> IGFR -> MADS -> CCEF result, takes up to five survivors in
their current CCEF order, and joins their unmodified passage strings with two
newlines. The resulting background is inserted into the same standard
system/user QA prompt and passed to the Phase-II decoder adapter as ordinary
token IDs. There are no memory embeddings and no compressor, ACR, CFRS, MTFRL,
or second retrieval round.

Decoding remains greedy, one beam, EOS or 64 new tokens. The context policy is
deterministic and lossless: passages and prompts are not truncated, and an
example fails closed if its complete prompt plus the 64-token output budget
exceeds `min(32768, loaded decoder capacity, finite tokenizer capacity)`. This
ceiling is intentionally independent of the 1,024-token compressed Phase-II
input limit. Evaluation output records the first-pass corpus indices, raw
document-token count, full prompt-token count, effective context ceiling, and
the versioned protocol metadata. The reported ARIA-NoComp row uses the full
16x Phase-II checkpoints; its approximately 590 tokens per passage and 2,950
raw context tokens per query are observed statistics rather than cutoffs.

## Training objective and frozen state

Phase I trains only the compressor adapter on four conditional-generation
categories. Phase II uses:

```text
L_QA + 0.10 L_MSE
```

`L_MSE` is computed in the teacher-forced QA pass between the sequence means
of valid memory states and valid query/gold-answer states, with a coordinate
mean over hidden dimensions. It is separate from the per-document CFRS score.

Phase II updates the QR, compressor, and generator LoRA adapters plus all of
`P_fb`. Language-model bases, `W_BGE`, the BGE encoder/index, and fixed
retrieval rules remain frozen. Retrieval and ranking remain hard in the forward
pass. On selected documents, identity-valued straight-through cosine factors
against their frozen BGE vectors expose backward paths from QA/MSE to QR and
`P_fb` without adding an auxiliary objective or changing inference. See
[training.md](training.md) for initialization, lengths, batches, learning
rates, and loss-path details.
