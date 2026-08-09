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
CFRS changes order, not set membership. Hard retrieval, sorting, index lookup,
and top-k decisions are detached/discrete.

## Query representation and first-pass retrieval

The Query Reasoner (QR) is the frozen language-model base plus a Phase-II
rank-16 `q_proj` LoRA. `q_rep` is its final-token hidden state for the native
tokenizer encoding of the query. The fixed, pre-fitted map `W_BGE` projects it
to the 1,024-dimensional BGE space.

QCA applies the 38 declarative rules in
`openrlhf/configs/qca_rules.json` to assign Simple, Multi-Aspect, or Multi-Hop
and to instantiate IGFR templates. Its confidence is logged but does not route
the model. QCA uses surface rules rather than `q_rep`.

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
mapped to `rho_i` in `[0.25, 1.0]`. A singleton receives 1.0; a multi-document
all-tied set receives 0.625. These rates are detached. At memory position
`t=1..K0_i`, training applies `g_it = sigmoid(10 * (rho_i*K0_i - t))` to the
raw state. Inference retains raw states with `g_it >= 0.5`, falling back to the
maximum-gate position if necessary. Removed memory is not redistributed.

## MTFRL

During training, MTFRL divides each document's gated-state sum by its gate
mass, then gives every first-pass document equal weight. At inference it means
the hard-retained raw states within each document and then across documents.
`P_fb` is a trainable two-layer GELU map with Xavier-uniform weights and zero
biases:

| Backbone | Dimensions |
|---|---|
| Mistral-7B / Llama-3-8B | 4096 -> 2048 -> 1024 |
| Qwen-2.5-14B | 5120 -> 2560 -> 1024 |

The L2-normalized feedback query is trained at temperature 0.05 with an
annotated-support contrastive objective. A detached copy performs exactly one
top-200 search over the frozen BGE index. The paper does not initialize this
head from `W_BGE`, an SVD, or another fitted projection.

## CFRS and generation

For every final survivor, CFRS disables the generator LoRA and uses the frozen
base decoder under teacher forcing to reconstruct the original truncated
passage from its final ACR-processed memory. At each valid passage target, it
sums squared error over the full next-token probability vector against the
one-hot target, then averages over targets. Padding and structural delimiters
are excluded.

The undetached error averages into `L_CFRS` and updates the compressor through
the frozen decoder. A detached reverse-min--max copy becomes a fidelity score;
if the error range is at most `1e-6`, every fidelity score is `0.5`. Final
ordering uses `0.70 * CCEF + 0.30 * fidelity` with stable CCEF tie order. CFRS
is not latent cross-attention and is distinct from the QA hidden-mean MSE loss.

The generator consumes the CFRS-ordered final memories. Evaluation decoding is
greedy, one beam, no sampling, and stops at EOS or 64 generated tokens.

## Training objective and frozen state

Phase I trains only the compressor adapter on four conditional-generation
categories. Phase II uses:

```text
L_QA + 0.10 L_MSE + 0.10 L_CFRS + 0.05 L_QR + 0.05 L_MTFRL
```

Phase II updates the QR, compressor, and generator LoRA adapters plus all of
`P_fb`. Language-model bases, `W_BGE`, the BGE encoder/index, and fixed
retrieval rules remain frozen. See [training.md](training.md) for
initialization, lengths, batches, learning rates, and loss-path details.
