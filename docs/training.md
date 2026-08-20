# Training

## W_BGE alignment

```bash
export BGE_FIT_QUERIES_PATH=/data/kilt_alignment_queries.jsonl
export BGE_FIT_EMBEDDINGS_PATH=/data/kilt_alignment_targets.pt
export BGE_PROJECTION_PATH=$PWD/artifacts/w_bge.pt
export TEST_URL_FILE=/data/official_test_urls.txt
export BASE_MODEL_REVISION=<40-character-hugging-face-commit>
bash scripts/fit_bge_projection.sh
```

This runs exactly 50,000 pairs, AdamW at `5e-4`, batch 128, two epochs, then
freezes the projection. When the alignment-target artifact was produced by the
revision-aware BGE builder (v2), the fitted W_BGE artifact carries the declared
and resolved BGE revision—or the local model tree digest—forward into every
checkpoint. Older v1 artifacts remain loadable but cannot make an exact BGE
revision claim. Keep `BASE_MODEL_REVISION` unchanged for W_BGE fitting and both
training phases so the projection cannot be paired with a different decoder.

## Phase I

```bash
export TEST_URL_FILE=/data/official_test_urls.txt
bash scripts/train_phase1.sh 16 42
```

Phase I trains the compressor adapter for three epochs with the conditional
generation objective in Equation 2 of the paper. It uses all four functional
categories, not a paraphrase-only corpus:

| Category | Rows | Target type |
|---|---:|---|
| SimpleQA | 2,000,000 | single-answer conditional generation |
| ComplexQA | 2,000,000 | multi-fact/reasoning conditional generation |
| Paraphrase | 1,966,291 | surface reformulation |
| Entity-Augmented | 1,842,174 | entity-centric conditional generation |
| **Total** | **7,808,465** | |

The effective batch is 128 for Mistral/Llama (16 for Qwen), the learning rate
is `1e-4` with cosine decay and 3% warmup, and the Phase-I input/target maxima
are 2,048/512 tokens. The passage maximum is 768 tokens. LoRA targets only
`q_proj`, with rank 16, alpha 32, dropout 0.10, Kaiming-uniform `A`, and zero
`B`.

Both Phase-I conditional generation and Phase-II answer generation use the
exact mean next-token cross-entropy over unmasked target positions in the
physical data-parallel minibatch. Rank-local means are rescaled from the
data-parallel target-token count before DeepSpeed gradient averaging. The
official launchers use one physical minibatch per optimizer step.

`BASE_MODEL_REVISION` is forwarded to the decoder and tokenizer; their
resolved commits must agree and are persisted in the checkpoint. Use the same
value for Phase II. A tag or branch is accepted, but an exact commit is the
reproducible choice.

## Phase II

```bash
export CORPUS_PATH=/data/kilt_corpus_no_test_overlap.jsonl
export CORPUS_EMBEDDINGS_PATH=/data/kilt_bge_aligned.pt
export BGE_PROJECTION_PATH=$PWD/artifacts/w_bge.pt
export TEST_URL_FILE=/data/official_test_urls.txt
bash scripts/train_phase2.sh 16 checkpoints/aria_phase1_seed42_cr16 42
```

Full ARIA Phase II executes both retrieval rounds in every batch and optimizes
the paper objective:

```text
L = L_QA + 0.10 L_MSE
```

`L_MSE` is the coordinate-wise squared distance between the sequence means of
valid memory-token and query/gold-answer hidden states in the final
teacher-forced QA decoder layer. It is symmetric: gradients reach both means.
The forward pass uses the exact hard document sets produced by retrieval and
ranking. Identity-valued straight-through cosine factors on selected documents
preserve those forward values while exposing gradients from QA/MSE to the QR
adapter and `P_fb`. No retrieval-specific auxiliary loss is added.

Phase II trains independent QR/compressor/generator `q_proj` LoRA adapters and
all parameters of `P_fb`. The compressor loads the corresponding Phase-I
ratio checkpoint, the QR adapter is copied from that compressor adapter, and
the generator receives a fresh Kaiming-`A`/zero-`B` LoRA. `P_fb` uses
Xavier-uniform weights and zero biases. Language-model bases, `W_BGE`, the BGE
encoder/index, retrieval rules, and hard selection/ranking operations remain
frozen or discrete.

Defaults are five epochs, 500 warmup steps, effective batch 32 for
Mistral/Llama or 16 for Qwen, and learning rate `2e-4` or `1.6e-4`, respectively.
The Phase-II input/target maxima are 1,024/128 tokens; passage/query maxima are
768/256. Training uses bfloat16 and gradient norm 1.0. MADS reuses the same
normalized projected QR state and frozen BGE document embeddings as the dense
retrieval path; no separate MiniLM model or semantic index belongs to the
paper protocol.

The main model uses soft ACR gates during training and hard thresholded gates
at inference. To train the hard-gate analysis checkpoint with the same forward
allocation and a sigmoid straight-through derivative, set
`ACR_TRAINING_GATE=hard_st` when invoking `scripts/train_phase2.sh`.

The default is the full paper model. QCA/AHR/IGFR/MADS/CCEF ablations are
fixed-checkpoint inference interventions: evaluate each label with the
corresponding five `full` checkpoints through `aria-ablate`; do not retrain
them. Train dedicated checkpoints only for the matched-retraining controls and
the matched CLaRa control:

```bash
# Appendix-A.31 matched rows: repeat independently for all five seeds.
RAG_CONFIGURATION=uniform_acr \
PHASE2_OUTPUT_DIR=checkpoints/uniform_acr_seed42_cr16 \
bash scripts/train_phase2.sh 16 checkpoints/aria_phase1_seed42_cr16 42

RAG_CONFIGURATION=static_second_retrieval \
PHASE2_OUTPUT_DIR=checkpoints/static_d2_seed42_cr16 \
bash scripts/train_phase2.sh 16 checkpoints/aria_phase1_seed42_cr16 42

RAG_CONFIGURATION=remove_all_coupling \
PHASE2_OUTPUT_DIR=checkpoints/all_off_matched_seed42_cr16 \
bash scripts/train_phase2.sh 16 checkpoints/aria_phase1_seed42_cr16 42

RAG_CONFIGURATION=clara_baseline \
PHASE1_OUTPUT_DIR=checkpoints/clara_phase1_seed42_cr16 \
bash scripts/train_phase1.sh 16 42

RAG_CONFIGURATION=clara_baseline \
PHASE2_OUTPUT_DIR=checkpoints/clara_seed42_cr16 \
bash scripts/train_phase2.sh 16 checkpoints/clara_phase1_seed42_cr16 42
```

Keep the Phase-II corpus/test-URL variables from the preceding block. The
`clara_baseline` must use its own all-linear Phase-I compressor checkpoint.
Phase II copies that adapter exactly into both QR and generator, freezes the
compressor/document representations, trains only QR and generator with answer
cross-entropy, and retains the ST top-5 formula even though the current training
artifact supplies five candidates. Evaluation exposes all 20 retained archive
candidates to the trained selector.

Training supports `full`, `clara_baseline`, and the four independently
retrained 16x coupling controls: `remove_cfrs`,
`uniform_acr`, `static_second_retrieval`, and `remove_all_coupling`.
`remove_cfrs` omits the fidelity reranking operation. `uniform_acr` replaces
adaptive scores with a score-independent 108-token target.
`static_second_retrieval` keeps D2=200 and union -> MADS -> CCEF but uses the
static QR/W_BGE query. `remove_all_coupling` combines those three changes. All
four use the same two-term training objective, Phase-I
initialization, Phase-II data/schedule/seeds, D1 construction and final document
ceiling as matched full.

The five retrieval-stage labels, `forward_path_off`, and the `fixed_*` labels
are inference-only and are rejected by the training CLI. `forward_path_off`
must reuse the aligned `full` checkpoint;
it disables CFRS, uses rho=1/full retention, and omits the second round. This is
the paper's 184-token fixed-checkpoint sensitivity row, not the separately
retrained 108-token `remove_all_coupling` row.

The versioned release convention for per-example uniform rounding and the
static query is
`min(1, 108 / sum(real K0))` for every real document and the original QR state
after frozen W_BGE for static D2. These fields and the effective loss weight
are serialized in `config.json`.

`scripts/train_all_cr.sh` expands the five paper compression ratios and five
independent run seeds `{42, 123, 456, 789, 2024}` for ARIA. Matched CLaRa needs
an additional all-linear Phase-I plus Phase-II run for every ratio/seed. The
launchers default to eight local worker processes, but
the current manuscript does not report a particular GPU model; record the
actual hardware used for every reproduced run.

## Paper-aligned implementation protocols

### CFRS

CFRS consumes each final passage's ACR-processed memory and the valid
non-memory states from the same compressor pass. It computes the coordinate
mean squared distance between the two sequence means. A detached reverse
min--max transform (tie fallback `0.5`, epsilon `1e-6`) is blended with CCEF at
weight `0.30` for stable ordering. CFRS is a ranking statistic, not a Phase-II
loss term.

### MTFRL

Training pools every soft-gated position with gate-mass normalization, takes a
mean within each first-pass document, and then gives each of the 1--5 documents
equal weight. Inference instead means only the hard-retained raw memory states.
The two-layer map is:

```text
P_fb(x) = W2 GELU(W1 x + b1) + b2
Mistral-7B/Llama-3-8B: 4096 -> 2048 -> 1024
Qwen-2.5-14B:          5120 -> 2560 -> 1024
```

Its weights are Xavier-uniform and biases are zero; the paper does not specify
an SVD-based initialization. The document-balanced pooled state is normalized
before projection, and the projected output is normalized for cosine search.
It performs exactly one BGE search for 200 documents; `D1 union D2` is then
reranked by MADS and CCEF. During Phase II, an identity-valued selected-document
cosine bridge supplies gradients from the two-term objective to this head.

### MADS and frozen retrieval state

MADS averages three per-query min--max-normalized axes with equal weights:
word-unigram TF--IDF cosine, cosine between normalized `W_BGE q_rep` and the
frozen BGE-large-en-v1.5 document vector, and spaCy named-entity coverage.
All-tied axes use `0.5`. It stably keeps 100 candidates before CCEF applies its
agreement multiplier, threshold `0.30`, floor `0.5`, and at-most-five survivor
rule.

The paper omits historical Hub and repository commits. Launchers persist the
resolved decoder/tokenizer and BGE commits (or a local-tree digest).
Each saved checkpoint also embeds a deterministic ZIP of the runnable Python,
shell, test, and dependency files, an ordered source-tree SHA-256, optional Git
HEAD plus dirty-state marker, and an `aria-checkpoint-v2` artifact manifest with
byte counts and SHA-256 values for every formal weight/config artifact. Strict
loading verifies these bytes before applying defaults or deserializing weights.
Do not claim bitwise paper reproducibility unless the checkpoint, source
archive, corpus/index hashes, model revisions, and dataset manifests all match.
