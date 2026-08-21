# Evaluation

The evaluator implements both full-corpus **Normal** and deterministic
**Oracle top-100** protocols. Paper-protocol answer scoring always reads an
explicit `aria-data --stage eval` artifact whose manifest declares a non-empty
`gold_answers` list of benchmark-provided aliases on every row. EM, CEM, and F1
take their independent maximum over that list, as required by Appendix A.35.

The `eval_processed_no_pos.jsonl` members in the external CLaRa candidate ZIPs
have only scalar `answer` values; neither those members nor their `with_pos`
companions contain benchmark alias annotations. They are therefore **candidate
artifacts only**, never a gold-answer source. Matched CLaRa joins
their fingerprinted BGE top-20 `docs` to the explicit evaluation artifact only
when the complete row count and every question match exactly. No aliases are
generated, normalized into new variants, or inferred from candidate text.
Full ARIA retrieves over the full KILT Wikipedia corpus, while the matched
CLaRa control consumes the external archived `docs` field.
Corpus fields and embedding hashes must follow the same
[data and artifact contract](data.md) used during training and inference.

Prepared evaluation artifacts use the same full-corpus retrieval rule:

```bash
EVAL_DATA_PATH=/data/aria/eval \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS='/data/{dataset}_kilt_bge.pt' \
bash scripts/evaluate.sh 16
```

The validated `(N, 1024)` BGE artifact is shared by AHR, the semantic axis of
MADS, and MTFRL. There is no separate MiniLM MADS artifact in the paper
protocol. See [the data contract](data.md) for its row-order, hash, model, and
revision requirements.

One checkpoint is one run. Five-seed statistics require five independently
trained checkpoints:

```bash
EVAL_DATA_PATH=/data/aria/eval \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS='/data/{dataset}_kilt_bge.pt' \
bash scripts/evaluate.sh 16
```

The evaluator fails closed if `EVAL_DATA_PATH`, its alias contract, or any
row-level alias list is missing. This is intentional: scalar-answer runs are
not labeled as paper results. It implements Appendix A.35 NFKC normalization,
multi-gold best matching, EM, substring CEM, and token F1. Cross-benchmark means are computed
per checkpoint before aggregating across seeds. To compare against a separately
saved baseline, set `BASELINE_RESULTS=/path/to/baseline.json`; the CLI then runs
10,000-resample paired two-sided bootstrap tests, pooling training seeds within
each stable example identity.

Greedy decoding uses one beam, no sampling, and stops at EOS or 64 generated
tokens. Phase-II targets may contain up to 128 tokens, but this must not be
mistaken for the evaluation generation limit. Token-level F1 is the paper's
primary metric; CEM is secondary.

When a prepared evaluation split supplies corpus-level `gold_doc_ids`, the
evaluator maps document IDs to canonical page URLs, retains the first ranked
passage per page, and records Normal Recall@5. It averages Recall only over
`Q_sup`, rows with at least one annotated support page; rows with an explicit
empty list remain in answer evaluation but not the Recall denominator. This
follows Appendix A.35: the five pages come from the first-pass retrieval
pipeline, before MTFRL's second round and CFRS. The external CLaRa archives
contain candidate-relative `pos_index`
annotations in some retained members, but lack positive labels aligned to
stable full-KILT corpus IDs. Their output therefore omits Recall@5 rather than
guessing an ID mapping.

Oracle is enabled with `RETRIEVAL_MODE=oracle` (or
`--retrieval_mode oracle`) and requires a prepared evaluation split containing
complete corpus-level `gold_doc_ids`. For each query, the evaluator computes a
stable BGE ranking (score descending, corpus index ascending on ties), retains
the first occurrence of each page until 100 unique pages are collected,
protects gold pages already present, evicts the lowest-ranked non-gold pages
from the tail, and appends missing gold-page representatives in annotation
order. Injected positives
receive no artificial rank advantage: MADS is the first stage that ranks the
fixed pool. Oracle replaces AHR/IGFR candidate acquisition, while QCA,
MADS/CCEF, ACR, compression, and CFRS retain their usual roles. MTFRL D2 is
restricted to the same page-unique pool.
Recall@1/@3/@5 are measured on the final post-CFRS order. Every prediction stores the
complete 100 corpus and page IDs, injected/evicted IDs, and a canonical pool SHA-256.
Oracle Recall and the Normal first-pass Recall diagnostic use different
candidate-pool protocols and must not be compared as the same endpoint.

Do not merge the two Appendix-A.31 protocols. The matched-retraining rows
`remove_cfrs`, `uniform_acr`, `static_second_retrieval`, and
`remove_all_coupling` require distinct checkpoints trained at 16x. They retain
two retrieval rounds, D2=200, union -> MADS -> CCEF, and the target 108-token
mean budget. `forward_path_off` is the complementary fixed-checkpoint
intervention: it validates against `full`, uses full retention (rho=1), skips
the second round, and has the reported 184-token mean. The legacy evaluator
combination `--no_cfrs --no_acr --no_mtfrl` resolves to
`forward_path_off`; use `--rag_configuration remove_all_coupling` for the
separately retrained all-off row.

The paper does not uniquely specify matched-uniform integer rounding or the
static-query construction. The release convention applies the same
score-independent ratio `min(1, 108 / sum(real K0))` to every real document,
then uses the standard soft training gate/hard inference threshold. Static D2
uses the original QR representation after frozen W_BGE. Checkpoints serialize
the convention names, target budget, allocation mode, and second-retrieval mode.
The counterfactual CLI still needs only the full-ARIA and independently trained
CLaRa checkpoint sets; its deprecated no-coupling argument is accepted only
when it repeats the aligned full paths. See `aria-ablate --help` and
`aria-counterfactual --help` for complete artifact arguments.

For the matched CLaRa control, the evaluator verifies and consumes the complete
BGE top-20 list retained in each upstream evaluation archive. The trained QR
adapter applies the hard-forward/soft-backward selector and sends its hard top-5
identities to the fixed compressor representations and generator. Each document
uses `max(1, floor(L_i/r))` memory states with masked pooling. Full
ARIA coupling controls instead retrieve from the explicitly aligned full KILT
corpus. The four complete-list archive digests are pinned in code and
verified before selection. Archive-local document IDs are SHA-256 hashes of the
canonical text; page IDs are hashes of the canonical title header. The retained
positive-index lists are independently fingerprinted, and CLaRa first-pass
Recall@5 is averaged only over rows with at least one positive page (`Q_sup`).

The top-20 candidate pool is an explicit **release convention**, not a claim
that the manuscript uniquely fixes `N=20`: Appendix A.37 specifies ST top-k
selection but does not uniquely state the preselection pool size. Checkpoints
record the training `N=5`, evaluation `N=20`, hard `k=5`, and both archive ID
schemes; evaluator JSON additionally records candidate/page order and positive
index fingerprints for the exact evaluated subset.

## External CLaRa archive set

The candidate ZIPs are not stored in Git and are not included in source or
wheel distributions. Obtain them only from a source whose redistribution and
dataset terms you are authorized to use, keep them outside the checkout, and
pass their common parent directory as `--clara_archive_dir` (or
`CLARA_ARCHIVE_DIR` in the shell wrappers). The directory is complete only when
these exact files pass SHA-256 verification:

| Benchmark | Filename | Required member | ZIP SHA-256 |
|---|---|---|---|
| NQ | `nq.zip` | `nq/eval_processed_no_pos.jsonl` | `7d26d5c29694cd81cccfcac4fd29c16ae7f245b4c554623cbe3c6ec8c3a0ad41` |
| HotpotQA | `hotpotqa.zip` | `hotpotqa/eval_processed_no_pos.jsonl` | `f46d7cfc23199f6cdff5e3ce1872ff150e6d940eb83b343cf37431cd740fa4db` |
| MuSiQue | `musique.zip` | `musique/eval_processed_no_pos.jsonl` | `85a55afc5c6067d00eef1888e13b598039a515f787791b23fbb495c35827e264` |
| 2WikiMultiHopQA | `2wiki.zip` | `2wiki/eval_processed_no_pos.jsonl` | `39cf44bcfa24938c40617ef5bba90235642bf02f537297f4055b3f6bc756846c` |

After ZIP verification, the loader independently hashes the ordered top-20
`docs` rows and ordered `pos_index` rows (canonical compact JSON plus newline):

| Benchmark | Candidate rows SHA-256 | Positive-index rows SHA-256 |
|---|---|---|
| NQ | `2e6126e5e7ab59401a0870a256d54dc7e188d97a5a8d3b9e53e14457001793f2` | `3fa5b5c2bee7bca30936d3d6f8704cd095e45143f8d4a947a6dc036bef0ecfe1` |
| HotpotQA | `6a8747f07642b438effb601c9f8e20fb5659eb21fe80bc3b8205df6646907776` | `3702710243f744979c8b3f3f2a59db724ec5e79267337b947c08089752a648b0` |
| MuSiQue | `d4b89b500dc3c6c7c324f4992a56c1c02e960ae06da08a7dad390d42aa17f136` | `02d3522d942e844c01dda8716892b24412c0793ef1fdd1814c365a8ec6bd1f35` |
| 2WikiMultiHopQA | `b8f6023a4df21bcdb36f5c6bd299e1bfc5bcb0ed1906de528fb784a138f2962f` | `e5bab2ed3d54d188fe3d91a4881c5d641b35ac25fbcfbdc36ab62458dfa2eba5` |

Every CLaRa run checks all four files, byte digests, and required members even
when evaluating one benchmark. It then validates the selected archive's
ordered candidate/positive fingerprints and exact row-count/question join.
There is no implicit repository-relative fallback.

```bash
aria-evaluate \
  --model_path_template '/checkpoints/clara_seed{seed}_cr{cr}' \
  --seeds 42 123 456 789 2024 \
  --rag_configuration clara_baseline \
  --dataset all --compression_rate 16 \
  --eval_data_path /data/aria/eval \
  --clara_archive_dir /external/clara-candidates
```

`aria-ablate --ablation clara_baseline` requires the same argument. The
counterfactual command always includes matched CLaRa and therefore always
requires it. Full ARIA and non-CLaRa ablations do not accept the argument.
