# Evaluation

The evaluator implements both full-corpus **Normal** and deterministic
**Oracle top-100** protocols. Paper-protocol answer scoring always reads an
explicit `aria-data --stage eval` artifact whose manifest declares one
non-empty scalar `answer` on every row. EM, CEM, and F1 all use that same
benchmark reference string.

The `eval_processed_no_pos.jsonl` members in the external CLaRa ZIPs are
candidate artifacts. Matched CLaRa joins their fingerprinted BGE top-20 `docs`
to the prepared evaluation artifact only when the complete row count and every
question match exactly; answer scoring continues to use the prepared scalar
reference. Full ARIA retrieves over the full KILT Wikipedia corpus, while the
matched CLaRa Normal control starts from the archived candidate texts.
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

The evaluator requires `EVAL_DATA_PATH`, its scalar-answer contract, and one
non-empty row-level `answer`. It implements Appendix A.35 NFKC normalization,
EM, substring CEM, and token F1 against that reference. Cross-benchmark means
are computed per checkpoint before aggregating across seeds. To compare against a separately
saved baseline, set `BASELINE_RESULTS=/path/to/baseline.json`; the CLI then runs
10,000-resample paired two-sided bootstrap tests, pooling training seeds within
each stable example identity.

Greedy decoding uses one beam, no sampling, and stops at EOS or 64 generated
tokens. Phase-II targets may contain up to 128 tokens, but this must not be
mistaken for the evaluation generation limit. Token-level F1 is the paper's
primary metric; CEM is secondary.

The Oracle-QCA endpoint is separate from Oracle top-100 retrieval. Supply
`--oracle_qca_labels /path/to/{dataset}_qca.json` with Normal retrieval, the
`full` configuration, 16x compression, and `--dataset all`. A JSON artifact maps each
`example_id` directly to `simple`, `multi_aspect`, or `multi_hop`; JSONL uses one
object per line with `example_id` and `question_type`. The evaluator intersects
the official split with that keyed annotation and evaluates the paper panel:
225 NQ, 257 HotpotQA, 84 MuSiQue, and 434 2Wiki examples. The CLI rejects
`--max_samples` and any panel whose per-benchmark matched count differs.
It first runs the ordinary 38-rule QCA, then replaces only
`QCAResult.question_type` before the first AHR candidate retrieval. Confidence,
matched rules, hop count, subquestions, and entity count remain the surface-QCA
values. Saved predictions contain both `qca_rule_type` and `qca_oracle_type`,
and result metadata stores both the external-file SHA-256 and matched-ID SHA-256.
No annotation labels are bundled with this repository.

The zero-shot sensitivity endpoint is enabled with `--qca_llm_mode qa` under
Normal/full/16x and `--dataset all`. The classifier uses the exact Mistral-7B
base revision recorded by the checkpoint with every LoRA adapter disabled. Its
versioned zero-shot prompt defines the three classes and requires
`Label: <class>` on the first line and `Rationale:` on the second. Decoding is
greedy to EOS or 64 new tokens. The parsed type alone replaces the surface-QCA
type before AHR; confidence, matched rules, entity count, hop count, and
subquestions are unchanged. Router outputs are cached by question and reused
across the five checkpoints. Predictions record the prompt version and hash,
raw output, parsed type, rationale, and measured router latency.

Use `--qca_llm_mode panel --qca_llm_labels '/path/to/{dataset}_qca.json'` for
the classification-only endpoint. It requires
the exact 225/257/84/434 primary panel, reports weighted F1 over all 1,000
queries, and saves each label-source SHA-256 and matched-ID SHA-256 separately
from the full-benchmark QA output.

When a prepared evaluation split supplies corpus-level `gold_doc_ids`, the
evaluator maps document IDs to canonical page URLs, retains the first ranked
passage per page, and records Normal Recall@5. It averages Recall only over
`Q_sup`, rows with at least one annotated support page; rows with an explicit
empty list remain in answer evaluation but not the Recall denominator. This
follows Appendix A.35: the five pages come from the first-pass retrieval
pipeline, before MTFRL's second round and CFRS. The external CLaRa archives
contain candidate-relative `pos_index` annotations in some retained members.
For CLaRa Normal, each archived candidate text must match exactly one prepared
full-corpus row; missing or ambiguous mappings stop evaluation. The hard top-5
document/page identities are measured against the same prepared corpus-level
gold pages and `Q_sup` denominator as ARIA. Archive `pos_index` is verified as
part of the external-artifact fingerprint but does not define Recall.

Oracle is enabled with `RETRIEVAL_MODE=oracle` (or
`--retrieval_mode oracle`) and requires a prepared evaluation split containing
complete corpus-level `gold_doc_ids`. For each query, the evaluator computes a
stable BGE ranking (score descending, corpus index ascending on ties), retains
the first occurrence of each page until 100 unique pages are collected,
protects gold pages already present, evicts the lowest-ranked non-gold pages
from the tail, and appends missing gold-page representatives in annotation
order. Injected positives receive no artificial rank advantage. ARIA applies
MADS/CCEF first and restricts MTFRL to the fixed pool; CLaRa applies its
trained-QR hard top-5 selector to the same 100 pages without an external
archive. Recall@1/@3/@5 use ARIA's final post-CFRS order or CLaRa's hard top-5
order. Every prediction stores the complete 100 corpus and page IDs,
injected/evicted IDs, selected IDs, and a canonical pool SHA-256.
Oracle Recall and the Normal first-pass Recall diagnostic use different
candidate-pool protocols and must not be compared as the same endpoint.

The paper's coupling table is evaluated as fixed-checkpoint inference on each
`full` checkpoint. Its runtime labels are `fixed_remove_cfrs`,
`fixed_uniform_acr`, `fixed_remove_mtfrl`, and `forward_path_off`; the last uses
full retention (rho=1) and disables the second retrieval round. The legacy
combination `--no_cfrs --no_acr --no_mtfrl` resolves to `forward_path_off`.

The independently trained `remove_cfrs`, `uniform_acr`,
`static_second_retrieval`, and `remove_all_coupling` configurations implement
additional 16x release controls with a 108-token target and static-D2 variants.
They are versioned separately and are not paper-table rows.
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
positive-index lists are independently fingerprinted for archive integrity.
Candidate texts are mapped uniquely to the full corpus, and CLaRa Recall@5 is
averaged against the prepared full-corpus gold pages over `Q_sup`.

The CLaRa protocol uses `N=5` candidates during training, `N=20` for Normal
evaluation, and hard `k=5` selection. Oracle replaces the Normal pool with the
shared `N=100` page-unique pool. Checkpoints record these sizes and both archive
ID schemes; evaluator JSON additionally records candidate/page order and
fingerprints for the exact evaluated set.

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

Every Normal CLaRa run checks all four files, byte digests, and required members
even when evaluating one benchmark. It then validates the selected archive's
ordered candidate/positive fingerprints and exact row-count/question join.
There is no implicit repository-relative fallback.

```bash
aria-evaluate \
  --model_path_template '/checkpoints/clara_seed{seed}_cr{cr}' \
  --seeds 42 123 456 789 2024 \
  --rag_configuration clara_baseline \
  --retrieval_mode normal \
  --dataset all --compression_rate 16 \
  --eval_data_path /data/aria/eval \
  --corpus_path /data/kilt_corpus.jsonl \
  --doc_embeddings '/data/{dataset}_kilt_bge.pt' \
  --clara_archive_dir /external/clara-candidates
```

CLaRa Oracle instead uses `--retrieval_mode oracle` with the same prepared
split, full corpus, and BGE artifacts, and omits `--clara_archive_dir`.

`aria-ablate --ablation clara_baseline` requires the same argument. The
counterfactual command always includes matched CLaRa and therefore always
requires it. Full ARIA and non-CLaRa ablations do not accept the argument.
