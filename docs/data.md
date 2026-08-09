# Data and artifact contract

ARIA validates every input artifact against the paper data and provenance
contract before materialization.

## Phase I

`aria-data --stage phase1` requires four explicit local or already-cached
sources after test-page URL de-duplication:

| Source | Rows |
|---|---:|
| SimpleQA-derived | 2,000,000 |
| ComplexQA-derived | 2,000,000 |
| Paraphrase | 1,966,291 |
| Entity-augmented | 1,842,174 |
| Total | 7,808,465 |

Each row carries an independently held-out target, distinct source/target IDs,
and a canonical page URL. The validation rules enforce distinct reconstruction
targets and exact source counts.

## Phase II

The four validated pools are sampled without replacement into five
class-balanced epoch views of 38,400 rows (9,600 per benchmark). Pass the five
fixed epoch-sampling seeds through `--epoch-seeds E0 E1 E2 E3 E4`; their values
are recorded in the manifest. Epoch-sampling seeds and the five independent
training-run seeds are separate experimental controls.

| Source pool | Rows |
|---|---:|
| NQ | 58,622 |
| HotpotQA | 90,185 |
| MuSiQue (augmented) | 168,745 |
| 2WikiMultiHopQA | 167,454 |

Set every path and seed to the provenance-backed values for the experiment:

```bash
aria-data --stage phase2 \
  --output-dir data/aria \
  --test-url-source local:/data/official_test_urls.txt \
  --epoch-seeds "$EPOCH_SEED_0" "$EPOCH_SEED_1" "$EPOCH_SEED_2" \
                "$EPOCH_SEED_3" "$EPOCH_SEED_4" \
  --training-retrieval-index-sha256 "$BGE_INDEX_SHA256" \
  --musique-audit-manifest /data/musique_audit_manifest.json \
  --phase2-nq-source local:/data/nq_train.jsonl \
  --phase2-hotpotqa-source local:/data/hotpotqa_train.jsonl \
  --phase2-musique-source local:/data/musique_augmented_train.jsonl \
  --phase2-2wikimultihopqa-source local:/data/2wiki_train.jsonl
```

Phase II also requires `--training-retrieval-index-sha256 HASH`, the SHA-256 of
the page-URL-deduplicated BGE-large-en-v1.5 corpus index that produced every
source row's ordered top-5 candidates. The manifest records that training index
plus a digest of all validated candidate-ID orders. Matched CLaRa training uses
this provenance to bind each candidate artifact to the fixed training index.
The former `--normal-retrieval-index-sha256` spelling remains a CLI alias only;
the manifest and checkpoint use the unambiguous training role.

Candidate evidence includes stable corpus document IDs. Retrieval diagnostics
use `gold_doc_ids` as corpus-level identifiers across all candidate lists.
For evaluation artifacts, every source row must provide `gold_doc_ids` as an
explicit, complete full-KILT annotation independent of `docs` and `pos_index`;
an empty list explicitly marks a row outside the Recall denominator `Q_sup`.
Every evaluation row must also provide a separate, non-empty list of
benchmark-provided answer aliases (the default input field is `gold_answers`).
The scalar `answer` and this explicit list are de-duplicated into the stored
`gold_answers`; the preparer does not invent spelling, normalization, or entity
variants. Use `--phase2-<benchmark>-gold-answers-key FIELD` when an official
source uses another field name. The evaluation manifest records
`benchmark_provided_gold_aliases`, and evaluators reject older manifests or
rows lacking that list.
Document IDs are mapped through the aligned corpus's canonical `page_url`, so
multiple supporting passages from one page count once.
This matters because an annotated positive may be absent from the source
candidate list altogether; deriving the Recall denominator from candidates
would silently drop it. Candidate `pos_index` labels are retained only as a
cross-check and must be a subset of the explicit IDs. The manifest pins this
contract so older candidate-derived artifacts are rejected instead of producing
inflated Recall@5. Oracle evaluation rejects empty support lists because its
fixed candidate pool is defined to contain at least one annotated gold page.
MuSiQue entity variants must record exact entity preservation, ROUGE-L ≥ 0.4,
unchanged answer/decomposition, and the Appendix A.33 human-audit manifest.
The reported 52,107 subquestions imply only
`52,107 - 19,938 = 32,169` strict `k-1` prefixes. The repository therefore
defines `deterministic-partial-state-v2`: it retains all prefixes, enumerates
non-empty prerequisite-hop subsets, and optionally exposes one unknown
frontier question with its answer replaced by `<MISSING>`. The final-hop answer
is never known evidence. Optional states are selected by a stable,
parent-balanced round robin until the exact 70,845-row target is reached.
State IDs and the selected partition are content-addressed.

Generate the partial-query artifact with:

```bash
aria-data --stage musique-partial \
  --output-dir data/aria \
  --phase2-musique-source local:/data/musique_originals.jsonl \
  --phase2-musique-source-id-key id \
  --musique-decomposition-key question_decomposition
```

Because each generated query differs from its parent, parent top-5 candidates
must not be copied. Rows remain marked `needs_candidate_retrieval=true` until
fresh BGE top-5 candidates from the Phase-II training index are attached; the
Phase-II normalizer rejects them until that marker is explicitly set to false.
The generation manifest records the exact count, capacity, state-ID digest,
and candidate-retrieval contract.

The exact evaluation split sizes are NQ 6,489, HotpotQA 7,384, MuSiQue 2,417,
and 2WikiMultiHopQA 12,576.

Materialize the four official, alias-bearing splits before evaluation:

```bash
aria-data --stage eval \
  --output-dir /data/aria \
  --eval-split validation \
  --phase2-nq-source local:/data/nq_eval_with_aliases.jsonl \
  --phase2-hotpotqa-source local:/data/hotpotqa_eval_with_aliases.jsonl \
  --phase2-musique-source local:/data/musique_eval_with_aliases.jsonl \
  --phase2-2wikimultihopqa-source local:/data/2wiki_eval_with_aliases.jsonl
```

Only use fields supplied by the benchmark or its official evaluation release.
If such an alias-bearing source is unavailable, the repository cannot reproduce
the paper's answer metrics and deliberately stops instead of falling back to
the external CLaRa archives' scalar answers. Those four ZIPs are candidate
artifacts, not evaluation-label artifacts; they are kept outside Git and passed
with `--clara_archive_dir` only for matched-CLaRa analyses. Their filenames,
members, and pinned SHA-256 values are listed in [evaluation.md](evaluation.md).

## Retrieval corpus

Every JSON/JSONL corpus row must contain:

```json
{"id":"stable-id","text":"passage text","page_url":"https://en.wikipedia.org/wiki/..."}
```

Training, inference, and evaluation resolve corpus aliases identically. Within
each group, the first non-empty string wins, in this exact order:

- text: `text`, `passage`, `content`, `document`;
- stable ID: `id`, `doc_id`, `document_id`, `passage_id`, `_id`;
- provenance URL: `page_url`, `url`, `wikipedia_url`.

The canonical text is stripped only at its beginning and end. Each
`text_sha256` value is SHA-256 over that remaining text's exact UTF-8 bytes;
internal spaces, tabs, and line breaks are preserved. The same canonical helper
fingerprints W_BGE alignment queries/passages and the fixed corpus stored in
checkpoint metadata, keeping W_BGE artifacts, manifests, and checkpoints under
one hashing scheme.

There are deliberately two retrieval-corpus roles. Phase-II training uses a
page-URL-deduplicated corpus from which every official test page has been
removed; checkpoints store this only under `aria_training_*` provenance.
Normal evaluation and inference instead use the complete KILT Wikipedia corpus.
Their corpus and BGE-index digests are recomputed and recorded as runtime
provenance, and are never required to equal the training digests. The runtime
commands reject the exact de-duplicated training corpus/index to prevent it from
being reported as full-corpus Normal retrieval.

The BGE artifact must be a dictionary containing `doc_embeddings` with shape
`(N, 1024)`, `document_ids` in identical order, per-row `text_sha256`,
`text_sha256_scheme: utf8-strip-v1`, and
`bge_model: BAAI/bge-large-en-v1.5` provenance. It must also carry
`index_sha256`, recomputed in bounded row chunks over the canonical float32
tensor. For Phase-II training it must match the training-retrieval manifest;
for Normal evaluation/inference it is independently validated against the
complete KILT corpus row order and is recorded in the evaluation provenance.
The dictionary form binds every vector row to its document ID and text hash.
A newline-delimited official test-URL artifact defines the disjoint evaluation
URL set used during corpus validation.

MADS does not require a second semantic artifact. Its semantic axis is the
cosine between the L2-normalized projected query `W_BGE q_rep` and the same
frozen BGE-large-en-v1.5 document row used by AHR and MTFRL. Consequently the
single `(N, 1024)` BGE artifact above must be available, row-aligned, and
provenance-validated for all three components. A MiniLM matrix is not part of
the paper protocol and must not be substituted for these BGE document vectors.

Build the separate BGE dense-retrieval artifact with:

```bash
aria-build-bge corpus \
  --input /data/kilt_corpus.jsonl \
  --output /data/kilt_bge.pt \
  --revision "$BGE_REVISION" \
  --device cuda --batch-size 128
```

`--model` accepts only the paper model name (the default) or an existing local
SentenceTransformer model directory. Set `BGE_REVISION` to a branch, tag, or,
preferably, an exact Hugging Face commit supplied by the artifact owner; this
repository does not guess the unpublished training commit. For a Hub model the
command records both `encoder_revision_declared` and the exact
`encoder_revision_resolved`, and refuses to publish when the loaded commit
cannot be proved. If `--revision` is omitted, the declared value is `main` and
the exact resolved commit is still recorded. For a local directory, revision
fields remain null and `encoder_source_sha256` fingerprints every file instead.

These fields are additive to the older artifact contract, so existing v1
artifacts remain loadable. New artifacts declare `aria-bge-artifact-v2`. The
`bge_model` field identifies the required model family, while `encoder_source`
identifies the actual Hub repository or local directory; they are intentionally
different metadata fields, not duplicate dictionary keys. The command
L2-normalizes and validates all 1,024-dimensional rows, computes the
loader-compatible `index_sha256`, publishes the result atomically, and refuses
to overwrite an existing output.

The paper's 50,000 W_BGE alignment pairs are fitted separately with
`scripts/fit_bge_projection.sh`. Each pair record must contain `question`,
`passage`, a stable `passage_id`, and `page_url`; the target embedding artifact
must carry matching IDs and passage hashes, declare both
`bge_model: BAAI/bge-large-en-v1.5` and
`text_sha256_scheme: utf8-strip-v1`, and include an `index_sha256` recomputed
over its canonical float32 tensor. Alignment URLs are disjoint from the
official test URLs. The saved artifact records the base model, BGE model,
optimizer protocol, seed, and query fingerprint.

Generate the aligned passage targets before fitting W_BGE with:

```bash
aria-build-bge alignment-targets \
  --input /data/kilt_alignment_50k.json \
  --output /data/kilt_alignment_targets.pt \
  --revision "$BGE_REVISION" \
  --device cuda --batch-size 128
```

The paper default enforces exactly 50,000 rows. The target artifact carries
passage IDs, text hashes, canonical page URLs, aggregate query/passage/source
fingerprints, model provenance, and the same canonical float32 index digest.
