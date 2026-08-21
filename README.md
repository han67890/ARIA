<div align="center">

# ARIA

### Score- and Memory-Conditioned Retrieval and Compression for Latent-Compression Retrieval-Augmented Generation

[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.11-3776ab.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-mixed%20AML%20%2B%20Apache--2.0%20%2B%20CC--BY--NC--4.0-lightgrey.svg)](#license-and-attribution)

</div>

**Author:** Yiheng Han, University of Electronic Science and Technology of China<br>
**Code:** [github.com/han67890/ARIA](https://github.com/han67890/ARIA)

ARIA is an end-to-end latent-compression retrieval-augmented generation system.
It extends [Apple CLaRa](https://github.com/apple/ml-clara) with a five-stage
retrieval pipeline and three retrieval↔compression couplings. This repository
keeps the upstream `openrlhf` package layout for checkpoint compatibility while
adding paper-protocol data, inference, evaluation, and reproducibility tooling.

## Method

The full algorithm runs:

```text
query
  → QR + frozen W_BGE
  → QCA
  → AHR (4,000 candidates)
  → IGFR for multi-hop gaps
  → MADS (100)
  → CCEF (top 5)
  → ACR + compression
  → MTFRL dense second round (200)
  → MADS + CCEF again (top 5)
  → ACR + compression again
  → CFRS reranking
  → generator
```

The three retrieval--compression operations are:

- **CFRS (compression → retrieval):** each final passage receives a
  reconstruction-fidelity error from its compressed memory. Reverse
  min--max-normalized errors are blended into the final ordering with weight
  `0.30`; normalization statistics are detached while the local error remains
  differentiable, supplying the compressor-side gradient specified by the
  submission method.
- **ACR (retrieval → compression):** CCEF relevance sets each document's
  scale in `[0.25, 1.0]`; a sigmoid gate with `beta=10` softly masks memory
  positions. The old-paper normalization retains its literal `+1e-6`
  denominator rather than adding singleton or all-tied special cases.
- **MTFRL (compression → retrieval):** the hard effective prefixes of the
  five first-pass memories are averaged and mapped by a trainable two-layer
  GELU projection into BGE space for one 200-document second retrieval. Its
  initialization is derived from the fitted `W_BGE` projection.

Training has two phases. Phase I retains all four target families and trains
the compressor LoRA to reconstruct each held-out target from memory tokens
alone; the task instruction is not part of the decoder condition. Phase II
trains the QR, compressor, and generator LoRA adapters plus the feedback
projection:

```text
L = L_QA + 0.10 L_MSE
```

`L_MSE` is the example mean of the squared L2 distance between the memory and
non-memory mean hidden states; it is not divided by hidden width. The
language-model bases, pre-fitted `W_BGE`, BGE encoder/index, and rules remain
frozen. See
[the method contract](docs/method.md) and [training guide](docs/training.md) for
the exact equations, loss paths, pool sizes, and defaults.

## Repository layout

The layout follows the public CLaRa research repository, with modern GitHub
packaging, tests, and CI added:

```text
.
├── .github/                 # automated CI, issue and PR templates
├── docs/                    # data/training/inference/evaluation guides
├── evaluation/              # upstream-compatible evaluation entry point/data
├── example/                 # data-contract input examples
├── openrlhf/
│   ├── cli/                 # data, train, infer, evaluate, ablation CLIs
│   ├── configs/             # machine-readable QCA paper table
│   ├── datasets/            # strict stage collators and schema validation
│   ├── models/modeling_aria.py
│   ├── trainer/
│   └── utils/
├── scripts/                 # one canonical set of paper protocol launchers
├── tests/                   # deterministic CPU tests
├── LICENSE / NOTICE / ACKNOWLEDGEMENTS
├── pyproject.toml
└── requirements.txt         # reconciled Linux/CUDA 12.1 environment
```

## Installation

### Linux/NVIDIA installation

```bash
git clone https://github.com/han67890/ARIA.git
cd ARIA
conda create -n aria python=3.10 -y
conda activate aria
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

The repository has one requirements file; `pyproject.toml` separately keeps the
package metadata and bounded direct dependencies. The packaged Linux stack is
reconciled around CUDA 12.1, PyTorch 2.3.1, DeepSpeed 0.14.0, Transformers
4.43.3, and `vllm==0.5.3.post1`.
The full paper-training environment targets Python 3.10. Lightweight package
and CI checks support Python 3.10 and 3.11; Python 3.12 is not claimed as a
supported environment.

The current manuscript does not pin Python-library versions or training
hardware. These are release-engineering choices for this repository and must
not be cited as paper-reported hyperparameters.

FlashAttention itself remains a separate `--no-build-isolation` installation
step after `requirements.txt`; its prerequisite packages are in the single
requirements file.

The core package avoids eagerly importing FlashAttention. Full training
requires a Linux/NVIDIA environment sized for the selected backbone; CPU-only
development installs cover static and deterministic unit checks only.

## Data preparation

ARIA validates explicit `local:` artifacts or already-cached `hf:` datasets
against the paper data contract. Every prepared output includes a provenance
manifest.

```bash
python -m openrlhf.cli.aria_data \
  --stage phase1 \
  --output-dir data/aria \
  --test-url-source local:/data/official_test_urls.txt \
  --phase1-simpleqa-source local:/data/simpleqa.arrow \
  --phase1-complexqa-source local:/data/complexqa.arrow \
  --phase1-paraphrase-source local:/data/paraphrase.arrow \
  --phase1-entity-augmented-source local:/data/entity.arrow
```

Phase II additionally requires four exact-count benchmark pools, the fixed
epoch-view seed schedule `(42, 123, 456, 789, 2024)`, stable corpus document IDs,
and deterministic MuSiQue augmentation metadata. The complete 168,745-row
MuSiQue pool is consumed as a versioned historical derived source supplied by
the artifact owner, together with its four-family content manifest; it is not
regenerated from the 19,938 original rows. See [the data contract](docs/data.md)
for schemas and counts.
Epoch-view sampling and independent training-run initialization are distinct
controls even though the paper uses the same five numeric seeds for both.
That contract also defines the shared training/inference/evaluation corpus-field
resolution order and the whitespace-preserving `text_sha256` calculation.

## W_BGE fitting and training

Fit the frozen H→1024 alignment projection before both phases:

```bash
export BGE_FIT_QUERIES_PATH=/data/kilt_50k_queries.jsonl
export BGE_FIT_EMBEDDINGS_PATH=/data/kilt_50k_bge.pt
export BGE_PROJECTION_PATH=$PWD/artifacts/w_bge.pt
export TEST_URL_FILE=/data/official_test_urls.txt
export BASE_MODEL_REVISION=<40-character-hugging-face-commit>
bash scripts/fit_bge_projection.sh
```

The fitting command consumes a versioned artifact of 50,000 aligned KILT
query/passage pairs and their frozen BGE vectors. Its source manifest and row-order digests
are required inputs; this repository does not derive those pairs from a raw
KILT dump.

Train one independent run:

```bash
export TEST_URL_FILE=/data/official_test_urls.txt
bash scripts/train_phase1.sh 16 42

export CORPUS_PATH=/data/kilt_corpus_no_test_overlap.jsonl
export CORPUS_EMBEDDINGS_PATH=/data/kilt_bge_with_alignment_metadata.pt
export BGE_PROJECTION_PATH=$PWD/artifacts/w_bge.pt
export TEST_URL_FILE=/data/official_test_urls.txt
bash scripts/train_phase2.sh 16 checkpoints/aria_phase1_seed42_cr16 42
```

The launchers fix the submission protocol's compression ratios, epoch counts,
rank-16 LoRA, fixed top-5 evidence set, and Phase-II MSE weight.
`scripts/train_all_cr.sh`
expands ratios `{4,16,32,64,128}` and run seeds `{42,123,456,789,2024}`.

## Inference

```bash
MODEL_PATH=checkpoints/aria_phase2_full_seed42_cr16 \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS=/data/kilt_bge.pt \
bash scripts/infer.sh \
  --question "Who wrote the novel whose adaptation won the award?"
```

Full ARIA uses `(N,1024)` BGE corpus embeddings and `W_BGE`, either bundled in
the checkpoint or supplied explicitly. The same frozen BGE document vectors
serve AHR dense retrieval, MADS's semantic axis, and MTFRL retrieval; MADS does
not use a separate MiniLM encoder or index. These aligned artifacts activate
the complete two-round retrieval path. Phase-II checkpoints bind the separate,
page-URL-deduplicated training corpus/index under `aria_training_*`; Normal
inference and evaluation validate and record the full-KILT corpus/index without
requiring it to equal those training-only fingerprints.

## Evaluation and analyses

Five-seed means require five distinct trained checkpoints:

```bash
EVAL_DATA_PATH=/data/aria/eval \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS='/data/{dataset}_kilt_bge.pt' \
bash scripts/evaluate.sh 16
```

The evaluator implements Appendix A.35 EM/CEM/F1 against one explicit
benchmark reference string `answer` per row, per-checkpoint benchmark
averaging, and paired two-sided bootstrap. `EVAL_DATA_PATH` is mandatory and
must use the scalar-answer manifest contract generated by
`aria-data --stage eval`. External CLaRa ZIP members supply candidates only;
their answers never replace the prepared benchmark reference.

The paper's ARIA-NoComp diagnostic uses the five full Phase-II 16x checkpoints
with Normal retrieval:

```bash
EVAL_DATA_PATH=/data/aria/eval \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS='/data/{dataset}_kilt_bge.pt' \
RAG_CONFIGURATION=no_compression \
bash scripts/evaluate.sh 16
```

It runs QCA -> AHR -> IGFR -> MADS -> CCEF once, concatenates the fixed top-five
raw passages into the decoder context, and bypasses compression, CFRS, ACR,
MTFRL, and the second retrieval round. No additional fine-tuning is performed;
the Phase-II checkpoint remains frozen. The reported average context is about
2,950 document tokens. The implementation reserves 64 generation tokens in the
32,768-token window and, only when needed, truncates the raw evidence tail while
preserving the standard system prompt, question, document order, and
two-newline separators.

It supports full-corpus Normal and a versioned Oracle top-100 protocol. Oracle
retains the first BGE-ranked passage per canonical page URL, injects missing
gold pages at the tail in annotation order. ARIA runs MADS/CCEF and restricts
MTFRL to that page-unique pool; CLaRa applies its trained-QR hard top-5 selector
to the same pool. Each materialized pool and its insertion/eviction provenance
is saved.
The distinct Oracle-QCA analysis accepts an external keyed JSON/JSONL label
artifact through `--oracle_qca_labels`; it is restricted to Normal/full/16x,
requires `--dataset all`, evaluates the fixed 225/257/84/434 benchmark panel,
and records the rule and
reference route for every prediction. The 1,000 reference labels are not bundled.
The zero-shot QCA sensitivity analysis uses `--qca_llm_mode qa` for the complete
four-benchmark QA endpoint and
`--qca_llm_mode panel --qca_llm_labels '/data/{dataset}.json'` for the exact
225/257/84/434 classification panel. It disables all LoRA adapters, greedily
decodes the versioned two-line label/rationale prompt from the fixed Mistral-7B
base, reuses the resulting routes across seeds, and replaces only the routed
category before AHR.

```bash
aria-evaluate \
  --model_path_template '/checkpoints/aria_seed{seed}_cr{cr}' \
  --seeds 42 123 456 789 2024 \
  --dataset all --compression_rate 16 --retrieval_mode normal \
  --rag_configuration full --qca_llm_mode qa \
  --eval_data_path /data/aria/eval \
  --corpus_path /data/kilt_corpus.jsonl \
  --doc_embeddings '/data/{dataset}_kilt_bge.pt'
```
The ablation and counterfactual CLIs use the same artifact contract, map every
seed to its distinct trained checkpoint, and verify the checkpoint's recorded
training configuration. The five retrieval-stage labels (`remove_qca`,
`remove_ahr`, `remove_igfr`, `remove_mads`, and `remove_ccef`) are the paper's
fixed-checkpoint interventions and therefore reuse the aligned `full`
checkpoint for each seed. The coupling rows likewise use
`fixed_remove_cfrs`, `fixed_uniform_acr`, `fixed_remove_mtfrl`, and
`forward_path_off` on those checkpoints. The last sets rho=1 and omits the
second retrieval round. The separately trained `remove_cfrs`, `uniform_acr`,
`static_second_retrieval`, and `remove_all_coupling` configurations remain
available as additional release controls and are not paper-table rows.

```bash
bash scripts/ablation.sh --help
bash scripts/counterfactual.sh --help
```

Matched CLaRa additionally requires the four upstream-derived candidate ZIPs
in an external directory. They are deliberately excluded from both Git and
Python packages. Preserve the exact filenames and pass the directory explicitly:

```bash
EVAL_DATA_PATH=/data/aria/eval \
CLARA_ARCHIVE_DIR=/external/clara-candidates \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS='/data/{dataset}_kilt_bge.pt' \
RAG_CONFIGURATION=clara_baseline \
MODEL_PATH_TEMPLATE='/checkpoints/clara_seed{seed}_cr{cr}' \
bash scripts/evaluate.sh 16
```

For Normal retrieval, the directory must contain `nq.zip`, `hotpotqa.zip`,
`musique.zip`, and `2wiki.zip`. Every matched-CLaRa command verifies all four
ZIP byte digests and required members, then rechecks the ordered top-20
candidate fingerprints and exact question alignment. Candidate text must map
uniquely to the prepared full corpus; Recall@5 uses the prepared corpus-level
gold pages and the same `Q_sup` denominator as ARIA. CLaRa Oracle takes the
shared top-100 pool and does not use these archives. See
[the evaluation guide](docs/evaluation.md) for checksums.

## Main result

ARIA achieves 44.50 average F1 at 16× Normal retrieval on NQ, HotpotQA,
MuSiQue, and 2WikiMultiHopQA with Mistral-7B, averaged over five runs. This is
the primary token-level F1 endpoint; CEM is a secondary metric.

## Testing and reproducibility

Install the lightweight development extras and run the CPU checks before a
pull request:

```bash
python -m pip install -e '.[dev]'
ruff check --select E9,F63,F7,F82 .
pytest -m 'not integration'
bash -n scripts/*.sh
```

Tests marked `integration` require external model, corpus, or accelerator
artifacts. The repository intentionally does not redistribute model weights,
the full KILT corpus/index, or the paper's complete training pools. See the
[reproducibility checklist](docs/reproducibility.md) and
[data contract](docs/data.md) before claiming paper-protocol results. Repository
owners should also complete the [public release checklist](docs/release.md),
including its full-history and licensing review, before publishing.
Git checkouts and Python distributions omit the large CLaRa candidate ZIPs.
Matched-CLaRa Normal analyses require an external archive set in addition to
the scalar-answer prepared evaluation artifact.

## Versioning and citation

The current package version is the `0.2.0` research snapshot. Releases follow
semantic versioning for the Python/API surface; changes that alter a paper
protocol or artifact schema are also called out explicitly in
[CHANGELOG.md](CHANGELOG.md). For citation metadata, use
[CITATION.cff](CITATION.cff) or:

```bibtex
@software{han2026aria,
  author  = {Yiheng Han},
  title   = {ARIA: Score- and Memory-Conditioned Retrieval and Compression
             for Latent-Compression Retrieval-Augmented Generation},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/han67890/ARIA},
  note    = {Research code and accompanying manuscript}
}
```

Add the paper DOI to the citation record when one becomes available; no DOI is
invented in this snapshot.

## License and attribution

This project is derived from Apple `ml-clara`. Inherited files retain Apple
copyright headers and are distributed under [LICENSE](LICENSE). See
[NOTICE](NOTICE) and [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for derivation and
third-party attribution and dataset/model terms. The Apple text has SPDX
identifier `AML`; SPDX does not mark it as OSI-approved. Original ARIA additions
Copyright 2026 Yiheng Han are licensed under Apache-2.0 only where the named
copyright holder owns the relevant rights. The upstream CLaRa acknowledgement
also identifies PISCO-derived CC BY-NC 4.0 material, so the distribution must
not be represented as a single-license or unrestricted commercial package.
See [LICENSES/README.md](LICENSES/README.md) for the exact scope. ARIA is an
independent research extension of CLaRa and is not endorsed by Apple.
