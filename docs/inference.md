# Inference

Full ARIA inference requires a Phase-II checkpoint, the fixed corpus, aligned
BGE embeddings, and either a bundled or explicit W_BGE artifact:

```bash
MODEL_PATH=checkpoints/aria_phase2_full_seed42_cr16 \
CORPUS_PATH=/data/kilt_corpus.jsonl \
DOC_EMBEDDINGS=/data/kilt_bge_aligned.pt \
bash scripts/infer.sh --question "Who wrote the novel whose film won the award?"
```

The row-aligned BGE artifact is validated against corpus IDs, text hashes,
model revision, normalization, and tensor digest before the pipeline is
attached. Its document vectors serve all three paper-defined consumers: AHR's
dense channel, MADS's semantic score, and MTFRL's second search. No MiniLM model
or second semantic index is required.

The public `generate_from_questions` path performs QR → QCA → AHR → optional
IGFR → MADS/CCEF → first ACR/compression → MTFRL D2 → union re-scoring → second
ACR/compression → CFRS → generator. The aligned dense corpus artifacts activate
both retrieval rounds and the complete ARIA scoring path.

Generation is greedy with one beam and no sampling, and stops at EOS or 64 new
tokens. The query is truncated to at most 256 tokens and each passage to at
most 768 tokens before compression. These evaluation limits differ from the
Phase-II training target maximum of 128 tokens.

Oracle evaluation uses the same canonical generator with a fixed pool: QCA is
retained, the first 100 page-unique BGE results plus deterministic missing-gold-
page tail injection replace AHR/IGFR, and MTFRL is constrained to that same
pool. See
[evaluation.md](evaluation.md) for ordering, provenance, and Recall scope.
