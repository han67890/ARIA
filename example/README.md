# Input-schema examples

These compact examples demonstrate the raw JSONL records accepted by ARIA's
data and retrieval loaders. They provide a quick reference for field names,
record nesting, ordered candidate documents, gold positions, and stable
retrieval-corpus identifiers.

| File | Purpose |
|---|---|
| `phase1_data.jsonl` | One representative Paraphrase-category conditional-generation row. |
| `phase2_data.jsonl` | One raw QA row with five ordered candidate documents and gold positions. |
| `corpus.jsonl` | The same five stable document IDs in retrieval-corpus form. |

The corpus row IDs and the Phase-II candidate IDs are aligned, making the files
useful for checking parsing, preprocessing, and retrieval-data integration.
Phase I is not paraphrase-only: the paper protocol combines SimpleQA,
ComplexQA, Paraphrase, and Entity-Augmented sources with the exact counts in
[`docs/data.md`](../docs/data.md). The compact file illustrates the shared
document/instruction/target and provenance fields for one category.
