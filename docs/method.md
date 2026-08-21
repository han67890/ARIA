# Submission-method contract

This document describes the method in the submitted manuscript
(`ARIA_old.tex`). Version 0.2.0 aligns the executable repository contract with
that submission.

## Requirements stated by the submission

### End-to-end path

```text
QR(q) -> QCA -> AHR -> optional IGFR -> MADS(top 100) -> CCEF(top 5)
      -> soft ACR/compression -> MTFRL search(top 200)
      -> MADS(top 100) -> CCEF(top 5) -> soft ACR/compression
      -> CFRS ordering -> generator
```

CCEF supplies the fixed five-document set used by ACR and MTFRL. MTFRL makes
one additional dense retrieval, after which MADS and CCEF run once more. CFRS
changes final order, not membership.

### QCA and AHR

QCA evaluates 38 weighted surface rules. A query is Multi-Hop only when a hop
rule fires and at least two named entities are present. A query is Multi-Aspect
only when an aspect rule fires and no hop rule fires. Thus, a hop match with
fewer than two entities is Simple even if an aspect rule also fires; all other
queries are Simple as well.

QCA confidence is the matched-rule weight divided by total rule weight. AHR
uses type-conditioned `(BM25, dense)` endpoints:

| QCA type | BM25 | Dense BGE |
|---|---:|---:|
| Simple | 0.75 | 0.25 |
| Multi-Aspect | 0.30 | 0.70 |
| Multi-Hop | 0.25 | 0.75 |

Low confidence falls back toward `(0.5, 0.5)`.

### ACR

For the five CCEF scores, ACR uses the literal old-paper normalization

```text
rho_i = rho_min + (rho_max-rho_min)
        * (s_i-min_j s_j) / (max_j s_j-min_j s_j+1e-6)
```

with `rho_min=0.25`, `rho_max=1.0`. At position `t`, the memory state is
multiplied by `sigmoid(10 * (rho_i * K_i - t))`. This is the differentiable
soft mask used by the submitted method.

The effective prefix length `T_i` is obtained from the `0.5` gate threshold for
MTFRL pooling and allocation auditing. The generator continues to receive the
soft sigmoid-gated states. ACR rates are detached inputs to the mask,
preserving the submission's separation between allocation and CFRS gradient
paths.

### CFRS

CFRS measures per-document conditional reconstruction fidelity from compressed
memory. The release computes the frozen decoder's teacher-forced next-token
squared-probability error, averaged over valid passage targets. Lower error is
better. Reverse min--max-normalized fidelity is blended with CCEF as

```text
0.70 * s_fused + 0.30 * fidelity
```

The reconstruction error remains differentiable to the compressor. In the
reverse normalization, the maximum, minimum, and
`Delta = max(error)-min(error)+1e-6` statistics are detached while the local
error is not, giving the appendix derivative `d fidelity_i / d error_i =
-1/Delta`. The literal formula maps an all-tied set to zero in the forward
pass and yields the local derivative `-1/1e-6`. Final sorting is hard in the
forward pass, while the score surrogate preserves this submitted CFRS gradient
path without adding an auxiliary loss.

### MTFRL

MTFRL operates on exactly five first-pass documents. For each document it
averages the hard effective prefix `m_i[1:T_i]`, then averages the five document
means. A two-layer GELU projection maps this summary to the 1,024-wide BGE
space and performs one top-200 dense search. Its initialization is derived from
the pre-fitted `W_BGE` map, as specified by the submission.

### Training objectives

Phase I trains only the compressor adapter. It retains the four SimpleQA,
ComplexQA, Paraphrase, and Entity-Augmented target families listed in the
submission appendix, while the decoder reconstructs each held-out target from
memory tokens alone. A task instruction is not part of the decoder condition.

Phase II uses exactly

```text
L = L_QA + 0.10 * L_MSE
```

`L_MSE` is the example mean of the squared L2 distance between the mean memory
hidden state and the mean non-memory query/answer hidden state in the same QA
forward pass. The squared norm is summed over hidden coordinates; there is no
`1/d_h` coordinate normalization. The complete scalar objective contains these
two terms.

### ARIA-NoComp

ARIA-NoComp is a fixed-checkpoint evaluation diagnostic. It runs the five
retrieval stages once and concatenates the fixed top-five raw passages into the
frozen Phase-II decoder context. Compression, CFRS, ACR, MTFRL, and the second
retrieval round are bypassed, with no additional fine-tuning. The direct prompt
uses a 32,768-token ceiling with a 64-token generation reserve; overflow is
removed only from the evidence tail, preserving the standard system text and
question.

## Implementation conventions for unspecified edges

These conventions make omitted edge cases deterministic without changing the
normal five-document paper path:

- The release turns the manuscript's qualitative confidence fallback into a
  continuous interpolation: `c=0` gives `(0.5, 0.5)` and `c=1` gives the
  type-conditioned endpoint. This adds no threshold or learned parameter.
- CCEF/MTFRL fail fast when thresholding cannot supply five real documents,
  rather than silently changing the paper's `N=5` average.
- MTFRL uses `T_i=max(1,floor(rho_i*K_i))` when a short document would
  otherwise yield an empty prefix, making the manuscript's `1/T_i` mean
  well-defined while leaving the generator's soft ACR states unchanged.
- Hard final CFRS sorting uses an identity-valued permutation surrogate. It
  keeps the submitted hard order in the forward pass and carries the
  appendix's CFRS score derivative.
- ARIA-NoComp uses deterministic raw-passage concatenation with two-newline
  separators and question-preserving evidence-tail truncation.
