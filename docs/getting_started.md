# Getting started

## Single reconciled environment

ARIA uses one requirements file for a Linux/NVIDIA release environment
reconciled around CUDA 12.1:

```bash
conda create -n aria python=3.10 -y
conda activate aria
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn==2.5.9.post1 --no-build-isolation
```

The unified Linux environment pins CUDA 12.1, PyTorch 2.3.1, DeepSpeed 0.14.0,
Transformers 4.43.3, and `vllm==0.5.3.post1`. FlashAttention is installed in the
final step with `--no-build-isolation`.

The current manuscript does not report exact library versions or a hardware
model. Treat the versions above as the repository's tested release stack, not
as manuscript hyperparameters. Full runs additionally require appropriately
sized NVIDIA accelerators, the pinned base model, fixed corpus/BGE vectors, and
prepared paper datasets. The spaCy English model and development checks are in
the same dependency file.

For a lightweight source checkout used only for CPU tests and static checks:

```bash
python -m pip install -e '.[dev]'
ruff check --select E9,F63,F7,F82 .
pytest -m 'not integration'
bash -n scripts/*.sh
```

The editable development install is not the paper training environment. Tests
marked `integration` require explicitly supplied model/data artifacts or an
accelerator and are excluded from the default CI job.

## Repository entry points

- `aria-data`: validate and materialize paper datasets.
- `aria-train`: fit either training phase.
- `aria-infer`: full two-round inference.
- `aria-evaluate`: five-checkpoint paper-protocol evaluation.
- `aria-ablate` and `aria-counterfactual`: paper analysis protocols.

The paper path uses `aria-build-bge` for the frozen BGE corpus/alignment
artifacts. MADS consumes these same document vectors; a separate MiniLM index
is not part of the method.

Matched-CLaRa evaluation additionally needs a complete external four-ZIP
candidate set supplied with `--clara_archive_dir`. These large archives are not
part of the checkout or package; see [evaluation](evaluation.md) for the exact
filenames, members, and SHA-256 values.

Continue with the [reproducibility checklist](reproducibility.md), then the
[method contract](method.md), [data](data.md), [training](training.md),
[inference](inference.md), and [evaluation](evaluation.md) guides.
