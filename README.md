# Sakhiya Call Center SFT — Qwen3.5-35B-A3B MoE

Fine-tuning Qwen3.5-35B-A3B (MoE) on Sakhiya Skin Clinic call center conversations using Unsloth + TRL.

## Hardware

H200 141 GB VRAM · CUDA 12.4 · bf16 · no quantization · LoRA rank 16

## Setup

### 1. Prerequisites

- Python 3.11 (use [pyenv](https://github.com/pyenv/pyenv) or the `.python-version` file)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and fill in your real keys:
#   HF_TOKEN / HUGGINGFACE_API_KEY  — HuggingFace token (read access for gated models)
#   OPENROUTER_API_KEY              — OpenRouter key (used by eval_judge.py)
```

### 3. Install dependencies

```bash
uv sync
```

This installs exact pinned versions from `uv.lock`, including Unsloth and Unsloth-Zoo from git.

### 4. Train

```bash
TORCH_LIB=.venv/lib/python3.11/site-packages/torch/lib
LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH uv run python train_gpu.py
```

The `LD_LIBRARY_PATH` prefix is required to avoid an NCCL symbol conflict (`ncclDevCommDestroy`) between torch's bundled libs and system libs.

**Key arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `unsloth/Qwen3.5-35B-A3B` | Base model |
| `--epochs` | `3` | Training epochs |
| `--lora_rank` | `16` | LoRA rank |
| `--push_to_hub` | `None` | HF repo to push merged weights (e.g. `yourname/sakhiya-qwen3-moe`) |

### 5. Evaluate

```bash
uv run python eval.py          # run inference on test set
uv run python eval_judge.py    # judge outputs via OpenRouter
```

## Project structure

```
train_gpu.py              # main SFT training script
eval.py                   # inference / evaluation
eval_judge.py             # LLM-as-judge scoring via OpenRouter
pyproject.toml            # pinned dependencies
uv.lock                   # exact lock file — reproducible installs
.env.example              # copy to .env and fill in keys
output/
  dataset/                # generated dataset (gitignored)
  checkpoints/            # model checkpoints (gitignored)
```

## Dependency notes

All versions are pinned to the [official Unsloth Qwen3.5-MoE notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_MoE.ipynb). Do not upgrade without testing — mismatched `triton`/`torch`/`unsloth` versions break training silently.
