"""
Phase 3: Export LoRA adapter to PEFT format → push to HF Hub → serve with vLLM.

No local base-model merge needed. Adapter is ~few hundred MB.
vLLM loads base model + adapter dynamically at serve time.

Steps:
  1. Download sampler weights from Tinker checkpoint (adapter only, ~few hundred MB)
  2. Convert to PEFT format (compatible with vLLM --lora-modules)
  3. (Optional) Push PEFT adapter to HuggingFace Hub

Usage:
  uv run python merge.py
  uv run python merge.py checkpoint=tinker://... hf_repo=myorg/sakhiya-agent
  uv run python merge.py skip_upload=True   # local only, no HF push

Serve:
  vllm serve Qwen/Qwen3.5-35B-A3B --enable-lora \\
    --lora-modules sakhiya=./output/peft/sakhiya-35B-r32
"""

import chz
from dotenv import load_dotenv
from pathlib import Path

from tinker_cookbook import weights

load_dotenv()

CHECKPOINT = "tinker://3ed8da41-fc3e-5835-a8b6-cbed2b5d222a:train:0/sampler_weights/final"
BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"


@chz.chz
class Config:
    checkpoint: str = CHECKPOINT
    base_model: str = BASE_MODEL
    output_path: str = "output/peft/sakhiya-35B-r32"
    hf_repo: str | None = None   # e.g. "miraiminds/sakhiya-agent-35B"
    hf_private: bool = True
    skip_upload: bool = False


def main(cfg: Config):
    output_path = Path(cfg.output_path)
    adapter_dir = str(output_path.parent / (output_path.stem + "-tinker-adapter"))

    print(f"[1/3] Downloading adapter from {cfg.checkpoint}")
    weights.download(
        tinker_path=cfg.checkpoint,
        output_dir=adapter_dir,
    )
    print(f"      Saved to {adapter_dir}")

    print(f"[2/3] Converting to PEFT format → {cfg.output_path}")
    weights.build_lora_adapter(
        base_model=cfg.base_model,
        adapter_path=adapter_dir,
        output_path=cfg.output_path,
    )
    print("      Done.")

    if not cfg.skip_upload and cfg.hf_repo:
        print(f"[3/3] Pushing to HuggingFace Hub → {cfg.hf_repo}")
        url = weights.publish_to_hf_hub(
            model_path=cfg.output_path,
            repo_id=cfg.hf_repo,
            private=cfg.hf_private,
        )
        print(f"      Published: {url}")
    else:
        print("[3/3] Skipping HF upload (pass hf_repo=myorg/mymodel to upload)")

    print()
    print("Serve with vLLM (on GPU machine):")
    print(f"  vllm serve {cfg.base_model} --trust-remote-code --enable-lora \\")
    print(f"    --lora-modules sakhiya={cfg.output_path}")
    print()
    print("Or if pushed to HF Hub:")
    print(f"  vllm serve {cfg.base_model} --trust-remote-code --enable-lora \\")
    print(f"    --lora-modules sakhiya={cfg.hf_repo or '<hf_repo>'}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
