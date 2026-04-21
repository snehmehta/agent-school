"""
Qwen3.5-35B-A3B SFT on Sakhiya call center data.
H200 141GB — runs bf16, no quantization needed.

Data: 231 real + 275 synthetic = ~506 conversations.
Train on agent (assistant) turns only — customer turns masked.

Usage:
  uv run python train_unsloth_h200.py
  uv run python train_unsloth_h200.py model=unsloth/Qwen3-8B epochs=3
"""

import json
import os
import re
from pathlib import Path

os.environ["UNSLOTH_MOE_DISABLE_AUTOTUNE"] = "1"

import chz
import torch
from datasets import Dataset
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

load_dotenv()

# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_LEAD = """You are an outbound call center agent at Sakhiya Skin Clinic (Doctor Sakhya's Plastic Surgery Centre). You are calling a NEW LEAD — someone who submitted an inquiry online for a skin or hair service. You only know their name and the service they inquired about.

Speak: Match customer language exactly (Hindi/Gujarati). Short acknowledgments (हाँ, जी, ओके, હા, ઓકે) are natural. Warm, brief, conversational — not scripted. Greet: "गुड [morning/afternoon/evening] [सर/मैम], सखिया स्किन क्लिनिक से [name] बात कर रही/रहा हूँ" or Gujarati equivalent.

Goals: Confirm inquiry → ask location → suggest nearest branch → answer questions → offer FREE counseling → book appointment → if busy, schedule callback.

Pricing: Hair Transplant ₹60,000 unlimited grafts | GFC ₹3,500–5,000/session | Hydra Facial ₹2,400 | All counseling FREE.
Branches — Ahmedabad: Nikol, Chandkheda, Bopal, Shivranjani, Bodakdev | Surat: Infinity Tower (Railway Station) | Navsari, Ankleshwar, Anand, Palanpur, Vatva.
Hair transplant surgery only in Surat; pre/post care in Ahmedabad.

Rules: NEVER invent prior visit history. Keep responses short — phone call. Never pressure."""

SYSTEM_FOLLOWUP = """You are an outbound call center agent at Sakhiya Skin Clinic (Doctor Sakhya's Plastic Surgery Centre). You are calling an EXISTING PATIENT who missed an appointment, has pending sessions, or hasn't returned.

Speak: Match customer language exactly (Hindi/Gujarati). Short acknowledgments (हाँ, जी, ओके, હા, ઓકે) are natural. Warm, brief, conversational. Greet: "गुड [morning/afternoon/evening] [सर/मैम], सखिया स्किन क्लिनिक से [name/branch] बात कर रही/रहा हूँ".

Goals: Ask how treatment is going → listen to their situation → reschedule appointment → if not ready, schedule callback reminder.

Pricing: Hair Transplant ₹60,000 | GFC ₹3,500–5,000/session | Package renewal 25% off | All counseling FREE.
Branches — Ahmedabad: Nikol, Chandkheda, Bopal, Shivranjani, Bodakdev | Surat: Infinity Tower | Navsari, Ankleshwar, Anand, Palanpur, Vatva.

Rules: ONLY reference treatment details the customer mentions first — NEVER invent session counts, dates, or specific history. Keep responses short. Never pressure."""

_LEAD_KWS = ["inquiry", "ઇન્ક્વ", "इंक्वायरी", "inquir", "new lead", "lead"]
_GUJ_RE = re.compile(r"[\u0A80-\u0AFF]")


def _classify(msgs: list[dict]) -> str:
    text = " ".join(m["content"] for m in msgs[:6]).lower()
    return "lead" if any(k in text for k in _LEAD_KWS) else "followup"


def _detect_lang(msgs: list[dict]) -> str:
    text = " ".join(m["content"] for m in msgs[:10])
    return "Gujarati" if _GUJ_RE.search(text) else "Hindi"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_conversations(
    real_path: str,
    synth_paths: list[str],
    seed: int = 42,
) -> list[list[dict]]:
    all_convos = []

    # Real data — no system prompt, inject one
    rp = Path(real_path)
    if rp.exists():
        with rp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                msgs = [m for m in rec.get("messages", []) if m["role"] != "system"]
                if len(msgs) < 4:
                    continue
                ct = _classify(msgs)
                sys_prompt = SYSTEM_LEAD if ct == "lead" else SYSTEM_FOLLOWUP
                all_convos.append([{"role": "system", "content": sys_prompt}] + msgs)

    # Synthetic data — already has system prompt
    for sp in synth_paths:
        sp = Path(sp)
        if not sp.exists():
            continue
        with sp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                msgs = rec.get("messages", [])
                if len(msgs) < 4:
                    continue
                all_convos.append(msgs)

    import random
    random.Random(seed).shuffle(all_convos)
    return all_convos


# ── Config ─────────────────────────────────────────────────────────────────────

@chz.chz
class Config:
    model: str = "unsloth/Qwen3.5-35B-A3B"     # MoE, ~70GB bf16 — fits H200 easily
    real_data: str = "output/sakhiya-skin-clinic/training/training_data.jsonl"
    synth_data: list[str] = chz.field(default_factory=lambda: [
        "output/dataset/synthetic-20260420-173115.jsonl",
    ])
    output_dir: str = "output/checkpoints/sakhiya-qwen3-moe"
    lora_rank: int = 32
    max_seq_length: int = 4096
    batch_size: int = 2
    grad_accum: int = 4                          # effective batch = 8
    epochs: int = 3
    lr: float = 2e-4
    seed: int = 42
    push_to_hub: str | None = None               # "username/model-name" to upload
    hf_token: str | None = None


def main(cfg: Config):
    print(f"Loading model: {cfg.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=False,         # H200 141GB — bf16 all the way
        fast_inference=False,       # MoE: not supported yet
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "gate_up_proj"],
        lora_alpha=cfg.lora_rank * 2,
        use_gradient_checkpointing=True,
        random_state=cfg.seed,
        bias="none",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    convos = load_all_conversations(cfg.real_data, cfg.synth_data, cfg.seed)
    print(f"Loaded {len(convos)} conversations")

    # Apply chat template — disable thinking for call center (no <think> needed)
    def format_convo(msgs: list[dict]) -> str:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,      # Qwen3: suppress <think> block
        )

    texts = [format_convo(c) for c in convos]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset: {len(dataset)} rows | sample length: {len(texts[0])} chars")
    print("\n--- First example preview ---")
    print(texts[0][:600])

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=cfg.seed,
            output_dir=cfg.output_dir,
            save_strategy="epoch",
            report_to="none",
            max_seq_length=cfg.max_seq_length,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    # Mask customer turns — only train on agent (assistant) responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    gpu = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {gpu.name} | VRAM: {gpu.total_memory/1024**3:.1f} GB")
    print(f"Training {cfg.epochs} epochs × {len(dataset)} samples "
          f"(effective batch={cfg.batch_size * cfg.grad_accum})")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(cfg.output_dir) / "final-lora"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nLoRA adapter saved to {out}")
    print(f"Next: merge with  model.save_pretrained_merged('{cfg.output_dir}/merged-bf16', tokenizer, save_method='merged_16bit')")

    if cfg.push_to_hub:
        token = cfg.hf_token or os.environ.get("HF_TOKEN")
        model.push_to_hub_merged(
            cfg.push_to_hub, tokenizer,
            save_method="merged_16bit", token=token,
        )
        print(f"Pushed to HF: {cfg.push_to_hub}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
