"""
Sakhiya call center SFT on combined dataset (real + synthetic).
Reads output/dataset/sakhiya_combined.jsonl — format:
  {"messages": [...], "source": "real"|"synthetic", "call_type": "lead"|"followup", "language": "Hindi"|"Gujarati"}

Usage:
  python train_gpu.py
  python train_gpu.py --model unsloth/Qwen3-8B --epochs 3
"""

import argparse
import json
import os
from pathlib import Path

os.environ["UNSLOTH_MOE_DISABLE_AUTOTUNE"] = "1"

import unsloth  # must be first to apply all patches
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


def load_combined_data(data_path: str, seed: int = 42) -> list[list[dict]]:
    convos = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if len(msgs) < 4:
                continue

            # Strip existing system prompt
            non_sys = [m for m in msgs if m["role"] != "system"]

            # Inject correct system prompt based on call_type
            call_type = rec.get("call_type", "followup")
            sys_prompt = SYSTEM_LEAD if call_type == "lead" else SYSTEM_FOLLOWUP
            convos.append([{"role": "system", "content": sys_prompt}] + non_sys)

    import random
    random.Random(seed).shuffle(convos)
    print(f"Loaded {len(convos)} conversations from {data_path}")
    return convos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3.5-35B-A3B")
    parser.add_argument("--data", default="output/dataset/sakhiya_combined.jsonl")
    parser.add_argument("--output_dir", default="output/checkpoints/sakhiya-qwen3-moe")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", default=None,
                        help="HF repo e.g. snehmehta/sakhiya-qwen3-moe")
    args = parser.parse_args()

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name} | VRAM: {gpu.total_memory/1024**3:.1f} GB")
    print(f"Loading model: {args.model}")

    model, processor = FastLanguageModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "gate_up_proj"],
        lora_alpha=args.lora_rank * 2,
        use_gradient_checkpointing=True,
        random_state=args.seed,
        bias="none",
    )

    convos = load_combined_data(args.data, args.seed)

    def format_convo(msgs):
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    texts = [format_convo(c) for c in convos]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset: {len(dataset)} rows | sample length: {len(texts[0])} chars")
    print("\n--- First example preview ---")
    print(texts[0][:600])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=args.output_dir,
            save_strategy="epoch",
            report_to="none",
            max_seq_length=args.max_seq_length,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print(f"\nTraining {args.epochs} epochs × {len(dataset)} samples "
          f"(effective batch={args.batch_size * args.grad_accum})")
    trainer.train()

    out = Path(args.output_dir) / "final-lora"
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nLoRA adapter saved to {out}")

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        model.push_to_hub_merged(
            args.push_to_hub, tokenizer,
            save_method="merged_16bit", token=token,
        )
        print(f"Pushed to HF: https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
