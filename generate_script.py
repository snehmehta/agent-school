"""
Step 1: Analyze real call recordings → generate comprehensive call center SOP.

Reads all training_data.jsonl, samples representative conversations,
feeds to teacher model → produces a detailed call center script covering:
- Conversation flow (lead + followup)
- Objection handling
- Rapport building
- Edge cases (bad audio, busy, wrong number, price pushback, etc.)

Usage:
  uv run python generate_script.py
  uv run python generate_script.py company=sakhiya-skin-clinic num_samples=50
  uv run python generate_script.py output=output/sop.md
"""

import json
import random
from pathlib import Path

import chz
from dotenv import load_dotenv
from openai import OpenAI

from train import OUTPUT_ROOT, _classify_call_type, _clean_messages

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEACHER_MODEL = "qwen/qwen3.6-plus"


@chz.chz
class Config:
    company: str | None = None
    num_samples: int = 60          # real convos to show teacher
    output: str = "output/sop.md"
    seed: int = 42


META_PROMPT = """You are an expert call center trainer. Below are {n} real transcribed phone calls from Sakhiya Skin Clinic — a chain of skin and hair treatment clinics in Gujarat, India.

These calls are in Hindi and Gujarati (mixed). Agents call patients for two purposes:
1. LEAD calls — new inquiries from Meta/social media ads (hair fall, hair transplant, skin treatment, laser, beard shaping, etc.)
2. FOLLOWUP calls — existing patients who missed appointments, have pending sessions, or haven't returned

Study these calls carefully. Then write a COMPREHENSIVE CALL CENTER SCRIPT / SOP (Standard Operating Procedure) that:

1. **Conversation Flow** — step-by-step how each call type should progress, with example lines in Hindi/Gujarati
2. **Opening** — how to greet, confirm identity, establish reason for call
3. **Rapport Building** — how to be warm without being scripted; use of short acknowledgments
4. **Information Gathering** — what to ask, in what order
5. **Objection Handling** — specific responses to every objection pattern you see in the data:
   - "Busy right now"
   - "Call back later"
   - "Too expensive"
   - "Already went to another clinic"
   - "Not interested"
   - "Bad timing (out of town, sick, etc.)"
   - "Wrong number / not me"
   - "Bad audio / can't hear"
   - Price negotiation patterns
6. **Closing** — how to book appointment, confirm date/time, end call
7. **Edge Cases** — how to handle: call drops, interrupted calls, customer switches language, customer is very terse (हाँ/हाँ only)
8. **Dos and Don'ts** — based on what you observe the BEST agents doing vs mistakes
9. **Sample Dialogues** — write 3-4 complete example conversations (lead Hindi, lead Gujarati, followup Hindi, followup Gujarati) that demonstrate ideal agent behavior

The SOP should be detailed enough that a NEW agent reading it could handle any situation correctly.
Include specific Hindi/Gujarati phrases that work well.

---

REAL CALL TRANSCRIPTS:

{calls}

---

Now write the comprehensive SOP:"""


def load_sample_calls(company: str | None, n: int, seed: int) -> list[dict]:
    if company:
        sources = [OUTPUT_ROOT / company / "training" / "training_data.jsonl"]
    else:
        sources = sorted(OUTPUT_ROOT.glob("*/training/training_data.jsonl"))

    all_convos = []
    for src in sources:
        if not src.exists():
            continue
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = _clean_messages(rec.get("messages", []))
                if not msgs:
                    continue
                call_type = _classify_call_type(msgs)
                all_convos.append({"messages": msgs, "call_type": call_type})

    rng = random.Random(seed)

    # Stratified sample: equal lead/followup
    leads = [c for c in all_convos if c["call_type"] == "lead"]
    followups = [c for c in all_convos if c["call_type"] == "followup"]
    half = n // 2
    sampled = rng.sample(leads, min(half, len(leads))) + rng.sample(followups, min(half, len(followups)))
    rng.shuffle(sampled)
    return sampled


def format_call(convo: dict, idx: int) -> str:
    lines = [f"\n--- Call {idx + 1} [{convo['call_type'].upper()}] ---"]
    for msg in convo["messages"]:
        tag = "Agent" if msg["role"] == "assistant" else "Customer"
        lines.append(f"{tag}: {msg['content']}")
    return "\n".join(lines)


def main(cfg: Config):
    print(f"Loading {cfg.num_samples} sample calls...")
    convos = load_sample_calls(cfg.company, cfg.num_samples, cfg.seed)
    print(f"  Loaded {len(convos)} conversations ({sum(1 for c in convos if c['call_type'] == 'lead')} lead, {sum(1 for c in convos if c['call_type'] == 'followup')} followup)")

    calls_text = "\n".join(format_call(c, i) for i, c in enumerate(convos))
    prompt = META_PROMPT.format(n=len(convos), calls=calls_text)

    print(f"Sending to teacher model ({TEACHER_MODEL})...")
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=__import__("os").environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8000,
        temperature=0.3,
        extra_headers={"X-Title": "agent-school-sop"},
    )

    sop = response.choices[0].message.content

    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sop, encoding="utf-8")
    print(f"SOP written to {out_path}  ({len(sop)} chars)")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
