"""
Step 2: Use teacher model + SOP → generate synthetic multi-turn conversations.

Each conversation includes the appropriate system prompt (lead or followup)
so the student learns to internalize the right behavior per call type.

Usage:
  uv run python generate_data.py                        # 2000 convos
  uv run python generate_data.py num=500 call_type=lead
  uv run python generate_data.py sop=output/sop.md num=100 dry_run=True
"""

import asyncio
import json
import random
import re
from pathlib import Path

import chz
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

REAL_CALLS_PATH = Path("output/sakhiya-skin-clinic/training/training_data.jsonl")

# Gujarati block unicode range U+0A80–U+0AFF
_GUJ_RE = re.compile(r"[\u0A80-\u0AFF]")

# Keywords that strongly indicate a LEAD call (new inquiry)
_LEAD_KEYWORDS = ["inquiry", "inquir", "ઇન્ક્વાયરી", "ઇન્ક્વ", "inquiry", "इंक्वायरी",
                  "new lead", "lead call", "first time", "पहली बार"]


def _detect_language(msgs: list[dict]) -> str:
    text = " ".join(m["content"] for m in msgs[:12])
    return "Gujarati" if _GUJ_RE.search(text) else "Hindi"


def _classify_lead_or_followup(msgs: list[dict]) -> str:
    text = " ".join(m["content"] for m in msgs[:6]).lower()
    if any(kw in text for kw in ["inquiry", "ઇન્ક્વ", "इंक्वायरी", "inquir"]):
        return "lead"
    # agent asks "how is treatment going" → followup
    if any(kw in text for kw in ["ટ્રીટ", "treatment", "appointment", "session", "ट्रीट"]):
        return "followup"
    return "lead"


def _format_real_call(msgs: list[dict], max_turns: int = 30) -> str:
    lines = []
    for m in msgs[:max_turns]:
        tag = "Agent" if m["role"] == "assistant" else "Customer"
        lines.append(f"{tag}: {m['content']}")
    return "\n".join(lines)


class RealCallBank:
    """Stratified index of real calls by (call_type, language)."""

    def __init__(self, path: Path, rng: random.Random):
        self._bank: dict[tuple[str, str], list[list[dict]]] = {}
        self._rng = rng
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msgs = [m for m in rec.get("messages", []) if m["role"] != "system"]
                if len(msgs) < 6:
                    continue
                lang = _detect_language(msgs)
                ct = _classify_lead_or_followup(msgs)
                key = (ct, lang)
                self._bank.setdefault(key, []).append(msgs)
        total = sum(len(v) for v in self._bank.values())
        print(f"RealCallBank: {total} calls indexed — {dict((k, len(v)) for k, v in self._bank.items())}")

    def sample(self, call_type: str, language: str, n: int = 2) -> list[list[dict]]:
        key = (call_type, language)
        pool = self._bank.get(key, [])
        if not pool:
            # fallback: any language with matching call_type
            pool = []
            for (ct, _), calls in self._bank.items():
                if ct == call_type:
                    pool.extend(calls)
        if not pool:
            return []
        return self._rng.sample(pool, min(n, len(pool)))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEACHER_MODEL = "qwen/qwen3-235b-a22b"

# Per-type system prompts injected into each generated conversation
SYSTEM_LEAD = """You are an outbound call center agent at Sakhiya Skin Clinic (Doctor Sakhya's Plastic Surgery Centre). You are calling a NEW LEAD — someone who submitted an inquiry online for a skin or hair service. You only know their name and the service they inquired about.

Speak: Match customer language exactly (Hindi/Gujarati). Short acknowledgments (हाँ, जी, ओके, હા, ઓકે) are natural. Warm, brief, conversational — not scripted. Greet: "गुड [morning/afternoon/evening] [सर/मैम], सखिया स्किन क्लिनिक से [name] बात कर रही/रहा हूँ" or Gujarati equivalent.

Goals: Confirm inquiry → ask location → suggest nearest branch → answer questions → offer FREE counseling → book appointment → if busy, schedule callback.

Pricing: Hair Transplant ₹60,000 unlimited grafts | GFC ₹3,500–5,000/session | Hydra Facial ₹2,400 | All counseling FREE.
Branches — Ahmedabad: Nikol, Chandkheda, Bopal, Shivranjani, Bodakdev | Surat: Infinity Tower (Railway Station) | Navsari, Ankleshwar, Anand, Palanpur, Vatva.
Hair transplant surgery only in Surat; pre/post care in Ahmedabad.

Rules: NEVER invent prior visit history. Keep responses short — phone call. Never pressure."""

SYSTEM_FOLLOWUP = """You are an outbound call center agent at Sakhiya Skin Clinic (Doctor Sakhya's Plastic Surgery Centre). You are calling an EXISTING PATIENT who missed an appointment, has pending sessions, or hasn't returned.

Speak: Match customer language exactly (Hindi/Gujarati). Short acknowledgments (हाँ, जी, ओके, হा, ઓકে) are natural. Warm, brief, conversational. Greet: "गुड [morning/afternoon/evening] [सर/मैम], सखिया स्किन क्लिनिक से [name/branch] बात कर रही/रहा हूँ".

Goals: Ask how treatment is going → listen to their situation → reschedule appointment → if not ready, schedule callback reminder.

Pricing: Hair Transplant ₹60,000 | GFC ₹3,500–5,000/session | Package renewal 25% off | All counseling FREE.
Branches — Ahmedabad: Nikol, Chandkheda, Bopal, Shivranjani, Bodakdev | Surat: Infinity Tower | Navsari, Ankleshwar, Anand, Palanpur, Vatva.

Rules: ONLY reference treatment details the customer mentions first — NEVER invent session counts, dates, or specific history. Keep responses short. Never pressure."""

# ── Scenario seeds ────────────────────────────────────────────────────────────

LEAD_SCENARIOS = [
    # service × language × customer_mood
    ("hair transplant", "Hindi", "interested but price-sensitive"),
    ("hair transplant", "Gujarati", "interested, asks about process"),
    ("hair transplant", "Hindi", "busy, asks to call back"),
    ("hair transplant", "Gujarati", "skeptical, visited another clinic"),
    ("hair fall / GFC treatment", "Hindi", "interested, wants to know sessions"),
    ("hair fall / GFC treatment", "Gujarati", "interested, far from city"),
    ("hair fall / GFC treatment", "Hindi", "very terse (हाँ हाँ only)"),
    ("skin treatment / acne", "Hindi", "interested, asks about cost"),
    ("skin treatment / acne", "Gujarati", "interested, asks about process"),
    ("laser hair removal", "Hindi", "price pushback"),
    ("laser hair removal", "Gujarati", "interested, asks number of sessions"),
    ("beard shaping", "Hindi", "interested, near Ahmedabad"),
    ("beard shaping", "Gujarati", "interested, from Surat"),
    ("Hydra Facial", "Hindi", "interested, asks about offers"),
    ("Hydra Facial", "Gujarati", "busy, will call back"),
    ("hair transplant", "Hindi", "wrong number / not the right person"),
    ("hair fall / GFC treatment", "Gujarati", "bad audio, call drops midway"),
    ("skin treatment", "Hindi", "customer in another city (Rajkot)"),
    ("hair transplant", "Gujarati", "customer from Palanpur"),
    ("hair transplant", "Hindi", "customer says already treated elsewhere"),
]

FOLLOWUP_SCENARIOS = [
    ("missed appointment - hair treatment", "Hindi", "apologizes, wants to reschedule"),
    ("missed appointment - skin treatment", "Gujarati", "was out of town"),
    ("pending GFC sessions", "Hindi", "forgot, wants to come soon"),
    ("pending GFC sessions", "Gujarati", "busy with work"),
    ("package renewal - beard", "Hindi", "interested in renewing"),
    ("package renewal - skin", "Gujarati", "asks about discount"),
    ("long gap since last visit", "Hindi", "had personal issues"),
    ("long gap since last visit", "Gujarati", "moved to another city temporarily"),
    ("post-treatment followup", "Hindi", "reports side effect (breakout)"),
    ("post-treatment followup", "Gujarati", "satisfied, wants next session"),
    ("missed appointment", "Hindi", "customer is sick"),
    ("missed appointment", "Gujarati", "bad audio throughout"),
    ("pending sessions - hair transplant", "Hindi", "scared about procedure"),
    ("pending sessions - laser", "Gujarati", "wants to know remaining sessions"),
    ("package renewal", "Hindi", "price negotiation"),
    ("missed appointment", "Gujarati", "very terse (ઠીક છે, ઠીક)"),
    ("followup after treatment", "Hindi", "unhappy with results"),
    ("rescheduling", "Gujarati", "keeps postponing"),
    ("post-treatment", "Hindi", "asks about cream / aftercare"),
    ("pending sessions", "Gujarati", "has not come in 3+ months"),
]

# ── Prompt template ────────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are simulating a real phone call between a call center agent from Sakhiya Skin Clinic and a customer.

CALL CENTER SOP (Standard Operating Procedure):
{sop}

---

TASK: Generate a realistic multi-turn phone conversation.

Call type: {call_type}
Scenario: {scenario}
Language: {language}
Customer mood/style: {mood}

NAMES — pick real Indian names (NEVER write [Name], [मेडम], [ટ્રીટ્મેન્ટ] or any placeholder):
- Agent names: Priya, Darshan, Komal, Riya, Kinjal, Bhavesh, Neha, Mansi
- Customer names: Rajesh, Sanjay, Prashant, Amit, Kavita, Meena, Jignesh, Payal, Rahul, Deepak, Hetal, Nidhi

FORMAT — write ONLY the dialogue, no narration, no stage directions:
Agent: [text]
Customer: [text]
Agent: [text]
...

Start with Customer answering the phone (e.g. "Haan?", "Hello?", "Ji?", "Hanji?").

LANGUAGE: {language}
Agents naturally code-switch for medical/English terms (FUE, grafts, sessions, etc.) even in Hindi/Gujarati speech.
Include natural fillers: हाँ, जी, ओके, हाँ हाँ, बिल्कुल / ઠીક, ઓકે, હા, હા હા, બિલ્કુલ

TURN COUNT:
- LEAD calls: 16–22 turns (customers ask MANY questions — technique, shaving, pain, results, cost breakdown, alternatives)
- FOLLOWUP calls: 8–14 turns (shorter, focused on rescheduling or status check)

LEAD CALL — agent must demonstrate REAL SALES TECHNIQUES:
1. Technical education: Explain FUE technique, graft survival, donor area, Sapphire vs standard FUE — educate don't just answer
2. Myth-busting: Correct misconceptions ("Turkish technique is just FUE with a different name", "pain is minimal with local anesthesia")
3. Social proof: "हमारे 22 साल का अनुभव है", "हजारों patients कर चुके हैं", specific success stories by treatment type
4. Objection handling WITH information: Never just say "aao milte hai" — first address the concern fully, THEN invite
5. Competitor handling: Acknowledge other clinics exist, explain Sakhiya's differentiator (technology, experience, unlimited grafts pricing)
6. Expectation management: Timeline, realistic results, what post-care looks like
7. WhatsApp offer: For IP/unknown number calls, offer to send clinic details on WhatsApp

FOLLOWUP CALL — rules:
- ONLY reference treatment details the customer mentions first
- NEVER invent session counts, dates, or specific history
- Ask how treatment is going, listen, then offer to reschedule

PRICING (use ONLY these, no invention):
- Hair Transplant: ₹60,000 unlimited grafts (Advanced FUE) | Sapphire FUE: ~₹50/graft
- GFC: ₹3,500–5,000/session | Package renewal: 25% off
- Hydra Facial: ₹2,400 | All counseling: FREE
- Dr. Sakhiya consultation: ₹2,000 (credited toward treatment)

STRICT RULES:
- NEVER use placeholders — always use real names
- No emojis
- Keep individual turns SHORT (1-3 sentences max) — this is a phone call
- End naturally: appointment booked, callback scheduled, or polite close
- Do NOT include any preamble — output only the raw dialogue

---

REAL CALL EXAMPLES (study tone, depth, language-switching, sales technique — then generate something NEW):

{real_examples}

---

Now generate a NEW conversation following all rules above. Do NOT copy the examples — create a fresh scenario:"""


@chz.chz
class Config:
    sop: str = "output/sop.md"
    output: str = "output/dataset/synthetic"   # will add timestamp + .jsonl
    num: int = 2000
    call_type: str | None = None   # 'lead' or 'followup' or None for both
    concurrency: int = 20
    max_tokens: int = 1400
    temperature: float = 0.85      # high diversity
    seed: int = 42
    dry_run: bool = False          # print first prompt, don't generate


def build_scenarios(call_type_filter: str | None, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    lead = [{"call_type": "lead", "scenario": s, "language": l, "mood": m}
            for s, l, m in LEAD_SCENARIOS]
    followup = [{"call_type": "followup", "scenario": s, "language": l, "mood": m}
                for s, l, m in FOLLOWUP_SCENARIOS]

    pool = []
    if call_type_filter in (None, "lead"):
        pool += lead
    if call_type_filter in (None, "followup"):
        pool += followup

    # Repeat pool with random variation until we have n scenarios
    scenarios = []
    while len(scenarios) < n:
        batch = rng.sample(pool, min(len(pool), n - len(scenarios)))
        scenarios.extend(batch)
    return scenarios[:n]


def parse_conversation(text: str) -> list[dict] | None:
    messages = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("Agent:"):
            content = line[len("Agent:"):].strip()
            if content:
                messages.append({"role": "assistant", "content": content})
        elif line.startswith("Customer:"):
            content = line[len("Customer:"):].strip()
            if content:
                messages.append({"role": "user", "content": content})

    # Validate: must start with user, end with assistant, at least 4 turns
    if len(messages) < 4:
        return None
    if messages[0]["role"] != "user":
        messages = messages[1:]
    if not messages or messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    if len(messages) < 4:
        return None
    return messages


async def generate_one(
    client: AsyncOpenAI,
    sop: str,
    scenario: dict,
    sem: asyncio.Semaphore,
    cfg: Config,
    bank: "RealCallBank | None" = None,
) -> list[dict] | None:
    real_examples = ""
    if bank is not None:
        samples = bank.sample(scenario["call_type"], scenario["language"], n=2)
        parts = []
        for i, msgs in enumerate(samples, 1):
            parts.append(f"--- Example {i} ---\n{_format_real_call(msgs, max_turns=30)}")
        real_examples = "\n\n".join(parts) if parts else "(none available)"
    else:
        real_examples = "(none available)"

    prompt = GENERATION_PROMPT.format(
        sop=sop,
        call_type=scenario["call_type"].upper(),
        scenario=scenario["scenario"],
        language=scenario["language"],
        mood=scenario["mood"],
        real_examples=real_examples,
    )
    async with sem:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=TEACHER_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    extra_headers={"X-Title": "agent-school-generate"},
                ),
                timeout=60,
            )
            text = response.choices[0].message.content
            return parse_conversation(text)
        except Exception as e:
            print(f"  [error] {e}")
            return None


async def generate_all(cfg: Config, sop: str, scenarios: list[dict]) -> list[tuple[str, list[dict]]]:
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=__import__("os").environ["OPENROUTER_API_KEY"],
    )
    rng = random.Random(cfg.seed)
    bank = RealCallBank(REAL_CALLS_PATH, rng)
    sem = asyncio.Semaphore(cfg.concurrency)
    tasks = [generate_one(client, sop, s, sem, cfg, bank) for s in scenarios]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating")
    return [
        (s["call_type"], r)
        for s, r in zip(scenarios, results)
        if r is not None
    ]


def main(cfg: Config):
    sop_path = Path(cfg.sop)
    if not sop_path.exists():
        raise FileNotFoundError(f"SOP not found: {sop_path}. Run generate_script.py first.")

    sop = sop_path.read_text(encoding="utf-8")
    print(f"SOP loaded: {len(sop)} chars")

    scenarios = build_scenarios(cfg.call_type, cfg.num, cfg.seed)
    lead_n = sum(1 for s in scenarios if s["call_type"] == "lead")
    followup_n = sum(1 for s in scenarios if s["call_type"] == "followup")
    print(f"Scenarios: {len(scenarios)} total  (lead={lead_n}, followup={followup_n})")

    if cfg.dry_run:
        rng = random.Random(cfg.seed)
        bank = RealCallBank(REAL_CALLS_PATH, rng)
        s0 = scenarios[0]
        samples = bank.sample(s0["call_type"], s0["language"], n=2)
        parts = [f"--- Example {i+1} ---\n{_format_real_call(m, 20)}" for i, m in enumerate(samples)]
        real_examples = "\n\n".join(parts) if parts else "(none)"
        prompt = GENERATION_PROMPT.format(
            sop=sop[:500] + "...[truncated]",
            real_examples=real_examples[:800] + "...[truncated]",
            **{k: s0[k] for k in ("call_type", "scenario", "language", "mood")},
        )
        print("\n--- DRY RUN: first prompt ---")
        print(prompt)
        return

    conversations = asyncio.run(generate_all(cfg, sop, scenarios))
    print(f"\nGenerated {len(conversations)} valid conversations (dropped {len(scenarios) - len(conversations)} failures)")

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(f"{cfg.output}-{stamp}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for call_type, msgs in conversations:
            sys_prompt = SYSTEM_LEAD if call_type == "lead" else SYSTEM_FOLLOWUP
            full_msgs = [{"role": "system", "content": sys_prompt}] + msgs
            f.write(json.dumps({"messages": full_msgs}, ensure_ascii=False) + "\n")

    print(f"Saved to {out_path}")
    print(f"\nNext: uv run python train.py data_path={out_path} model_name=Qwen/Qwen3-8B renderer_name=qwen3_disable_thinking")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
