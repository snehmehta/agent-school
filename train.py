"""
Phase 2: LoRA fine-tune student model on exported call transcripts using Tinker.

Two stages:
  1. prepare: aggregate every `output/<company>/training/training_data.jsonl`
     into a single normalized JSONL (only `messages`, cleaned, role-validated).
     Because qwen3_5_disable_thinking does NOT satisfy has_extension_property,
     we explode each multi-turn conversation [u1,a1,...,un,an] into N growing
     prefix sub-conversations so every assistant turn is trained as LAST_ASSISTANT_MESSAGE
     with the correct token prefix. train_on_what is set to LAST_ASSISTANT_MESSAGE.
  2. train:   hand that JSONL to tinker-cookbook's supervised trainer.

Usage:
  uv run python train.py                             # prepare + train, all companies
  uv run python train.py company=sakhiya-skin-clinic # single company
  uv run python train.py prepare_only=True           # only build dataset
  uv run python train.py model_name=Qwen/Qwen3-8B lora_rank=16 learning_rate=1e-4
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import chz
from dotenv import load_dotenv
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

load_dotenv()


OUTPUT_ROOT = Path("output")
DATASET_DIR = OUTPUT_ROOT / "dataset"

# System prompts per call type
SYSTEM_FOLLOWUP = """You are an outbound call center agent at Sakhiya Skin Clinic (also called Doctor Sakhya's Plastic Surgery Centre).

## Your role
You are calling an EXISTING PATIENT for a follow-up. You know:
- Their name (address them as "[Name] सर" or "[Name] मैम/મેમ")
- The treatment/package they have taken (only reference what is explicitly mentioned in the conversation)
- That they missed an appointment, have pending sessions, or haven't returned since their last visit

## How to speak
- Match the customer's language exactly: respond in Hindi if they speak Hindi, Gujarati if they speak Gujarati
- Short acknowledgments ("हाँ", "जी", "ओके", "હા", "ઓકે") are normal and natural — use them freely
- Be warm, brief, conversational — like a real call center agent, not a formal script
- Greet: "गुड मॉर्निंग/आफ्टरनून/इवनिंग [सर/मैम], सखिया स्किन क्लिनिक से [your name] बात कर रही/रहा हूँ"

## Your goals (in order)
1. Remind them of their pending appointment / sessions / follow-up
2. Reschedule if they missed — ask preferred date and time
3. If they're not interested right now, offer current promotions as a soft upsell
4. If they're busy or out of town, ask when to call back — never pressure

## Pricing you can mention
- Hydra Facial: ₹2,400 (special offer, normally ₹4,000)
- GFC (hair treatment): ₹3,000–4,000 per session
- Hair Transplant: ₹60,000 for unlimited grafts
- Package renewals: 25% off if renewing now
- All counseling/consultations: FREE

## Branches
Ahmedabad: Shivranjani, Chandkheda, Bhopol, Borkadav
Surat: Infinity Mall, Railway Station area
Also: Anand, Ankleshwar, Vatva, Navsari
When customer gives their location, suggest the nearest branch.

## Rules
- NEVER invent treatments, sessions, or visit dates not mentioned in the conversation
- If customer is unavailable: "ठीक है, कब कॉल बैक दूं?" then end politely
- Keep responses concise — this is a phone call, not a message
"""

SYSTEM_LEAD = """You are an outbound call center agent at Sakhiya Skin Clinic (also called Doctor Sakhya's Plastic Surgery Centre).

## Your role
You are calling a NEW LEAD — someone who submitted an inquiry online (via Meta/social media) for a skin or hair service. You only know:
- Their name (from the inquiry)
- The service they inquired about (hair fall, hair transplant, skin treatment, beard shaping, laser hair removal, etc.)
- Nothing else — no visit history, no prior treatments

## How to speak
- Match the customer's language exactly: respond in Hindi if they speak Hindi, Gujarati if they speak Gujarati
- Short acknowledgments ("हाँ", "जी", "ओके", "हाँ हाँ", "હા", "ઓકે") are normal and natural
- Be warm, friendly, conversational — not scripted or robotic
- Greet: "गुड मॉर्निंग/आफ्टरनून/इवनिंग [सर/मैम], सखिया स्किन क्लिनिक से [your name] बात कर रही/रहा हूँ"
- Confirm you're speaking to the right person: "[Name] सर/मैम जोड़े बात है?"

## Your goals (in order)
1. Confirm their inquiry and interest
2. Ask where they're calling from → suggest nearest clinic branch
3. Answer basic questions about the service, process, and pricing
4. Offer FREE counseling at the clinic — emphasize no charge for consultation
5. Book an appointment: ask for preferred date and time
6. If busy/not ready: ask when to call back — never pressure

## Pricing you can share
- Hair Transplant: ₹60,000 for unlimited grafts (limited offer)
- GFC hair treatment: ₹3,000–4,000 per session, minimum 6 sessions recommended
- Hydra Facial: ₹2,400 (special offer, normally ₹4,000)
- Laser Hair Removal: package-based pricing, discuss at clinic
- All initial consultations/counseling: FREE

## Branches
Ahmedabad: Shivranjani, Chandkheda, Bhopol, Borkadav
Surat: Infinity Mall (near Railway Station)
Also: Anand, Ankleshwar, Vatva, Navsari
When customer gives location, map them to the nearest branch.

## Rules
- NEVER invent prior visit history, previous treatments, or patient details
- Do not over-explain — answer what's asked, then move toward booking
- If customer says wrong inquiry topic: "कोई बात नहीं सर, [correct service] के लिए भी हम कर सकते हैं"
- If unavailable: "ठीक है, कब कॉल बैक दूं?" then end politely
- Keep responses concise — phone call, not a message
"""

# Follow-up signals: agent references prior context in early turns
_FOLLOWUP_KEYWORDS = [
    # Hindi
    "अपॉइंटमेंट बुक", "आए नहीं", "आ नहीं पाए", "फॉलोअप", "फॉलो अप",
    "लास्ट", "ट्रीटमेंट लिया", "पैकेज लिया", "रिन्यू", "वापस नहीं आए",
    "चेकअप करवाए", "सेशन बाकी",
    # Gujarati
    "અપોઇન્ટ", "ફોલોઅપ", "ટ્રીટમેન્ટ", "આવ્યા નહોતા", "પેકેજ",
    "રિન્યૂ", "છેલ્લે", "સેશન",
]

# Lead signals: agent asks for name/location or references inquiry without prior context
_LEAD_KEYWORDS = [
    "इंक्वायरी", "inquiry", "ઇન્ક્વાયરી",
    "नाम जान", "नाम बता", "कहाँ से", "ક્યાંથી",
    "क्लिनिक पर आना", "ક્લિનિક પર આવવ",
]


def _classify_call_type(messages: list[dict]) -> str:
    """Return 'followup' or 'lead' based on early assistant turns."""
    # Scan first 6 assistant turns for context signals
    assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"][:6]
    early_text = " ".join(assistant_turns).lower()

    followup_hits = sum(1 for kw in _FOLLOWUP_KEYWORDS if kw.lower() in early_text)
    lead_hits = sum(1 for kw in _LEAD_KEYWORDS if kw.lower() in early_text)

    return "followup" if followup_hits > lead_hits else "lead"


def _clean_messages(messages: list[dict]) -> list[dict] | None:
    """Drop empties, strip, merge consecutive same-role, ensure user-first
    and assistant-last, require at least one user+assistant pair."""
    cleaned: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += " " + content
        else:
            cleaned.append({"role": role, "content": content})

    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()

    if len(cleaned) < 2:
        return None
    if not any(m["role"] == "assistant" for m in cleaned):
        return None
    return cleaned


def _explode_to_prefixes(
    messages: list[dict], system_prompt: str | None
) -> list[list[dict]]:
    """Explode [u1,a1,u2,a2,...,un,an] into N prefix sub-conversations.

    Each sub-conversation ends on an assistant turn so LAST_ASSISTANT_MESSAGE
    trains the correct turn with the right token prefix (works around
    has_extension_property=False on qwen3_5_disable_thinking renderer).
    """
    prefix: list[dict] = []
    if system_prompt:
        prefix.append({"role": "system", "content": system_prompt})

    examples: list[list[dict]] = []
    for msg in messages:
        prefix = prefix + [msg]
        if msg["role"] == "assistant":
            examples.append(list(prefix))
    return examples


def prepare_dataset(company: str | None, system_prompt: str | None = None) -> Path:  # noqa: ARG001
    """Aggregate exports into `output/dataset/<stamp>.jsonl`. Returns path.

    Classifies each conversation as 'followup' or 'lead' and applies the
    appropriate system prompt. Explodes multi-turn convos into per-assistant-turn
    sub-conversations for LAST_ASSISTANT_MESSAGE training.

    system_prompt arg is ignored — per-type prompts are used instead.
    """
    if company:
        sources = [OUTPUT_ROOT / company / "training" / "training_data.jsonl"]
    else:
        sources = sorted(OUTPUT_ROOT.glob("*/training/training_data.jsonl"))

    sources = [p for p in sources if p.exists()]
    if not sources:
        raise FileNotFoundError(
            f"No training_data.jsonl found under {OUTPUT_ROOT}/ (company={company})"
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = company or "all"
    out_path = DATASET_DIR / f"{tag}-{stamp}.jsonl"

    kept_convos = dropped = written = 0
    type_counts: dict[str, int] = {"followup": 0, "lead": 0}
    per_source: dict[str, tuple[int, int]] = {}
    with out_path.open("w", encoding="utf-8") as out_f:
        for src in sources:
            s_kept = s_dropped = 0
            with src.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        s_dropped += 1
                        continue
                    msgs = _clean_messages(rec.get("messages", []))
                    if msgs is None:
                        s_dropped += 1
                        continue
                    call_type = _classify_call_type(msgs)
                    sys_prompt = SYSTEM_FOLLOWUP if call_type == "followup" else SYSTEM_LEAD
                    type_counts[call_type] += 1
                    for sub in _explode_to_prefixes(msgs, sys_prompt):
                        out_f.write(
                            json.dumps({"messages": sub}, ensure_ascii=False) + "\n"
                        )
                        written += 1
                    s_kept += 1
            per_source[str(src)] = (s_kept, s_dropped)
            kept_convos += s_kept
            dropped += s_dropped

    print(
        f"[prepare] wrote {out_path}  "
        f"convos={kept_convos}  sub-examples={written}  dropped={dropped}  "
        f"followup={type_counts['followup']}  lead={type_counts['lead']}"
    )
    for src, (k, d) in per_source.items():
        print(f"  - {src}: convos={k} dropped={d}")
    if written == 0:
        raise RuntimeError("No usable conversations after cleaning.")
    return out_path


@chz.chz
class CLIConfig:
    company: str | None = None
    data_path: str | None = None
    prepare_only: bool = False

    system_prompt: str | None = (
        "You are a helpful customer-service agent. "
        "Respond in the language and style of the human agent examples."
    )

    model_name: str = "Qwen/Qwen3.5-4B"
    load_checkpoint_path: str | None = None

    learning_rate: float = 1e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 4
    lora_rank: int = 32

    base_url: str | None = None
    save_every: int = 20
    eval_every: int = 5

    # Default renderer disables Qwen3.5 thinking traces — user wants reasoning off.
    renderer_name: str | None = "qwen3_5_disable_thinking"
    # Dataset is pre-exploded into per-assistant-turn sub-convos, so LAST is correct.
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE
    max_length: int = 8192
    batch_size: int = 32

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None


def cli_main(cfg: CLIConfig):
    if cfg.data_path:
        data_path = Path(cfg.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"data_path not found: {data_path}")
    else:
        data_path = prepare_dataset(cfg.company, cfg.system_prompt)

    if cfg.prepare_only:
        print(f"[prepare_only] dataset ready at {data_path}")
        return

    model_slug = cfg.model_name.replace("/", "-")
    tag = cfg.company or "all"
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"agent-school-{tag}-{model_slug}-r{cfg.lora_rank}"
        f"-lr{cfg.learning_rate}-bs{cfg.batch_size}-{stamp}"
    )
    log_path = cfg.log_path or f"output/runs/{run_name}"
    wandb_name = cfg.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cfg.model_name,
        explicit_renderer_name=cfg.renderer_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
    )

    common_cfg = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        train_on_what=cfg.train_on_what,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_cfg,
        file_path=str(data_path),
    )

    train_cfg = train.Config(
        log_path=log_path,
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        dataset_builder=dataset,
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        base_url=cfg.base_url,
        wandb_project=cfg.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
    )
    print(f"[train] data={data_path}  log={log_path}")
    asyncio.run(train.main(train_cfg))


if __name__ == "__main__":
    chz.entrypoint(cli_main)
