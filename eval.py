"""
Replay evaluation: take real conversations from training data,
feed user turns to the model, compare model response vs ground truth.

Usage:
  uv run python eval.py                          # random 10 convos via Tinker
  uv run python eval.py num=5 call_type=lead
  uv run python eval.py company=sakhiya-skin-clinic num=3
  uv run python eval.py use_vllm=True base_url=http://GPU_IP:8000/v1 model=sakhiya
"""

import json
import random
from pathlib import Path

import chz
from dotenv import load_dotenv

from train import (
    SYSTEM_FOLLOWUP,
    SYSTEM_LEAD,
    OUTPUT_ROOT,
    _classify_call_type,
    _clean_messages,
)

load_dotenv()

CHECKPOINT_35B = "tinker://3ed8da41-fc3e-5835-a8b6-cbed2b5d222a:train:0/sampler_weights/final"


@chz.chz
class Config:
    # Tinker mode (default)
    checkpoint: str = CHECKPOINT_35B
    tinker_model: str = "Qwen/Qwen3.5-35B-A3B"
    renderer_name: str = "qwen3_5_disable_thinking"

    # vLLM mode
    use_vllm: bool = False
    base_url: str = "http://localhost:8000/v1"
    model: str = "sakhiya"

    # Eval settings
    company: str | None = None
    call_type: str | None = None    # 'lead' or 'followup'
    num: int = 10
    max_turns: int = 8
    max_tokens: int = 150
    temperature: float = 0.3
    seed: int = 42


def load_conversations(company: str | None, call_type_filter: str | None) -> list[dict]:
    if company:
        sources = [OUTPUT_ROOT / company / "training" / "training_data.jsonl"]
    else:
        sources = sorted(OUTPUT_ROOT.glob("*/training/training_data.jsonl"))

    convos = []
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
                if msgs is None:
                    continue
                call_type = _classify_call_type(msgs)
                if call_type_filter and call_type != call_type_filter:
                    continue
                convos.append({"messages": msgs, "call_type": call_type, "source": str(src)})
    return convos


def replay_convo_tinker(sampling_client, renderer, cfg: Config, convo: dict) -> None:
    import tinker
    from tinker_cookbook import renderers

    msgs = convo["messages"]
    call_type = convo["call_type"]
    system_prompt = SYSTEM_FOLLOWUP if call_type == "followup" else SYSTEM_LEAD

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
            i += 2
        else:
            i += 1

    pairs = pairs[: cfg.max_turns]
    if not pairs:
        return

    print(f"\n{'=' * 70}")
    print(f"[{call_type.upper()}]  turns={len(pairs)}  src={Path(convo['source']).parent.parent.name}")
    print("=" * 70)

    import asyncio

    history: list[renderers.Message] = [renderers.Message(role="system", content=system_prompt)]

    for turn_idx, (user_turn, ground_truth) in enumerate(pairs):
        history.append(renderers.Message(role="user", content=user_turn))

        inp = renderer.build_generation_prompt(history)
        params = tinker.SamplingParams(
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            stop=renderer.get_stop_sequences(),
        )

        async def sample():
            result = await sampling_client.sample_async(
                prompt=inp, sampling_params=params, num_samples=1
            )
            return result

        result = asyncio.run(sample())
        parsed, _ = renderer.parse_response(list(result.sequences[0].tokens))
        from tinker_cookbook.renderers import format_content_as_string
        model_reply = format_content_as_string(parsed["content"]).strip()

        print(f"\n  Turn {turn_idx + 1}")
        print(f"  U:  {user_turn}")
        print(f"  GT: {ground_truth}")
        print(f"  M:  {model_reply}")

        history.append(renderers.Message(role="assistant", content=ground_truth))


def replay_convo_vllm(client, cfg: Config, convo: dict) -> None:
    msgs = convo["messages"]
    call_type = convo["call_type"]
    system_prompt = SYSTEM_FOLLOWUP if call_type == "followup" else SYSTEM_LEAD

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
            i += 2
        else:
            i += 1

    pairs = pairs[: cfg.max_turns]
    if not pairs:
        return

    print(f"\n{'=' * 70}")
    print(f"[{call_type.upper()}]  turns={len(pairs)}  src={Path(convo['source']).parent.parent.name}")
    print("=" * 70)

    history: list[dict] = []
    for turn_idx, (user_turn, ground_truth) in enumerate(pairs):
        history.append({"role": "user", "content": user_turn})

        response = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": system_prompt}] + history,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        model_reply = response.choices[0].message.content.strip()

        print(f"\n  Turn {turn_idx + 1}")
        print(f"  U:  {user_turn}")
        print(f"  GT: {ground_truth}")
        print(f"  M:  {model_reply}")

        history.append({"role": "assistant", "content": ground_truth})


def main(cfg: Config):
    convos = load_conversations(cfg.company, cfg.call_type)
    if not convos:
        print("No conversations found.")
        return

    rng = random.Random(cfg.seed)
    sample = rng.sample(convos, min(cfg.num, len(convos)))
    print(f"Evaluating {len(sample)} conversations  (pool={len(convos)}, call_type={cfg.call_type or 'all'})")

    if cfg.use_vllm:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key="dummy")
        print(f"Mode: vLLM  model={cfg.model} @ {cfg.base_url}")
        for convo in sample:
            replay_convo_vllm(client, cfg, convo)
    else:
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        print(f"Mode: Tinker  checkpoint={cfg.checkpoint}")
        tokenizer = get_tokenizer(cfg.tinker_model)
        renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=cfg.checkpoint)

        for convo in sample:
            replay_convo_tinker(sampling_client, renderer, cfg, convo)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
