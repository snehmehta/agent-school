"""
Test trained checkpoint with correct disable_thinking renderer.
Uses real multi-turn conversations from training data as test cases.

Usage:
  uv run python test_checkpoint.py
  uv run python test_checkpoint.py cfg.checkpoint=tinker://...
  uv run python test_checkpoint.py cfg.call_type=followup
"""

import asyncio

import chz
import tinker
from dotenv import load_dotenv

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from train import SYSTEM_FOLLOWUP, SYSTEM_LEAD

load_dotenv()

CHECKPOINT_35B = "tinker://e3b53866-bdce-569c-af96-7b8edd895103:train:0/sampler_weights/final"
CHECKPOINT_4B  = "tinker://128e341c-4cc7-5e4e-9387-15674004c675:train:0/sampler_weights/final"

# Real multi-turn conversations from training data.
# Each entry: (call_type, history_so_far, expected_next_user_turn_label)
# history = alternating user/assistant messages; model predicts next assistant turn.

MULTI_TURN_TESTS = [
    # Lead call — Hindi — hair fall inquiry
    {
        "label": "LEAD-HI: hair fall inquiry",
        "call_type": "lead",
        "history": [
            {"role": "user",      "content": "हेलो।"},
            {"role": "assistant", "content": "हेलो जी, गुड मॉर्निंग। सखिया स्किन क्लिनिक से बात कर रही हूँ।"},
            {"role": "user",      "content": "गुड मॉर्निंग।"},
            {"role": "assistant", "content": "जी सर, आपने हेयर फॉल के लिए इंक्वायरी की थी।"},
            {"role": "user",      "content": "हाँ, हाँ।"},
        ],
    },
    # Lead call — Hindi — customer pushes back on price
    {
        "label": "LEAD-HI: price pushback",
        "call_type": "lead",
        "history": [
            {"role": "user",      "content": "हेलो।"},
            {"role": "assistant", "content": "हेलो जी, सखिया स्किन क्लिनिक से बात कर रही हूँ। आपने हेयर ट्रांसप्लांट के लिए इंक्वायरी की थी।"},
            {"role": "user",      "content": "हाँ।"},
            {"role": "assistant", "content": "जी सर, एक बार क्लिनिक पर आकर डॉक्टर से मिल सकते हो? काउंसलिंग फ्री है।"},
            {"role": "user",      "content": "कितने का पड़ेगा पूरा ट्रांसप्लांट?"},
        ],
    },
    # Follow-up — Hindi — missed appointment
    {
        "label": "FOLLOWUP-HI: missed appointment",
        "call_type": "followup",
        "history": [
            {"role": "user",      "content": "हाँ।"},
            {"role": "assistant", "content": "हाँ गुड इवनिंग मैम, सखिया स्किन क्लीनिक से प्राची बात कर रही हूँ।"},
            {"role": "user",      "content": "हाँ बोलो।"},
            {"role": "assistant", "content": "मैम, आपने अपॉइंटमेंट बुक करवाई थी पर आप आ नहीं पाए थे, कब आओगे?"},
            {"role": "user",      "content": "एक्चुअली पीरियड्स की वजह से नहीं आ पाई। कल का कर दो।"},
        ],
    },
    # Follow-up — Hindi — package renewal
    {
        "label": "FOLLOWUP-HI: package renewal",
        "call_type": "followup",
        "history": [
            {"role": "user",      "content": "हेलो।"},
            {"role": "assistant", "content": "हाँ गुड इवनिंग सर, सख्या स्किन क्लीनिक से प्राची बात कर रही हूँ।"},
            {"role": "user",      "content": "बोलिए।"},
            {"role": "assistant", "content": "जी सर, आपने बियर्ड शेपिंग का पैकेज लिया था, उसको रिन्यू नहीं करवाया?"},
            {"role": "user",      "content": "यस, सोच रहा हूँ। कितने का है?"},
        ],
    },
    # Lead call — Gujarati — hair transplant inquiry
    {
        "label": "LEAD-GUJ: hair transplant",
        "call_type": "lead",
        "history": [
            {"role": "user",      "content": "હલો."},
            {"role": "assistant", "content": "હા, ગુડ મોર્નિંગ સર. સખ્યા સ્કિન ક્લિનિકથી વાત કરું છું."},
            {"role": "user",      "content": "હા બોલો."},
            {"role": "assistant", "content": "સર, તમે હેર ટ્રાન્સપ્લાન્ટ માટે ઇન્ક્વાયરી કરી હતી, ક્યાંથી વાત કરો છો?"},
            {"role": "user",      "content": "સુરતથી."},
        ],
    },
    # Lead call — Gujarati — customer asks about process
    {
        "label": "LEAD-GUJ: asks process",
        "call_type": "lead",
        "history": [
            {"role": "user",      "content": "હલો."},
            {"role": "assistant", "content": "હા ગુડ આફ્ટરનૂન, સખ્યા ક્લિનિકથી વાત કરું છું. હેર ટ્રાન્સપ્લાન્ટ ઇન્ક્વાયરી કરી હતી."},
            {"role": "user",      "content": "હા."},
            {"role": "assistant", "content": "ક્યારે ફાવે ક્લિનિક પર આવવું?"},
            {"role": "user",      "content": "પ્રોસેસ કેટલા ટાઇમ લે? અને ખર્ચ કેટલો?"},
        ],
    },
]


@chz.chz
class Config:
    checkpoint: str = CHECKPOINT_35B
    model: str = "Qwen/Qwen3.5-35B-A3B"
    renderer_name: str = "qwen3_5_disable_thinking"
    call_type: str | None = None  # filter to 'lead' or 'followup' only
    max_tokens: int = 200
    temperature: float = 0.5


def main(cfg: Config):
    tokenizer = get_tokenizer(cfg.model)
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=cfg.checkpoint)

    tests = MULTI_TURN_TESTS
    if cfg.call_type:
        tests = [t for t in tests if t["call_type"] == cfg.call_type]

    async def run():
        for test in tests:
            sys_prompt = SYSTEM_FOLLOWUP if test["call_type"] == "followup" else SYSTEM_LEAD
            msgs = [renderers.Message(role="system", content=sys_prompt)]
            for m in test["history"]:
                msgs.append(renderers.Message(role=m["role"], content=m["content"]))

            inp = renderer.build_generation_prompt(msgs)
            params = tinker.SamplingParams(
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                stop=renderer.get_stop_sequences(),
            )
            result = await sampling_client.sample_async(
                prompt=inp, sampling_params=params, num_samples=1
            )
            parsed, _ = renderer.parse_response(list(result.sequences[0].tokens))
            response_text = renderers.format_content_as_string(parsed["content"])

            print(f"\n{'='*60}")
            print(f"[{test['label']}]")
            # Print last 2 turns of history for context
            for m in test["history"][-2:]:
                tag = "U" if m["role"] == "user" else "A"
                print(f"  {tag}: {m['content']}")
            print(f"  >> {response_text}")

    asyncio.run(run())
    print(f"\n{'='*60}")
    print(f"checkpoint: {cfg.checkpoint}")
    print(f"model:      {cfg.model}  temp={cfg.temperature}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
