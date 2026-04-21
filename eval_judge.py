"""
LLM-judge eval: replay real conversations, score agent responses via OpenRouter.
Usage: python eval_judge.py
"""
import json, os, random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "qwen/qwen3-235b-a22b"
DATA_PATH = "output/dataset/sakhiya_combined.jsonl"
NUM_EVAL = 20
SEED = 42

client = OpenAI(base_url=OPENROUTER_BASE, api_key=os.environ["OPENROUTER_API_KEY"])

# Load real conversations only for eval
records = []
with open(DATA_PATH) as f:
    for line in f:
        rec = json.loads(line.strip())
        if rec.get("source") == "real":
            records.append(rec)

rng = random.Random(SEED)
sample = rng.sample(records, min(NUM_EVAL, len(records)))
print(f"Evaluating {len(sample)} real conversations with LLM judge")

JUDGE_PROMPT = """You are evaluating a call center agent's response at Sakhiya Skin Clinic.

Conversation so far:
{history}

Customer said: {user_turn}

Agent responded: {agent_response}

Ground truth (what a real agent said): {ground_truth}

Score the agent response 1-5 on:
1. Language match (Hindi/Gujarati matching customer)
2. Brevity (phone call appropriate, not too long)
3. Helpfulness (addresses customer concern)
4. SOP compliance (correct pricing, no invented history)
5. Natural tone (warm, not scripted)

Reply with JSON only: {{"scores": {{"language":N,"brevity":N,"helpfulness":N,"sop":N,"tone":N}}, "avg": N.N, "comment": "one line"}}"""

total_scores = []

for rec in sample:
    msgs = [m for m in rec["messages"] if m["role"] != "system"]
    sys_msg = next((m for m in rec["messages"] if m["role"] == "system"), None)

    # Take first 3 user/assistant pairs for eval speed
    pairs = []
    i = 0
    while i < len(msgs) - 1 and len(pairs) < 3:
        if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i+1]["content"]))
            i += 2
        else:
            i += 1

    for turn_idx, (user_turn, gt) in enumerate(pairs):
        history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs[:turn_idx*2])

        # Simulate a reasonable agent response (placeholder - replace with actual model output)
        # For now judge the ground truth against itself to establish baseline
        prompt = JUDGE_PROMPT.format(
            history=history[:500],
            user_turn=user_turn,
            agent_response=gt,  # TODO: replace with model output when vLLM running
            ground_truth=gt,
        )

        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=200, temperature=0.1,
                extra_headers={"X-Title": "sakhiya-eval"},
            )
            result = json.loads(resp.choices[0].message.content.strip().strip("```json").strip("```"))
            avg = result["avg"]
            total_scores.append(avg)
            print(f"  [{rec['call_type']} {rec['language']}] turn={turn_idx+1} score={avg:.1f} — {result['comment']}")
        except Exception as e:
            print(f"  judge error: {e}")

if total_scores:
    print(f"\nMean score: {sum(total_scores)/len(total_scores):.2f}/5.0 over {len(total_scores)} turns")
    print("PASS" if sum(total_scores)/len(total_scores) >= 3.5 else "FAIL")
