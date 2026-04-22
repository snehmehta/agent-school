# Agent School

A repo where we use tinker to train model using student teacher distillation where teacher are call recording of human agents


## Abstract flow

We get call recording from the company put it into raw-data/{company_name}
||
Do a transcribe dirazation 
||
Manual review + assisted with resemblyzer to pick which is user (query) which is assistant response
||
Do a proper lora adaptor training using tinker 
||
Merge those training into a model and covert that into FP8
||
Server those model


Let's go!

## Plan of action: 

### Phase 1: Data preparation

- [x] This a one proper TUI so have a proper base foundation.
- [x] Get recording from the company
- [x] Transcribe with dirzation on (have speaker1 and speaker2) using xai stt model, to save on price we'll use batching and to handle interruption we'll log and resume things.
- [x] Use resemblyzer of one of the reviewed by human in TUI to decide between user query tagged as user and human agent response to tag as assistant. Add a feature in TUI such that it can be reviewed and toggeled if wronged by resemblyzer.



### Phase 2: Training


- [x] Check out reference /Users/snehmehta/work/miraiminds/agent-school/reference-cookbook it's single turn distillation but we are trying to do multi turn distillation
- [x] Entire 231 conversation between customer (user) and human agent (assistant) output/sakhiya-skin-clinic/training/training_data.jsonl has been transcribe.
- [x] `train.py` built for multi-turn SFT on full conversations. Aggregates every `output/<company>/training/training_data.jsonl`, cleans (strip metadata, drop empties, merge consecutive same-role, enforce user-first / assistant-last, optional system prompt) → single JSONL → `tinker_cookbook.supervised.train` + `FromConversationFileBuilder` with LoRA. `train_on_what=ALL_ASSISTANT_MESSAGES` so every assistant turn contributes loss (true multi-turn, not last-turn-only).
- [x] Default model `Qwen/Qwen3.5-4B` with renderer `qwen3_5_disable_thinking` (reasoning off). Override via `chz` args: `uv run python train.py company=sakhiya-skin-clinic lora_rank=32 learning_rate=1e-4 num_epochs=4`.
- [ ] Run training against Tinker (needs `TINKER_API_KEY`; optional `wandb_project=`). Tune `lora_rank` / `lr` / `num_epochs` on eval loss.
- [ ] Merge LoRA adaptor into base model, convert to FP8, serve.
