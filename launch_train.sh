#!/usr/bin/env bash
# Launch Sakhiya LoRA training in nohup mode — survives SSH disconnects.
# Usage: ./launch_train.sh [--push_to_hub snehmehta/sakhiya-qwen3-moe]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TORCH_LIB="$SCRIPT_DIR/.venv/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export UNSLOTH_MOE_DISABLE_AUTOTUNE=1
export UNSLOTH_MOE_LORA_MERGED=1  # bypass buggy down_proj extractor in separated LoRA path

LOG_FILE="$SCRIPT_DIR/logs/train.log"
PID_FILE="$SCRIPT_DIR/logs/train.pid"
mkdir -p "$SCRIPT_DIR/logs"

# Kill any existing training run
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing existing training run (PID $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
    fi
fi

echo "=== Sakhiya Qwen3.5-35B-A3B MoE LoRA Training ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# shellcheck disable=SC2086
nohup "$SCRIPT_DIR/.venv/bin/python" -u train_gpu.py \
    --model "unsloth/Qwen3.5-35B-A3B" \
    --data "output/dataset/sakhiya_combined.jsonl" \
    --output_dir "output/checkpoints/sakhiya-qwen3-moe" \
    --lora_rank 16 \
    --max_seq_length 4096 \
    --batch_size 2 \
    --grad_accum 4 \
    --epochs 3 \
    --lr 1e-4 \
    --eval_ratio 0.05 \
    "$@" \
    >> "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Training started in background — PID $TRAIN_PID"
echo "Log:  tail -f $LOG_FILE"
echo "Kill: kill \$(cat $PID_FILE)"
