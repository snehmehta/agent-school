"""
ML Engineer training monitor — parses logs/train.log and shows a dashboard.
Run: python monitor_training.py
"""

import re
import sys
import time
from pathlib import Path
from datetime import datetime

LOG = Path("logs/train.log")
PID = Path("logs/train.pid")


def is_alive():
    if not PID.exists():
        return False
    pid = PID.read_text().strip()
    return Path(f"/proc/{pid}").exists()


def parse_log(text):
    stats = {}

    # GPU from nvidia-smi line in header
    m = re.search(r"GPU: (.+)", text)
    if m:
        stats["gpu"] = m.group(1).strip()

    # Training loss lines: {'loss': 2.3, 'grad_norm': ..., 'learning_rate': ..., 'epoch': 0.5, ...}
    loss_matches = re.findall(
        r"'loss':\s*([\d.]+).*?'learning_rate':\s*([\S]+).*?'epoch':\s*([\d.]+)", text
    )
    if loss_matches:
        recent = loss_matches[-5:]  # last 5 steps
        stats["recent_losses"] = [(float(l), float(e)) for l, _, e in recent]
        stats["current_loss"] = float(loss_matches[-1][0])
        stats["current_lr"] = loss_matches[-1][1]
        stats["current_epoch"] = float(loss_matches[-1][2])
        stats["total_loss_reports"] = len(loss_matches)

    # Eval loss
    eval_matches = re.findall(r"'eval_loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)", text)
    if eval_matches:
        stats["eval_losses"] = [(float(l), float(e)) for l, e in eval_matches]
        stats["best_eval_loss"] = min(l for l, _ in eval_matches)

    # Step info: [step/total_steps]
    step_matches = re.findall(r"\[(\d+)/(\d+)", text)
    if step_matches:
        last = step_matches[-1]
        stats["step"] = int(last[0])
        stats["total_steps"] = int(last[1])
        stats["pct"] = 100 * int(last[0]) / max(int(last[1]), 1)

    # Time per step (it/s)
    its_matches = re.findall(r"([\d.]+)it/s", text)
    if its_matches:
        stats["its"] = float(its_matches[-1])

    # Errors
    errors = re.findall(r"(?:Error|Exception|Traceback)[^\n]*", text, re.IGNORECASE)
    if errors:
        stats["errors"] = errors[-3:]

    # Final save
    if "LoRA adapter saved to" in text:
        stats["done"] = True

    return stats


def format_bar(pct, width=40):
    filled = int(width * pct / 100)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:.1f}%"


def show_dashboard(stats, log_tail):
    print("\033[2J\033[H", end="")  # clear screen
    print(f"{'='*60}")
    print(f"  SAKHIYA Qwen3.5-35B MoE LoRA — Training Monitor")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Process: {'ALIVE ✓' if is_alive() else 'STOPPED ✗'}")
    print(f"{'='*60}")

    if "gpu" in stats:
        print(f"\n  GPU: {stats['gpu']}")

    if "step" in stats:
        print(f"\n  Progress: step {stats['step']}/{stats['total_steps']}")
        print(f"  {format_bar(stats['pct'])}")
        if "current_epoch" in stats:
            print(f"  Epoch: {stats['current_epoch']:.2f}/3.0")

    if "current_loss" in stats:
        print(f"\n  Train Loss:  {stats['current_loss']:.4f}  (LR: {stats['current_lr']})")
        if "recent_losses" in stats:
            trend = "  ".join(f"{l:.3f}" for l, _ in stats["recent_losses"])
            print(f"  Last 5:      {trend}")

    if "eval_losses" in stats:
        ev_str = "  ".join(f"e{e:.0f}={l:.4f}" for l, e in stats["eval_losses"])
        print(f"\n  Eval Loss:   {ev_str}")
        print(f"  Best eval:   {stats['best_eval_loss']:.4f}")

    if "its" in stats:
        its = stats["its"]
        if "step" in stats and "total_steps" in stats:
            remaining = (stats["total_steps"] - stats["step"]) / max(its, 0.001)
            h, m = divmod(int(remaining), 3600)
            m //= 60
            print(f"\n  Speed: {its:.2f} it/s  |  ETA: {h}h {m}m")

    if stats.get("done"):
        print("\n  ✓ TRAINING COMPLETE — LoRA adapter saved.")

    if "errors" in stats:
        print(f"\n  ERRORS DETECTED:")
        for e in stats["errors"]:
            print(f"    {e}")

    print(f"\n{'─'*60}")
    print("  Last 20 log lines:")
    print(f"{'─'*60}")
    for line in log_tail:
        print(f"  {line}")
    print(f"{'='*60}")
    print("  Ctrl+C to stop monitoring (training continues in background)")


def main():
    print("Watching logs/train.log ... (Ctrl+C to exit)")
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    while True:
        if not LOG.exists():
            print(f"Waiting for {LOG} ...")
            time.sleep(5)
            continue

        text = LOG.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in text.splitlines() if l.strip()]
        log_tail = lines[-20:]
        stats = parse_log(text)
        show_dashboard(stats, log_tail)

        if stats.get("done") and not is_alive():
            print("\nTraining finished. Exiting monitor.")
            break

        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped. Training continues in background.")
        print(f"Resume: python monitor_training.py")
        print(f"Log:    tail -f logs/train.log")
