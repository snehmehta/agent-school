"""
Batch processor: splits files into batches, processes each batch with
concurrent workers, checkpoints after every file so runs are resumable.
"""
import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

from transcriber import get_output_path, load_progress, save_progress, transcribe_file

BATCH_SIZE = 10
WORKERS = 3


@dataclass
class BatchResult:
    file: Path
    status: str          # "done" | "error" | "skipped"
    error: str = ""


@dataclass
class BatchProgress:
    total: int
    done: int = 0
    errors: int = 0
    current_batch: int = 0
    total_batches: int = 0
    current_file: str = ""

    @property
    def pending(self) -> int:
        return self.total - self.done - self.errors


def _make_batches(files: list[Path], size: int) -> list[list[Path]]:
    return [files[i : i + size] for i in range(0, len(files), size)]


async def _process_one(
    mp3_path: Path,
    api_key: str,
    progress: dict,
    on_update: Callable[[BatchResult], None],
) -> BatchResult:
    key = str(mp3_path)

    if progress.get(key) == "done":
        result = BatchResult(file=mp3_path, status="skipped")
        on_update(result)
        return result

    try:
        segments = await transcribe_file(mp3_path, api_key)
        out = get_output_path(mp3_path)
        out.write_text(
            json.dumps(
                {"file": mp3_path.name, "status": "done", "segments": segments},
                ensure_ascii=False,
                indent=2,
            )
        )
        progress[key] = "done"
        save_progress(progress)
        result = BatchResult(file=mp3_path, status="done")
    except Exception as e:
        progress[key] = f"error: {e}"
        save_progress(progress)
        result = BatchResult(file=mp3_path, status="error", error=str(e))

    on_update(result)
    return result


async def run_batch(
    files: list[Path],
    api_key: str,
    *,
    batch_size: int = BATCH_SIZE,
    workers: int = WORKERS,
    on_result: Callable[[BatchResult, BatchProgress], None] | None = None,
    stop_event: asyncio.Event | None = None,
) -> BatchProgress:
    """
    Process files in batches. Each batch runs `workers` concurrent requests.
    Checkpoints progress after every file. Skips already-done files.
    Returns final BatchProgress.
    """
    progress = load_progress()
    batches = _make_batches(files, batch_size)

    # Count real pending (skip already done)
    pending = [f for f in files if progress.get(str(f)) != "done"]
    stats = BatchProgress(
        total=len(files),
        done=len(files) - len(pending),
        total_batches=len(batches),
    )

    sem = asyncio.Semaphore(workers)

    def _on_update(result: BatchResult) -> None:
        stats.current_file = result.file.name
        if result.status == "done":
            stats.done += 1
        elif result.status == "error":
            stats.errors += 1
        if on_result:
            on_result(result, stats)

    for batch_idx, batch in enumerate(batches):
        if stop_event and stop_event.is_set():
            break

        stats.current_batch = batch_idx + 1

        async def _bounded(f: Path) -> BatchResult:
            if stop_event and stop_event.is_set():
                return BatchResult(file=f, status="skipped")
            async with sem:
                return await _process_one(f, api_key, progress, _on_update)

        await asyncio.gather(*[_bounded(f) for f in batch])

    return stats
