import asyncio
import json
import tempfile
from pathlib import Path
import httpx

PROGRESS_FILE = "output/progress.json"


def load_progress() -> dict:
    p = Path(PROGRESS_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_progress(progress: dict) -> None:
    p = Path(PROGRESS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, indent=2))


def get_output_path(mp3_path: Path) -> Path:
    company = mp3_path.parent.name
    out_dir = Path("output") / company
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / mp3_path.with_suffix(".json").name


async def transcribe_file(mp3_path: Path, api_key: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=300) as client:
        with open(mp3_path, "rb") as f:
            response = await client.post(
                "https://api.x.ai/v1/stt",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (mp3_path.name, f, "audio/mpeg")},
                data={"diarize": "true", "language": "hi"},
            )
        response.raise_for_status()
        data = response.json()

    return _words_to_segments(data.get("words", []), data.get("text", ""))


def _words_to_segments(words: list[dict], full_text: str) -> list[dict]:
    if not words:
        return [{"speaker": "speaker_0", "text": full_text, "start": 0, "end": 0}]

    segments = []
    current_speaker = words[0].get("speaker", 0)
    current_words = []
    current_start = words[0].get("start", 0)
    current_end = words[0].get("end", 0)

    for word in words:
        speaker = word.get("speaker", 0)
        if speaker != current_speaker:
            segments.append({
                "speaker": f"speaker_{current_speaker}",
                "text": " ".join(current_words),
                "start": current_start,
                "end": current_end,
            })
            current_speaker = speaker
            current_words = [word["text"]]
            current_start = word.get("start", 0)
            current_end = word.get("end", 0)
        else:
            current_words.append(word["text"])
            current_end = word.get("end", current_end)

    if current_words:
        segments.append({
            "speaker": f"speaker_{current_speaker}",
            "text": " ".join(current_words),
            "start": current_start,
            "end": current_end,
        })

    return segments


def _transcribe_sarvam_sync(mp3_path: Path, api_key: str) -> list[dict]:
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=api_key)
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="transcribe",
        language_code="unknown",
        with_diarization=True,
        num_speakers=2,
    )
    job.upload_files(file_paths=[str(mp3_path)])
    job.start()
    job.wait_until_complete()

    with tempfile.TemporaryDirectory() as tmp_dir:
        job.download_outputs(output_dir=tmp_dir)
        out_files = list(Path(tmp_dir).glob("*.json"))
        if not out_files:
            raise RuntimeError("Sarvam returned no output files")
        data = json.loads(out_files[0].read_text())

    return _sarvam_to_segments(data)


def _sarvam_to_segments(data: dict) -> list[dict]:
    # Diarized transcript format
    diarized = data.get("diarized_transcript", {})
    entries = diarized.get("entries", [])
    if entries:
        segments = []
        for entry in entries:
            speaker_id = entry.get("speaker_id", "SPEAKER_00")
            # Normalize "SPEAKER_00" → "speaker_0"
            idx = speaker_id.replace("SPEAKER_", "").lstrip("0") or "0"
            segments.append({
                "speaker": f"speaker_{idx}",
                "text": entry.get("transcript", ""),
                "start": entry.get("start", 0),
                "end": entry.get("end", 0),
            })
        return segments

    # Fallback: flat transcript
    text = data.get("transcript", data.get("text", ""))
    return [{"speaker": "speaker_0", "text": text, "start": 0, "end": 0}]


async def transcribe_file_sarvam(mp3_path: Path, api_key: str) -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _transcribe_sarvam_sync, mp3_path, api_key)


async def transcribe_file_google(mp3_path: Path, api_key: str) -> list[dict]:
    raise NotImplementedError("Google transcription not yet implemented")


async def transcribe_file_deepgram(mp3_path: Path, api_key: str) -> list[dict]:
    raise NotImplementedError("Deepgram transcription not yet implemented")


async def process_file(mp3_path: Path, api_key: str, progress: dict, provider: str = "xai") -> dict:
    key = str(mp3_path)
    out_path = get_output_path(mp3_path)

    try:
        if provider == "sarvam":
            segments = await transcribe_file_sarvam(mp3_path, api_key)
        elif provider == "google":
            segments = await transcribe_file_google(mp3_path, api_key)
        elif provider == "deepgram":
            segments = await transcribe_file_deepgram(mp3_path, api_key)
        else:
            segments = await transcribe_file(mp3_path, api_key)
        result = {
            "file": mp3_path.name,
            "status": "done",
            "segments": segments,
        }
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        progress[key] = "done"
        save_progress(progress)
        return result
    except Exception as e:
        progress[key] = f"error: {e}"
        save_progress(progress)
        raise
