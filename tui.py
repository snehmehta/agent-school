import asyncio
import json
import os
import subprocess
from pathlib import Path

from rich.markup import escape as markup_escape

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Label,
    ListView,
    ListItem,
    Button,
    Static,
    ProgressBar,
    RichLog,
    Select,
)
from textual.reactive import reactive

from transcriber import load_progress, process_file

RAW_DATA_DIR = Path("raw-data")
MAX_CONCURRENT = 3
SPEAKERS = ["speaker_0", "speaker_1"]


class FileItem(ListItem):
    def __init__(self, mp3_path: Path, status: str = "pending"):
        super().__init__()
        self.mp3_path = mp3_path
        self.file_status = status

    def compose(self) -> ComposeResult:
        icon = self._icon()
        yield Label(f"{icon} {self.mp3_path.name}", id="file-label")

    def _icon(self) -> str:
        return {"pending": "○", "running": "⟳", "done": "✓"}.get(
            self.file_status, "✗"
        )

    def update_status(self, status: str) -> None:
        self.file_status = status
        label = self.query_one("#file-label", Label)
        label.update(f"{self._icon()} {self.mp3_path.name}")
        self.set_class(status == "done", "done")
        self.set_class(status == "running", "running")
        self.set_class(status not in ("pending", "done", "running"), "error")


class AgentSchoolApp(App):
    CSS = """
    Screen {
        background: $surface;
    }
    #sidebar {
        width: 40%;
        border-right: solid $primary;
    }
    #main-panel {
        width: 60%;
    }
    #stats-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary;
    }
    #file-list {
        height: 1fr;
    }
    #controls {
        height: 5;
        padding: 1;
        background: $panel;
        border-top: solid $primary;
    }
    #controls Button {
        margin-right: 1;
    }
    #audio-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary;
        layout: horizontal;
    }
    #audio-bar Button {
        margin-right: 1;
        min-width: 8;
    }
    #audio-status {
        padding: 0 1;
        content-align: left middle;
        height: 1fr;
    }
    #speaker-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary;
        layout: horizontal;
    }
    #speaker-bar Button {
        margin-right: 1;
        min-width: 14;
    }
    #speaker-roles-label {
        padding: 0 1;
        content-align: left middle;
        width: 1fr;
    }
    #transcript-log {
        height: 1fr;
        padding: 1;
    }
    #progress-bar {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-top: solid $primary;
    }
    #provider-bar {
        height: 4;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary;
    }
    #provider-bar Label {
        padding: 1 0 0 0;
        width: auto;
    }
    #provider-select {
        width: 1fr;
        margin: 0 0 0 1;
    }
    .done { color: $success; }
    .running { color: $warning; }
    .error { color: $error; }
    .active-role { background: $accent; }
    ListItem { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("t", "transcribe_selected", "Transcribe selected"),
        Binding("a", "transcribe_all", "Transcribe all"),
        Binding("s", "stop", "Stop"),
        Binding("p", "play_audio", "Play audio"),
        Binding("x", "stop_audio", "Stop audio"),
        Binding("e", "export_training", "Export training data"),
        Binding("j", "list_down", "Next file", show=False),
        Binding("k", "list_up", "Prev file", show=False),
        Binding("r", "swap_roles", "Swap roles [r]", show=True),
    ]

    _running = reactive(False)
    _done_count = reactive(0)
    _total = reactive(0)

    PROVIDERS = [
        ("xAI (Grok)", "xai"),
        ("Sarvam", "sarvam"),
        ("Google", "google"),
        ("Deepgram", "deepgram"),
    ]

    def __init__(self, company: str, api_keys: dict[str, str]):
        super().__init__()
        self.company = company
        self.api_keys = api_keys
        # Default to first provider that has a key
        self.provider = next(
            (p for p in ("xai", "sarvam", "google", "deepgram") if api_keys.get(p)),
            "xai",
        )
        self.api_key = api_keys.get(self.provider, "")
        self.mp3_dir = RAW_DATA_DIR / company
        self.progress = load_progress()
        self._tasks: list[asyncio.Task] = []
        self._stop_flag = False
        self._current_mp3: Path | None = None
        self._audio_proc: subprocess.Popen | None = None
        # Default: speaker_0 = user, speaker_1 = assistant
        self._speaker_roles: dict[str, str] = {
            "speaker_0": "user",
            "speaker_1": "assistant",
        }

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static(f"[bold]{self.company}[/bold]", id="stats-bar")
                yield ListView(id="file-list")
                with Horizontal(id="provider-bar"):
                    yield Label("Provider:")
                    yield Select(
                        [(label, value) for label, value in self.PROVIDERS],
                        value=self.provider,
                        id="provider-select",
                    )
                with Horizontal(id="controls"):
                    yield Button("Transcribe [t]", id="btn-selected", variant="primary")
                    yield Button("All [a]", id="btn-all", variant="success")
                    yield Button("Stop [s]", id="btn-stop", variant="error")
            with Vertical(id="main-panel"):
                with Horizontal(id="audio-bar"):
                    yield Button("▶ Play [p]", id="btn-play", variant="primary")
                    yield Button("■ Stop [x]", id="btn-stop-audio", variant="error")
                    yield Button("Export [e]", id="btn-export", variant="success")
                    yield Static("No file selected", id="audio-status")
                with Horizontal(id="speaker-bar"):
                    yield Button("⇄ Swap Roles [r]", id="btn-swap-roles", variant="warning")
                    yield Static("spk0=user  spk1=asst", id="speaker-roles-label")
                yield RichLog(id="transcript-log", highlight=True, markup=True)
                yield ProgressBar(id="progress-bar", show_eta=True)
        yield Footer()

    def on_mount(self) -> None:
        files = sorted(self.mp3_dir.glob("*.mp3"))
        self._total = len(files)
        list_view = self.query_one("#file-list", ListView)

        done = 0
        for f in files:
            status = "pending"
            key = str(f)
            if self.progress.get(key) == "done":
                status = "done"
                done += 1
            elif self.progress.get(key, "").startswith("error"):
                status = "error"
            list_view.append(FileItem(f, status))

        self._done_count = done
        self._update_stats()
        self._update_speaker_label()

    def _update_stats(self) -> None:
        stats = self.query_one("#stats-bar", Static)
        stats.update(
            f"[bold]{self.company}[/bold]  "
            f"[green]{self._done_count}[/green]/{self._total} done  "
            f"{'[yellow]running[/yellow]' if self._running else ''}"
        )
        bar = self.query_one("#progress-bar", ProgressBar)
        bar.update(total=self._total, progress=self._done_count)

    def _update_speaker_label(self) -> None:
        r0 = self._speaker_roles["speaker_0"]
        r1 = self._speaker_roles["speaker_1"]
        label = self.query_one("#speaker-roles-label", Static)
        label.update(f"[cyan]spk0={r0}[/cyan]   [magenta]spk1={r1}[/magenta]")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "provider-select":
            return
        self.provider = str(event.value)
        self.api_key = self.api_keys.get(self.provider, "")
        log = self.query_one("#transcript-log", RichLog)
        label = next((l for l, v in self.PROVIDERS if v == self.provider), self.provider)
        if self.api_key:
            log.write(f"[green]Provider → {label}[/green]")
        else:
            log.write(f"[red]No API key for {label} — set in .env[/red]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-selected":
            self.action_transcribe_selected()
        elif bid == "btn-all":
            self.action_transcribe_all()
        elif bid == "btn-stop":
            self.action_stop()
        elif bid == "btn-play":
            self.action_play_audio()
        elif bid == "btn-stop-audio":
            self.action_stop_audio()
        elif bid == "btn-export":
            self.action_export_training()
        elif bid == "btn-swap-roles":
            self.action_swap_roles()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, FileItem):
            self._current_mp3 = item.mp3_path
            self._show_transcript(item.mp3_path)
            status_widget = self.query_one("#audio-status", Static)
            status_widget.update(f"[dim]{item.mp3_path.name}[/dim]")

    def _show_transcript(self, mp3_path: Path) -> None:
        from transcriber import get_output_path
        log = self.query_one("#transcript-log", RichLog)
        log.clear()
        out = get_output_path(mp3_path)
        if not out.exists():
            log.write(f"[dim]No transcript yet for {mp3_path.name}[/dim]")
            return
        data = json.loads(out.read_text())
        log.write(f"[bold]{mp3_path.name}[/bold]\n")
        for seg in data.get("segments", []):
            speaker = seg["speaker"]
            role = self._speaker_roles.get(speaker, "unknown")
            if role == "user":
                color = "cyan"
                role_tag = "[cyan]user[/cyan]"
            elif role == "assistant":
                color = "magenta"
                role_tag = "[magenta]asst[/magenta]"
            else:
                color = "white"
                role_tag = "[white]?[/white]"
            start = seg.get("start", 0)
            safe_text = markup_escape(seg["text"])
            log.write(
                f"[{color}][{speaker}][/{color}] {role_tag} [{start:.1f}s]  {safe_text}"
            )

    def action_play_audio(self) -> None:
        if not self._current_mp3:
            return
        self.action_stop_audio()
        status = self.query_one("#audio-status", Static)
        status.update(f"[yellow]Playing {self._current_mp3.name}...[/yellow]")
        try:
            self._audio_proc = subprocess.Popen(
                ["afplay", str(self._current_mp3)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            asyncio.create_task(self._watch_audio())
        except FileNotFoundError:
            status.update("[red]afplay not found (macOS only)[/red]")

    async def _watch_audio(self) -> None:
        if not self._audio_proc:
            return
        await asyncio.get_event_loop().run_in_executor(None, self._audio_proc.wait)
        status = self.query_one("#audio-status", Static)
        if self._current_mp3:
            status.update(f"[dim]{self._current_mp3.name}[/dim]")

    def action_stop_audio(self) -> None:
        if self._audio_proc and self._audio_proc.poll() is None:
            self._audio_proc.terminate()
            self._audio_proc = None
        if self._current_mp3:
            status = self.query_one("#audio-status", Static)
            status.update(f"[dim]{self._current_mp3.name}[/dim]")

    def action_list_down(self) -> None:
        self.query_one("#file-list", ListView).action_cursor_down()

    def action_list_up(self) -> None:
        self.query_one("#file-list", ListView).action_cursor_up()

    def action_swap_roles(self) -> None:
        r0 = self._speaker_roles["speaker_0"]
        r1 = self._speaker_roles["speaker_1"]
        self._speaker_roles["speaker_0"] = r1
        self._speaker_roles["speaker_1"] = r0
        self._update_speaker_label()
        if self._current_mp3:
            self._show_transcript(self._current_mp3)

    def action_transcribe_selected(self) -> None:
        lv = self.query_one("#file-list", ListView)
        if lv.highlighted_child and isinstance(lv.highlighted_child, FileItem):
            item = lv.highlighted_child
            if item.file_status not in ("running",):
                asyncio.create_task(self._run_single(item))

    def action_transcribe_all(self) -> None:
        if self._running:
            return
        self._stop_flag = False
        asyncio.create_task(self._run_all())

    def action_stop(self) -> None:
        self._stop_flag = True
        self._running = False
        self._update_stats()

    def action_export_training(self) -> None:
        if not self._current_mp3:
            log = self.query_one("#transcript-log", RichLog)
            log.write("[red]No file selected[/red]")
            return
        from transcriber import get_output_path
        out = get_output_path(self._current_mp3)
        if not out.exists():
            log = self.query_one("#transcript-log", RichLog)
            log.write("[red]No transcript to export[/red]")
            return

        data = json.loads(out.read_text())
        segments = data.get("segments", [])

        messages = []
        for seg in segments:
            speaker = seg["speaker"]
            role = self._speaker_roles.get(speaker, "user")
            text = seg["text"].strip()
            if not text:
                continue
            # Merge consecutive same-role messages
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += " " + text
            else:
                messages.append({"role": role, "content": text})

        record = {
            "metadata": {
                "company": self.company,
                "file": self._current_mp3.name,
                "speaker_roles": self._speaker_roles.copy(),
            },
            "messages": messages,
        }

        export_dir = Path("output") / self.company / "training"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / "training_data.jsonl"

        with open(export_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log = self.query_one("#transcript-log", RichLog)
        log.write(f"[green]Exported → {export_path}[/green]")

    async def _run_single(self, item: FileItem) -> None:
        log = self.query_one("#transcript-log", RichLog)
        was_done = item.file_status == "done"
        item.update_status("running")
        log.write(f"[yellow]Transcribing {item.mp3_path.name}...[/yellow]")
        try:
            await process_file(item.mp3_path, self.api_key, self.progress, self.provider)
            item.update_status("done")
            if not was_done:
                self._done_count += 1
            self._update_stats()
            self._show_transcript(item.mp3_path)
            log.write(f"[green]Done: {item.mp3_path.name}[/green]")
        except Exception as e:
            item.update_status("error")
            log.write(f"[red]Error {item.mp3_path.name}: {e}[/red]")

    async def _run_all(self) -> None:
        self._running = True
        self._update_stats()
        log = self.query_one("#transcript-log", RichLog)

        items: list[FileItem] = [
            i for i in self.query(FileItem)
            if i.file_status not in ("done",)
        ]
        log.write(f"[bold]Queued {len(items)} files (concurrency={MAX_CONCURRENT})[/bold]")

        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def _bounded(item: FileItem) -> None:
            if self._stop_flag:
                return
            async with sem:
                if self._stop_flag:
                    return
                await self._run_single(item)

        await asyncio.gather(*[_bounded(i) for i in items])
        self._running = False
        self._update_stats()
        log.write("[bold green]All done![/bold green]")
