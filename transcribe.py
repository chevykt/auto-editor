"""faster-whisper adapter.

The only module that knows a speech model exists. Its entire contract with the
rest of the project is `list[Word]` plus a duration, which is what makes every
downstream stage testable without audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from config import EditorConfig
from models import Word

CACHE_VERSION = 1


def cache_path_for(video: str | Path) -> Path:
    video = Path(video)
    return video.with_name(video.stem + ".transcript.json")


def save_cache(path: str | Path, words: list[Word], duration: float,
               model: str) -> Path:
    path = Path(path)
    path.write_text(json.dumps({
        "version": CACHE_VERSION,
        "model": model,
        "duration": duration,
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
    }, indent=2), encoding="utf-8")
    return path


def load_cached(path: str | Path,
                model: str) -> tuple[list[Word] | None, float | None]:
    """Return (words, duration), or (None, None) on any kind of miss."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    if data.get("version") != CACHE_VERSION or data.get("model") != model:
        return None, None
    words = [Word(w["text"], w["start"], w["end"]) for w in data["words"]]
    return words, data["duration"]


def probe_duration(video: str | Path) -> float:
    """Media length in seconds, via ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg: winget install Gyan.FFmpeg")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _run_whisper(video: Path, cfg: EditorConfig) -> list[Word]:
    """Load the model and transcribe. Isolated so tests can stub it out."""
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg.model, device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(video), word_timestamps=True,
                                   language=cfg.language)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append(Word(text, float(w.start), float(w.end)))
    return words


def transcribe(video: str | Path,
               cfg: EditorConfig) -> tuple[list[Word], float]:
    """Word-level transcript plus media duration, cached beside the video."""
    video = Path(video)
    cache = cache_path_for(video)

    if cfg.use_cache:
        words, duration = load_cached(cache, cfg.model)
        if words is not None and duration is not None:
            return words, duration

    duration = probe_duration(video)
    words = _run_whisper(video, cfg)
    save_cache(cache, words, duration, cfg.model)
    return words, duration
