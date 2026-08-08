"""faster-whisper adapter.

The only module that knows a speech model exists. Its entire contract with the
rest of the project is `list[Word]` plus a duration, which is what makes every
downstream stage testable without audio.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from config import EditorConfig
from models import Word, normalize

# Bump this whenever transcription behaviour changes in EITHER direction.
# Reverting a change needs a bump just as much as making one: v4 was cut with
# an experimental decoding setting that lost a mid-line restart, and because
# the revert kept version 4 the cache stayed valid and a render was built from
# the bad transcript. The key covers model, language and prompt - it cannot
# see decoding parameters, so this number is the only guard.
CACHE_VERSION = 5

ProgressFn = Callable[[float], None]

# Whisper is trained on clean, readable transcripts, so it normalises speech:
# it strips fillers, collapses repeated words, and repairs mid-word restarts.
# That is the behaviour it was built for, and it destroys exactly the signal
# the disfluency detectors need. Whisper conditions on initial_prompt
# stylistically, so a deliberately disfluent sample biases it toward verbatim
# output. The cost is that it can also hallucinate fillers that were never
# spoken - which is why this is opt-in and the report lists every cut.
VERBATIM_PROMPT = (
    "Um, so, uh - I was gonna, gonna say that the the config- configuration, "
    "it's, it's not, not quite right yet."
)


def effective_prompt(cfg: EditorConfig) -> str | None:
    """The initial_prompt actually handed to Whisper, if any."""
    if cfg.initial_prompt is not None:
        return cfg.initial_prompt
    return VERBATIM_PROMPT if cfg.verbatim else None


def transcription_key(cfg: EditorConfig) -> str:
    """Identity of a transcription run.

    Anything that changes the words Whisper produces must appear here, or a
    cache hit would silently serve a transcript made under different settings.
    """
    parts = [cfg.model, cfg.language or "auto", effective_prompt(cfg) or ""]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


# Transcripts live here rather than beside the video. Module-level so tests can
# point it somewhere temporary instead of polluting a real cache.
CACHE_DIR = Path.home() / ".auto-editor" / "transcripts"

_HASH_CHUNK = 1024 * 1024


def content_hash(video: str | Path) -> str:
    """Identity of a recording by its BYTES, not its path.

    The web UI copies every upload into a fresh job folder, so a
    location-based cache never hit and each re-run cost a full transcription -
    minutes, to try a different script split. Hashing means the same recording
    is recognised wherever it sits.

    Reading 124MB costs a fraction of a second against the minutes a
    re-transcription costs, so the whole file is hashed rather than sampled.
    """
    digest = hashlib.sha256()
    with open(video, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def cache_path_for(video: str | Path, key: str | None = None) -> Path:
    """Where this recording's transcript lives, for these settings.

    The settings key is in the FILENAME, not just the contents, so transcripts
    for different models coexist instead of overwriting each other every time
    you switch.
    """
    name = content_hash(video)
    if key:
        name = f"{name}.{key}"
    return CACHE_DIR / f"{name}.transcript.json"


def save_cache(path: str | Path, words: list[Word], duration: float,
               key: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": CACHE_VERSION,
        "key": key,
        "duration": duration,
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
    }, indent=2), encoding="utf-8")
    return path


def load_cached(path: str | Path,
                key: str) -> tuple[list[Word] | None, float | None]:
    """Return (words, duration), or (None, None) on any kind of miss."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    if data.get("version") != CACHE_VERSION or data.get("key") != key:
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


def _decode(model, video: Path, cfg: EditorConfig,
            on_progress: ProgressFn | None = None,
            duration: float | None = None) -> list[Word]:
    """Drain the segment generator into a flat word list.

    faster-whisper returns a lazy generator, so any GPU library failure
    surfaces here during iteration rather than at model construction. That is
    why the fallback in _run_whisper wraps this call, not just the constructor.
    """
    # Tried and rejected: condition_on_previous_text=False plus temperature=0
    # in verbatim mode. The theory was sound - Whisper conditions on its own
    # increasingly tidy output, which should dilute the disfluent prompt - but
    # measured on a real recording it made capture WORSE, dropping from 99 to
    # 96 words and losing a mid-line restart the default settings had caught.
    # Left at Whisper's defaults on the evidence.
    segments, _ = model.transcribe(str(video), word_timestamps=True,
                                   language=cfg.language,
                                   initial_prompt=effective_prompt(cfg))
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            # Whisper emits standalone punctuation as its own token - a mid-word
            # restart can arrive as 'config', '-', 'configuration', the hyphen
            # carrying no audio at all. Dropping it here keeps every downstream
            # stage from having to know that speech can be interrupted by a
            # token that is not speech.
            if text and normalize(text):
                words.append(Word(text, float(w.start), float(w.end)))

        if on_progress and duration:
            # the final segment can run past the probed duration
            on_progress(min(seg.end / duration, 1.0))

    return words


def _device_attempts(cfg: EditorConfig) -> list[tuple[str, str]]:
    """(device, compute_type) pairs to try, in order."""
    if cfg.device != "auto":
        return [(cfg.device, cfg.compute_type)]
    # CUDA is worth trying, but a machine can report a GPU while lacking the
    # CUDA runtime entirely, so CPU must always be reachable as a fallback.
    return [("cuda", "float16"), ("cpu", "int8")]


def _run_whisper(video: Path, cfg: EditorConfig,
                 on_progress: ProgressFn | None = None,
                 duration: float | None = None) -> list[Word]:
    """Load the model and transcribe. Isolated so tests can stub it out."""
    from faster_whisper import WhisperModel

    attempts = _device_attempts(cfg)
    for i, (device, compute_type) in enumerate(attempts):
        last = i == len(attempts) - 1
        try:
            model = WhisperModel(cfg.model, device=device,
                                 compute_type=compute_type)
            return _decode(model, video, cfg, on_progress, duration)
        except (RuntimeError, ValueError) as exc:
            if last:
                raise
            print(f"  {device} unavailable ({exc}); falling back to "
                  f"{attempts[i + 1][0]}", file=sys.stderr)

    raise RuntimeError("no transcription device available")


def transcribe(video: str | Path, cfg: EditorConfig,
               on_progress: ProgressFn | None = None
               ) -> tuple[list[Word], float]:
    """Word-level transcript plus media duration, cached beside the video.

    on_progress receives a fraction from 0.0 to 1.0 as transcription advances.
    A cache hit reports 1.0 once and loads no model.
    """
    video = Path(video)
    key = transcription_key(cfg)
    cache = cache_path_for(video, key)

    if cfg.use_cache:
        words, duration = load_cached(cache, key)
        if words is not None and duration is not None:
            if on_progress:
                on_progress(1.0)
            return words, duration

    duration = probe_duration(video)
    words = _run_whisper(video, cfg, on_progress, duration)
    save_cache(cache, words, duration, key)
    return words, duration
