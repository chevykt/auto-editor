"""Every tuning knob, one place. Nothing in this project hardcodes a threshold."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

# Two knobs are expressed as negative flags on the command line but as positive
# fields here. A config file uses the positive form, which reads better, so the
# value is inverted on the way to argparse.
_INVERTED_FLAGS = {"clean_stutters": "no_clean_stutters",
                   "use_cache": "no_cache"}


@dataclass
class EditorConfig:
    # Alignment
    threshold: float = 0.75
    review_margin: float = 0.05
    lookahead_s: float = 30.0
    window_tolerance: float = 0.4   # window sizes searched: wc*(1±tolerance)
    tiebreak: str = "latest"        # "latest" | "score"
    retake_lookahead_lines: int = 5  # how far ahead to check for a rival claim

    # Mid-line restarts: an abandoned attempt that matches the START of a line
    partial_retakes: bool = True
    partial_retake_threshold: float = 0.8
    partial_retake_max_gap_s: float = 3.0
    partial_retake_min_words: int = 2

    # Silence
    silence_ms: int = 600

    # Disfluency
    clean_stutters: bool = True
    duplicate_window_s: float = 4.0
    max_disfluency_words: int = 5
    repeat_max_gap_s: float = 1.5
    restart_max_gap_s: float = 1.0
    min_fragment_chars: int = 3
    min_restart_growth: int = 3     # full word must exceed fragment by this many chars

    # Tokens long enough to be hiding a false start Whisper absorbed.
    # Flagged in the report, never cut unless approved. 5x is deliberately
    # conservative: on the one recording with a known case the true
    # positive ran 6.4x the median and the next candidate was 4.3x.
    flag_suspects: bool = True
    suspect_ratio: float = 5.0
    suspect_min_duration_s: float = 0.9
    suspect_min_words: int = 20

    # Reorder mode: rebuild from matched windows in script order
    reorder: bool = False
    # Reorder cuts at word boundaries rather than into silence, so segments
    # breathe outward a little to avoid clipping. Bounded by the neighbouring
    # word, so a segment can never swallow speech that belongs to something else.
    reorder_pad_ms: int = 60

    # Cut list
    # 30ms, not the 75ms the design spec proposed. Calibrated against a real
    # recording: a duplicated "the" ran 220ms, and 75ms pulled in from each
    # side left only 70ms actually removed - two thirds of the stutter stayed
    # in. 30ms still lands the cut inside the gap without clipping a plosive.
    padding_ms: int = 30
    # Silence cuts are bounded by speech on both sides, so they can afford - and
    # need - a generous pull-in to protect a breath or a plosive. A disfluency
    # cut is a short span where that same pull-in would eat what we are removing.
    silence_padding_ms: int = 75
    min_segment_ms: int = 150

    # Render
    crossfade_ms: int = 75
    video_codec: str = "libx264"
    video_crf: int = 18
    video_preset: str = "medium"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    # Transcription
    model: str = "medium"
    language: str | None = None
    use_cache: bool = True
    device: str = "auto"            # "auto" | "cuda" | "cpu"
    compute_type: str = "int8"      # only consulted when device is not "auto"
    verbatim: bool = False          # bias Whisper toward transcribing disfluencies
    initial_prompt: str | None = None   # overrides the built-in verbatim prompt

    @property
    def silence_s(self) -> float:
        return self.silence_ms / 1000.0

    @property
    def padding_s(self) -> float:
        return self.padding_ms / 1000.0

    @property
    def silence_padding_s(self) -> float:
        return self.silence_padding_ms / 1000.0

    @property
    def reorder_pad_s(self) -> float:
        return self.reorder_pad_ms / 1000.0

    @property
    def min_segment_s(self) -> float:
        return self.min_segment_ms / 1000.0

    @property
    def crossfade_s(self) -> float:
        return self.crossfade_ms / 1000.0


def tunable_names() -> set[str]:
    """Knob names a config file may set."""
    return {f.name for f in fields(EditorConfig)}


def load_config_file(path: str | Path) -> dict:
    """Tuning knobs from a TOML file, as argparse defaults.

    Only knobs live here - actions like --auto, --batch and --out stay on the
    command line. A run's *settings* are worth saving and reusing; what a
    particular invocation *does* is not.

    Unknown keys are an error rather than being ignored, so a typo surfaces
    instead of silently leaving a threshold at its default.
    """
    path = Path(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    unknown = set(data) - tunable_names()
    if unknown:
        raise ValueError(
            f"unknown keys in {path}: {', '.join(sorted(unknown))}\n"
            f"valid keys: {', '.join(sorted(tunable_names()))}")

    defaults: dict = {}
    for key, value in data.items():
        if key in _INVERTED_FLAGS:
            defaults[_INVERTED_FLAGS[key]] = not value
        else:
            defaults[key] = value
    return defaults
