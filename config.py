"""Every tuning knob, one place. Nothing in this project hardcodes a threshold."""
from __future__ import annotations

from dataclasses import dataclass


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

    # Cut list
    # 30ms, not the 75ms the design spec proposed. Calibrated against a real
    # recording: a duplicated "the" ran 220ms, and 75ms pulled in from each
    # side left only 70ms actually removed - two thirds of the stutter stayed
    # in. 30ms still lands the cut inside the gap without clipping a plosive.
    padding_ms: int = 30
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
    def min_segment_s(self) -> float:
        return self.min_segment_ms / 1000.0

    @property
    def crossfade_s(self) -> float:
        return self.crossfade_ms / 1000.0
