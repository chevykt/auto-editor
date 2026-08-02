"""Gap-based silence detection.

Uses gaps between consecutive word timestamps rather than a separate ffmpeg
silencedetect pass, reusing data already fetched during transcription.
"""
from __future__ import annotations

from config import EditorConfig
from models import CutReason, CutSpan, Word


def detect_silence(words: list[Word], cfg: EditorConfig,
                   duration: float | None = None) -> list[CutSpan]:
    """Find gaps with no speech at or above cfg.silence_ms.

    duration: total media length. When supplied, trailing dead air after the
    last word is also cut. Leading dead air is always checked.
    """
    if not words:
        return []

    cuts: list[CutSpan] = []
    min_gap = cfg.silence_s

    if words[0].start >= min_gap:
        cuts.append(CutSpan(0.0, words[0].start, {CutReason.SILENCE},
                            f"leading silence {words[0].start:.2f}s"))

    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if gap >= min_gap:
            cuts.append(CutSpan(a.end, b.start, {CutReason.SILENCE},
                                f"gap {gap:.2f}s"))

    if duration is not None:
        tail = duration - words[-1].end
        if tail >= min_gap:
            cuts.append(CutSpan(words[-1].end, duration, {CutReason.SILENCE},
                                f"trailing silence {tail:.2f}s"))

    return cuts
