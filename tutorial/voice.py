"""Keon's voiceover, cleaned and timed: the fixed timeline the picture is fitted to.

The auto-editor engine does the cleaning (silence, losing takes and stutters
removed), so he can fumble and retake freely while recording. Every time here
is in the CLEANED voiceover's seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from aligner import align_script
from config import EditorConfig
from edit import analyse
from models import Span, Word, normalize
from render import render_audio
from script_parser import parse_script
from transcribe import transcribe
from tutorial.plan import Plan, script_text

ANCHOR_THRESHOLD = 0.8


class VoiceError(RuntimeError):
    """The read does not match the script well enough to time the video by."""

    def __init__(self, report: dict):
        bad = [f"beat {line['index'] + 1}: {line['status']}"
               for line in report["lines"] if line["status"] != "ok"]
        super().__init__("voiceover does not match the script - " + "; ".join(bad))
        self.report = report


@dataclass(frozen=True)
class VoiceTiming:
    duration: float
    windows: tuple[tuple[float, float], ...]
    anchors: dict[tuple[int, int], float]


def to_clean_time(t: float, kept: list[Span]) -> float | None:
    """Where t lands once the removed spans are gone; None if t was removed."""
    removed_before = 0.0
    previous_end = 0.0
    for seg in kept:
        removed_before += seg.start - previous_end
        if seg.start <= t <= seg.end:
            return t - removed_before
        previous_end = seg.end
    return None


def _next_clean_time(t: float, kept: list[Span]) -> float:
    """to_clean_time, or the next kept instant when t itself was cut."""
    for seg in kept:
        if seg.end >= t:
            return to_clean_time(max(t, seg.start), kept)
    return sum(seg.duration for seg in kept)


def beat_windows(starts: list[float], duration: float) -> tuple[tuple[float, float], ...]:
    """Windows that tile the whole voiceover: talk before the first beat belongs
    to it, and talk between beats belongs to the beat before."""
    edges = [0.0, *starts[1:], duration]
    return tuple((edges[i], edges[i + 1]) for i in range(len(starts)))


def find_anchor(words: list[Word], word_range: tuple[int, int],
                phrase: str) -> float | None:
    """Start of the phrase as spoken within word_range, or None if not said."""
    target = normalize(phrase)
    size = len(phrase.split())
    lo, hi = word_range
    best, best_start = 0.0, None
    for i in range(lo, hi - size + 1):
        spoken = " ".join(normalize(w.text) for w in words[i:i + size])
        score = fuzz.ratio(target, spoken) / 100.0
        if score > best:
            best, best_start = score, words[i].start
    return best_start if best >= ANCHOR_THRESHOLD else None


def clean_voiceover(voice: Path, plan: Plan, cfg: EditorConfig,
                    out_wav: Path) -> tuple[VoiceTiming, list[str]]:
    """Clean the read into out_wav and time every beat and anchor against it."""
    text = script_text(plan)
    words, duration = transcribe(voice, cfg)
    report = analyse(words, text, duration, cfg, str(voice))
    if any(line["status"] != "ok" for line in report["lines"]):
        raise VoiceError(report)

    kept = [Span(s["start"], s["end"]) for s in report["segments"]]
    render_audio(voice, out_wav, kept, cfg)

    # analyse() aligned these same words already; it reports times, not the
    # word ranges the anchor search needs, so the alignment is run again.
    lines = parse_script(text)
    if len(lines) != len(plan.beats):
        raise ValueError("a beat's 'say' contains a blank line; keep each beat "
                         "one paragraph")
    matches, _ = align_script(words, lines, cfg)

    starts: list[float] = []
    anchors: dict[tuple[int, int], float] = {}
    warnings: list[str] = []
    for b, (beat, match) in enumerate(zip(plan.beats, matches)):
        starts.append(_next_clean_time(words[match.word_range[0]].start, kept))
        for s, step in enumerate(beat.steps):
            if not step.at:
                continue
            spoken = find_anchor(words, match.word_range, step.at)
            if spoken is None:
                warnings.append(f"beat {b + 1}, step {s + 1}: {step.at!r} was not "
                                "said, so this step is spread evenly instead")
            else:
                anchors[(b, s)] = _next_clean_time(spoken, kept)

    clean_duration = sum(seg.duration for seg in kept)
    return VoiceTiming(clean_duration, beat_windows(starts, clean_duration),
                       anchors), warnings
