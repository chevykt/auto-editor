"""Repeats that Whisper collapsed into a single word token.

Whisper normalises speech. Say "the the" and it often returns one `the` - but
with timestamps spanning BOTH attempts, because the acoustic model still saw
them. The transcript loses the repeat; the audio does not.

Measured on a real recording: Whisper emitted one `the` at 10.00-10.40, and
inside that span sat 70ms of speech, 150ms of near-silence, then the second
attempt. No transcript-level detector can see that. This module looks at the
audio instead.

The impure/pure split matters. `audio_envelope` shells out to ffmpeg; every
other function here is arithmetic over a plain list of numbers, so the
detection logic is testable with hand-built envelopes and no media at all -
the same property that makes the rest of the pipeline debuggable.
"""
from __future__ import annotations

import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import EditorConfig
from models import CutReason, CutSpan, Word

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Envelope:
    """Frame-by-frame loudness. `frames[i]` covers `i * frame_s` seconds in."""
    frames: list[float]
    frame_s: float

    def index_at(self, seconds: float) -> int:
        return int(seconds / self.frame_s)


@dataclass(frozen=True)
class Gap:
    """A run of near-silence sitting inside one word token."""
    word_index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def audio_envelope(video: str | Path, frame_ms: int = 10) -> Envelope:
    """RMS loudness per frame, via one ffmpeg pass. The only impure function here."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg")

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        capture_output=True, check=True).stdout

    samples = struct.unpack(f"<{len(raw) // 2}h", raw[:len(raw) // 2 * 2])
    size = max(1, int(SAMPLE_RATE * frame_ms / 1000))

    frames: list[float] = []
    for i in range(0, len(samples) - size + 1, size):
        chunk = samples[i:i + size]
        frames.append(math.sqrt(sum(s * s for s in chunk) / size))

    return Envelope(frames=frames, frame_s=size / SAMPLE_RATE)


def silence_threshold(envelope: Envelope, floor_pct: float = 0.15) -> float:
    """Loudness below which a frame counts as silence.

    Derived from the recording rather than fixed, so it survives different mic
    levels and room tone. Sits a fraction of the way from the noise floor up to
    the speech level, both taken as percentiles so a single click or a breath
    cannot move it.
    """
    if not envelope.frames:
        return 0.0

    ordered = sorted(envelope.frames)
    noise = ordered[len(ordered) // 10]              # 10th percentile
    speech = ordered[len(ordered) * 9 // 10]         # 90th percentile
    return noise + floor_pct * max(0.0, speech - noise)


def find_gaps(envelope: Envelope, words: list[Word],
              cfg: EditorConfig) -> list[Gap]:
    """Runs of silence inside word tokens, long enough to separate two attempts.

    A gap touching either edge of the token is ignored: that is boundary slop
    in Whisper's timestamps, not a pause between attempts. Only interior
    silence with real speech on both sides counts.
    """
    if not envelope.frames or not words:
        return []

    threshold = silence_threshold(envelope)
    minimum = cfg.intraword_min_gap_ms / 1000.0
    flank = max(1, int(cfg.intraword_flank_ms / (envelope.frame_s * 1000)))
    gaps: list[Gap] = []

    for index, word in enumerate(words):
        lo = envelope.index_at(word.start)
        hi = min(envelope.index_at(word.end), len(envelope.frames))
        if hi - lo < 3:
            continue

        run_start: int | None = None
        for i in range(lo, hi):
            quiet = envelope.frames[i] <= threshold
            if quiet and run_start is None:
                run_start = i
            elif not quiet and run_start is not None:
                _keep(gaps, index, run_start, i, lo, hi, envelope,
                      minimum, threshold, flank)
                run_start = None
        # a run reaching the token's end is edge slop, not an interior gap

    return gaps


def _keep(gaps: list[Gap], index: int, first: int, last: int,
          lo: int, hi: int, envelope: Envelope, minimum: float,
          threshold: float, flank: int) -> None:
    """Record a silence run only if real speech sits on BOTH sides of it.

    Whisper's word tokens routinely run into the following pause, so a word
    ending a phrase has trailing silence inside its own span. Rejecting only
    leading silence let those through, and the first real run flagged
    'minutes.', 'from', 'part.' and 'you' as stutters - cutting three seconds
    of ordinary speech.

    Requiring `flank` loud frames before AND after the gap is what separates a
    pause between two attempts from a token that simply overhangs a pause.
    """
    if (last - first) * envelope.frame_s < minimum:
        return

    loud_before = sum(1 for i in range(lo, first)
                      if envelope.frames[i] > threshold)
    loud_after = sum(1 for i in range(last, hi)
                     if envelope.frames[i] > threshold)
    if loud_before < flank or loud_after < flank:
        return

    gaps.append(Gap(word_index=index,
                    start=round(first * envelope.frame_s, 6),
                    end=round(last * envelope.frame_s, 6)))


def detect_collapsed_repeats(words: list[Word], gaps: list[Gap],
                             cfg: EditorConfig) -> list[CutSpan]:
    """Cut the earlier attempt out of a token that holds more than one.

    The cut runs from the token's start to the end of its LAST interior gap,
    so "the the the" collapsed into one token still leaves only the final
    attempt. Capped by intraword_max_cut_s: a collapsed repeat of a short word
    is short, and anything longer is more likely a mis-timestamped token than
    a stutter.
    """
    if not cfg.intraword_repeats:
        return []

    last_gap: dict[int, Gap] = {}
    for gap in gaps:
        current = last_gap.get(gap.word_index)
        if current is None or gap.end > current.end:
            last_gap[gap.word_index] = gap

    cuts: list[CutSpan] = []
    for index, gap in sorted(last_gap.items()):
        word = words[index]
        if gap.end - word.start > cfg.intraword_max_cut_s:
            continue
        cuts.append(CutSpan(
            word.start, gap.end, {CutReason.DISFLUENCY},
            f"collapsed repeat: {word.text!r} spoken twice in one token"))

    return cuts
