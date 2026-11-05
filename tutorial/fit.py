"""Fitting the recorded footage to the voiceover.

The voiceover is fixed; the picture bends to it. Within each beat, fixed points
pin the footage to the voice: the beat's start to its window's start, each
anchored click to the word that names it, and the beat's end to its window's
end. Between two fixed points the footage is shown as it happened. If there is
time to spare it waits first, so a click lands on its word rather than early.
If there is too little, waiting (page loads) is cut before anything is sped up,
and speed-ups stop at MAX_SPEED: past that, the beat needs fewer steps or more
narration, not a frantic video.
"""
from __future__ import annotations

from dataclasses import dataclass

from tutorial.plan import Plan
from tutorial.runner import SETTLE_S, StepEvent
from tutorial.voice import VoiceTiming

MAX_SPEED = 2.0
KEEP_OF_WAIT_S = 0.2        # a cut wait still shows its start, so the load is seen
_WAIT_KINDS = ("wait_for", "pause", "goto")
OPENING_WORD_S = 0.05       # an anchor this close to the last fixed point needs no pin


class FitError(ValueError):
    """A beat has more action than its narration can carry."""


@dataclass(frozen=True)
class Play:
    src_start: float        # capture seconds
    src_end: float
    speed: float


@dataclass(frozen=True)
class Hold:
    src_at: float           # the capture frame shown
    seconds: float


Piece = Play | Hold


def _cut_waits(s0: float, s1: float, waits: list[tuple[float, float]],
               target: float) -> list[tuple[float, float]]:
    """(s0, s1) with the late part of its longest waits removed, until it fits
    target or nothing more can go. Returns the parts to keep, in order."""
    inside = sorted(((max(a, s0), min(b, s1)) for a, b in waits if b > s0 and a < s1),
                    key=lambda w: w[1] - w[0], reverse=True)
    length, removed = s1 - s0, []
    for a, b in inside:
        if length <= target:
            break
        cut = min(b - a - KEEP_OF_WAIT_S, length - target)
        if cut > 0:
            removed.append((b - cut, b))
            length -= cut

    kept, t = [], s0
    for a, b in sorted(removed):
        kept.append((t, a))
        t = b
    kept.append((t, s1))
    return [(a, b) for a, b in kept if b > a]


def fit_beat(span: tuple[float, float], waits: list[tuple[float, float]],
             marks: list[tuple[float, float]],
             window: tuple[float, float]) -> list[Piece]:
    """Pieces that show span (capture seconds) across window (voice seconds)."""
    points = [(span[0], window[0]), *marks, (span[1], window[1])]
    pieces: list[Piece] = []
    for i, ((s0, t0), (s1, t1)) in enumerate(zip(points, points[1:])):
        source, target = s1 - s0, t1 - t0
        last = i == len(points) - 2
        if source <= target:
            play = [Play(s0, s1, 1)] if source > 0 else []
            hold = [Hold(s1 if last else s0, target - source)] if target > source else []
            pieces += play + hold if last else hold + play
            continue

        kept = _cut_waits(s0, s1, waits, target)
        length = sum(b - a for a, b in kept)
        speed = max(1, length / target)
        if speed > MAX_SPEED:
            raise FitError(f"{source:.1f}s of action for {target:.1f}s of narration "
                           f"would need {speed:.1f}x speed (limit {MAX_SPEED:g}x)")
        pieces += [Play(a, b, speed) for a, b in kept]
    return pieces


def fit_all(events: list[StepEvent], t0: float, timing: VoiceTiming,
            plan: Plan) -> tuple[list[Piece], list[str]]:
    """Pieces for the whole video, beat after beat, plus warnings."""
    pieces: list[Piece] = []
    warnings: list[str] = []
    last_frame = 0.0
    for b, window in enumerate(timing.windows):
        beat_events = [e for e in events if e.beat == b]
        if not beat_events:
            pieces.append(Hold(last_frame, window[1] - window[0]))
            continue

        span = (beat_events[0].start - t0, beat_events[-1].end - t0 + SETTLE_S)
        waits = [(e.start - t0, e.end - t0) for e in beat_events if e.kind in _WAIT_KINDS]
        marks: list[tuple[float, float]] = []
        for e in beat_events:
            said = timing.anchors.get((b, e.step))
            if said is None:
                continue
            done = e.action - t0
            previous = marks[-1] if marks else (span[0], window[0])
            if previous[0] < done < span[1] and previous[1] < said < window[1]:
                marks.append((done, said))
            elif abs(said - previous[1]) < OPENING_WORD_S:
                pass        # the word opens the beat; the step already starts with it
            else:
                warnings.append(f"beat {b + 1}, step {e.step + 1}: its word is said "
                                "outside where this step can happen, so it is not "
                                "pinned to it")
        try:
            pieces += fit_beat(span, waits, marks, window)
        except FitError as exc:
            raise FitError(f"beat {b + 1}: {exc}") from exc
        last_frame = span[1]
    return pieces, warnings
