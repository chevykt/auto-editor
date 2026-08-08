"""Tokens long enough to be hiding a discarded attempt.

Whisper normalises speech. Say "and 80 to 2— and 80 to 120" and it can return a
single `and`, with timestamps spanning the whole false start. Nothing in the
transcript shows the fumble; the only trace is that `and` ran 2.06s against a
0.32s median.

These are reported as SUSPECTS, never cut. A long token has innocent
explanations - a drawn-out word, a trailing pause absorbed into the timestamp -
and a previous attempt at cutting this class automatically produced six wrong
cuts on a real recording. Flagging costs nothing when wrong; cutting does not.

The proposed boundary is an estimate: everything up to roughly one word's width
before the token ends, on the assumption that the surviving attempt is the last
one. Worth checking by ear before applying, which is exactly what the review
screen is for.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from config import EditorConfig
from models import CutReason, CutSpan, Word


@dataclass(frozen=True)
class Suspect:
    word_index: int
    text: str
    start: float
    end: float
    ratio: float          # duration as a multiple of the median

    @property
    def duration(self) -> float:
        return self.end - self.start


def find_suspects(words: list[Word], cfg: EditorConfig) -> list[Suspect]:
    """Tokens whose duration is an extreme outlier for this recording.

    Both tests must pass. The ratio catches what is abnormal for this speaker;
    the absolute floor stops a recording of very short words from flagging
    ordinary ones just because the median is tiny.

    Needs enough words for the median to mean anything - a handful of tokens
    cannot establish what "normal" is.
    """
    if not cfg.flag_suspects or len(words) < cfg.suspect_min_words:
        return []

    durations = [w.end - w.start for w in words]
    median = statistics.median(durations)
    if median <= 0:
        return []

    suspects: list[Suspect] = []
    for index, word in enumerate(words):
        duration = word.end - word.start
        ratio = duration / median
        if ratio >= cfg.suspect_ratio and duration >= cfg.suspect_min_duration_s:
            suspects.append(Suspect(index, word.text, word.start, word.end,
                                    round(ratio, 2)))

    return suspects


def suspect_cuts(suspects: list[Suspect], words: list[Word],
                 cfg: EditorConfig) -> list[CutSpan]:
    """What would be removed if a suspect were approved.

    The surviving attempt is assumed to be the last one, so the span runs from
    the token's start to roughly one word's width before its end. That is an
    estimate, not a measurement - it is why these arrive unchecked.
    """
    if not suspects:
        return []

    median = statistics.median([w.end - w.start for w in words])
    cuts: list[CutSpan] = []

    for suspect in suspects:
        end = suspect.end - median
        if end <= suspect.start:
            continue
        cuts.append(CutSpan(
            suspect.start, round(end, 6), {CutReason.SUSPECT},
            f"possible hidden false start: {suspect.text!r} spans "
            f"{suspect.duration:.2f}s, {suspect.ratio:g}x the median"))

    return cuts
