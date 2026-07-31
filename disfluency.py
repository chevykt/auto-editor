"""Narrow, local detection of stutters and false starts.

Every detector here is deliberately conservative. Leaving a minor stutter in
is a small cost; trimming an intentional ad-lib is a large one. All three
require temporal adjacency, not merely shared vocabulary, which is what stops
an ad-lib that reuses a nearby script word from being eaten.
"""
from __future__ import annotations

from config import EditorConfig
from models import CutReason, CutSpan, Word, normalize

FILLERS = frozenset({"uh", "um", "er", "ah", "erm", "hmm", "mm"})


def detect_immediate_repeats(words: list[Word], lo: int, hi: int,
                             cfg: EditorConfig) -> list[CutSpan]:
    """The same word twice in a row, optionally with fillers between.

    Cuts the earlier occurrence and any fillers, leaving the clean one.
    """
    cuts: list[CutSpan] = []
    i = lo
    while i < hi - 1:
        first = normalize(words[i].text)
        if not first:
            i += 1
            continue

        j = i + 1
        while j < hi and normalize(words[j].text) in FILLERS:
            j += 1

        if j < hi and normalize(words[j].text) == first:
            if words[j].start - words[i].end <= cfg.repeat_max_gap_s:
                cuts.append(CutSpan(
                    words[i].start, words[j].start, {CutReason.DISFLUENCY},
                    f"immediate repeat: {first!r}"))
                i = j
                continue
        i += 1
    return cuts
