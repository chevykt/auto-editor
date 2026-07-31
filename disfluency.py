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


def detect_truncated_restarts(words: list[Word], lo: int, hi: int,
                              cfg: EditorConfig) -> list[CutSpan]:
    """A cut-off word fragment immediately followed by the full word.

    Requires the fragment to be a literal prefix of the next word, at least
    cfg.min_fragment_chars long, and the full word to be meaningfully longer.
    Those guards keep it off legitimate sequences like "in international".
    """
    cuts: list[CutSpan] = []
    for i in range(lo, hi - 1):
        frag = normalize(words[i].text)
        full = normalize(words[i + 1].text)

        if len(frag) < cfg.min_fragment_chars:
            continue
        if not full.startswith(frag) or full == frag:
            continue
        if len(full) - len(frag) < cfg.min_restart_growth:
            continue
        if words[i + 1].start - words[i].end > cfg.restart_max_gap_s:
            continue

        cuts.append(CutSpan(
            words[i].start, words[i + 1].start, {CutReason.DISFLUENCY},
            f"truncated restart: {frag!r} -> {full!r}"))
    return cuts
