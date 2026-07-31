"""Fuzzy sliding-window alignment of script lines against the transcript.

Two ideas carry this module:

1. Windows are scored with rapidfuzz rather than compared exactly, because a
   good take still paraphrases slightly.
2. Lines are matched in script order against a forward-moving cursor, so two
   similarly-worded lines cannot steal each other's occurrence.
"""
from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from config import EditorConfig
from models import (CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine,
                    Span, Word, normalize)


@dataclass(frozen=True)
class Candidate:
    score: float
    start_idx: int
    end_idx: int      # exclusive


def _window_text(words: list[Word], i: int, j: int) -> str:
    return " ".join(normalize(w.text) for w in words[i:j])


def find_candidates(words: list[Word], lo: int, hi: int, line_text: str,
                    cfg: EditorConfig) -> list[Candidate]:
    """Score every plausible window in words[lo:hi] against line_text.

    Window sizes range over the line's word count +/- cfg.window_tolerance,
    since a spoken take rarely has exactly the scripted word count.
    """
    target = normalize(line_text)
    wc = len(line_text.split())
    if wc == 0 or hi - lo <= 0:
        return []

    smallest = max(1, int(wc * (1.0 - cfg.window_tolerance)))
    largest = max(smallest, int(wc * (1.0 + cfg.window_tolerance)) + 1)

    out: list[Candidate] = []
    for size in range(smallest, largest + 1):
        for i in range(lo, hi - size + 1):
            score = fuzz.token_sort_ratio(target, _window_text(words, i, i + size)) / 100.0
            if score >= cfg.threshold:
                out.append(Candidate(score, i, i + size))
    return out


def suppress_overlaps(cands: list[Candidate]) -> list[Candidate]:
    """Collapse overlapping windows into one best window per distinct attempt.

    Returned in timeline order. Each survivor represents a separate spoken
    attempt at the line - which is exactly what retake resolution consumes.
    """
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (-c.score, c.start_idx)):
        if any(not (c.end_idx <= k.start_idx or c.start_idx >= k.end_idx) for k in kept):
            continue
        kept.append(c)
    return sorted(kept, key=lambda c: c.start_idx)
