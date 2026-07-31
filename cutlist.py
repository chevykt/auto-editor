"""Turn raw detector output into a final, renderable cut list.

The three detectors run independently and will produce colliding spans - a
disfluency inside a losing retake, silence abutting both. Merging is not
cosmetic: overlapping spans produce negative-duration ffmpeg trims.
"""
from __future__ import annotations

from config import EditorConfig
from models import CutSpan, Span


def merge_spans(cuts: list[CutSpan]) -> list[CutSpan]:
    """Merge overlapping and touching spans, unioning reasons and details.

    Returns new objects; the input list and its members are not mutated.
    """
    if not cuts:
        return []

    out: list[CutSpan] = []
    for c in sorted(cuts, key=lambda c: (c.start, c.end)):
        if out and c.start <= out[-1].end:
            last = out[-1]
            last.end = max(last.end, c.end)
            last.reasons |= set(c.reasons)
            details = [d for d in (last.detail, c.detail) if d]
            last.detail = "; ".join(details)
        else:
            out.append(CutSpan(c.start, c.end, set(c.reasons), c.detail))
    return out
