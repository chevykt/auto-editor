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


def apply_padding(cuts: list[CutSpan], cfg: EditorConfig) -> list[CutSpan]:
    """Pull every cut boundary inward so the cut lands inside the gap.

    Pulling inward makes the removed span smaller, never larger, so a cut
    cannot clip a breath or a plosive on either side. Spans too short to
    absorb the padding collapse to nothing and are dropped rather than
    inverting into a negative duration.
    """
    pad = cfg.padding_s
    if pad <= 0:
        return [CutSpan(c.start, c.end, set(c.reasons), c.detail) for c in cuts]

    out: list[CutSpan] = []
    for c in cuts:
        start, end = c.start + pad, c.end - pad
        if end - start <= 1e-9:
            continue
        out.append(CutSpan(round(start, 6), round(end, 6),
                           set(c.reasons), c.detail))
    return out
