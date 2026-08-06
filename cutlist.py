"""Turn raw detector output into a final, renderable cut list.

The three detectors run independently and will produce colliding spans - a
disfluency inside a losing retake, silence abutting both. Merging is not
cosmetic: overlapping spans produce negative-duration ffmpeg trims.
"""
from __future__ import annotations

from config import EditorConfig
from models import CutReason, CutSpan, Span


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


def padding_for(reasons: set[CutReason], cfg: EditorConfig) -> tuple[float, float]:
    """Signed offsets for (start, end) of a cut with these reasons.

    Positive moves a boundary later. Direction depends on what the span
    CONTAINS, which is the thing that is easy to get backwards:

    A silence span contains no speech, so shrinking it inward keeps a margin
    of quiet on each side and cannot clip a breath or a plosive.

    A disfluency or retake span IS the speech being removed. Shrinking it
    inward leaves the edges of the fumble audible - you still hear the front
    of the abandoned word. Those spans grow outward instead, so the whole
    fragment goes. The cost is up to padding_ms of the neighbouring keeper,
    which is the right trade: a clipped syllable is less distracting than
    hearing the mistake you were trying to remove.

    A mixed span counts as speech, because it contains some.
    """
    if reasons == {CutReason.SILENCE}:
        pad = cfg.silence_padding_s
        return (+pad, -pad)          # inward: shrink into the quiet
    pad = cfg.padding_s
    return (-pad, +pad)              # outward: swallow the whole fragment


def apply_padding(cuts: list[CutSpan], cfg: EditorConfig,
                  sources: list[CutSpan] | None = None) -> list[CutSpan]:
    """Pull every cut boundary inward so the cut lands inside the gap.

    Pulling inward makes the removed span smaller, never larger, so a cut
    cannot clip a breath or a plosive on either side. Spans too short to
    absorb the padding collapse to nothing and are dropped rather than
    inverting into a negative duration.

    Each boundary is padded by the reason that produced it, not by the merged
    span's combined reasons. A silence gap that merged with an adjacent
    stutter has a silence boundary at one end and a disfluency boundary at the
    other; padding both the same way is wrong at whichever end loses.

    sources: the pre-merge cuts, used to attribute each boundary. Defaults to
    `cuts` itself, which is correct when nothing has been merged yet.
    """
    origins = list(sources) if sources is not None else list(cuts)

    def offset_at(value: float, attr: str) -> float:
        """The signed shift for one boundary, from whichever cut produced it."""
        index = 0 if attr == "start" else 1
        for origin in origins:
            if abs(getattr(origin, attr) - value) < 1e-9:
                return padding_for(origin.reasons, cfg)[index]
        return padding_for({CutReason.DISFLUENCY}, cfg)[index]

    out: list[CutSpan] = []
    for c in cuts:
        start = max(0.0, c.start + offset_at(c.start, "start"))
        end = c.end + offset_at(c.end, "end")
        if end - start <= 1e-9:
            continue
        out.append(CutSpan(round(start, 6), round(end, 6),
                           set(c.reasons), c.detail))
    return out


def keep_segments(cuts: list[CutSpan], duration: float) -> list[Span]:
    """The complement of the cut list: everything that survives, in order."""
    segs: list[Span] = []
    t = 0.0
    for c in sorted(cuts, key=lambda c: c.start):
        if c.start > t:
            segs.append(Span(round(t, 6), round(c.start, 6)))
        t = max(t, c.end)
    if t < duration:
        segs.append(Span(round(t, 6), round(duration, 6)))
    return segs


def _absorb_slivers(cuts: list[CutSpan], duration: float,
                    cfg: EditorConfig) -> list[CutSpan]:
    """Delete surviving segments too short to be worth keeping.

    Single pass over the segment list, so absorbing one sliver can never
    trigger a chain that swallows a long neighbouring segment.
    """
    floor = cfg.min_segment_s
    if floor <= 0 or not cuts:
        return cuts

    extra: list[CutSpan] = []
    for seg in keep_segments(cuts, duration):
        # only interior slivers - never trim the very start or end of the video
        if seg.duration < floor and seg.start > 0.0 and seg.end < duration:
            extra.append(CutSpan(seg.start, seg.end, set(), "sliver absorbed"))

    return merge_spans(cuts + extra) if extra else cuts


def build_cutlist(cuts: list[CutSpan], duration: float,
                  cfg: EditorConfig) -> list[CutSpan]:
    """Merge, pad, then absorb slivers. This order matters.

    Merging first means padding is applied to final boundaries rather than to
    interior seams that are about to disappear. Sliver absorption runs last,
    on the geometry the renderer will actually see.
    """
    merged = merge_spans(cuts)
    padded = apply_padding(merged, cfg, sources=cuts)
    return _absorb_slivers(padded, duration, cfg)
