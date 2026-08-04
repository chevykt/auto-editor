"""Reconstructive mode: rebuild the video from matched windows, in script order.

This is the opposite of the default. The subtractive pipeline starts from the
full recording and removes only what it is confident about, so anything it does
not recognise - an ad-lib, an aside, extra explanation - survives untouched.
Reorder mode keeps ONLY what matched a script line and discards everything
else, then concatenates those windows in script order rather than the order
they were spoken.

That trade is only worth making when lines were deliberately recorded out of
sequence and need reassembling. Any ad-libbed content is lost, and the natural
flow between beats goes with it. Off by default for exactly that reason.
"""
from __future__ import annotations

from config import EditorConfig
from models import CutSpan, LineMatch, Span, Word


def _subtract(segment: Span, cuts: list[CutSpan]) -> list[Span]:
    """Split a segment around any cuts falling inside it, preserving order.

    A matched window can still contain a stutter. Reorder mode keeps the
    window, so the disfluency has to be removed from within it rather than
    by the cut list the subtractive path would have used.
    """
    pieces: list[Span] = []
    t = segment.start

    for cut in sorted(cuts, key=lambda c: c.start):
        if cut.end <= segment.start or cut.start >= segment.end:
            continue
        if cut.start > t:
            pieces.append(Span(round(t, 6), round(min(cut.start, segment.end), 6)))
        t = max(t, cut.end)

    if t < segment.end:
        pieces.append(Span(round(t, 6), round(segment.end, 6)))
    return pieces


def matched_window(words: list[Word], match: LineMatch,
                   cfg: EditorConfig) -> Span | None:
    """The span of a line's winning match, padded outward into surrounding silence.

    Reorder mode cuts at word boundaries, not into a natural gap, so a window
    taken exactly at the timestamps clips the attack of the first word and the
    tail of the last. The pad is bounded by the neighbouring word, so it can
    never reach into speech belonging to something else.
    """
    if match.word_range is None:
        return None

    i, j = match.word_range
    pad = cfg.reorder_pad_s

    start = words[i].start - pad
    start = max(words[i - 1].end, start) if i > 0 else max(0.0, start)

    end = words[j - 1].end + pad
    if j < len(words):
        end = min(words[j].start, end)

    return Span(round(start, 6), round(end, 6))


def reorder_segments(words: list[Word], matches: list[LineMatch],
                     cuts: list[CutSpan], cfg: EditorConfig) -> list[Span]:
    """Surviving segments for reorder mode, in script order.

    Missed lines contribute nothing - there is no take to reassemble. The
    report still flags them, so a hole in the script is visible rather than
    silently producing a shorter video.
    """
    segments: list[Span] = []

    for match in sorted(matches, key=lambda m: m.line_idx):
        window = matched_window(words, match, cfg)
        if window is None:
            continue
        segments.extend(_subtract(window, cuts))

    return segments
