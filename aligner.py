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

    The primary score (token_sort_ratio) is order-insensitive by design, so a
    paraphrase that reorders clauses still matches. That insensitivity has a
    side effect: when a line is retaken with only a word or two changed, a
    window straddling the boundary between the rejected attempt's tail and
    the new attempt's head can contain the exact same bag of words as the
    true phrase and score an identical tie (e.g. "button click the blue"
    vs. "click the blue button" - same four words). An order-sensitive nudge
    (plain, position-aware ratio) is folded in at a scale far below any
    threshold or rounding boundary, purely to break that tie toward whichever
    window actually preserves the line's word order. It never changes which
    windows clear cfg.threshold, and it is exactly zero for a true match
    (order-ratio 1.0), so it never moves an already-exact score off of 1.0.
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
            window = _window_text(words, i, i + size)
            primary = fuzz.token_sort_ratio(target, window) / 100.0
            order = fuzz.ratio(target, window) / 100.0
            score = primary - (1.0 - order) * 1e-6
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


def resolve_retakes(cands: list[Candidate],
                    cfg: EditorConfig) -> tuple[Candidate, list[Candidate]]:
    """Pick the winning attempt at a line; the rest are losing retakes.

    "latest" rounds scores to 3dp before comparing, so two attempts that are
    functionally equally good resolve toward the later one - the take the
    speaker actually settled on. "score" keeps raw score and prefers the
    earlier attempt on an exact tie.
    """
    if not cands:
        raise ValueError("resolve_retakes requires at least one candidate")

    if cfg.tiebreak == "latest":
        winner = max(cands, key=lambda c: (round(c.score, 3), c.start_idx))
    else:
        winner = max(cands, key=lambda c: (c.score, -c.start_idx))

    losers = [c for c in cands if c is not winner]
    return winner, losers


def mean_seconds_per_word(words: list[Word]) -> float:
    """Average pace across the whole transcript, used to size lookahead."""
    if len(words) < 2:
        return 0.5
    span = words[-1].end - words[0].start
    return max(span / len(words), 0.05)


def _word_index_after(words: list[Word], t: float) -> int:
    """First index whose word starts at or after t; len(words) if none."""
    for i, w in enumerate(words):
        if w.start >= t:
            return i
    return len(words)


def _score_line_against(words: list[Word], cand: Candidate, text: str) -> float:
    return fuzz.token_sort_ratio(
        normalize(text), _window_text(words, cand.start_idx, cand.end_idx)) / 100.0


def _drop_candidates_owed_to_later_lines(
        words: list[Word], cands: list[Candidate], line: ScriptLine,
        later: list[ScriptLine], cfg: EditorConfig) -> list[Candidate]:
    """Stop a line from swallowing an occurrence a later line has a claim to.

    Two identical script lines produce two occurrences in the recording. Without
    this, line N sees both, treats them as a retake of itself, keeps the later
    one, and leaves line N+1 with nothing. A candidate is a genuine retake only
    if no upcoming line matches it at least as well.

    The first candidate is always kept - this line has to match something.
    """
    if len(cands) < 2 or not later:
        return cands

    horizon = later[:cfg.retake_lookahead_lines]
    kept = [cands[0]]
    for cand in cands[1:]:
        own = _score_line_against(words, cand, line.text)
        claimed = False
        for other in horizon:
            rival = _score_line_against(words, cand, other.text)
            if rival >= own and rival >= cfg.threshold:
                claimed = True
                break
        if not claimed:
            kept.append(cand)
    return kept


def align_script(words: list[Word], lines: list[ScriptLine],
                 cfg: EditorConfig) -> tuple[list[LineMatch], list[CutSpan]]:
    """Match every script line, in order, against the transcript.

    Returns the per-line matches and the RETAKE_LOSER cut spans produced by
    lines that were attempted more than once.

    The cursor only advances on a successful match, so one unmatchable line
    cannot cascade into failing every line after it.
    """
    matches: list[LineMatch] = []
    cuts: list[CutSpan] = []
    if not words:
        return ([LineMatch(l.index, None, 0.0, None, MatchStatus.MISS)
                 for l in lines], [])

    spw = mean_seconds_per_word(words)
    cursor = 0

    for pos, line in enumerate(lines):
        lookahead = max(cfg.lookahead_s, 3.0 * line.word_count * spw)
        region_end = len(words)
        if cursor < len(words):
            region_end = _word_index_after(words, words[cursor].start + lookahead)
            region_end = max(region_end, cursor + 1)

        scoped = True
        cands = suppress_overlaps(
            find_candidates(words, cursor, region_end, line.text, cfg))

        if not cands:
            scoped = False
            cands = suppress_overlaps(
                find_candidates(words, 0, len(words), line.text, cfg))

        if not cands:
            matches.append(LineMatch(line.index, None, 0.0, None,
                                     MatchStatus.MISS, scoped=False))
            continue

        cands = _drop_candidates_owed_to_later_lines(
            words, cands, line, lines[pos + 1:], cfg)

        winner, losers = resolve_retakes(cands, cfg)

        for loser in losers:
            end = words[winner.start_idx].start if loser.start_idx < winner.start_idx \
                else words[loser.end_idx - 1].end
            cuts.append(CutSpan(
                words[loser.start_idx].start, end, {CutReason.RETAKE_LOSER},
                f"losing take of line {line.index} (score {loser.score:.2f})"))

        status = (MatchStatus.REVIEW
                  if winner.score < cfg.threshold + cfg.review_margin
                  else MatchStatus.OK)

        matches.append(LineMatch(
            line_idx=line.index,
            span=Span(words[winner.start_idx].start, words[winner.end_idx - 1].end),
            score=winner.score,
            word_range=(winner.start_idx, winner.end_idx),
            status=status,
            scoped=scoped,
            matched_text=" ".join(w.text for w in words[winner.start_idx:winner.end_idx]),
        ))
        cursor = winner.end_idx

    return matches, cuts
