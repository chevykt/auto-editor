"""Fixture helpers.

`words()` is the reason the pure core is testable without video or Whisper:
it turns a compact string into a realistic word-timestamp list.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Word  # noqa: E402


def words(text: str, start: float = 0.0, dur: float = 0.3,
          gaps: list[float] | None = None) -> list[Word]:
    """Build a word list from a space-separated string.

    dur:  duration of every word.
    gaps: silence AFTER each word; length should be len(tokens) - 1.
          Missing entries default to 0.0.
    """
    toks = text.split()
    gaps = list(gaps or [])
    gaps += [0.0] * (len(toks) - 1 - len(gaps))

    out: list[Word] = []
    t = start
    for i, tok in enumerate(toks):
        out.append(Word(tok, round(t, 6), round(t + dur, 6)))
        t += dur
        if i < len(gaps):
            t += gaps[i]
    return out
