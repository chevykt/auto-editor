"""Script text -> list[ScriptLine]."""
from __future__ import annotations

import re

from models import ScriptLine

_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def parse_script(text: str, split_sentences: bool = False) -> list[ScriptLine]:
    """Split a voiceover script into beats.

    Default: blank lines separate beats, and newlines inside a beat are
    collapsed to spaces (most voiceover scripts are one paragraph per beat).
    split_sentences: treat every sentence as its own beat instead, for scripts
    written as a single block.
    """
    if split_sentences:
        flat = " ".join(text.split())
        chunks = _SENTENCE_END.split(flat)
    else:
        chunks = _BLANK_LINE.split(text)

    lines: list[ScriptLine] = []
    for chunk in chunks:
        collapsed = " ".join(chunk.split())
        if collapsed:
            lines.append(ScriptLine(index=len(lines), text=collapsed))
    return lines
