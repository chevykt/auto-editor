"""Shared data model. Imports nothing from this project, by design."""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from enum import Enum

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and surrounding whitespace.

    Used for every text comparison in the project so that alignment and
    disfluency detection agree on what "the same word" means.
    """
    return text.lower().translate(_PUNCT).strip()


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Span:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class CutReason(str, Enum):
    SILENCE = "silence"
    RETAKE_LOSER = "retake_loser"
    DISFLUENCY = "disfluency"
    # Reported, never applied unless approved. See suspects.py.
    SUSPECT = "suspect"


@dataclass
class CutSpan:
    start: float
    end: float
    reasons: set[CutReason] = field(default_factory=set)
    detail: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class MatchStatus(str, Enum):
    OK = "ok"
    REVIEW = "review"
    MISS = "miss"


@dataclass
class ScriptLine:
    index: int
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class LineMatch:
    line_idx: int
    span: Span | None
    score: float
    word_range: tuple[int, int] | None
    status: MatchStatus
    scoped: bool = True
    matched_text: str = ""
