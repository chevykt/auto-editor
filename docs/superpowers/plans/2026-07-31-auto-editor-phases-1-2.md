# Script-Matched Video Auto-Editor — Implementation Plan (Phases 1 & 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that removes dead air, false starts, and stutters from a tutorial screen recording while preserving intentional ad-libs, producing a review report and then a rendered cut.

**Architecture:** `transcribe.py` is a thin adapter returning `list[Word]`; every module downstream is a pure function over word lists, so all alignment/disfluency/cut logic is tested with hand-authored fixtures — no video, no Whisper, no ffmpeg. The algorithm is subtractive: build a cut-list of silence + losing retakes + disfluencies, and keep everything else in original order.

**Tech Stack:** Python 3.14.6, `faster-whisper` 1.2.1 (via `ctranslate2` 4.8.1), `rapidfuzz` 3.14.5, ffmpeg 8.1.2, pytest.

**Approved spec:** `docs/superpowers/specs/2026-07-31-auto-editor-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `models.py` | Shared dataclasses + `normalize()`. Zero dependencies on other project modules — breaks the aligner/disfluency/cutlist import cycle. |
| `config.py` | `EditorConfig` dataclass. Every tuning knob, one place. |
| `script_parser.py` | Script text → `list[ScriptLine]`. |
| `silence.py` | Word gaps → `SILENCE` cut spans. |
| `aligner.py` | Sliding-window fuzzy match, monotonic cursor, retake resolution. |
| `disfluency.py` | Three narrow local detectors. |
| `cutlist.py` | Merge, pad, keep-segments, sliver absorption. |
| `report.py` | JSON (machine, re-loadable) + Markdown (human). |
| `transcribe.py` | faster-whisper adapter + transcript cache. Only module that touches the model. |
| `render.py` | ffmpeg filter graph + invocation. Only module that shells out. |
| `edit.py` | CLI parsing and orchestration. |
| `tests/conftest.py` | `words()` fixture builder — the thing that makes the pure core testable. |

Ordering is dependency-driven: `models` → leaf modules → `cutlist` → `report` → CLI (phase 1 complete) → `render` (phase 2).

---

## Task 1: Project scaffolding and shared models

**Files:**
- Create: `requirements.txt`
- Create: `models.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Create `requirements.txt`**

```
faster-whisper==1.2.1
rapidfuzz==3.14.5
pytest==8.3.4
```

- [ ] **Step 2: Install dependencies**

Run: `python -m pip install -r requirements.txt`
Expected: installs `ctranslate2-4.8.1-cp314-cp314-win_amd64.whl` and `rapidfuzz-3.14.5-cp314-cp314-win_amd64.whl` without building from source. If pip tries to build a wheel from source, stop — that means a cp314 wheel is missing and the Python version needs revisiting.

- [ ] **Step 3: Write the failing test**

Create `tests/test_models.py`:

```python
from models import Word, Span, CutSpan, CutReason, MatchStatus, normalize


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Config-,") == "config"
    assert normalize("  The  ") == "the"
    assert normalize("!!!") == ""


def test_span_duration():
    assert Span(1.0, 2.5).duration == 1.5


def test_cutspan_duration_and_reasons_are_mutable_set():
    c = CutSpan(1.0, 2.0, {CutReason.SILENCE}, "gap")
    c.reasons.add(CutReason.DISFLUENCY)
    assert c.duration == 1.0
    assert c.reasons == {CutReason.SILENCE, CutReason.DISFLUENCY}


def test_word_is_hashable_and_frozen():
    w = Word("hello", 0.0, 0.4)
    assert {w}
    assert w.text == "hello"


def test_enums_serialise_as_plain_strings():
    assert CutReason.SILENCE.value == "silence"
    assert MatchStatus.REVIEW.value == "review"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 5: Write `models.py`**

```python
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
```

- [ ] **Step 6: Write the test fixture helper**

Create `tests/__init__.py` as an empty file, then create `tests/conftest.py`:

```python
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
```

- [ ] **Step 7: Create `pytest.ini`**

Every later test file does `from conftest import words`. Because `tests/` is a package (it has `__init__.py`), pytest resolves the fixture module as `tests.conftest`, and the bare import fails. Putting both the project root and `tests/` on `pythonpath` fixes it for the whole suite.

```ini
[pytest]
testpaths = tests
# rootdir makes `import models` work; tests/ makes `from conftest import words`
# work even though tests/ is a package.
pythonpath = . tests
markers =
    smoke: end-to-end tests requiring ffmpeg and real media (slow)
addopts = -m "not smoke"
```

- [ ] **Step 8: Write a test guarding the import contract**

Create `tests/test_importpath.py`:

```python
"""Guards the test-import contract the whole suite depends on.

Every other test file does `from conftest import words`. That only works
because pytest.ini puts both the project root and tests/ on pythonpath.
If this file fails, no other test file can import its fixtures.
"""
from conftest import words
from models import Word


def test_conftest_is_importable_as_a_top_level_module():
    assert words("hello world")


def test_words_builds_contiguous_timestamps():
    ws = words("a b c", dur=0.3)
    assert [(w.start, w.end) for w in ws] == [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9)]


def test_words_applies_gaps_after_each_token():
    ws = words("a b c", dur=0.3, gaps=[0.5, 0.0])
    assert ws[1].start == 0.8
    assert ws[2].start == 1.1


def test_words_gaps_shorter_than_token_list_are_padded():
    ws = words("a b c", dur=0.3, gaps=[0.5])
    assert ws[2].start == 1.1


def test_words_returns_model_instances():
    assert all(isinstance(w, Word) for w in words("a b"))
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/ -v`
Expected: 10 passed

- [ ] **Step 10: Commit**

```bash
git add requirements.txt models.py pytest.ini tests/
git commit -m "feat: add shared data model and test fixture helper"
```

---

## Task 2: Configuration

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from config import EditorConfig


def test_defaults_match_spec():
    c = EditorConfig()
    assert c.threshold == 0.75
    assert c.silence_ms == 600
    assert c.padding_ms == 75
    assert c.model == "medium"
    assert c.clean_stutters is True
    assert c.tiebreak == "latest"
    assert c.duplicate_window_s == 4.0
    assert c.max_disfluency_words == 5
    assert c.min_segment_ms == 150
    assert c.crossfade_ms == 75
    assert c.review_margin == 0.05
    assert c.lookahead_s == 30.0


def test_second_derived_properties():
    c = EditorConfig(silence_ms=600, padding_ms=75, min_segment_ms=150,
                     crossfade_ms=75)
    assert c.silence_s == 0.6
    assert c.padding_s == 0.075
    assert c.min_segment_s == 0.15
    assert c.crossfade_s == 0.075
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write `config.py`**

```python
"""Every tuning knob, one place. Nothing in this project hardcodes a threshold."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EditorConfig:
    # Alignment
    threshold: float = 0.75
    review_margin: float = 0.05
    lookahead_s: float = 30.0
    window_tolerance: float = 0.4   # window sizes searched: wc*(1±tolerance)
    tiebreak: str = "latest"        # "latest" | "score"
    retake_lookahead_lines: int = 5  # how far ahead to check for a rival claim

    # Silence
    silence_ms: int = 600

    # Disfluency
    clean_stutters: bool = True
    duplicate_window_s: float = 4.0
    max_disfluency_words: int = 5
    repeat_max_gap_s: float = 1.5
    restart_max_gap_s: float = 1.0
    min_fragment_chars: int = 3
    min_restart_growth: int = 3     # full word must exceed fragment by this many chars

    # Cut list
    padding_ms: int = 75
    min_segment_ms: int = 150

    # Render
    crossfade_ms: int = 75
    video_codec: str = "libx264"
    video_crf: int = 18
    video_preset: str = "medium"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    # Transcription
    model: str = "medium"
    language: str | None = None
    use_cache: bool = True

    @property
    def silence_s(self) -> float:
        return self.silence_ms / 1000.0

    @property
    def padding_s(self) -> float:
        return self.padding_ms / 1000.0

    @property
    def min_segment_s(self) -> float:
        return self.min_segment_ms / 1000.0

    @property
    def crossfade_s(self) -> float:
        return self.crossfade_ms / 1000.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add EditorConfig with all tuning knobs"
```

---

## Task 3: Script parsing

**Files:**
- Create: `script_parser.py`
- Test: `tests/test_script_parser.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_script_parser.py`:

```python
from script_parser import parse_script


def test_splits_on_blank_lines_by_default():
    text = "First beat here.\nStill first beat.\n\nSecond beat.\n\n\nThird beat."
    lines = parse_script(text)
    assert [l.text for l in lines] == [
        "First beat here. Still first beat.",
        "Second beat.",
        "Third beat.",
    ]
    assert [l.index for l in lines] == [0, 1, 2]


def test_split_sentences_mode():
    text = "Click the button. Then wait! Does it work? Yes."
    lines = parse_script(text, split_sentences=True)
    assert [l.text for l in lines] == [
        "Click the button.",
        "Then wait!",
        "Does it work?",
        "Yes.",
    ]


def test_ignores_empty_and_whitespace_only_blocks():
    assert parse_script("\n\n   \n\nOnly beat.\n\n   \n") == [
        type(parse_script("Only beat.")[0])(index=0, text="Only beat.")
    ]


def test_empty_script_returns_empty_list():
    assert parse_script("   \n\n  ") == []


def test_word_count_property():
    assert parse_script("one two three")[0].word_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_script_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'script_parser'`

- [ ] **Step 3: Write `script_parser.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_script_parser.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add script_parser.py tests/test_script_parser.py
git commit -m "feat: add script parser with blank-line and sentence modes"
```

---

## Task 4: Silence detection

**Files:**
- Create: `silence.py`
- Test: `tests/test_silence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_silence.py`:

```python
from config import EditorConfig
from models import CutReason
from silence import detect_silence
from conftest import words


def test_gap_at_or_above_threshold_is_cut():
    # three words, 0.7s gap after the first, 0.1s after the second
    ws = words("alpha beta gamma", dur=0.3, gaps=[0.7, 0.1])
    cuts = detect_silence(ws, EditorConfig(silence_ms=600))
    assert len(cuts) == 1
    assert cuts[0].start == 0.3
    assert cuts[0].end == 1.0
    assert cuts[0].reasons == {CutReason.SILENCE}


def test_gap_below_threshold_is_kept():
    ws = words("alpha beta", dur=0.3, gaps=[0.5])
    assert detect_silence(ws, EditorConfig(silence_ms=600)) == []


def test_gap_exactly_at_threshold_is_cut():
    ws = words("alpha beta", dur=0.3, gaps=[0.6])
    assert len(detect_silence(ws, EditorConfig(silence_ms=600))) == 1


def test_leading_silence_is_cut_when_long_enough():
    ws = words("alpha beta", start=2.0, dur=0.3)
    cuts = detect_silence(ws, EditorConfig(silence_ms=600))
    assert cuts[0].start == 0.0
    assert cuts[0].end == 2.0


def test_trailing_silence_is_cut_when_duration_known():
    ws = words("alpha beta", dur=0.3)          # ends at 0.6
    cuts = detect_silence(ws, EditorConfig(silence_ms=600), duration=5.0)
    assert cuts[-1].start == 0.6
    assert cuts[-1].end == 5.0


def test_no_trailing_cut_without_duration():
    ws = words("alpha beta", dur=0.3)
    assert detect_silence(ws, EditorConfig(silence_ms=600)) == []


def test_empty_word_list():
    assert detect_silence([], EditorConfig()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_silence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'silence'`

- [ ] **Step 3: Write `silence.py`**

```python
"""Gap-based silence detection.

Uses gaps between consecutive word timestamps rather than a separate ffmpeg
silencedetect pass, reusing data already fetched during transcription.
"""
from __future__ import annotations

from config import EditorConfig
from models import CutReason, CutSpan, Word


def detect_silence(words: list[Word], cfg: EditorConfig,
                   duration: float | None = None) -> list[CutSpan]:
    """Find gaps with no speech at or above cfg.silence_ms.

    duration: total media length. When supplied, trailing dead air after the
    last word is also cut. Leading dead air is always checked.
    """
    if not words:
        return []

    cuts: list[CutSpan] = []
    min_gap = cfg.silence_s

    if words[0].start >= min_gap:
        cuts.append(CutSpan(0.0, words[0].start, {CutReason.SILENCE},
                            f"leading silence {words[0].start:.2f}s"))

    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if gap >= min_gap:
            cuts.append(CutSpan(a.end, b.start, {CutReason.SILENCE},
                                f"gap {gap:.2f}s"))

    if duration is not None:
        tail = duration - words[-1].end
        if tail >= min_gap:
            cuts.append(CutSpan(words[-1].end, duration, {CutReason.SILENCE},
                                f"trailing silence {tail:.2f}s"))

    return cuts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_silence.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add silence.py tests/test_silence.py
git commit -m "feat: add gap-based silence detection with leading/trailing handling"
```

---

## Task 5: Alignment — candidate window search

**Files:**
- Create: `aligner.py`
- Test: `tests/test_aligner_candidates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aligner_candidates.py`:

```python
from aligner import find_candidates, suppress_overlaps
from config import EditorConfig
from conftest import words


def test_exact_match_scores_one():
    ws = words("click the blue button now")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    assert cands
    assert max(c.score for c in cands) == 1.0


def test_paraphrase_still_matches_above_threshold():
    ws = words("now click on the blue button")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    assert cands
    assert max(c.score for c in cands) >= 0.75


def test_unrelated_text_produces_no_candidates():
    ws = words("completely different words entirely")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    assert cands == []


def test_search_is_confined_to_word_range():
    ws = words("click the blue button and then click the blue button")
    # restrict to the second half only
    cands = find_candidates(ws, 6, len(ws), "click the blue button", EditorConfig())
    assert cands
    assert all(c.start_idx >= 6 for c in cands)


def test_suppress_overlaps_keeps_best_of_each_cluster():
    ws = words("click the blue button pause click the blue button")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    kept = suppress_overlaps(cands)
    assert len(kept) == 2
    assert kept[0].start_idx == 0
    assert kept[1].start_idx == 5


def test_suppress_overlaps_on_empty_list():
    assert suppress_overlaps([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aligner_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aligner'`

- [ ] **Step 3: Write the candidate-search half of `aligner.py`**

```python
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
from models import Word, normalize


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
    true phrase and score an identical tie ("button click the blue" vs.
    "click the blue button" - same four words, both 100.0). An order-sensitive
    nudge is folded in at a scale far below any threshold or rounding
    boundary, purely to break that tie toward the window that preserves word
    order. It is exactly zero for a true match (order-ratio 1.0), so an exact
    match still scores exactly 1.0.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aligner_candidates.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add aligner.py tests/test_aligner_candidates.py
git commit -m "feat: add fuzzy sliding-window candidate search with overlap suppression"
```

---

## Task 6: Alignment — retake resolution

**Files:**
- Modify: `aligner.py` (append)
- Test: `tests/test_aligner_retakes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aligner_retakes.py`:

```python
import pytest

from aligner import Candidate, resolve_retakes
from config import EditorConfig


def test_highest_score_wins():
    cands = [Candidate(0.80, 0, 4), Candidate(0.95, 10, 14)]
    winner, losers = resolve_retakes(cands, EditorConfig())
    assert winner.start_idx == 10
    assert [l.start_idx for l in losers] == [0]


def test_tie_breaks_toward_later_occurrence_by_default():
    cands = [Candidate(0.90, 0, 4), Candidate(0.90, 10, 14)]
    winner, losers = resolve_retakes(cands, EditorConfig(tiebreak="latest"))
    assert winner.start_idx == 10
    assert [l.start_idx for l in losers] == [0]


def test_near_tie_within_rounding_also_breaks_later():
    cands = [Candidate(0.9004, 0, 4), Candidate(0.9001, 10, 14)]
    winner, _ = resolve_retakes(cands, EditorConfig(tiebreak="latest"))
    assert winner.start_idx == 10


def test_score_tiebreak_prefers_earlier_on_exact_tie():
    cands = [Candidate(0.90, 0, 4), Candidate(0.90, 10, 14)]
    winner, losers = resolve_retakes(cands, EditorConfig(tiebreak="score"))
    assert winner.start_idx == 0
    assert [l.start_idx for l in losers] == [10]


def test_single_candidate_has_no_losers():
    winner, losers = resolve_retakes([Candidate(0.9, 0, 4)], EditorConfig())
    assert winner.start_idx == 0
    assert losers == []


def test_empty_candidates_raises():
    with pytest.raises(ValueError):
        resolve_retakes([], EditorConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aligner_retakes.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_retakes' from 'aligner'`

- [ ] **Step 3: Append `resolve_retakes` to `aligner.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aligner_retakes.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add aligner.py tests/test_aligner_retakes.py
git commit -m "feat: add retake resolution with latest and score tie-break rules"
```

---

## Task 7: Alignment — monotonic cursor with fallback

**Files:**
- Modify: `aligner.py` (append)
- Test: `tests/test_aligner_align.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aligner_align.py`:

```python
from aligner import align_script
from config import EditorConfig
from models import CutReason, MatchStatus, ScriptLine
from conftest import words


def lines(*texts):
    return [ScriptLine(i, t) for i, t in enumerate(texts)]


def test_matches_lines_in_order():
    ws = words("open the settings panel then click the blue button")
    matches, cuts = align_script(ws, lines("open the settings panel",
                                           "click the blue button"),
                                 EditorConfig())
    assert [m.status for m in matches] == [MatchStatus.OK, MatchStatus.OK]
    assert matches[0].word_range == (0, 4)
    assert matches[1].word_range[0] >= 4
    assert cuts == []


def test_similar_lines_do_not_steal_each_others_occurrence():
    # the same phrase twice; two script lines must take one each
    ws = words("click the blue button click the blue button")
    matches, _ = align_script(ws, lines("click the blue button",
                                        "click the blue button"),
                              EditorConfig())
    assert matches[0].word_range == (0, 4)
    assert matches[1].word_range == (4, 8)


def test_retake_marks_loser_for_cut():
    # first attempt is a poorer paraphrase, second is exact
    ws = words("click the red button click the blue button")
    matches, cuts = align_script(ws, lines("click the blue button"),
                                 EditorConfig())
    assert matches[0].word_range == (4, 8)
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.RETAKE_LOSER}
    assert cuts[0].start == ws[0].start
    # cut runs from the loser's start up to the winner's start
    assert cuts[0].end == ws[4].start


def test_unmatchable_line_is_a_miss_and_does_not_move_cursor():
    ws = words("open the settings panel then click the blue button")
    matches, _ = align_script(ws, lines("open the settings panel",
                                        "absolutely nothing like this exists",
                                        "click the blue button"),
                              EditorConfig())
    assert matches[1].status == MatchStatus.MISS
    assert matches[1].span is None
    # the miss must not cascade - line 2 still matches
    assert matches[2].status == MatchStatus.OK
    assert matches[2].word_range[0] >= 4


def test_low_score_match_is_flagged_for_review():
    ws = words("click the blue button")
    cfg = EditorConfig(threshold=0.5, review_margin=0.45)
    matches, _ = align_script(ws, lines("click a blue widget"), cfg)
    assert matches[0].status == MatchStatus.REVIEW


def test_out_of_scope_line_found_via_full_recording_fallback():
    # line 1 appears BEFORE line 0 in the recording, so the forward cursor
    # cannot find it and must fall back to a full search
    ws = words("click the blue button then open the settings panel")
    matches, _ = align_script(ws, lines("open the settings panel",
                                        "click the blue button"),
                              EditorConfig())
    assert matches[1].status != MatchStatus.MISS
    assert matches[1].scoped is False


def test_matched_text_is_populated():
    ws = words("click the blue button")
    matches, _ = align_script(ws, lines("click the blue button"), EditorConfig())
    assert matches[0].matched_text == "click the blue button"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aligner_align.py -v`
Expected: FAIL with `ImportError: cannot import name 'align_script' from 'aligner'`

- [ ] **Step 3: Append the alignment driver to `aligner.py`**

```python
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
        claimed = any(
            _score_line_against(words, cand, other.text) >= own
            and _score_line_against(words, cand, other.text) >= cfg.threshold
            for other in horizon)
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
            # cut the losing attempt plus the dead space before the winner
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
```

- [ ] **Step 4: Update the imports at the top of `aligner.py`**

Replace the `from models import ...` line with:

```python
from models import (CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine,
                    Span, Word, normalize)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_aligner_align.py -v`
Expected: 7 passed

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add aligner.py tests/test_aligner_align.py
git commit -m "feat: add monotonic alignment cursor with full-recording fallback"
```

---

## Task 8: Disfluency — immediate word repeats

**Files:**
- Create: `disfluency.py`
- Test: `tests/test_disfluency_repeats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_disfluency_repeats.py`:

```python
from config import EditorConfig
from disfluency import detect_immediate_repeats
from models import CutReason
from conftest import words


def test_adjacent_duplicate_word_is_cut():
    ws = words("click the the button")
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.DISFLUENCY}
    assert cuts[0].start == ws[1].start
    assert cuts[0].end == ws[2].start


def test_repeat_with_filler_between_is_cut_including_filler():
    ws = words("click uh click the button")
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].start == ws[0].start
    assert cuts[0].end == ws[2].start


def test_repeat_separated_by_long_pause_is_left_alone():
    # 2s gap - likely deliberate, not a stutter
    ws = words("the the button", gaps=[2.0, 0.0])
    assert detect_immediate_repeats(ws, 0, len(ws), EditorConfig()) == []


def test_punctuation_and_case_differences_still_count_as_repeat():
    ws = words("The the button")
    assert len(detect_immediate_repeats(ws, 0, len(ws), EditorConfig())) == 1


def test_non_repeat_is_untouched():
    ws = words("click the blue button")
    assert detect_immediate_repeats(ws, 0, len(ws), EditorConfig()) == []


def test_triple_repeat_cuts_both_earlier_copies():
    ws = words("the the the button")
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 2


def test_range_is_respected():
    ws = words("the the button and the the widget")
    cuts = detect_immediate_repeats(ws, 3, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].start == ws[4].start
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_disfluency_repeats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'disfluency'`

- [ ] **Step 3: Write `disfluency.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_disfluency_repeats.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add disfluency.py tests/test_disfluency_repeats.py
git commit -m "feat: add immediate word repeat detection"
```

---

## Task 9: Disfluency — truncated-word restarts

**Files:**
- Modify: `disfluency.py` (append)
- Test: `tests/test_disfluency_restarts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_disfluency_restarts.py`:

```python
from config import EditorConfig
from disfluency import detect_truncated_restarts
from models import CutReason
from conftest import words


def test_fragment_followed_by_full_word_is_cut():
    ws = words("the config- configuration screen")
    cuts = detect_truncated_restarts(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.DISFLUENCY}
    assert cuts[0].start == ws[1].start
    assert cuts[0].end == ws[2].start


def test_short_fragment_below_min_chars_is_ignored():
    # "in" is a prefix of "international" but too short to be a safe signal
    ws = words("in international waters")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_small_growth_is_ignored():
    # "the" -> "there" grows by only 2 chars; likely real speech
    ws = words("the there button")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_identical_words_are_not_a_restart():
    # that is a repeat, handled by the other detector
    ws = words("config config screen")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_long_pause_between_breaks_the_signal():
    ws = words("config- configuration screen", gaps=[1.5, 0.0])
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_non_prefix_is_ignored():
    ws = words("setup configuration screen")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_disfluency_restarts.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_truncated_restarts'`

- [ ] **Step 3: Append to `disfluency.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_disfluency_restarts.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add disfluency.py tests/test_disfluency_restarts.py
git commit -m "feat: add truncated-word restart detection with prefix guards"
```

---

## Task 10: Disfluency — duplicate phrases and the public entry point

**Files:**
- Modify: `disfluency.py` (append)
- Test: `tests/test_disfluency_phrases.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_disfluency_phrases.py`:

```python
from config import EditorConfig
from disfluency import detect_duplicate_phrases, detect_disfluencies
from models import CutReason
from conftest import words


def test_repeated_phrase_within_window_cuts_earlier_copy():
    ws = words("open the settings open the settings panel")
    cuts = detect_duplicate_phrases(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].start == ws[0].start
    assert cuts[0].end == ws[3].start


def test_repeat_outside_window_is_kept():
    # 6s apart with a 4s window - intentional callback, not a stutter
    ws = words("open the settings open the settings panel",
               gaps=[0, 0, 6.0, 0, 0, 0])
    assert detect_duplicate_phrases(ws, 0, len(ws), EditorConfig()) == []


def test_longer_repeat_preferred_over_shorter():
    ws = words("click the blue button click the blue button now")
    cuts = detect_duplicate_phrases(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].end == ws[4].start


def test_phrase_longer_than_cap_is_not_cut():
    cfg = EditorConfig(max_disfluency_words=2)
    ws = words("one two three one two three")
    cuts = detect_duplicate_phrases(ws, 0, len(ws), cfg)
    # only a 2-gram may be cut, never the full 3-word clause
    assert all(c.end - c.start <= ws[2].start for c in cuts)


def test_single_word_repeat_is_left_to_the_repeat_detector():
    ws = words("the the button")
    assert detect_duplicate_phrases(ws, 0, len(ws), EditorConfig()) == []


def test_detect_disfluencies_runs_all_three_and_sorts():
    ws = words("the the config- configuration open the settings open the settings now")
    cuts = detect_disfluencies(ws, 0, len(ws), EditorConfig())
    assert len(cuts) >= 3
    assert cuts == sorted(cuts, key=lambda c: c.start)
    assert all(c.reasons == {CutReason.DISFLUENCY} for c in cuts)


def test_detect_disfluencies_disabled_by_flag():
    ws = words("the the button")
    cfg = EditorConfig(clean_stutters=False)
    assert detect_disfluencies(ws, 0, len(ws), cfg) == []


def test_emphasis_repeat_with_pause_survives():
    # "no, no, don't click that" spoken deliberately, with a beat between
    ws = words("no no dont click that", gaps=[1.8, 0, 0, 0])
    cuts = detect_disfluencies(ws, 0, len(ws), EditorConfig())
    assert cuts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_disfluency_phrases.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_duplicate_phrases'`

- [ ] **Step 3: Append to `disfluency.py`**

```python
def detect_duplicate_phrases(words: list[Word], lo: int, hi: int,
                             cfg: EditorConfig) -> list[CutSpan]:
    """A short phrase repeated seconds apart, where the second run continues cleanly.

    Longer n-grams are preferred, so "click the blue button" repeated is cut as
    one span rather than four. Single words are excluded - those belong to
    detect_immediate_repeats, which understands fillers.
    """
    cuts: list[CutSpan] = []
    claimed: set[int] = set()

    for size in range(cfg.max_disfluency_words, 1, -1):
        i = lo
        while i + 2 * size <= hi:
            if any(k in claimed for k in range(i, i + size)):
                i += 1
                continue

            first = [normalize(w.text) for w in words[i:i + size]]
            if any(not t for t in first):
                i += 1
                continue

            found = False
            for j in range(i + size, hi - size + 1):
                if words[j].start - words[i + size - 1].end > cfg.duplicate_window_s:
                    break
                if [normalize(w.text) for w in words[j:j + size]] == first:
                    cuts.append(CutSpan(
                        words[i].start, words[j].start, {CutReason.DISFLUENCY},
                        f"duplicate phrase: {' '.join(first)!r}"))
                    claimed.update(range(i, j + size))
                    i = j + size
                    found = True
                    break
            if not found:
                i += 1

    return sorted(cuts, key=lambda c: c.start)


def detect_disfluencies(words: list[Word], lo: int, hi: int,
                        cfg: EditorConfig) -> list[CutSpan]:
    """Run every detector over words[lo:hi]. Honours cfg.clean_stutters."""
    if not cfg.clean_stutters:
        return []
    cuts = (detect_immediate_repeats(words, lo, hi, cfg)
            + detect_truncated_restarts(words, lo, hi, cfg)
            + detect_duplicate_phrases(words, lo, hi, cfg))
    return sorted(cuts, key=lambda c: c.start)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_disfluency_phrases.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add disfluency.py tests/test_disfluency_phrases.py
git commit -m "feat: add duplicate-phrase detection and unified disfluency entry point"
```

---

## Task 11: Cut list — overlap merging

**Files:**
- Create: `cutlist.py`
- Test: `tests/test_cutlist_merge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cutlist_merge.py`:

```python
from cutlist import merge_spans
from models import CutReason, CutSpan


def test_non_overlapping_spans_are_untouched():
    cuts = [CutSpan(0.0, 1.0, {CutReason.SILENCE}),
            CutSpan(2.0, 3.0, {CutReason.DISFLUENCY})]
    assert len(merge_spans(cuts)) == 2


def test_overlapping_spans_merge_and_union_reasons():
    cuts = [CutSpan(0.0, 2.0, {CutReason.RETAKE_LOSER}, "loser"),
            CutSpan(1.0, 3.0, {CutReason.DISFLUENCY}, "stutter")]
    merged = merge_spans(cuts)
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0.0, 3.0)
    assert merged[0].reasons == {CutReason.RETAKE_LOSER, CutReason.DISFLUENCY}
    assert "loser" in merged[0].detail and "stutter" in merged[0].detail


def test_touching_spans_merge():
    cuts = [CutSpan(0.0, 1.0, {CutReason.SILENCE}),
            CutSpan(1.0, 2.0, {CutReason.SILENCE})]
    assert len(merge_spans(cuts)) == 1


def test_fully_contained_span_does_not_shrink_the_outer_one():
    cuts = [CutSpan(0.0, 5.0, {CutReason.RETAKE_LOSER}),
            CutSpan(1.0, 2.0, {CutReason.DISFLUENCY})]
    merged = merge_spans(cuts)
    assert (merged[0].start, merged[0].end) == (0.0, 5.0)


def test_unsorted_input_is_handled():
    cuts = [CutSpan(5.0, 6.0, {CutReason.SILENCE}),
            CutSpan(0.0, 1.0, {CutReason.SILENCE})]
    merged = merge_spans(cuts)
    assert [c.start for c in merged] == [0.0, 5.0]


def test_does_not_mutate_input():
    original = CutSpan(0.0, 2.0, {CutReason.SILENCE}, "a")
    merge_spans([original, CutSpan(1.0, 3.0, {CutReason.DISFLUENCY}, "b")])
    assert original.end == 2.0
    assert original.reasons == {CutReason.SILENCE}


def test_empty_input():
    assert merge_spans([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cutlist_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cutlist'`

- [ ] **Step 3: Write `cutlist.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cutlist_merge.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add cutlist.py tests/test_cutlist_merge.py
git commit -m "feat: add cut span overlap merging with reason unioning"
```

---

## Task 12: Cut list — padding with inversion clamp

**Files:**
- Modify: `cutlist.py` (append)
- Test: `tests/test_cutlist_padding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cutlist_padding.py`:

```python
from config import EditorConfig
from cutlist import apply_padding
from models import CutReason, CutSpan


def test_boundaries_are_pulled_inward():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert out[0].start == 1.075
    assert out[0].end == 2.925


def test_span_shorter_than_twice_padding_is_clamped_not_inverted():
    cuts = [CutSpan(1.0, 1.05, {CutReason.DISFLUENCY})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert not out or out[0].end >= out[0].start


def test_span_exactly_twice_padding_collapses_to_nothing_and_is_dropped():
    cuts = [CutSpan(1.0, 1.15, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert out == []


def test_zero_padding_is_a_passthrough():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=0))
    assert (out[0].start, out[0].end) == (1.0, 3.0)


def test_reasons_and_detail_survive_padding():
    cuts = [CutSpan(1.0, 3.0, {CutReason.RETAKE_LOSER}, "loser")]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert out[0].reasons == {CutReason.RETAKE_LOSER}
    assert out[0].detail == "loser"


def test_applies_to_every_reason_type_alike():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE}),
            CutSpan(5.0, 7.0, {CutReason.DISFLUENCY}),
            CutSpan(9.0, 11.0, {CutReason.RETAKE_LOSER})]
    out = apply_padding(cuts, EditorConfig(padding_ms=100))
    assert [round(c.start, 3) for c in out] == [1.1, 5.1, 9.1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cutlist_padding.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_padding' from 'cutlist'`

- [ ] **Step 3: Append to `cutlist.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cutlist_padding.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add cutlist.py tests/test_cutlist_padding.py
git commit -m "feat: add cut boundary padding with inversion clamp"
```

---

## Task 13: Cut list — keep segments and sliver absorption

**Files:**
- Modify: `cutlist.py` (append)
- Test: `tests/test_cutlist_segments.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cutlist_segments.py`:

```python
from config import EditorConfig
from cutlist import build_cutlist, keep_segments
from models import CutReason, CutSpan


def test_keep_segments_are_the_complement_of_cuts():
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(4.0, 5.0, {CutReason.SILENCE})]
    segs = keep_segments(cuts, duration=6.0)
    assert [(s.start, s.end) for s in segs] == [(0.0, 1.0), (2.0, 4.0), (5.0, 6.0)]


def test_cut_at_the_very_start_produces_no_leading_segment():
    cuts = [CutSpan(0.0, 2.0, {CutReason.SILENCE})]
    segs = keep_segments(cuts, duration=5.0)
    assert [(s.start, s.end) for s in segs] == [(2.0, 5.0)]


def test_cut_running_to_the_end_produces_no_trailing_segment():
    cuts = [CutSpan(3.0, 5.0, {CutReason.SILENCE})]
    segs = keep_segments(cuts, duration=5.0)
    assert [(s.start, s.end) for s in segs] == [(0.0, 3.0)]


def test_no_cuts_yields_the_whole_timeline():
    assert [(s.start, s.end) for s in keep_segments([], duration=5.0)] == [(0.0, 5.0)]


def test_sliver_between_two_cuts_is_absorbed():
    # 0.05s survivor between two cuts, below the 150ms floor
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(2.05, 3.0, {CutReason.DISFLUENCY})]
    final = build_cutlist(cuts, duration=5.0, cfg=EditorConfig(min_segment_ms=150,
                                                              padding_ms=0))
    assert len(final) == 1
    assert (final[0].start, final[0].end) == (1.0, 3.0)


def test_segment_above_the_floor_survives():
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(2.5, 3.0, {CutReason.DISFLUENCY})]
    final = build_cutlist(cuts, duration=5.0, cfg=EditorConfig(min_segment_ms=150,
                                                              padding_ms=0))
    assert len(final) == 2


def test_absorption_does_not_cascade_into_deleting_everything():
    # a long survivor sits after two slivers; it must not be swallowed
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(2.05, 3.0, {CutReason.SILENCE}),
            CutSpan(3.05, 4.0, {CutReason.SILENCE})]
    final = build_cutlist(cuts, duration=20.0, cfg=EditorConfig(min_segment_ms=150,
                                                               padding_ms=0))
    segs = keep_segments(final, duration=20.0)
    assert (0.0, 1.0) in [(s.start, s.end) for s in segs]
    assert (4.0, 20.0) in [(s.start, s.end) for s in segs]


def test_build_cutlist_merges_pads_and_absorbs_in_order():
    cuts = [CutSpan(4.0, 5.0, {CutReason.SILENCE}),
            CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(1.5, 2.5, {CutReason.DISFLUENCY})]
    final = build_cutlist(cuts, duration=10.0, cfg=EditorConfig(padding_ms=50))
    assert [round(c.start, 3) for c in final] == [1.05, 4.05]
    assert final[0].reasons == {CutReason.SILENCE, CutReason.DISFLUENCY}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cutlist_segments.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_cutlist' from 'cutlist'`

- [ ] **Step 3: Append to `cutlist.py`**

```python
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
    padded = apply_padding(merged, cfg)
    return _absorb_slivers(padded, duration, cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cutlist_segments.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add cutlist.py tests/test_cutlist_segments.py
git commit -m "feat: add keep-segment computation and sliver absorption"
```

---

## Task 14: Report — JSON

**Files:**
- Create: `report.py`
- Test: `tests/test_report_json.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_json.py`:

```python
import json

from config import EditorConfig
from models import (CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine,
                    Span)
from report import build_report, load_report, write_json
from conftest import words


def sample():
    ws = words("click the blue button")
    lines = [ScriptLine(0, "click the blue button")]
    matches = [LineMatch(0, Span(0.0, 1.2), 1.0, (0, 4), MatchStatus.OK,
                         matched_text="click the blue button")]
    cuts = [CutSpan(1.5, 2.5, {CutReason.SILENCE}, "gap 1.00s"),
            CutSpan(3.0, 3.4, {CutReason.DISFLUENCY}, "immediate repeat: 'the'")]
    return ws, lines, matches, cuts


def test_report_round_trips_through_json(tmp_path):
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, duration=10.0,
                       cfg=EditorConfig(), video="raw.mp4")
    path = tmp_path / "r.json"
    write_json(rep, path)

    loaded = load_report(path)
    assert loaded["duration"] == 10.0
    assert len(loaded["cuts"]) == 2
    assert loaded["cuts"][0]["reasons"] == ["silence"]


def test_json_is_valid_and_human_readable(tmp_path):
    ws, lines, matches, cuts = sample()
    path = tmp_path / "r.json"
    write_json(build_report(lines, matches, cuts, ws, 10.0, EditorConfig(),
                            "raw.mp4"), path)
    text = path.read_text(encoding="utf-8")
    json.loads(text)
    assert "\n  " in text          # indented, so it can be hand-edited


def test_report_records_config_used():
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, 10.0,
                       EditorConfig(threshold=0.9), "raw.mp4")
    assert rep["config"]["threshold"] == 0.9


def test_report_counts_and_totals_cut_time():
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, 10.0, EditorConfig(), "raw.mp4")
    assert rep["summary"]["cut_count"] == 2
    assert round(rep["summary"]["cut_seconds"], 3) == 1.4
    assert round(rep["summary"]["output_seconds"], 3) == 8.6


def test_unmatched_spans_are_listed_as_likely_adlibs():
    ws = words("click the blue button and by the way this is improvised")
    lines = [ScriptLine(0, "click the blue button")]
    matches = [LineMatch(0, Span(ws[0].start, ws[3].end), 1.0, (0, 4),
                         MatchStatus.OK, matched_text="click the blue button")]
    rep = build_report(lines, matches, [], ws, duration=5.0,
                       cfg=EditorConfig(), video="raw.mp4")
    assert rep["adlibs"]
    assert "improvised" in rep["adlibs"][0]["text"]


def test_miss_is_recorded_explicitly():
    ws, lines, _, _ = sample()
    matches = [LineMatch(0, None, 0.0, None, MatchStatus.MISS)]
    rep = build_report(lines, matches, [], ws, 5.0, EditorConfig(), "raw.mp4")
    assert rep["lines"][0]["status"] == "miss"
    assert rep["summary"]["misses"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report_json.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'report'`

- [ ] **Step 3: Write `report.py`**

```python
"""Review artifacts.

The JSON report is the editable, re-loadable one: it is what --from-report
consumes, so a human can delete a cut they disagree with and render from the
result. The Markdown report is for reading.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from config import EditorConfig
from cutlist import keep_segments
from models import CutSpan, LineMatch, ScriptLine, Word


def _adlib_spans(matches: list[LineMatch], cuts: list[CutSpan],
                 words: list[Word]) -> list[dict]:
    """Words that matched no script line and were not cut - probable ad-libs."""
    claimed: set[int] = set()
    for m in matches:
        if m.word_range:
            claimed.update(range(*m.word_range))

    def is_cut(w: Word) -> bool:
        return any(c.start <= w.start and w.end <= c.end for c in cuts)

    out: list[dict] = []
    run: list[Word] = []
    for i, w in enumerate(words):
        if i not in claimed and not is_cut(w):
            run.append(w)
        elif run:
            out.append({"start": run[0].start, "end": run[-1].end,
                        "text": " ".join(x.text for x in run)})
            run = []
    if run:
        out.append({"start": run[0].start, "end": run[-1].end,
                    "text": " ".join(x.text for x in run)})
    return out


def build_report(lines: list[ScriptLine], matches: list[LineMatch],
                 cuts: list[CutSpan], words: list[Word], duration: float,
                 cfg: EditorConfig, video: str) -> dict:
    """Assemble the full review report as a plain dict."""
    cut_seconds = sum(c.duration for c in cuts)
    by_line = {m.line_idx: m for m in matches}

    return {
        "video": video,
        "duration": duration,
        "config": asdict(cfg),
        "summary": {
            "line_count": len(lines),
            "misses": sum(1 for m in matches if m.status.value == "miss"),
            "reviews": sum(1 for m in matches if m.status.value == "review"),
            "cut_count": len(cuts),
            "cut_seconds": cut_seconds,
            "output_seconds": duration - cut_seconds,
        },
        "lines": [
            {
                "index": l.index,
                "script": l.text,
                "status": by_line[l.index].status.value,
                "score": by_line[l.index].score,
                "scoped": by_line[l.index].scoped,
                "matched_text": by_line[l.index].matched_text,
                "start": by_line[l.index].span.start if by_line[l.index].span else None,
                "end": by_line[l.index].span.end if by_line[l.index].span else None,
            }
            for l in lines if l.index in by_line
        ],
        "cuts": [
            {
                "start": c.start,
                "end": c.end,
                "duration": c.duration,
                "reasons": sorted(r.value for r in c.reasons),
                "detail": c.detail,
            }
            for c in cuts
        ],
        "adlibs": _adlib_spans(matches, cuts, words),
        "segments": [{"start": s.start, "end": s.end}
                     for s in keep_segments(cuts, duration)],
    }


def write_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report_json.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add report.py tests/test_report_json.py
git commit -m "feat: add JSON review report with adlib detection and segments"
```

---

## Task 15: Report — Markdown

**Files:**
- Modify: `report.py` (append)
- Test: `tests/test_report_markdown.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_markdown.py`:

```python
from config import EditorConfig
from models import (CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine,
                    Span)
from report import build_report, render_markdown, write_markdown
from conftest import words


def sample_report():
    ws = words("click the blue button and this bit is improvised")
    lines = [ScriptLine(0, "click the blue button"),
             ScriptLine(1, "a line that never got said")]
    matches = [
        LineMatch(0, Span(0.0, 1.2), 1.0, (0, 4), MatchStatus.OK,
                  matched_text="click the blue button"),
        LineMatch(1, None, 0.0, None, MatchStatus.MISS),
    ]
    cuts = [CutSpan(2.0, 2.6, {CutReason.SILENCE}, "gap 0.60s"),
            CutSpan(3.0, 3.4, {CutReason.DISFLUENCY}, "immediate repeat: 'the'"),
            CutSpan(5.0, 6.0, {CutReason.RETAKE_LOSER}, "losing take of line 0")]
    return build_report(lines, matches, cuts, ws, 20.0, EditorConfig(), "raw.mp4")


def test_markdown_contains_every_script_line():
    md = render_markdown(sample_report())
    assert "click the blue button" in md
    assert "a line that never got said" in md


def test_misses_are_called_out():
    md = render_markdown(sample_report())
    assert "MISS" in md


def test_silence_is_summarised_not_enumerated():
    md = render_markdown(sample_report())
    # the silence cut's detail should not appear line by line
    assert "gap 0.60s" not in md
    assert "silence" in md.lower()


def test_non_silence_cuts_are_listed_individually():
    md = render_markdown(sample_report())
    assert "immediate repeat" in md
    assert "losing take of line 0" in md


def test_adlibs_section_present():
    md = render_markdown(sample_report())
    assert "improvised" in md


def test_write_markdown_creates_the_file(tmp_path):
    path = write_markdown(sample_report(), tmp_path / "r.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Auto-Editor Review")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report_markdown.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_markdown' from 'report'`

- [ ] **Step 3: Append to `report.py`**

```python
def _ts(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    m, s = divmod(float(seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def render_markdown(report: dict) -> str:
    """Human review document.

    Silence is summarised rather than enumerated - it is never a judgment
    call, and listing hundreds of gaps would bury the cuts that are.
    """
    s = report["summary"]
    out: list[str] = [
        "# Auto-Editor Review",
        "",
        f"**Source:** `{report['video']}`  ",
        f"**Duration:** {_ts(report['duration'])} -> "
        f"{_ts(s['output_seconds'])} "
        f"({s['cut_seconds']:.1f}s removed across {s['cut_count']} cuts)  ",
        f"**Script lines:** {s['line_count']}  "
        f"({s['misses']} missed, {s['reviews']} needing review)",
        "",
        "## Script lines",
        "",
        "| # | Status | Score | Time | Matched text |",
        "|---|---|---|---|---|",
    ]

    for line in report["lines"]:
        flag = {"ok": "ok", "review": "**REVIEW**", "miss": "**MISS**"}[line["status"]]
        if not line["scoped"] and line["status"] != "miss":
            flag += " (out of order)"
        text = line["matched_text"] or f"_{line['script']}_"
        out.append(f"| {line['index']} | {flag} | {line['score']:.2f} | "
                   f"{_ts(line['start'])} | {text} |")

    silence = [c for c in report["cuts"] if c["reasons"] == ["silence"]]
    judged = [c for c in report["cuts"] if c["reasons"] != ["silence"]]

    out += [
        "",
        "## Cuts",
        "",
        f"Silence: **{len(silence)} cuts**, "
        f"{sum(c['duration'] for c in silence):.1f}s total.",
        "",
        "Everything below is a judgment call. Check that none of these ate an "
        "intentional ad-lib.",
        "",
        "| Time | Length | Reason | Detail |",
        "|---|---|---|---|",
    ]
    for c in judged:
        out.append(f"| {_ts(c['start'])} | {c['duration']:.2f}s | "
                   f"{', '.join(c['reasons'])} | {c['detail']} |")
    if not judged:
        out.append("| _none_ | | | |")

    out += ["", "## Kept, but not in the script", "",
            "Probable ad-libs. These were left in deliberately.", ""]
    if report["adlibs"]:
        for a in report["adlibs"]:
            out.append(f"- `{_ts(a['start'])}` {a['text']}")
    else:
        out.append("_none_")

    return "\n".join(out) + "\n"


def write_markdown(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_report_markdown.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add report.py tests/test_report_markdown.py
git commit -m "feat: add markdown review report with silence summarised"
```

---

## Task 16: Transcription adapter and cache

**Files:**
- Create: `transcribe.py`
- Test: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_transcribe.py`:

```python
import json

from config import EditorConfig
from models import Word
from transcribe import (cache_path_for, load_cached, probe_duration,
                        save_cache, transcribe)


def test_cache_path_sits_beside_the_video(tmp_path):
    video = tmp_path / "raw.mp4"
    assert cache_path_for(video).name == "raw.transcript.json"
    assert cache_path_for(video).parent == tmp_path


def test_cache_round_trip(tmp_path):
    ws = [Word("hello", 0.0, 0.4), Word("world", 0.5, 0.9)]
    path = tmp_path / "raw.transcript.json"
    save_cache(path, ws, duration=12.5, model="medium")

    loaded, duration = load_cached(path, model="medium")
    assert loaded == ws
    assert duration == 12.5


def test_cache_miss_on_different_model(tmp_path):
    path = tmp_path / "raw.transcript.json"
    save_cache(path, [Word("hi", 0.0, 0.2)], duration=1.0, model="small")
    assert load_cached(path, model="medium") == (None, None)


def test_cache_miss_on_absent_file(tmp_path):
    assert load_cached(tmp_path / "nope.json", model="medium") == (None, None)


def test_corrupt_cache_is_treated_as_a_miss(tmp_path):
    path = tmp_path / "raw.transcript.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_cached(path, model="medium") == (None, None)


def test_transcribe_uses_cache_without_loading_the_model(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    save_cache(cache_path_for(video), [Word("cached", 0.0, 0.5)],
               duration=3.0, model="medium")

    def explode(*a, **k):
        raise AssertionError("model must not be loaded on a cache hit")

    monkeypatch.setattr("transcribe._run_whisper", explode)
    ws, duration = transcribe(video, EditorConfig(model="medium"))
    assert [w.text for w in ws] == ["cached"]
    assert duration == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transcribe'`

- [ ] **Step 3: Write `transcribe.py`**

```python
"""faster-whisper adapter.

The only module that knows a speech model exists. Its entire contract with the
rest of the project is `list[Word]` plus a duration, which is what makes every
downstream stage testable without audio.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from config import EditorConfig
from models import Word

CACHE_VERSION = 1


def cache_path_for(video: str | Path) -> Path:
    video = Path(video)
    return video.with_suffix("").with_name(video.stem + ".transcript.json")


def save_cache(path: str | Path, words: list[Word], duration: float,
               model: str) -> Path:
    path = Path(path)
    path.write_text(json.dumps({
        "version": CACHE_VERSION,
        "model": model,
        "duration": duration,
        "words": [{"text": w.text, "start": w.start, "end": w.end} for w in words],
    }, indent=2), encoding="utf-8")
    return path


def load_cached(path: str | Path,
                model: str) -> tuple[list[Word] | None, float | None]:
    """Return (words, duration), or (None, None) on any kind of miss."""
    path = Path(path)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    if data.get("version") != CACHE_VERSION or data.get("model") != model:
        return None, None
    words = [Word(w["text"], w["start"], w["end"]) for w in data["words"]]
    return words, data["duration"]


def probe_duration(video: str | Path) -> float:
    """Media length in seconds, via ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg: winget install Gyan.FFmpeg")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def _run_whisper(video: Path, cfg: EditorConfig) -> list[Word]:
    """Load the model and transcribe. Isolated so tests can stub it out."""
    from faster_whisper import WhisperModel

    model = WhisperModel(cfg.model, device="auto", compute_type="auto")
    segments, _ = model.transcribe(str(video), word_timestamps=True,
                                   language=cfg.language)
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            if text:
                words.append(Word(text, float(w.start), float(w.end)))
    return words


def transcribe(video: str | Path,
               cfg: EditorConfig) -> tuple[list[Word], float]:
    """Word-level transcript plus media duration, cached beside the video."""
    video = Path(video)
    cache = cache_path_for(video)

    if cfg.use_cache:
        words, duration = load_cached(cache, cfg.model)
        if words is not None and duration is not None:
            return words, duration

    duration = probe_duration(video)
    words = _run_whisper(video, cfg)
    save_cache(cache, words, duration, cfg.model)
    return words, duration
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_transcribe.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add transcribe.py tests/test_transcribe.py
git commit -m "feat: add faster-whisper adapter with model-aware transcript cache"
```

---

## Task 17: CLI — phase 1 complete (report only)

**Files:**
- Create: `edit.py`
- Test: `tests/test_edit_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_edit_pipeline.py`:

```python
import pytest

from config import EditorConfig
from edit import analyse, build_parser, config_from_args
from models import MatchStatus
from conftest import words


def test_analyse_produces_matches_and_cuts():
    ws = words("click the the blue button", gaps=[0, 0, 0, 0])
    script = "click the blue button"
    report = analyse(ws, script, duration=10.0, cfg=EditorConfig(),
                    video="raw.mp4")
    assert report["lines"][0]["status"] in ("ok", "review")
    assert report["summary"]["cut_count"] >= 1


def test_analyse_flags_a_missing_line():
    ws = words("click the blue button")
    report = analyse(ws, "click the blue button\n\nnothing resembling this",
                     duration=10.0, cfg=EditorConfig(), video="raw.mp4")
    assert report["lines"][1]["status"] == "miss"
    assert report["summary"]["misses"] == 1


def test_analyse_keeps_adlibs():
    ws = words("click the blue button and honestly this part was made up")
    report = analyse(ws, "click the blue button", duration=10.0,
                     cfg=EditorConfig(), video="raw.mp4")
    assert any("made up" in a["text"] for a in report["adlibs"])


def test_parser_defaults_match_config():
    args = build_parser().parse_args(["--script", "s.txt", "--video", "v.mp4"])
    cfg = config_from_args(args)
    assert cfg.threshold == 0.75
    assert cfg.silence_ms == 600
    assert cfg.clean_stutters is True


def test_no_clean_stutters_flag_disables_detection():
    args = build_parser().parse_args(
        ["--script", "s.txt", "--video", "v.mp4", "--no-clean-stutters"])
    assert config_from_args(args).clean_stutters is False


def test_dry_run_and_auto_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--script", "s.txt", "--video", "v.mp4", "--dry-run", "--auto"])


def test_reorder_and_export_xml_are_rejected_as_not_implemented():
    for flag in ("--reorder", "--export-xml"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["--script", "s.txt", "--video", "v.mp4", flag])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_edit_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'edit'`

- [ ] **Step 3: Write `edit.py`**

```python
"""CLI entrypoint and orchestration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aligner import align_script
from config import EditorConfig
from cutlist import build_cutlist
from disfluency import detect_disfluencies
from models import Word
from report import build_report, load_report, write_json, write_markdown
from script_parser import parse_script
from silence import detect_silence
from transcribe import transcribe


class _NotImplementedFlag(argparse.Action):
    """Reject a specced-but-unbuilt flag loudly instead of ignoring it."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            f"{option_string} is not implemented in this build (phases 1-2 only). "
            "See docs/superpowers/specs for the deferred scope.")


def build_parser() -> argparse.ArgumentParser:
    d = EditorConfig()
    p = argparse.ArgumentParser(
        prog="edit.py",
        description="Remove dead air, false starts, and stutters from a "
                    "script-matched tutorial recording, keeping ad-libs.")

    p.add_argument("--script", required=True, help="Voiceover script (.txt)")
    p.add_argument("--video", required=True, help="Raw screen recording")
    p.add_argument("--out", help="Rendered output (required with --auto/--from-report)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Report only, no render (default)")
    mode.add_argument("--auto", action="store_true",
                      help="Render immediately, skipping the review gate")

    p.add_argument("--from-report", help="Render from an approved JSON report")
    p.add_argument("--report-path", help="Stem for the .json/.md report pair")

    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--review-margin", type=float, default=d.review_margin)
    p.add_argument("--lookahead-s", type=float, default=d.lookahead_s)
    p.add_argument("--tiebreak", choices=["latest", "score"], default=d.tiebreak)
    p.add_argument("--silence-ms", type=int, default=d.silence_ms)
    p.add_argument("--padding-ms", type=int, default=d.padding_ms)
    p.add_argument("--min-segment-ms", type=int, default=d.min_segment_ms)
    p.add_argument("--crossfade-ms", type=int, default=d.crossfade_ms)
    p.add_argument("--duplicate-window-s", type=float, default=d.duplicate_window_s)
    p.add_argument("--max-disfluency-words", type=int, default=d.max_disfluency_words)
    p.add_argument("--model", default=d.model)
    p.add_argument("--language", default=None)
    p.add_argument("--split-sentences", action="store_true")
    p.add_argument("--no-clean-stutters", action="store_true",
                   help="Disable disfluency detection entirely")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore any cached transcript")

    p.add_argument("--reorder", nargs=0, action=_NotImplementedFlag,
                   help=argparse.SUPPRESS)
    p.add_argument("--export-xml", nargs=0, action=_NotImplementedFlag,
                   help=argparse.SUPPRESS)
    return p


def config_from_args(args: argparse.Namespace) -> EditorConfig:
    return EditorConfig(
        threshold=args.threshold,
        review_margin=args.review_margin,
        lookahead_s=args.lookahead_s,
        tiebreak=args.tiebreak,
        silence_ms=args.silence_ms,
        padding_ms=args.padding_ms,
        min_segment_ms=args.min_segment_ms,
        crossfade_ms=args.crossfade_ms,
        duplicate_window_s=args.duplicate_window_s,
        max_disfluency_words=args.max_disfluency_words,
        clean_stutters=not args.no_clean_stutters,
        model=args.model,
        language=args.language,
        use_cache=not args.no_cache,
    )


def analyse(words: list[Word], script_text: str, duration: float,
            cfg: EditorConfig, video: str,
            split_sentences: bool = False) -> dict:
    """The whole of phase 1: transcript + script in, review report out."""
    lines = parse_script(script_text, split_sentences=split_sentences)

    raw = detect_silence(words, cfg, duration=duration)
    matches, retake_cuts = align_script(words, lines, cfg)
    raw += retake_cuts

    for m in matches:
        if m.word_range:
            raw += detect_disfluencies(words, m.word_range[0], m.word_range[1], cfg)

    cuts = build_cutlist(raw, duration, cfg)
    return build_report(lines, matches, cuts, words, duration, cfg, video)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    video = Path(args.video)

    if args.from_report:
        report = load_report(args.from_report)
    else:
        if not video.exists():
            print(f"error: video not found: {video}", file=sys.stderr)
            return 2
        script_text = Path(args.script).read_text(encoding="utf-8")

        print(f"Transcribing {video.name} (model={cfg.model})...", file=sys.stderr)
        words, duration = transcribe(video, cfg)
        print(f"  {len(words)} words, {duration:.1f}s", file=sys.stderr)

        report = analyse(words, script_text, duration, cfg, str(video),
                         split_sentences=args.split_sentences)

        stem = Path(args.report_path) if args.report_path \
            else video.with_suffix("").with_name(video.stem + ".report")
        write_json(report, stem.with_suffix(".json"))
        write_markdown(report, stem.with_suffix(".md"))

        s = report["summary"]
        print(f"\nReport: {stem.with_suffix('.md')}", file=sys.stderr)
        print(f"  {s['cut_count']} cuts, {s['cut_seconds']:.1f}s removed "
              f"-> {s['output_seconds']:.1f}s output", file=sys.stderr)
        print(f"  {s['misses']} missed lines, {s['reviews']} needing review",
              file=sys.stderr)

    if not (args.auto or args.from_report):
        print("\nReview the report, then re-run with --auto, or render the "
              "approved report with --from-report.", file=sys.stderr)
        return 0

    if not args.out:
        print("error: --out is required when rendering", file=sys.stderr)
        return 2

    from render import render_video
    render_video(video, Path(args.out), report, cfg)
    print(f"\nRendered: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_edit_pipeline.py -v`
Expected: 7 passed

- [ ] **Step 5: Verify the CLI help renders**

Run: `python edit.py --help`
Expected: usage text listing every flag, exit code 0

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add edit.py tests/test_edit_pipeline.py
git commit -m "feat: add CLI and analysis pipeline (phase 1 complete)"
```

---

## Task 18: Render — filter graph construction

**Files:**
- Create: `render.py`
- Test: `tests/test_render_graph.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_graph.py`:

```python
import pytest

from config import EditorConfig
from models import Span
from render import build_filter_graph


def test_single_segment_graph():
    graph = build_filter_graph([Span(1.0, 5.0)], EditorConfig(crossfade_ms=0))
    assert "trim=start=1.000000:end=5.000000" in graph
    assert "atrim=start=1.000000:end=5.000000" in graph
    assert "concat=n=1:v=1:a=1[outv][outa]" in graph


def test_multi_segment_graph_labels_every_segment():
    segs = [Span(0.0, 2.0), Span(4.0, 6.0), Span(8.0, 10.0)]
    graph = build_filter_graph(segs, EditorConfig(crossfade_ms=0))
    for i in range(3):
        assert f"[v{i}]" in graph
        assert f"[a{i}]" in graph
    assert "concat=n=3:v=1:a=1[outv][outa]" in graph


def test_timestamps_are_reset_per_segment():
    graph = build_filter_graph([Span(5.0, 7.0)], EditorConfig(crossfade_ms=0))
    assert "setpts=PTS-STARTPTS" in graph
    assert "asetpts=PTS-STARTPTS" in graph


def test_fades_are_applied_when_crossfade_enabled():
    graph = build_filter_graph([Span(0.0, 4.0)], EditorConfig(crossfade_ms=75))
    assert "afade=t=in:st=0:d=0.075" in graph
    assert "afade=t=out:st=3.925:d=0.075" in graph


def test_fade_is_skipped_on_segments_too_short_to_hold_one():
    graph = build_filter_graph([Span(0.0, 0.1)], EditorConfig(crossfade_ms=75))
    assert "afade" not in graph


def test_no_fade_when_crossfade_is_zero():
    graph = build_filter_graph([Span(0.0, 4.0)], EditorConfig(crossfade_ms=0))
    assert "afade" not in graph


def test_empty_segments_raises():
    with pytest.raises(ValueError):
        build_filter_graph([], EditorConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'render'`

- [ ] **Step 3: Write `render.py`**

```python
"""ffmpeg rendering of the surviving segments, in original order.

Frame-accurate cutting of video and audio together requires re-encoding;
stream copy cannot do it.

Audio uses a short per-segment afade rather than a true acrossfade. A real
crossfade overlaps adjacent segments, shortening total audio by the fade
length at every cut, which accumulates into audible A/V desync over hundreds
of cuts. Per-segment fades suppress clicks identically while preserving exact
durations.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from config import EditorConfig
from models import Span


def build_filter_graph(segments: list[Span], cfg: EditorConfig) -> str:
    """filter_complex graph selecting and concatenating the surviving segments."""
    if not segments:
        raise ValueError("cannot render: no surviving segments")

    fade = cfg.crossfade_s
    parts: list[str] = []

    for i, seg in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={seg.start:.6f}:end={seg.end:.6f},"
            f"setpts=PTS-STARTPTS[v{i}]")

        achain = (f"[0:a]atrim=start={seg.start:.6f}:end={seg.end:.6f},"
                  f"asetpts=PTS-STARTPTS")
        # a fade needs room at both ends; skip it on very short segments
        if fade > 0 and seg.duration > 2 * fade:
            out_at = round(seg.duration - fade, 6)
            achain += (f",afade=t=in:st=0:d={fade:g}"
                       f",afade=t=out:st={out_at:g}:d={fade:g}")
        parts.append(f"{achain}[a{i}]")

    streams = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
    parts.append(f"{streams}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    return ";\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_render_graph.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add render.py tests/test_render_graph.py
git commit -m "feat: add ffmpeg filter graph builder with per-segment audio fades"
```

---

## Task 19: Render — ffmpeg invocation

**Files:**
- Modify: `render.py` (append)
- Test: `tests/test_render_invoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_render_invoke.py`:

```python
import pytest

from config import EditorConfig
from models import Span
from render import build_ffmpeg_command, render_video


def test_command_uses_filter_complex_script_not_inline():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", "graph.txt", EditorConfig())
    assert "-filter_complex_script" in cmd
    assert "-filter_complex" not in cmd
    assert cmd[cmd.index("-filter_complex_script") + 1] == "graph.txt"


def test_command_maps_the_concat_outputs():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", "graph.txt", EditorConfig())
    assert "[outv]" in cmd and "[outa]" in cmd


def test_command_honours_codec_config():
    cfg = EditorConfig(video_codec="libx265", video_crf=22, audio_bitrate="128k")
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", "graph.txt", cfg)
    assert "libx265" in cmd
    assert "22" in cmd
    assert "128k" in cmd


def test_command_overwrites_without_prompting():
    assert "-y" in build_ffmpeg_command("in.mp4", "out.mp4", "g.txt", EditorConfig())


def test_render_refuses_a_report_with_no_segments(tmp_path):
    report = {"segments": [], "duration": 10.0}
    with pytest.raises(ValueError, match="no surviving segments"):
        render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                     EditorConfig())


def test_render_writes_the_graph_to_a_file(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        graph_path = cmd[cmd.index("-filter_complex_script") + 1]
        seen["graph"] = open(graph_path, encoding="utf-8").read()

        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr("render.subprocess.run", fake_run)
    monkeypatch.setattr("render.shutil.which", lambda n: "ffmpeg")

    report = {"segments": [{"start": 0.0, "end": 2.0},
                           {"start": 4.0, "end": 6.0}], "duration": 10.0}
    render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                 EditorConfig())
    assert "concat=n=2" in seen["graph"]


def test_missing_ffmpeg_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr("render.shutil.which", lambda n: None)
    report = {"segments": [{"start": 0.0, "end": 2.0}], "duration": 10.0}
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                     EditorConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render_invoke.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_ffmpeg_command'`

- [ ] **Step 3: Append to `render.py`**

```python
def build_ffmpeg_command(video: str | Path, out: str | Path,
                         graph_path: str | Path,
                         cfg: EditorConfig) -> list[str]:
    """Assemble the ffmpeg argv.

    The graph goes in via -filter_complex_script rather than inline: a few
    hundred cuts produces a filter string far past the Windows 8191-character
    command-line limit, which would fail only on long recordings.
    """
    return [
        "ffmpeg", "-y",
        "-i", str(video),
        "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", cfg.video_codec,
        "-preset", cfg.video_preset,
        "-crf", str(cfg.video_crf),
        "-c:a", cfg.audio_codec,
        "-b:a", cfg.audio_bitrate,
        str(out),
    ]


def render_video(video: str | Path, out: str | Path, report: dict,
                 cfg: EditorConfig) -> Path:
    """Render the surviving segments of `report` to `out`."""
    segments = [Span(s["start"], s["end"]) for s in report.get("segments", [])]
    if not segments:
        raise ValueError("cannot render: no surviving segments in report")

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg")

    graph = build_filter_graph(segments, cfg)

    tmp = Path(tempfile.gettempdir()) / f"auto_editor_graph_{Path(out).stem}.txt"
    tmp.write_text(graph, encoding="utf-8")

    try:
        cmd = build_ffmpeg_command(video, out, tmp, cfg)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}")
    finally:
        tmp.unlink(missing_ok=True)

    return Path(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_render_invoke.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add render.py tests/test_render_invoke.py
git commit -m "feat: add ffmpeg invocation via filter_complex_script"
```

---

## Task 20: End-to-end smoke test against real media

**Files:**
- Create: `tests/test_smoke_endtoend.py`

This is the only test that touches ffmpeg. It generates its own media, so it needs no sample recording, but it is slow — mark it so the fast suite stays fast.

- [ ] **Step 1: Write the smoke test**

Create `tests/test_smoke_endtoend.py`:

```python
"""End-to-end check against real media. Slow; excluded from the default run.

Run with: python -m pytest tests/test_smoke_endtoend.py -v -m smoke
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from config import EditorConfig
from models import Span
from render import render_video
from transcribe import probe_duration

pytestmark = pytest.mark.smoke

HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A 10-second test clip: colour bars plus a sine tone."""
    path = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        str(path),
    ], capture_output=True, check=True)
    return path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_probe_duration_reads_real_media(clip):
    assert 9.5 < probe_duration(clip) < 10.5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_removes_the_cut_spans(clip, tmp_path):
    # keep 0-2s and 6-10s: a 6-second output from a 10-second source
    report = {"duration": 10.0,
              "segments": [{"start": 0.0, "end": 2.0},
                           {"start": 6.0, "end": 10.0}]}
    out = tmp_path / "out.mp4"
    render_video(clip, out, report, EditorConfig())

    assert out.exists()
    assert 5.5 < probe_duration(out) < 6.5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_survives_a_graph_past_the_windows_arg_limit(clip, tmp_path):
    # 300 segments produces a filter string well over 8191 characters
    segments = [{"start": round(i * 0.03, 3), "end": round(i * 0.03 + 0.02, 3)}
                for i in range(300)]
    report = {"duration": 10.0, "segments": segments}
    out = tmp_path / "many.mp4"
    render_video(clip, out, report, EditorConfig(crossfade_ms=0))
    assert out.exists()
```

- [ ] **Step 2: Confirm the marker is already registered**

`pytest.ini` was created in Task 1 and already contains the `smoke` marker and `addopts = -m "not smoke"`. Verify it, do not recreate it:

Run: `python -c "print(open('pytest.ini').read())"`
Expected: output contains `smoke:` and `addopts = -m \"not smoke\"`

- [ ] **Step 3: Run the fast suite and confirm smoke tests are excluded**

Run: `python -m pytest tests/ -v`
Expected: all pass, smoke tests deselected

- [ ] **Step 4: Run the smoke tests**

Run: `python -m pytest tests/test_smoke_endtoend.py -v -m smoke`
Expected: 3 passed. The third takes ~30s.

- [ ] **Step 5: Commit**

```bash
git add tests/test_smoke_endtoend.py
git commit -m "test: add end-to-end smoke tests against generated media"
```

---

## Task 21: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Script-Matched Tutorial Video Auto-Editor

Removes dead air, false starts, and stutters from a screen recording where you
read a voiceover script aloud — while keeping everything you actually meant to
say, including ad-libs that appear nowhere in the script.

## How it decides what to cut

The algorithm is **subtractive**. It starts from the full recording and removes
only what it is confident about:

| Category | Action |
|---|---|
| Clean recitation of a script line, or the winning take of a retaken line | keep |
| Silence above the gap threshold | cut |
| Losing attempt at a line that was retaken | cut |
| Stutter, immediate repeat, or truncated-word restart | cut |
| Anything else that doesn't match the script — ad-libs, asides, extra explanation | **keep**, and list in the report |

The rejected alternative, "keep only what matches the script," would delete
intentional ad-libs. Nothing is reordered.

Disfluency detection is deliberately conservative: it requires temporal
adjacency, not merely a shared word, and it caps how much it will remove. A
missed stutter is a small cost; an eaten ad-lib is a large one. Every
judgment-call cut is listed in the report so you can check.

## Install

```bash
python -m pip install -r requirements.txt
winget install Gyan.FFmpeg
```

Requires Python 3.11+ and ffmpeg on PATH.

## Use

Analyse and produce a review report — this is the default, nothing is rendered:

```bash
python edit.py --script script.txt --video raw_recording.mp4
```

That writes `raw_recording.report.md` (read this) and `raw_recording.report.json`
(edit this). Read the Markdown, check the judgment-call cuts, then render:

```bash
python edit.py --script script.txt --video raw_recording.mp4 --out final.mp4 --auto
```

If you disagree with a cut, delete it from the JSON report's `segments` list and
render the approved version directly — no re-transcription:

```bash
python edit.py --video raw_recording.mp4 --script script.txt --from-report raw_recording.report.json --out final.mp4
```

The transcript is cached beside the video as `<name>.transcript.json`, so
re-running with different thresholds costs no Whisper time. `--no-cache` forces
a fresh transcription.

## Tuning

| Flag | Default | What it does |
|---|---|---|
| `--threshold` | 0.75 | Fuzzy score a window must hit to count as a line match |
| `--review-margin` | 0.05 | Band above the threshold flagged as REVIEW rather than OK |
| `--silence-ms` | 600 | Shortest gap treated as cuttable silence |
| `--padding-ms` | 75 | How far each cut boundary is pulled inward, so cuts don't clip breaths |
| `--min-segment-ms` | 150 | Surviving slivers shorter than this are absorbed |
| `--crossfade-ms` | 75 | Per-segment audio fade at cut points; 0 disables |
| `--duplicate-window-s` | 4.0 | Max separation for a repeat to count as a stutter rather than deliberate |
| `--max-disfluency-words` | 5 | Hard cap on how much any one disfluency cut may remove |
| `--tiebreak` | latest | Retake tie-break: `latest` or `score` |
| `--lookahead-s` | 30.0 | Floor on how far ahead each line is searched |
| `--model` | medium | Whisper model size |
| `--no-clean-stutters` | off | Turn disfluency detection off entirely |
| `--split-sentences` | off | Split the script per sentence instead of per blank line |

## Known limits

- Whisper sometimes smooths over very fast stutters and won't transcribe every
  repeated fragment, so disfluency detection is best-effort. The report shows
  what it did catch.
- A deliberate emphasis repeat ("no, no, don't click that") is not reliably
  distinguishable from an accidental stutter. The narrow window and the
  explicit disfluency-cut list in the report are the safety net. If one gets
  cut wrongly, delete it from the JSON and use `--from-report`.
- `--reorder` (rebuild in script order) and `--export-xml` (Premiere timeline)
  are specced but not built. Both are rejected at the command line rather than
  silently ignored.

## Tests

```bash
python -m pytest tests/ -v
```

The whole analysis core is pure functions over word lists, so it tests with
hand-written fixtures — no video, no Whisper, no ffmpeg. The end-to-end tests
generate their own media and are excluded by default:

```bash
python -m pytest tests/test_smoke_endtoend.py -v -m smoke
```
````

- [ ] **Step 2: Verify the documented commands parse**

Run: `python edit.py --help`
Expected: exit 0, every flag in the README table appears

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README covering the subtractive model and tuning knobs"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §3.1 seam / testability | 1, 16 |
| §3.2 shared data model | 1 |
| §4 pipeline steps 1–9 | 16, 3, 4, 5–7, 8–10, 11–13, 14–15, 18–19 |
| §5.1 monotonic cursor + fallback | 7 |
| §5.2 OK/REVIEW/MISS | 7 |
| §6.1 overlap merging | 11 |
| §6.2 padding + inversion clamp | 12 |
| §6.3 sliver absorption | 13 |
| §7 three disfluency detectors + guards | 8, 9, 10 |
| §8 review gate, cache, `--from-report` | 16, 17 |
| §9 JSON + Markdown reports | 14, 15 |
| §10.1 afade not acrossfade | 18 |
| §10.2 `-filter_complex_script` | 19, 20 |
| §11 every CLI flag | 17 |
| §12 fixture-driven tests | every task; 20 for smoke |
| §14 `--reorder`/`--export-xml` rejected | 17 |

No gaps.

**Placeholder scan:** No TBD/TODO. Every code step contains complete, runnable code. Every test step contains real assertions.

**Type consistency verified:**
- `Candidate.start_idx` / `.end_idx` / `.score` — consistent across Tasks 5, 6, 7.
- `LineMatch.word_range` is `(start, end)` exclusive — produced in Task 7, consumed in Tasks 14 and 17.
- `CutSpan.reasons` is a mutable `set` — required by the merge in Task 11; `Word` and `Span` are frozen, `CutSpan` is not. Deliberate, tested in Task 1.
- `build_cutlist(cuts, duration, cfg)` — defined Task 13, called Task 17 with the same signature.
- `render_video(video, out, report, cfg)` — defined Task 19, called Task 17.
- `detect_disfluencies(words, lo, hi, cfg)` — defined Task 10, called Task 17.
- `report["segments"]` — written in Task 14, consumed by `render_video` in Task 19 and by the smoke test in Task 20.

**One ordering note for the implementer:** Task 17 imports `render` lazily inside `main()`, so the CLI works before Task 19 exists. Do not hoist that import to the top of `edit.py`.
