# Script-Matched Tutorial Video Auto-Editor — Design

Date: 2026-07-31
Status: Approved
Source spec: `auto-editor-spec.md`
Scope this pass: Build phases 1 and 2 (report + subtractive render). Phases 3–5 out of scope.

## 1. Problem

Given a voiceover script and a screen recording of someone reading that script aloud while
demoing something, produce an edited video with dead air, false starts, and stutters removed —
while preserving everything the speaker actually intended to say, including deliberate ad-libs
that appear nowhere in the script.

## 2. Core principle: subtractive, not reconstructive

Every moment in the recording falls into one of four categories:

| Category | Action |
|---|---|
| Script match (clean recitation, or winning take of a retaken line) | keep |
| Local disfluency (immediate repeat, truncated restart, losing retake) | cut |
| Unmatched-but-intentional (ad-lib, aside, extra explanation) | **keep**, and report |
| Silence / dead air | cut |

The algorithm starts from the full recording timeline and removes only spans confidently
identified as silence, a losing retake, or a local disfluency. Everything else stays in its
original position and original order.

The rejected alternative — "keep only what matches the script, cut everything else" — would
delete intentional ad-libs. That behaviour is deferred to an opt-in `--reorder` mode, which is
out of scope for this pass.

## 3. Architecture

### 3.1 The seam

`transcribe.py` is a thin adapter whose only job is to return `list[Word]`. Every module
downstream of it is a pure function over word lists.

This is the central architectural decision. It means the alignment, retake, disfluency,
padding, and cut-list logic — the entire risky part of the system — is testable with
hand-authored fixtures, with no video file, no Whisper model, and no ffmpeg. Correctness is
established before the tool is ever pointed at a real recording.

### 3.2 Shared data model (`models.py`)

The source spec lists modules but not the contract between them. Defining the types in one
place is required: without it, `aligner`, `disfluency`, and `cutlist` form an import cycle.

```
Word      { text: str, start: float, end: float }
Span      { start: float, end: float }
CutReason enum: SILENCE | RETAKE_LOSER | DISFLUENCY
CutSpan   { start: float, end: float, reasons: set[CutReason], detail: str }
MatchStatus enum: OK | REVIEW | MISS
LineMatch { line_idx: int, span: Span|None, score: float,
            word_range: tuple[int,int]|None, status: MatchStatus, scoped: bool }
```

All times are floating-point seconds relative to the start of the recording.

### 3.3 Modules

```
auto-editor/
  edit.py            # CLI entrypoint, orchestration, review gate
  config.py          # all tuning knobs as one dataclass
  models.py          # shared dataclasses above
  transcribe.py      # faster-whisper adapter -> list[Word]; transcript caching
  script_parser.py   # script.txt -> list[ScriptLine]
  silence.py         # gap-based silence detection
  aligner.py         # fuzzy sliding-window matching + retake resolution
  disfluency.py      # local repeat / restart / duplicate-phrase detection
  cutlist.py         # merge, pad, sliver-absorb -> final cut list
  render.py          # ffmpeg trim/atrim + concat over surviving segments
  report.py          # JSON + Markdown review reports
  tests/             # pytest, fixture-driven
  requirements.txt
```

`config.py`, `models.py`, and `tests/` are additions to the source spec's layout. The rest
follows it. `xml_export.py` is deferred with phase 4.

## 4. Pipeline

1. **Transcribe** audio with word-level timestamps (`faster-whisper`). Cache to
   `<video>.transcript.json`; reuse on subsequent runs unless `--no-cache`.
2. **Parse script** into lines/beats. Split on blank lines by default; `--split-sentences`
   splits a single block on sentence boundaries instead.
3. **Detect silence** from gaps between consecutive word timestamps. Gaps at or above
   `--silence-ms` become `SILENCE` cut spans.
4. **Align** each script line against the transcript with `rapidfuzz` over a sliding window,
   collecting every window scoring at or above `--threshold` as a candidate match.
5. **Resolve retakes**: among candidates for one line, keep the best score; tie-break toward
   the later occurrence (`--tiebreak`). Losing attempts, and the gap between a loser and the
   winner, become `RETAKE_LOSER` cut spans.
6. **Detect disfluencies** within and immediately around each winning match. Redundant
   fragments become `DISFLUENCY` cut spans.
7. **Build cut list**: merge, pad, absorb slivers (section 6).
8. **Report** — always written, before any rendering.
9. **Render** — only when `--auto` or `--from-report` is passed.

## 5. Alignment design

### 5.1 Monotonic search cursor with full-recording fallback

The source spec calls out the failure mode — two similarly-worded script lines matching the
wrong occurrence — and prescribes scoping the search "near where each line is expected in
time", without specifying how that expectation is formed.

Resolution: process lines in script order, maintaining a cursor at the end of the previous
line's winning match. Each line searches forward from that cursor, bounded by a lookahead
window. Because line *N+1* only searches after line *N*'s match, similar lines cannot steal
each other's occurrence, and retakes are still found naturally (both attempts of line *N* fall
after line *N-1*).

The lookahead window extends from the cursor forward by
`max(--lookahead-s, 3 x estimated_line_duration)`, where `estimated_line_duration` is the
line's word count multiplied by the mean seconds-per-word observed across the whole transcript.
The floor keeps short lines searchable across a long pause; the multiplier lets long lines
scale up. Both attempts of a retaken line normally fall inside this window.

The scoping is a preference, not a constraint. If the scoped region yields no candidate at or
above threshold, the search falls back to the entire recording and the resulting match is
marked `scoped=False` in the report. Without this fallback, a single missed line would shift
the cursor wrongly and cascade failures through every subsequent line.

### 5.2 Match status

- `OK` — score comfortably above threshold.
- `REVIEW` — score within `--review-margin` of threshold. Flagged distinctly in the report; a
  bare pass is not treated as a clean match.
- `MISS` — no candidate anywhere, including after fallback. Reported as a hard miss. Never
  silently dropped.

## 6. Cut-list construction

### 6.1 Overlap merging

Not mentioned in the source spec, and required for correctness. The three detectors run
independently and will produce colliding spans — a disfluency inside a losing retake, silence
abutting both. Feeding overlapping spans to the renderer produces negative-duration trims.

After collection: sort by start, merge overlapping and adjacent spans, and union their
`reasons` sets so the report still explains why each merged span was removed.

### 6.2 Padding

Each cut span is first trimmed to the surrounding word timestamps, then pulled *inward* by
`--padding-ms` on each side so the cut lands inside the silent gap rather than clipping a
breath or a plosive. Pulling inward makes the cut span smaller, never larger.

Guard: if a span is shorter than twice the padding, the pull-in is clamped so the span can
never invert. Applies uniformly to silence, retake-loser, and disfluency cuts.

### 6.3 Sliver absorption

After padding, any surviving keep-segment shorter than `--min-segment-ms` is absorbed into the
cut list, merging the two cuts that surround it. Applied in a single pass over the segment list
so absorption cannot cascade into deleting long stretches.

## 7. Disfluency detection

Three narrow detectors, all restricted to short spans immediately adjacent to a clean
continuation:

1. **Immediate word repeats** — the same or near-identical token twice in a row, optionally
   separated by a short filler (`uh`, `um`, `er`, `ah`). Cut the earlier occurrence.
2. **Truncated-word restarts** — a short fragment immediately followed by the full word, where
   the fragment is a literal prefix of the full word and at least 2 characters
   ("config-, configuration"). Cut the fragment.
3. **Short duplicate phrases seconds apart** — an n-gram repeated within
   `--duplicate-window-s`, where the second occurrence continues cleanly. Cut the earlier
   occurrence plus the gap before the clean continuation.

Guards: no detector may cut more than `--max-disfluency-words` tokens. All three require
temporal adjacency, not merely shared vocabulary — this is what keeps an ad-lib that happens to
reuse a nearby script word from being eaten.

Deliberate bias: leaving a minor stutter in is preferable to trimming an intentional ad-lib.
Genuine emphasis repeats ("no, no, don't click that") are not reliably distinguishable from
accidental stutters, so the window stays narrow and the report's explicit disfluency-cut list
is the safety net.

Whisper sometimes smooths over fast stutters and will not transcribe every repeated fragment,
so this stage is best-effort by construction. `--clean-stutters` (default on) disables it
entirely.

## 8. Review gate

The source spec specifies `--dry-run` defaulting to true alongside `--auto`, which is
redundant, and describes a review gate that is impractical as written: approving a report would
mean re-running Whisper over the whole recording.

Resolution:

- The report is **always** written. There is no mode that renders without producing one.
- `--auto` renders in the same run, skipping the gate.
- The transcript is cached to `<video>.transcript.json`, so a second run costs no transcription.
- `--from-report <path>` renders directly from an approved JSON report, whose cut list may be
  hand-edited first.

`--from-report` is the only meaningful addition beyond the source spec. It converts "report
first, render second" from a stated intent into a workflow that can actually be used, and it
makes the JSON report the reviewable, editable artifact rather than a read-only summary.

## 9. Reports

Two files written from one run, sharing the `--report-path` stem:

- **JSON** — machine-readable, and the input to `--from-report`. Contains the full word list
  reference, every `LineMatch`, and every `CutSpan` with reasons.
- **Markdown** — human review. Per script line: matched text, timestamp, confidence, and a flag
  for `REVIEW` or `MISS`. Then every non-silence cut span with its reason, silence summarised as
  a count and total duration to keep the report short. Then every unmatched-but-kept span, so
  ad-libs are easy to eyeball.

## 10. Rendering

Frame-accurate cutting of video and audio together requires re-encoding; stream copy cannot do
it. The render builds the surviving segments in original order via an ffmpeg `filter_complex`
chain of `trim` / `atrim` + `concat`.

### 10.1 Audio fades instead of crossfades

The source spec asks for a 50–100 ms audio crossfade at each cut. A true `acrossfade` overlaps
adjacent segments, which shortens total audio duration by the fade length at every cut. Across
a few hundred cuts this accumulates into audible audio/video desynchronisation.

Instead, a short `afade` in/out is applied to each surviving segment before concatenation. This
suppresses clicks identically for voiceover material, while preserving exact segment durations
and therefore perfect A/V sync. Controlled by `--crossfade-ms`; 0 disables it. Video cuts are
hard, which is appropriate for tutorial content.

This is a deliberate, approved departure from the source spec's wording, made to satisfy its
actual intent.

### 10.2 Windows command-line limit

A recording with a few hundred cuts produces a `filter_complex` string well past Windows'
8191-character command-line limit. The filter graph is therefore written to a temporary file
and passed via `-filter_complex_script`, not inline. Without this the tool would fail only on
long recordings — precisely the ones that matter.

## 11. CLI

```
python edit.py --script script.txt --video raw.mp4 --out final.mp4 [flags]
```

`--script` and `--video` are required. `--out` is required only when rendering (`--auto` or
`--from-report`); a report-only run does not need it.

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | 0.75 | Fuzzy match confidence for a script line |
| `--silence-ms` | 600 | Minimum gap counting as cuttable silence |
| `--padding-ms` | 75 | Inward pull-in at every cut boundary |
| `--model` | medium | Whisper model size |
| `--dry-run` | on | Report only, no render. The default; accepted explicitly for clarity |
| `--auto` | off | Render immediately, skipping the review gate |
| `--from-report` | — | Render from an approved JSON report |
| `--lookahead-s` | 30.0 | Floor on the forward search window from the cursor |
| `--clean-stutters` | on | Toggle disfluency detection |
| `--tiebreak` | latest | Retake tie-break: `latest` or `score` |
| `--duplicate-window-s` | 4.0 | Max separation for a duplicate-phrase disfluency |
| `--max-disfluency-words` | 5 | Cap on tokens any disfluency cut may remove |
| `--min-segment-ms` | 150 | Surviving segments below this are absorbed |
| `--crossfade-ms` | 75 | Per-segment audio fade; 0 disables |
| `--review-margin` | 0.05 | Band above threshold flagged as REVIEW |
| `--split-sentences` | off | Split script on sentences instead of blank lines |
| `--no-cache` | off | Ignore any cached transcript |
| `--report-path` | `<video>.report` | Stem for the `.json` / `.md` report pair |

Every tuning knob in the source spec is exposed. None are hardcoded.

`--dry-run` and `--auto` are mutually exclusive; passing both is an argument error rather than
a silent precedence rule.

## 12. Testing

Fixture-driven pytest, exploiting the seam in section 3.1. A helper constructs `list[Word]`
from a compact string so cases read as the phenomenon under test:

```python
words("the the button", gaps=[0.3, 0.3])
```

Coverage targets, all without video or Whisper:

- silence detection at, above, and below threshold
- alignment: exact, paraphrased, below-threshold miss, REVIEW band
- similar-lines disambiguation via the monotonic cursor
- cursor fallback when a line is missed mid-script
- retake resolution and both tie-break rules
- each disfluency detector, plus negative cases: emphasis repeats, ad-libs sharing vocabulary
- overlap merging with unioned reasons
- padding clamp on spans shorter than 2x padding
- sliver absorption without cascade

Render and transcription are exercised by a smoke test against a short generated clip, kept
separate from the pure-logic suite.

## 13. Environment

- Python 3.14.6; `faster-whisper` wheel availability on 3.14 is verified as the first
  implementation step rather than assumed.
- ffmpeg 8.1.2 and ffprobe installed and verified on PATH.
- `auto-editor` is an existing PyPI package name. Fine as a local directory; this project is
  not to be published under it.

## 14. Out of scope

Phase 3 (`--reorder` script-order reconstruction), phase 4 (Premiere XML export), and phase 5
(batch mode, config file). `--reorder` and `--export-xml` are rejected at argument-parse time
with a clear not-yet-implemented message rather than being silently ignored.
