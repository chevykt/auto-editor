# Script-Matched Tutorial Video Auto-Editor

A local web app and CLI that automatically removes dead air, false starts and
stutters from screen-recorded tutorial videos, while preserving improvised
content the speaker meant to keep. Cuts can be rendered directly, or exported
as an editable timeline to DaVinci Resolve or Premiere Pro.

Python 3.14 · faster-whisper · rapidfuzz · ffmpeg · FastAPI · ~3,400 lines ·
387 tests · 60 commits

---

## The problem

Recording a tutorial voiceover produces a take full of dead air, stumbles,
half-finished words, and lines delivered two or three times before one lands.
Cleaning that up by hand is slow and mechanical.

The obvious automation — "keep what matches my script, cut everything else" —
is wrong. It deletes every ad-lib, aside and piece of improvised explanation,
which in a tutorial is often the most valuable material.

## Core design decision

The tool is **subtractive, not reconstructive**. It starts from the full
recording and removes only spans it is confident about:

| Category | Action |
|---|---|
| Clean delivery of a script line, or the winning take of a retaken one | keep |
| Silence above a threshold | cut |
| Losing attempt at a line that was retaken | cut |
| Stutter, immediate repeat, truncated-word restart | cut |
| Anything unmatched — ad-libs, asides, extra explanation | **keep**, and report |

Everything stays in its original position. Nothing is reordered by default.

The governing bias: **a missed stutter is a small cost; a deleted ad-lib is a
large one.** Every judgment-call cut is itemised in a review report so it can
be checked before rendering.

## Pipeline

1. **Transcribe** — faster-whisper, word-level timestamps, cached
2. **Parse script** — into beats
3. **Detect silence** — from gaps between word timestamps
4. **Align** — fuzzy sliding-window matching of each script line against the
   transcript, using a forward-moving cursor so similarly-worded lines cannot
   steal each other's occurrence
5. **Resolve retakes** — pick the winning attempt, mark the losers
6. **Detect disfluencies** — immediate repeats, truncated-word restarts,
   duplicated phrases, mid-line restarts
7. **Build cut list** — merge overlaps, pad boundaries, absorb slivers
8. **Report** — JSON (machine-editable) and Markdown (human-readable)
9. **Finish** — render with ffmpeg trim/concat, or export a timeline for manual
   editing

## Interfaces

**Web UI** (`python -m web`) — drag in a recording, paste the script, watch
real transcription progress, then review every proposed cut on a timeline:
play it, uncheck it, watch the output duration update, then render or export
to DaVinci Resolve. Runs entirely on localhost; nothing leaves the machine.

**CLI** — the full feature set, including batch processing over a folder,
TOML config files, script-order reconstruction mode, and DaVinci Resolve
(FCPXML) and Premiere Pro (XML) timeline export. Exports reference the whole
source recording, so any cut can be undone by dragging a clip's edge.

## Architecture

`transcribe.py` is a thin adapter whose only contract with the rest of the
system is `list[Word]` plus a duration. Every stage downstream of it —
alignment, retake resolution, disfluency detection, cut-list construction — is
a **pure function over word lists**.

That single boundary is why 365 of the 387 tests run with no video file, no
speech model and no ffmpeg, in about four seconds. The remaining 22 are
end-to-end tests that generate their own media.

The web layer imports the engine directly rather than shelling out to the CLI,
so the two cannot drift about what should be cut. A job state machine enforces
the review gate structurally: there is no transition from `ready` to `done`,
so a job physically cannot render without an explicit approval call.

## Engineering decisions worth noting

**Padding direction depends on what a span contains.** A silence cut is
bounded by speech, so its boundaries shrink inward to avoid clipping a breath.
A disfluency cut *is* the speech being removed, so its boundaries grow outward
— shrinking them leaves the edges of the fumble audible. Each boundary of a
merged span follows whichever cut produced it.

**A feature was built, measured, and deleted.** An audio-level pass to recover
repeats that the speech model had collapsed into a single token flagged six
cuts on real footage. Checked by ear, five were wrong, and the sixth removed
its target along with a neighbouring word. It was removed rather than shipped
disabled, with the finding and the reasoning recorded so it would not be
retried on intuition.

**Uncertain detections are flagged, not applied.** Where a signal is
suggestive but not conclusive, the tool reports it as an unchecked row in the
review screen rather than acting on it. On a 274-word recording this produced
one flag, positioned at a known false start, left for the user to confirm by
ear.

**The script decides what counts as a stumble.** "The theory" and "and Android"
have the same shape as a cut-off word followed by the full word. A code review
caught the detector removing such pairs, so any repetition the script itself
contains is now treated as intended speech.

**Retakes are bounded in time.** On one real recording, two deliveries of a
line about a minute apart were treated as a retake, deleting 29 seconds of
demonstration. Matches are now only retakes when seconds apart, and output on
that recording went from 96 to 107 seconds at report level.

**Failure modes are made loud.** Network errors, missing dependencies,
unreadable inputs and queue contention all surface as specific messages rather
than a page that appears to do nothing. Several of these were found by using
the tool on real footage rather than by testing.

**End-to-end tests earn their keep.** A review run showed every render failing
after a machine upgrade to ffmpeg 9, which had removed a flag the tool relied
on. All unit tests still passed; only the 22 tests using real ffmpeg caught it.

## Testing

- 365 unit tests — pure logic, hand-built fixtures, no external dependencies
- 22 end-to-end tests — real ffmpeg, generated media, full HTTP round trips
- Tests encode *reasons*, not just behaviour: several assert safety properties
  (an emphasis repeat must survive, an ad-lib sharing vocabulary must not be
  cut) and carry the measurement that justified a threshold, so a future
  change fails with the case to re-measure rather than a bare mismatch.

Validated against real recordings throughout. Roughly a dozen defects were
found and fixed during development, several of which only surfaced under real
use — a large green test suite coexisting with a tool that was, in one case,
writing its output where the user would never find it.
