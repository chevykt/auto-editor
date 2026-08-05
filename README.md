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
```

```bash
winget install Gyan.FFmpeg
```

Requires Python 3.11+ and ffmpeg on PATH. Developed against Python 3.14.6 and
ffmpeg 8.1.2.

## Use

Analyse and produce a review report. This is the default — nothing is rendered:

```bash
python edit.py --script script.txt --video raw_recording.mp4
```

That writes `raw_recording.report.md` (read this) and
`raw_recording.report.json` (edit this). Read the Markdown, check the
judgment-call cuts, then render:

```bash
python edit.py --script script.txt --video raw_recording.mp4 --out final.mp4 --auto
```

If you disagree with a cut, delete it from the JSON report's `segments` list
and render the approved version directly — no re-transcription:

```bash
python edit.py --video raw_recording.mp4 --from-report raw_recording.report.json --out final.mp4
```

No `--script` needed there — the cut list is already decided, so the script
would never be read.

The transcript is cached beside the video as `<name>.transcript.json`, so
re-running with different thresholds costs no Whisper time. `--no-cache` forces
a fresh transcription. Changing `--model` invalidates the cache automatically.

## Reading the report

The Markdown report has three sections:

- **Script lines** — every line, with its status. `ok` is a clean match.
  `**REVIEW**` means it only just cleared the threshold — worth a look.
  `**MISS**` means the line was never found anywhere in the recording. A line
  is never silently dropped without appearing here.
- **Cuts** — silence is summarised as a count and a total, because it is never
  a judgment call. Everything else is listed individually with a reason. This
  list is the safety net: scan it to confirm no cut ate an intentional ad-lib.
- **Kept, but not in the script** — probable ad-libs, left in deliberately.

## Batch mode (`--batch`)

Process a whole folder of recordings in one go:

```bash
python edit.py --batch recordings/ --verbatim --auto --out-dir edited/
```

Each recording is paired with the same-stem `.txt` beside it — `lesson.mp4`
with `lesson.txt`. Recordings with no matching script are **named and skipped**,
not silently ignored; a batch that quietly processes four of five files is worse
than one that tells you which it missed.

Outputs land beside each recording, or under `--out-dir` if given:
`lesson.cut.mp4`, `lesson.report.md`, `lesson.report.json`. A render is named
`.cut.mp4` and excluded from discovery, so re-running a batch never edits its
own output.

One failed recording doesn't abandon the rest. Failures are collected, reported
at the end, and the exit code is non-zero if any job failed — so a scripted run
can tell success from partial success.

## Config file (`--config`)

Put the tuning knobs in TOML instead of retyping flags:

```toml
# auto-editor.toml
verbatim = true
threshold = 0.75
padding_ms = 30
silence_padding_ms = 75
clean_stutters = true
model = "medium"
```

```bash
python edit.py --batch recordings/ --config auto-editor.toml --auto
```

Precedence is **defaults < config file < command line**, so a flag you type
always wins over the file.

Only *knobs* live in the file — every field of `EditorConfig` is settable.
Actions like `--auto`, `--batch`, `--out` and `--export-xml` stay on the command
line: a run's settings are worth saving and reusing, but what a particular
invocation *does* is not.

Unknown keys are an error listing the valid ones, rather than being ignored. A
typo like `treshold` should tell you, not silently leave the threshold at its
default.

## Premiere timeline export (`--export-xml`)

Instead of trusting the auto-cut, hand the cuts to an editor and nudge them:

```bash
python edit.py --script script.txt --video raw.mp4 --export-xml
```

Writes `raw.report.xml` alongside the reports — an FCP7 `xmeml` sequence that
Premiere Pro imports directly (File → Import). You get one video track and one
audio track, each cut into the surviving segments, all referencing your
original file. Nothing is re-encoded and nothing is destroyed, so trimming a
cut by a few frames is just dragging an edge.

Works with `--auto` too, if you want the rendered MP4 *and* the timeline.
Works in `--reorder` mode as well, where the clips are laid out in script
order. Combine with `--from-report` to export a timeline from a cut list you
already hand-edited.

Frame rate, dimensions and duration are read from the source with `ffprobe`, so
the sequence matches your footage. NTSC rates are handled properly — 29.97 is
written as timebase 30 with the NTSC flag set, which is what the format
requires; writing `29.97` directly produces a sequence Premiere silently
misreads.

## Reorder mode (`--reorder`)

The opposite approach, and off by default. Instead of removing what it's
confident about, it keeps **only** what matched a script line and concatenates
those windows in **script order** rather than the order you spoke them.

```bash
python edit.py --script script.txt --video raw.mp4 --out final.mp4 --reorder --auto
```

Use it only when lines were deliberately recorded out of sequence and need
reassembling. The trade-off is real and unavoidable:

| | Default (subtractive) | `--reorder` |
|---|---|---|
| Order | as spoken | as scripted |
| Ad-libs, asides, extra explanation | **kept** | **discarded** |
| Natural flow between beats | preserved | lost |
| Unmatched content | reported, kept | reported, removed |

Measured on a real 80-second recording: the default produced 34.7s and kept a
13-second ad-lib; `--reorder` produced 21.8s and threw it away. Both are
correct — they answer different questions.

Disfluencies are still removed from inside the windows it keeps, so a stutter
in a matched line doesn't survive just because the line did. Missed lines
contribute nothing and are listed under "Holes in the reassembly", so a gap in
the script is visible rather than silently yielding a shorter video.

The report says which mode produced it, and in reorder mode the unmatched
section is headed **"Discarded"** rather than "Kept" — the same content, the
opposite fate.

## Tuning

| Flag | Default | What it does |
|---|---|---|
| `--threshold` | 0.75 | Fuzzy score a window must hit to count as a line match |
| `--review-margin` | 0.05 | Band above the threshold flagged as REVIEW rather than OK |
| `--silence-ms` | 600 | Shortest gap treated as cuttable silence |
| `--padding-ms` | 30 | Pull-in at disfluency and retake boundaries |
| `--silence-padding-ms` | 75 | Pull-in at silence boundaries |
| `--min-segment-ms` | 150 | Surviving slivers shorter than this are absorbed |
| `--crossfade-ms` | 75 | Per-segment audio fade at cut points; 0 disables |
| `--duplicate-window-s` | 4.0 | Max separation for a repeat to count as a stutter rather than deliberate |
| `--max-disfluency-words` | 5 | Hard cap on how much any one disfluency cut may remove |
| `--tiebreak` | latest | Retake tie-break: `latest` or `score` |
| `--lookahead-s` | 30.0 | Floor on how far ahead each line is searched |
| `--model` | medium | Whisper model size |
| `--language` | auto | Force a language instead of auto-detecting |
| `--no-clean-stutters` | off | Turn disfluency detection off entirely |
| `--export-xml` | off | Also write a Premiere-compatible XML timeline |
| `--reorder` | off | Rebuild in script order, discarding unmatched content |
| `--reorder-pad-ms` | 60 | Outward breathing room on each reordered window |
| `--split-sentences` | off | Split the script per sentence instead of per blank line |
| `--no-cache` | off | Ignore any cached transcript |
| `--report-path` | `<video>.report` | Stem for the `.json` / `.md` report pair |

`--dry-run` (the default) and `--auto` are mutually exclusive; passing both is
an error rather than a silent precedence rule.

`--video` is required unless using `--batch`. `--script` is required except with
`--from-report`. `--out` is required whenever something is actually rendered,
and `--batch` uses `--out-dir` instead since it writes one output per recording.
All of these, plus missing input files, are checked before transcription
starts — a typo shouldn't cost you a full Whisper run to discover.

## Design notes

Two decisions that are easy to get wrong, and why they went the way they did:

**Audio uses per-segment fades, not crossfades.** A true `acrossfade` overlaps
adjacent segments, shortening total audio by the fade length at *every* cut.
Across a few hundred cuts that accumulates into audible audio/video desync.
Per-segment `afade` suppresses clicks identically while preserving exact
durations. A smoke test renders 50 cuts with fades on and asserts the audio and
video stream durations still agree.

**Cut boundaries are padded per reason, not uniformly.** A silence cut is
bounded by speech on both sides, so pulling in generously protects a breath or
a plosive. A disfluency cut is a short span where that same pull-in would eat
most of what you're removing — a duplicated "the" measured 220ms on a real
recording, and a 75ms pull-in from each side left only 70ms actually removed.
Each boundary is padded by the reason that produced it, so a silence gap that
merged with an adjacent stutter gets the generous value at one end and the
tight value at the other.

**The ffmpeg filter graph goes to a file.** A few hundred cuts produces a
`filter_complex` string far past the Windows 8191-character command-line limit,
so it is passed via `-filter_complex_script`. Inlining it would fail only on
long recordings — exactly the ones this tool exists for.

## Known limits

- Whisper sometimes smooths over very fast stutters and won't transcribe every
  repeated fragment, so disfluency detection is best-effort. The report shows
  what it did catch.
- A deliberate emphasis repeat ("no, no, don't click that") is not reliably
  distinguishable from an accidental stutter. The narrow window and the
  explicit disfluency-cut list in the report are the safety net. If one gets
  cut wrongly, delete it from the JSON and use `--from-report`.
- Two script lines with near-identical wording are disambiguated by matching in
  script order against a forward-moving cursor. If a line is genuinely spoken
  out of order, it is found by a full-recording fallback and flagged
  `(out of order)` in the report.
- Batch mode processes recordings sequentially, not in parallel. Transcription
  dominates the runtime, and Whisper already uses the available cores.
- The XML export is verified against the format's structural rules and against
  real footage metadata, but it has not been round-tripped through Premiere
  itself. If a sequence imports oddly, the report's JSON remains the source of
  truth for what should have been cut.

## Tests

```bash
python -m pytest tests/ -v
```

138 tests, no video, no Whisper, no ffmpeg. The whole analysis core is pure
functions over word lists, so it tests with hand-written fixtures.

The 9 end-to-end tests generate their own media and are excluded by default:

```bash
python -m pytest tests/ -v -m smoke
```

Those run the full CLI against a real video with the transcript cache
pre-seeded, which exercises parsing, alignment, retakes, disfluencies, the cut
list, both reports, and the ffmpeg render — without downloading a model.
