# Local Web UI — Design

Date: 2026-08-05
Status: Approved
Builds on: `2026-07-31-auto-editor-design.md` (the CLI engine, phases 1-5, complete)

## 1. Problem

The engine works but only through a CLI. Editing a recording means typing a
command, reading a Markdown report, hand-editing JSON to disagree with a cut,
and running a second command to render. The review gate — the design's central
safety mechanism — is the most awkward part of the whole workflow.

A local web UI turns that around. Drop in a video, paste the script, watch the
analysis run, then approve or reject each cut *while listening to it*, and
render. Reviewing becomes the easy path rather than the tedious one.

## 2. Scope

**In:** upload, script entry, progress reporting, interactive review, render,
download, XML export.

**Out:** `--reorder` mode, batch processing, waveform rendering,
authentication, multi-user, remote deployment. All remain CLI concerns or
deferred.

The UI covers the common path. The CLI keeps every capability.

## 3. Constraints from real use

- Recordings are **100-200 MB**. Browser upload is fine at this size; nothing
  needs designing around multi-gigabyte files.
- Transcription is slow — roughly 2 minutes for 80 seconds of audio on CPU.
  Progress reporting is a requirement, not a nicety.
- Single user, single machine, one job at a time.

## 4. Architecture

```
web/
  __main__.py    python -m web  -> start server, open browser
  server.py      FastAPI routes, static serving, range-aware video streaming
  jobs.py        job state machine and queue
  static/
    index.html
    app.js
    style.css
```

The engine is **not modified**, with one exception (section 7). `server.py`
imports `transcribe()`, `analyse()` and `render_video()` and calls them
directly. No subprocess, no duplicated pipeline. If the CLI and the UI ever
disagree about what should be cut, that is a bug in one shared code path rather
than a divergence between two.

Frontend is vanilla HTML/CSS/JS. No npm, no bundler, no build step: the point
is that `python -m web` is the entire install story.

## 5. Job lifecycle

A job moves through fixed states, and the UI polls `GET /api/jobs/{id}`:

```
queued -> probing -> transcribing -> analysing -> ready
                                                    |
                                          (user approves cuts)
                                                    v
                                            rendering -> done
```

`failed` is reachable from any state and carries a message.

`ready` is the review gate. A job never renders on its own; it waits for an
explicit approval call. This is the whole design principle, enforced by the
state machine rather than by convention.

### State

Jobs live in memory, keyed by id, with their files under
`web/work/<job-id>/` — gitignored, since it holds uploaded video. A server
restart loses in-flight jobs, which is acceptable for a single-user local tool.
The report JSON is written to disk, so a finished job's results survive even
though its in-memory record does not.

Disk cost per job is roughly three times the source: the upload, the transcript
cache, and the render. `DELETE /api/jobs/{id}` removes the folder.

One job runs at a time. Transcription saturates the CPU, so concurrency would
make every job slower rather than the set faster. Additional submissions queue.

## 6. HTTP API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/jobs` | multipart: video file + script text. Returns job id |
| GET | `/api/jobs/{id}` | state, progress 0-1, and the report once ready |
| GET | `/api/jobs/{id}/video` | the source, **with HTTP range support** |
| POST | `/api/jobs/{id}/render` | body: indices of cuts to **keep applied** (i.e. spans still to be removed from the video). Starts rendering |
| GET | `/api/jobs/{id}/download` | the rendered file |
| GET | `/api/jobs/{id}/xml` | Premiere timeline for the approved cut list |
| DELETE | `/api/jobs/{id}` | remove the job and its working files |

### Upload streams to disk

FastAPI will buffer an entire upload in memory if allowed to. At 200 MB per
job that is avoidable waste, so the request body is written to the work folder
in chunks.

### Range requests are load-bearing

Clicking a cut seeks the `<video>` element to that timestamp. Without HTTP
range support the browser refetches the whole 200 MB file on every seek, and
the central feature of the review screen feels broken. Starlette's
`FileResponse` implements ranges; the test suite asserts a `206 Partial
Content` with a correct `Content-Range` rather than trusting it.

## 7. The one engine change

`transcribe.py` gains an optional progress callback:

```python
ProgressFn = Callable[[float], None]      # receives a fraction, 0.0 to 1.0

def transcribe(video, cfg, on_progress: ProgressFn | None = None
               ) -> tuple[list[Word], float]
```

`_decode` already iterates a generator of segments, each carrying an `end`
timestamp. Dividing that by the media duration gives a true completion
fraction, reported as it goes. The fraction is clamped to 1.0, since Whisper's
final segment can overshoot the probed duration slightly.

Default `None` preserves current behaviour exactly, so the CLI is unaffected.
On a cache hit the callback is invoked once with `1.0` and no model is loaded.

A spinner would have been free, but a two-minute wait with no sense of
progress reads as a hang.

## 8. Review screen

The reason the UI exists.

**Player** — the source video, served with range support so seeking is instant.

**Timeline** — a proportional bar of the full duration, segments coloured by
what happens to them: kept, silence, disfluency, retake. The shape of the edit
at a glance.

**Judgment calls** — every non-silence cut as a row: timestamp, reason badge,
detail, a play button that seeks the player there, and a checkbox. Unchecking
keeps that span in the final video. Silence is summarised as a count and total,
matching the Markdown report's reasoning: silence is never a judgment call, and
listing hundreds of gaps would bury the cuts that are.

**Kept, not in your script** — probable ad-libs, read-only. Confirms the tool
preserved what you improvised.

**Output duration** updates live as cuts are toggled, so the cost of keeping
something is visible before committing.

### Applying the toggles

Unchecking a cut drops it from the list; the surviving cuts go through the
existing `keep_segments()` to produce new segments. The same function the CLI
uses, so approval cannot drift from the engine's own arithmetic.

Cuts are identified by index into `report["cuts"]`, which is stable for a given
report.

## 9. Errors

Every failure surfaces in the UI with a usable message, never a traceback:

- ffmpeg or ffprobe missing — install instructions
- unreadable or non-media upload — reject at probe, before transcription
- no script text — reject before the job is created
- Whisper failure — the message, with the CUDA-to-CPU fallback already handled
  in the engine
- render failure — ffmpeg's stderr tail

A failed job keeps its files so the problem can be inspected.

## 10. Testing

The engine's 253 unit and 17 smoke tests keep passing unchanged; the one
signature change is backward compatible.

New tests:

- **Job state machine** — pure transitions, no HTTP, no media. Every state
  reachable, `ready` never advancing to `rendering` without approval, failure
  reachable from anywhere.
- **API** — FastAPI's `TestClient` against a tiny generated clip: upload,
  poll, approve, render, download. Transcription stubbed, since Whisper's
  behaviour is already covered by the engine's tests.
- **Range requests** — a `Range` header returns `206` with the right
  `Content-Range` and body length. Explicitly tested because seeking silently
  degrades rather than failing loudly.
- **Toggle arithmetic** — unchecking a cut produces exactly the segments
  `keep_segments()` produces for the remaining cuts.
- **Upload is streamed** — a large upload does not load fully into memory.

Frontend logic is deliberately thin enough not to need its own test runner. The
one piece worth isolating — recomputing output duration from toggled cuts — is
a pure function tested through the API's approval endpoint.

## 11. Dependencies

Added: `fastapi`, `uvicorn`, `python-multipart` (upload parsing),
`httpx` (test client).

No frontend dependencies.

## 12. Deliberate omissions

- **Waveform display.** Substantial work; playing the cut answers the same
  question better.
- **Persistent job history.** Reports persist; the job list does not.
- **`--reorder` in the UI.** It discards ad-libs. Behind a toggle in a review
  screen it invites an expensive misclick. It stays a CLI flag.
- **Editing cut boundaries.** Toggling a cut on or off covers the real need.
  Frame-level adjustment is what the XML export is for.
