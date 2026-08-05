# Local Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app where you drop in a video, paste the script, watch analysis progress, approve or reject each cut while listening to it, and render.

**Architecture:** FastAPI server importing the existing engine directly — no subprocess, no duplicated pipeline. A job state machine enforces the review gate: `ready` cannot reach `rendering` without an explicit approval call. Frontend is vanilla HTML/CSS/JS with no build step.

**Tech Stack:** Python 3.14.6, FastAPI 0.141, uvicorn 0.52, python-multipart, httpx (tests). No frontend dependencies.

**Approved spec:** `docs/superpowers/specs/2026-08-05-local-web-ui-design.md`

**Verified before planning:** all four packages are pure-Python wheels (no cp314 build risk), and Starlette's `FileResponse` was confirmed to serve `206 Partial Content` with a correct `Content-Range` and exact byte slice — so range support needs a regression test, not an implementation.

---

## File Structure

| File | Responsibility |
|---|---|
| `web/__init__.py` | package marker |
| `web/jobs.py` | `Job`, `JobState`, transition rules, `JobStore`, `Worker`, the analysis pipeline, and toggle arithmetic. No HTTP. |
| `web/server.py` | FastAPI routes and static serving. No business logic. |
| `web/__main__.py` | `python -m web` — start uvicorn, open a browser |
| `web/static/index.html` | both screens, hidden/shown by JS |
| `web/static/style.css` | styling |
| `web/static/app.js` | upload, polling, review rendering, approval |
| `transcribe.py` | **modified**: optional progress callback |

`jobs.py` holds everything testable without a server; `server.py` is a thin
translation layer. That split is what lets the state machine and the toggle
arithmetic be tested as pure functions.

---

## Task 1: Progress callback in the engine

**Files:**
- Modify: `transcribe.py`
- Modify: `requirements.txt`
- Test: `tests/test_transcribe.py` (append)

- [ ] **Step 1: Add web dependencies to `requirements.txt`**

```
faster-whisper==1.2.1
rapidfuzz==3.14.5
pytest==8.3.4
fastapi==0.141.1
uvicorn==0.52.1
python-multipart==0.0.32
httpx==0.28.1
```

- [ ] **Step 2: Install**

Run: `python -m pip install -r requirements.txt`
Expected: all wheels are `py3-none-any`, nothing builds from source.

- [ ] **Step 3: Write the failing test**

Append to `tests/test_transcribe.py`:

```python
def test_progress_is_reported_as_a_fraction(monkeypatch, tmp_path):
    """A two-minute wait with no feedback reads as a hang."""
    import faster_whisper

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSegment:
        def __init__(self, end, words):
            self.end, self.words = end, words

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return [FakeSegment(5.0, [FakeWord(" one", 0.0, 1.0)]),
                    FakeSegment(10.0, [FakeWord(" two", 5.0, 6.0)])], None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr("transcribe.probe_duration", lambda v: 10.0)

    seen = []
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    transcribe(video, EditorConfig(use_cache=False), on_progress=seen.append)

    assert seen == [0.5, 1.0]


def test_progress_never_exceeds_one(monkeypatch, tmp_path):
    """Whisper's final segment can overshoot the probed duration."""
    import faster_whisper

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSegment:
        end = 12.0
        words = [FakeWord(" over", 0.0, 1.0)]

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return [FakeSegment()], None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr("transcribe.probe_duration", lambda v: 10.0)

    seen = []
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    transcribe(video, EditorConfig(use_cache=False), on_progress=seen.append)

    assert seen == [1.0]


def test_a_cache_hit_reports_complete_without_loading_a_model(tmp_path,
                                                              monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    cfg = EditorConfig()
    save_cache(cache_path_for(video), [Word("hi", 0.0, 0.2)], 3.0,
               transcription_key(cfg))

    def explode(*a, **k):
        raise AssertionError("must not load a model on a cache hit")

    monkeypatch.setattr("transcribe._run_whisper", explode)

    seen = []
    transcribe(video, cfg, on_progress=seen.append)
    assert seen == [1.0]


def test_omitting_the_callback_keeps_current_behaviour(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    cfg = EditorConfig()
    save_cache(cache_path_for(video), [Word("hi", 0.0, 0.2)], 3.0,
               transcription_key(cfg))
    ws, duration = transcribe(video, cfg)
    assert [w.text for w in ws] == ["hi"]
    assert duration == 3.0
```

- [ ] **Step 4: Run to verify it fails**

Run: `python -m pytest tests/test_transcribe.py -k progress -v`
Expected: FAIL with `TypeError: transcribe() got an unexpected keyword argument 'on_progress'`

- [ ] **Step 5: Thread the callback through `transcribe.py`**

Replace `_decode`, `_run_whisper` and `transcribe` with:

```python
def _decode(model, video: Path, cfg: EditorConfig,
            on_progress: ProgressFn | None = None,
            duration: float | None = None) -> list[Word]:
    """Drain the segment generator into a flat word list.

    faster-whisper returns a lazy generator, so any GPU library failure
    surfaces here during iteration rather than at model construction. That is
    why the fallback in _run_whisper wraps this call, not just the constructor.
    """
    segments, _ = model.transcribe(str(video), word_timestamps=True,
                                   language=cfg.language,
                                   initial_prompt=effective_prompt(cfg))
    words: list[Word] = []
    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()
            # Whisper emits standalone punctuation as its own token - a mid-word
            # restart can arrive as 'config', '-', 'configuration', the hyphen
            # carrying no audio at all. Dropping it here keeps every downstream
            # stage from having to know that speech can be interrupted by a
            # token that is not speech.
            if text and normalize(text):
                words.append(Word(text, float(w.start), float(w.end)))

        if on_progress and duration:
            # the final segment can run past the probed duration
            on_progress(min(seg.end / duration, 1.0))

    return words


def _run_whisper(video: Path, cfg: EditorConfig,
                 on_progress: ProgressFn | None = None,
                 duration: float | None = None) -> list[Word]:
    """Load the model and transcribe. Isolated so tests can stub it out."""
    from faster_whisper import WhisperModel

    attempts = _device_attempts(cfg)
    for i, (device, compute_type) in enumerate(attempts):
        last = i == len(attempts) - 1
        try:
            model = WhisperModel(cfg.model, device=device,
                                 compute_type=compute_type)
            return _decode(model, video, cfg, on_progress, duration)
        except (RuntimeError, ValueError) as exc:
            if last:
                raise
            print(f"  {device} unavailable ({exc}); falling back to "
                  f"{attempts[i + 1][0]}", file=sys.stderr)

    raise RuntimeError("no transcription device available")


def transcribe(video: str | Path, cfg: EditorConfig,
               on_progress: ProgressFn | None = None
               ) -> tuple[list[Word], float]:
    """Word-level transcript plus media duration, cached beside the video.

    on_progress receives a fraction from 0.0 to 1.0 as transcription advances.
    A cache hit reports 1.0 once and loads no model.
    """
    video = Path(video)
    cache = cache_path_for(video)
    key = transcription_key(cfg)

    if cfg.use_cache:
        words, duration = load_cached(cache, key)
        if words is not None and duration is not None:
            if on_progress:
                on_progress(1.0)
            return words, duration

    duration = probe_duration(video)
    words = _run_whisper(video, cfg, on_progress, duration)
    save_cache(cache, words, duration, key)
    return words, duration
```

Add the type alias near the top of `transcribe.py`, after the imports:

```python
ProgressFn = Callable[[float], None]
```

and add `Callable` to the imports:

```python
from collections.abc import Callable
```

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/ -q`
Expected: 257 passed (253 existing + 4 new)

- [ ] **Step 7: Commit**

```bash
git add transcribe.py requirements.txt tests/test_transcribe.py
git commit -m "feat: report transcription progress via an optional callback"
```

---

## Task 2: Job state machine

**Files:**
- Create: `web/__init__.py`
- Create: `web/jobs.py`
- Test: `tests/test_web_jobs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_jobs.py`:

```python
import pytest

from web.jobs import InvalidTransition, Job, JobState


def make(tmp_path) -> Job:
    return Job(id="abc123", video=tmp_path / "raw.mp4", script="hello")


def test_a_new_job_is_queued(tmp_path):
    assert make(tmp_path).state is JobState.QUEUED


def test_the_happy_path_runs_in_order(tmp_path):
    job = make(tmp_path)
    for state in (JobState.PROBING, JobState.TRANSCRIBING, JobState.ANALYSING,
                  JobState.READY, JobState.RENDERING, JobState.DONE):
        job.advance(state)
    assert job.state is JobState.DONE


def test_skipping_a_state_is_rejected(tmp_path):
    job = make(tmp_path)
    with pytest.raises(InvalidTransition):
        job.advance(JobState.READY)


def test_ready_cannot_jump_straight_to_done(tmp_path):
    """The review gate. Rendering must be asked for, never assumed."""
    job = make(tmp_path)
    for state in (JobState.PROBING, JobState.TRANSCRIBING, JobState.ANALYSING,
                  JobState.READY):
        job.advance(state)
    with pytest.raises(InvalidTransition):
        job.advance(JobState.DONE)


def test_failure_is_reachable_from_any_state(tmp_path):
    for reached in (JobState.QUEUED, JobState.TRANSCRIBING, JobState.READY):
        job = make(tmp_path)
        while job.state is not reached:
            job.advance(next(iter(_forward_from(job.state))))
        job.fail("boom")
        assert job.state is JobState.FAILED
        assert job.error == "boom"


def _forward_from(state):
    from web.jobs import ALLOWED_TRANSITIONS
    return ALLOWED_TRANSITIONS[state]


def test_terminal_states_go_nowhere(tmp_path):
    job = make(tmp_path)
    job.fail("boom")
    with pytest.raises(InvalidTransition):
        job.advance(JobState.PROBING)


def test_as_dict_carries_what_the_ui_polls_for(tmp_path):
    job = make(tmp_path)
    job.progress = 0.42
    payload = job.as_dict()
    assert payload["id"] == "abc123"
    assert payload["state"] == "queued"
    assert payload["progress"] == 0.42
    assert payload["report"] is None
    assert payload["error"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web'`

- [ ] **Step 3: Create `web/__init__.py`**

Create it as an empty file.

- [ ] **Step 4: Create `web/jobs.py`**

```python
"""Job lifecycle for the web UI. Everything here is testable without a server.

The state machine exists to enforce the review gate structurally rather than by
convention: there is no transition from ready to done, so a job cannot render
without an explicit approval call.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class JobState(str, Enum):
    QUEUED = "queued"
    PROBING = "probing"
    TRANSCRIBING = "transcribing"
    ANALYSING = "analysing"
    READY = "ready"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"


# Forward transitions only. Failure is handled separately by fail(), because it
# is reachable from everywhere and listing it in every row would obscure the
# shape of the happy path.
ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.QUEUED: {JobState.PROBING},
    JobState.PROBING: {JobState.TRANSCRIBING},
    JobState.TRANSCRIBING: {JobState.ANALYSING},
    JobState.ANALYSING: {JobState.READY},
    JobState.READY: {JobState.RENDERING},
    JobState.RENDERING: {JobState.DONE},
    JobState.DONE: set(),
    JobState.FAILED: set(),
}


class InvalidTransition(Exception):
    """Raised when code tries to move a job somewhere it cannot go."""


@dataclass
class Job:
    id: str
    video: Path
    script: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    report: dict | None = None
    output: Path | None = None
    error: str | None = None

    @property
    def directory(self) -> Path:
        return self.video.parent

    def advance(self, to: JobState) -> None:
        if to not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition(
                f"cannot go from {self.state.value} to {to.value}")
        self.state = to

    def fail(self, message: str) -> None:
        """Reachable from any state, including the terminal ones."""
        self.state = JobState.FAILED
        self.error = message

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "report": self.report,
            "error": self.error,
            "has_output": self.output is not None and self.output.exists(),
        }
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_web_jobs.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add web/__init__.py web/jobs.py tests/test_web_jobs.py
git commit -m "feat: add job state machine enforcing the review gate"
```

---

## Task 3: Job store, worker, pipeline and toggle arithmetic

**Files:**
- Modify: `web/jobs.py` (append)
- Test: `tests/test_web_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_pipeline.py`:

```python
import pytest

from config import EditorConfig
from models import Word
from web.jobs import (JobState, JobStore, Worker, approved_segments,
                      run_analysis)


def fake_transcribe(words, duration):
    def _fn(video, cfg, on_progress=None):
        if on_progress:
            on_progress(1.0)
        return words, duration
    return _fn


def fake_analyse(report):
    def _fn(words, script, duration, cfg, video, split_sentences=False):
        return report
    return _fn


REPORT = {
    "duration": 10.0,
    "cuts": [
        {"start": 1.0, "end": 2.0, "reasons": ["silence"], "detail": "gap"},
        {"start": 4.0, "end": 5.0, "reasons": ["disfluency"], "detail": "the"},
    ],
    "segments": [],
}


# --- the pipeline --------------------------------------------------------

def test_the_pipeline_stops_at_ready(tmp_path):
    """The review gate. Analysis must never render on its own."""
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")

    run_analysis(job, EditorConfig(),
                 transcribe_fn=fake_transcribe([Word("hi", 0.0, 0.2)], 10.0),
                 analyse_fn=fake_analyse(REPORT))

    assert job.state is JobState.READY
    assert job.report == REPORT


def test_progress_reaches_the_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")

    def transcribing(video, cfg, on_progress=None):
        on_progress(0.5)
        return [Word("hi", 0.0, 0.2)], 10.0

    run_analysis(job, EditorConfig(), transcribe_fn=transcribing,
                 analyse_fn=fake_analyse(REPORT))
    assert job.progress == 1.0, "completes at 1.0 once analysed"


def test_a_transcription_failure_fails_the_job_with_a_message(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")

    def broken(video, cfg, on_progress=None):
        raise RuntimeError("ffprobe not found on PATH")

    run_analysis(job, EditorConfig(), transcribe_fn=broken,
                 analyse_fn=fake_analyse(REPORT))

    assert job.state is JobState.FAILED
    assert "ffprobe" in job.error


# --- toggle arithmetic ---------------------------------------------------

def test_keeping_every_cut_matches_the_engines_own_segments():
    segments = approved_segments(REPORT, {0, 1})
    assert [(s.start, s.end) for s in segments] == [
        (0.0, 1.0), (2.0, 4.0), (5.0, 10.0)]


def test_unchecking_a_cut_puts_that_span_back():
    segments = approved_segments(REPORT, {0})
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.0), (2.0, 10.0)]


def test_unchecking_everything_keeps_the_whole_recording():
    segments = approved_segments(REPORT, set())
    assert [(s.start, s.end) for s in segments] == [(0.0, 10.0)]


def test_an_out_of_range_index_is_ignored():
    segments = approved_segments(REPORT, {0, 99})
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.0), (2.0, 10.0)]


# --- store ---------------------------------------------------------------

def test_create_makes_a_directory_per_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")
    assert job.directory.is_dir()
    assert job.video.name == "raw.mp4"
    assert store.get(job.id) is job


def test_a_path_in_the_filename_cannot_escape_the_job_directory(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("../../evil.mp4", "hello")
    assert job.video.name == "evil.mp4"
    assert job.video.parent.parent == tmp_path


def test_delete_removes_the_job_and_its_files(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")
    job.video.write_bytes(b"x")

    assert store.delete(job.id) is True
    assert store.get(job.id) is None
    assert not job.directory.exists()


def test_deleting_an_unknown_job_is_false(tmp_path):
    assert JobStore(tmp_path).delete("nope") is False


def test_the_worker_runs_submitted_work(tmp_path):
    done = []
    worker = Worker()
    worker.submit(lambda: done.append(1))
    worker.join()
    assert done == [1]


def test_the_worker_survives_a_failing_job(tmp_path):
    done = []
    worker = Worker()
    worker.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    worker.submit(lambda: done.append(2))
    worker.join()
    assert done == [2], "a failed job must not stop the queue"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'JobStore' from 'web.jobs'`

- [ ] **Step 3: Append to `web/jobs.py`**

```python
def approved_segments(report: dict, keep_applied: set[int]) -> list[Span]:
    """Surviving segments once the user has toggled cuts off.

    keep_applied holds indices into report["cuts"] that should STILL be
    removed from the video. Anything not listed is put back.

    Runs the survivors through the engine's own keep_segments, so an approved
    cut list cannot drift from what the CLI would have produced.
    """
    cuts = [CutSpan(cut["start"], cut["end"])
            for i, cut in enumerate(report["cuts"]) if i in keep_applied]
    return keep_segments(cuts, report["duration"])


def run_analysis(job: Job, cfg: EditorConfig,
                 transcribe_fn=transcribe, analyse_fn=analyse) -> None:
    """Take a job from queued to ready. Never further.

    Stopping at ready is the review gate: rendering is a separate, explicit
    call. The two function arguments exist so tests can drive the pipeline
    without a speech model.
    """
    try:
        job.advance(JobState.PROBING)
        job.advance(JobState.TRANSCRIBING)

        words, duration = transcribe_fn(
            job.video, cfg,
            on_progress=lambda fraction: setattr(job, "progress", fraction))

        job.advance(JobState.ANALYSING)
        job.report = analyse_fn(words, job.script, duration, cfg,
                                str(job.video))
        job.progress = 1.0
        job.advance(JobState.READY)
    except Exception as exc:                          # noqa: BLE001
        job.fail(str(exc))


def run_render(job: Job, cfg: EditorConfig, keep_applied: set[int],
               render_fn=render_video) -> None:
    """Render the approved cut list. Only callable from ready."""
    try:
        job.advance(JobState.RENDERING)
        report = dict(job.report)
        report["segments"] = [{"start": s.start, "end": s.end}
                              for s in approved_segments(report, keep_applied)]

        job.output = job.directory / "edited.mp4"
        render_fn(job.video, job.output, report, cfg)
        job.advance(JobState.DONE)
    except Exception as exc:                          # noqa: BLE001
        job.fail(str(exc))


class JobStore:
    """Jobs in memory, their files on disk under one directory each."""

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, script: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        directory = self.work_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)

        # Path(...).name strips any directory component, so an uploaded
        # filename cannot write outside its own job folder.
        safe = Path(filename).name or "upload.mp4"

        job = Job(id=job_id, video=directory / safe, script=script)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.directory, ignore_errors=True)
        return True


class Worker:
    """One job at a time.

    Transcription saturates the CPU, so running jobs concurrently would make
    each one slower without finishing the set any sooner.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, fn) -> None:
        self._queue.put(fn)

    def join(self) -> None:
        """Block until the queue drains. For tests."""
        self._queue.join()

    def _loop(self) -> None:
        while True:
            fn = self._queue.get()
            try:
                fn()
            except Exception:                         # noqa: BLE001
                # run_analysis and run_render already record failure on the
                # job; swallowing here keeps one bad job from killing the
                # worker thread and stalling every job behind it.
                pass
            finally:
                self._queue.task_done()
```

- [ ] **Step 4: Update the imports at the top of `web/jobs.py`**

Replace the import block with:

```python
from __future__ import annotations

import queue
import shutil
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import EditorConfig
from cutlist import keep_segments
from edit import analyse
from models import CutSpan, Span
from render import render_video
from transcribe import transcribe
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_web_pipeline.py -v`
Expected: 13 passed

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 277 passed

- [ ] **Step 7: Commit**

```bash
git add web/jobs.py tests/test_web_pipeline.py
git commit -m "feat: add job store, serial worker, pipeline and toggle arithmetic"
```

---

## Task 4: Server — create a job and poll it

**Files:**
- Create: `web/server.py`
- Test: `tests/test_web_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_api.py`:

```python
import io

import pytest
from fastapi.testclient import TestClient

from web import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A server whose work directory and pipeline are both under test control."""
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)

    def instant(job, cfg):
        job.advance(server.JobState.PROBING)
        job.advance(server.JobState.TRANSCRIBING)
        job.advance(server.JobState.ANALYSING)
        job.report = {
            "duration": 10.0,
            "mode": "subtractive",
            "summary": {"output_seconds": 8.0},
            "cuts": [
                {"start": 1.0, "end": 2.0, "reasons": ["silence"],
                 "detail": "gap", "duration": 1.0},
                {"start": 4.0, "end": 5.0, "reasons": ["disfluency"],
                 "detail": "the", "duration": 1.0},
            ],
            "lines": [],
            "adlibs": [],
            "segments": [{"start": 0.0, "end": 1.0}],
        }
        job.progress = 1.0
        job.advance(server.JobState.READY)

    monkeypatch.setattr(server, "run_analysis", instant)
    return TestClient(server.app)


def upload(client, name="raw.mp4", data=b"\x00" * 2048, script="hello world"):
    return client.post("/api/jobs",
                       files={"video": (name, io.BytesIO(data), "video/mp4")},
                       data={"script": script})


def test_upload_creates_a_job(client):
    response = upload(client)
    assert response.status_code == 200
    assert response.json()["id"]


def test_the_uploaded_bytes_land_on_disk(client):
    job_id = upload(client, data=b"abcdef").json()["id"]
    job = server.STORE.get(job_id)
    assert job.video.read_bytes() == b"abcdef"


def test_an_empty_script_is_rejected_before_a_job_exists(client):
    response = upload(client, script="   ")
    assert response.status_code == 400
    assert "script" in response.json()["detail"].lower()


def test_polling_returns_state_and_report(client):
    job_id = upload(client).json()["id"]
    server.WORKER.join()

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "ready"
    assert body["progress"] == 1.0
    assert len(body["report"]["cuts"]) == 2


def test_polling_an_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_the_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.server'`

- [ ] **Step 3: Create `web/server.py`**

```python
"""FastAPI routes. Translation only - the logic lives in web.jobs."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import EditorConfig
from web.jobs import (JobState, JobStore, Worker, run_analysis, run_render)

STATIC = Path(__file__).resolve().parent / "static"
WORK = Path(__file__).resolve().parent / "work"

# 1 MiB at a time: a 200 MB upload must never be buffered whole in memory.
UPLOAD_CHUNK = 1024 * 1024

app = FastAPI(title="Auto-Editor")
STORE = JobStore(WORK)
WORKER = Worker()


def editor_config() -> EditorConfig:
    """Settings for a UI run.

    verbatim is on because Whisper otherwise tidies away the stutters and
    restarts the review screen exists to show.
    """
    return EditorConfig(verbatim=True)


def _job_or_404(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...),
                     script: str = Form(...)) -> dict:
    if not script.strip():
        raise HTTPException(status_code=400,
                            detail="script is empty - paste your script")

    job = STORE.create(video.filename or "upload.mp4", script)

    with job.video.open("wb") as handle:
        while chunk := await video.read(UPLOAD_CHUNK):
            handle.write(chunk)

    cfg = editor_config()
    WORKER.submit(lambda: run_analysis(job, cfg))
    return {"id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_or_404(job_id).as_dict()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
```

- [ ] **Step 4: Create the static directory so the mount succeeds**

Create `web/static/index.html` with a placeholder that Task 8 replaces:

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Auto-Editor</title></head>
<body><p>Loading…</p></body></html>
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_web_api.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add web/server.py web/static/index.html tests/test_web_api.py
git commit -m "feat: add job creation with streamed upload, and polling"
```

---

## Task 5: Serving the video with range support

**Files:**
- Modify: `web/server.py` (append before the `app.mount` line)
- Test: `tests/test_web_range.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_range.py`:

```python
"""Range support is load-bearing, not an optimisation.

Clicking a cut seeks the video element to that timestamp. Without ranges the
browser refetches the whole 200 MB file on every seek, so the central feature
of the review screen degrades silently rather than failing loudly. That is
exactly the kind of thing worth asserting.
"""
import io

import pytest
from fastapi.testclient import TestClient

from web import server

PAYLOAD = bytes(range(256)) * 400          # 102400 bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "run_analysis", lambda job, cfg: None)
    return TestClient(server.app)


@pytest.fixture
def job_id(client):
    return client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(PAYLOAD), "video/mp4")},
        data={"script": "hello"}).json()["id"]


def test_the_whole_file_is_served_without_a_range_header(client, job_id):
    response = client.get(f"/api/jobs/{job_id}/video")
    assert response.status_code == 200
    assert len(response.content) == len(PAYLOAD)


def test_the_server_advertises_range_support(client, job_id):
    response = client.get(f"/api/jobs/{job_id}/video")
    assert response.headers.get("accept-ranges") == "bytes"


def test_a_range_request_returns_206_with_only_that_slice(client, job_id):
    response = client.get(f"/api/jobs/{job_id}/video",
                          headers={"Range": "bytes=1000-1999"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 1000-1999/{len(PAYLOAD)}"
    assert response.content == PAYLOAD[1000:2000]


def test_an_open_ended_range_runs_to_the_end(client, job_id):
    start = len(PAYLOAD) - 500
    response = client.get(f"/api/jobs/{job_id}/video",
                          headers={"Range": f"bytes={start}-"})
    assert response.status_code == 206
    assert response.content == PAYLOAD[start:]


def test_video_for_an_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope/video").status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_range.py -v`
Expected: FAIL with 404 on every route — `/video` does not exist yet

- [ ] **Step 3: Append the route to `web/server.py`**

Insert before the `app.mount(...)` line:

```python
@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    """The source recording.

    FileResponse implements HTTP range requests, which is what makes seeking
    to a cut instant instead of refetching the whole file. Verified by
    tests/test_web_range.py rather than assumed.
    """
    job = _job_or_404(job_id)
    if not job.video.exists():
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(job.video, media_type="video/mp4")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_web_range.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add web/server.py tests/test_web_range.py
git commit -m "feat: serve the source video with HTTP range support"
```

---

## Task 6: Approval, render, download and delete

**Files:**
- Modify: `web/server.py` (append before the `app.mount` line)
- Test: `tests/test_web_render.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_render.py`:

```python
import io

import pytest
from fastapi.testclient import TestClient

from web import server

READY_REPORT = {
    "duration": 10.0,
    "mode": "subtractive",
    "summary": {"output_seconds": 8.0},
    "cuts": [
        {"start": 1.0, "end": 2.0, "reasons": ["silence"], "detail": "gap",
         "duration": 1.0},
        {"start": 4.0, "end": 5.0, "reasons": ["disfluency"], "detail": "the",
         "duration": 1.0},
    ],
    "lines": [],
    "adlibs": [],
    "segments": [],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "run_analysis", lambda job, cfg: None)

    rendered = {}

    def fake_render(video, out, report, cfg):
        rendered["report"] = report
        out.write_bytes(b"rendered video")

    monkeypatch.setattr(server, "render_video", fake_render)
    client = TestClient(server.app)
    client.rendered = rendered
    return client


@pytest.fixture
def ready_job(client):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00" * 64), "video/mp4")},
        data={"script": "hello"}).json()["id"]

    job = server.STORE.get(job_id)
    for state in (server.JobState.PROBING, server.JobState.TRANSCRIBING,
                  server.JobState.ANALYSING, server.JobState.READY):
        job.advance(state)
    job.report = dict(READY_REPORT)
    return job_id


def test_approving_every_cut_renders_the_expected_segments(client, ready_job):
    response = client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0, 1]})
    assert response.status_code == 200
    server.WORKER.join()

    segments = client.rendered["report"]["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [
        (0.0, 1.0), (2.0, 4.0), (5.0, 10.0)]


def test_unchecking_a_cut_puts_that_span_back(client, ready_job):
    client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0]})
    server.WORKER.join()

    segments = client.rendered["report"]["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [(0.0, 1.0), (2.0, 10.0)]


def test_the_job_reaches_done(client, ready_job):
    client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0, 1]})
    server.WORKER.join()
    assert client.get(f"/api/jobs/{ready_job}").json()["state"] == "done"


def test_rendering_before_ready_is_rejected(client):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00"), "video/mp4")},
        data={"script": "hello"}).json()["id"]

    response = client.post(f"/api/jobs/{job_id}/render", json={"cuts": []})
    assert response.status_code == 409
    assert "ready" in response.json()["detail"].lower()


def test_download_returns_the_rendered_file(client, ready_job):
    client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0, 1]})
    server.WORKER.join()

    response = client.get(f"/api/jobs/{ready_job}/download")
    assert response.status_code == 200
    assert response.content == b"rendered video"


def test_download_before_rendering_is_404(client, ready_job):
    assert client.get(f"/api/jobs/{ready_job}/download").status_code == 404


def test_delete_removes_the_job(client, ready_job):
    assert client.delete(f"/api/jobs/{ready_job}").status_code == 200
    assert client.get(f"/api/jobs/{ready_job}").status_code == 404


def test_deleting_an_unknown_job_is_404(client):
    assert client.delete("/api/jobs/nope").status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_render.py -v`
Expected: FAIL — `/render` returns 404, and `server.render_video` does not exist to patch

- [ ] **Step 3: Append to `web/server.py`**

Insert before the `app.mount(...)` line:

```python
class RenderRequest(BaseModel):
    """Indices into report["cuts"] that should STILL be removed.

    Anything absent has been unchecked in the review screen and stays in the
    finished video.
    """
    cuts: list[int]


@app.post("/api/jobs/{job_id}/render")
def start_render(job_id: str, request: RenderRequest) -> dict:
    job = _job_or_404(job_id)
    if job.state is not JobState.READY:
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.state.value}, not ready to render")

    cfg = editor_config()
    keep = set(request.cuts)
    WORKER.submit(lambda: run_render(job, cfg, keep, render_fn=render_video))
    return {"state": "rendering"}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if job.output is None or not job.output.exists():
        raise HTTPException(status_code=404, detail="nothing rendered yet")
    return FileResponse(job.output, media_type="video/mp4",
                        filename=f"{job.video.stem}.edited.mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not STORE.delete(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    return {"deleted": job_id}
```

- [ ] **Step 4: Update the imports at the top of `web/server.py`**

Add `render_video` and `BaseModel`, so the import block reads:

```python
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import EditorConfig
from render import render_video
from web.jobs import (JobState, JobStore, Worker, run_analysis, run_render)
```

`render_video` is imported into `server` specifically so tests can patch
`server.render_video` and pass it through to `run_render`.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_web_render.py -v`
Expected: 8 passed

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 296 passed

- [ ] **Step 7: Commit**

```bash
git add web/server.py tests/test_web_render.py
git commit -m "feat: add approval, render, download and delete routes"
```

---

## Task 7: XML export route

**Files:**
- Modify: `web/server.py` (append before the `app.mount` line)
- Test: `tests/test_web_xml.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_web_xml.py`:

```python
import io
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from web import server
from xml_export import VideoMeta


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "run_analysis", lambda job, cfg: None)
    monkeypatch.setattr(
        server, "probe_video_meta",
        lambda video: VideoMeta(fps=25.0, width=1920, height=1080,
                                duration=10.0))
    return TestClient(server.app)


@pytest.fixture
def ready_job(client):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00" * 64), "video/mp4")},
        data={"script": "hello"}).json()["id"]

    job = server.STORE.get(job_id)
    for state in (server.JobState.PROBING, server.JobState.TRANSCRIBING,
                  server.JobState.ANALYSING, server.JobState.READY):
        job.advance(state)
    job.report = {
        "duration": 10.0,
        "cuts": [{"start": 1.0, "end": 2.0, "reasons": ["silence"],
                  "detail": "gap", "duration": 1.0}],
        "segments": [],
    }
    return job_id


def test_xml_reflects_the_approved_cut_list(client, ready_job):
    response = client.post(f"/api/jobs/{ready_job}/xml", json={"cuts": [0]})
    assert response.status_code == 200

    root = ET.fromstring(response.text)
    clips = root.findall("./sequence/media/video/track/clipitem")
    assert len(clips) == 2, "one cut splits the timeline into two clips"


def test_unchecking_the_cut_yields_a_single_clip(client, ready_job):
    response = client.post(f"/api/jobs/{ready_job}/xml", json={"cuts": []})
    root = ET.fromstring(response.text)
    assert len(root.findall("./sequence/media/video/track/clipitem")) == 1


def test_xml_before_ready_is_rejected(client):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00"), "video/mp4")},
        data={"script": "hello"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/xml",
                       json={"cuts": []}).status_code == 409
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_web_xml.py -v`
Expected: FAIL — `/xml` returns 404

- [ ] **Step 3: Append to `web/server.py`**

Insert before the `app.mount(...)` line:

```python
@app.post("/api/jobs/{job_id}/xml")
def export_xml(job_id: str, request: RenderRequest) -> Response:
    """A Premiere timeline for the cut list as currently approved."""
    job = _job_or_404(job_id)
    if job.state is not JobState.READY:
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.state.value}, not ready to export")

    segments = approved_segments(job.report, set(request.cuts))
    xml = build_xml(job.video, segments, probe_video_meta(job.video))
    return Response(
        content=xml, media_type="application/xml",
        headers={"Content-Disposition":
                 f'attachment; filename="{job.video.stem}.xml"'})
```

- [ ] **Step 4: Update the imports at the top of `web/server.py`**

```python
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
...
from web.jobs import (JobState, JobStore, Worker, approved_segments,
                      run_analysis, run_render)
from xml_export import build_xml, probe_video_meta
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_web_xml.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add web/server.py tests/test_web_xml.py
git commit -m "feat: add Premiere XML export for the approved cut list"
```

---

## Task 8: Front end — markup and styling

**Files:**
- Replace: `web/static/index.html`
- Create: `web/static/style.css`

- [ ] **Step 1: Write `web/static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auto-Editor</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
<main>
  <h1>Auto-Editor</h1>

  <section id="start">
    <div id="drop" class="drop">
      <p id="drop-text">Drop a recording here, or click to choose</p>
      <input type="file" id="file" accept="video/*" hidden>
    </div>
    <label for="script">Your script</label>
    <textarea id="script" rows="10"
      placeholder="Paste the script you read aloud. One beat per paragraph, separated by blank lines."></textarea>
    <button id="analyse" disabled>Analyse</button>
    <p id="start-error" class="error" hidden></p>
  </section>

  <section id="working" hidden>
    <p id="stage">Queued</p>
    <div class="bar"><div id="bar-fill"></div></div>
    <p class="hint">Transcribing runs about 1.5x realtime on CPU. The first run
      downloads the model.</p>
  </section>

  <section id="review" hidden>
    <div class="summary">
      <span id="filename"></span>
      <span id="duration"></span>
    </div>

    <video id="player" controls preload="metadata"></video>
    <div id="timeline" class="timeline"></div>
    <div class="legend">
      <span><i class="sw kept"></i>kept</span>
      <span><i class="sw silence"></i>silence</span>
      <span><i class="sw disfluency"></i>disfluency</span>
      <span><i class="sw retake"></i>retake</span>
    </div>

    <p id="silence-summary" class="hint"></p>

    <h2>Judgment calls</h2>
    <p class="hint">Uncheck anything you want to keep. Play each one to hear it.</p>
    <div id="cuts"></div>

    <h2>Kept, not in your script</h2>
    <p class="hint">Probable ad-libs. These stay in.</p>
    <div id="adlibs"></div>

    <h2>Script lines</h2>
    <div id="lines"></div>

    <div class="actions">
      <button id="render">Render</button>
      <button id="xml" class="secondary">Download XML</button>
    </div>
    <p id="review-error" class="error" hidden></p>
  </section>

  <section id="finished" hidden>
    <p id="finished-text"></p>
    <video id="result" controls preload="metadata"></video>
    <div class="actions">
      <a id="download" class="button">Download</a>
      <button id="again" class="secondary">Edit another</button>
    </div>
  </section>
</main>
<script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `web/static/style.css`**

```css
:root {
  --bg: #faf9f7;
  --panel: #ffffff;
  --line: #e3e1dc;
  --text: #2c2c2a;
  --muted: #6b6a66;
  --kept: #9fe1cb;
  --silence: #f5c4b3;
  --disfluency: #fac775;
  --retake: #f4c0d1;
  --accent: #1d9e75;
  --danger: #a32d2d;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
}

main { max-width: 760px; margin: 0 auto; padding: 32px 20px 64px; }

h1 { font-size: 22px; font-weight: 500; margin: 0 0 24px; }
h2 { font-size: 16px; font-weight: 500; margin: 28px 0 4px; }

.hint { color: var(--muted); font-size: 13px; margin: 4px 0 12px; }
.error { color: var(--danger); font-size: 14px; }

.drop {
  border: 1.5px dashed var(--line);
  border-radius: 10px;
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  color: var(--muted);
  background: var(--panel);
}
.drop.over { border-color: var(--accent); color: var(--accent); }
.drop.loaded { border-style: solid; color: var(--text); }

label { display: block; margin: 20px 0 6px; font-size: 14px; }

textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  font: inherit;
  background: var(--panel);
  resize: vertical;
}

button, .button {
  font: inherit;
  padding: 9px 18px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  text-decoration: none;
  display: inline-block;
}
button.secondary, .button.secondary { background: var(--panel); color: var(--text); }
button:disabled { opacity: 0.45; cursor: not-allowed; }
#analyse { margin-top: 16px; }

.bar { height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; }
#bar-fill { height: 100%; width: 0; background: var(--accent); transition: width .3s; }

.summary {
  display: flex; justify-content: space-between;
  margin-bottom: 12px; font-size: 14px;
}

video { width: 100%; border-radius: 8px; background: #000; display: block; }

.timeline { display: flex; height: 26px; border-radius: 4px; overflow: hidden; margin: 8px 0 6px; }
.timeline div { min-width: 1px; }
.timeline .kept { background: var(--kept); }
.timeline .silence { background: var(--silence); }
.timeline .disfluency { background: var(--disfluency); }
.timeline .retake { background: var(--retake); }

.legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
.sw.kept { background: var(--kept); }
.sw.silence { background: var(--silence); }
.sw.disfluency { background: var(--disfluency); }
.sw.retake { background: var(--retake); }

.row {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 12px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: none;
  font-size: 14px;
}
.row:first-child { border-top: 1px solid var(--line); border-radius: 8px 8px 0 0; }
.row:last-child { border-radius: 0 0 8px 8px; }
.row:only-child { border-radius: 8px; }
.row.off { opacity: 0.5; }

.time { font-family: ui-monospace, monospace; font-size: 12px; color: var(--muted); min-width: 56px; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 6px; white-space: nowrap; }
.badge.disfluency { background: #faeeda; color: #854f0b; }
.badge.retake { background: #fbeaf0; color: #993556; }
.badge.silence { background: #faece7; color: #993c1d; }
.detail { flex: 1; }
.play { border: none; background: none; color: var(--muted); cursor: pointer; padding: 2px 6px; }

.status { font-size: 11px; padding: 2px 8px; border-radius: 6px; }
.status.ok { background: #e1f5ee; color: #0f6e56; }
.status.review { background: #faeeda; color: #854f0b; }
.status.miss { background: #fcebeb; color: #a32d2d; }

.actions { display: flex; gap: 10px; margin-top: 24px; }
.empty { color: var(--muted); font-size: 14px; padding: 10px 0; }
```

- [ ] **Step 3: Verify the page is served**

Run: `python -m pytest tests/test_web_api.py::test_the_index_page_is_served -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add web/static/index.html web/static/style.css
git commit -m "feat: add web UI markup and styling"
```

---

## Task 9: Front end — behaviour

**Files:**
- Create: `web/static/app.js`

- [ ] **Step 1: Write `web/static/app.js`**

```javascript
const $ = (id) => document.getElementById(id);
const show = (id) => $(id).hidden = false;
const hide = (id) => $(id).hidden = true;

let file = null;
let job = null;
let report = null;
let applied = new Set();

const ts = (s) => {
  const m = Math.floor(s / 60);
  const r = (s % 60).toFixed(2).padStart(5, "0");
  return `${String(m).padStart(2, "0")}:${r}`;
};

const reasonOf = (cut) =>
  cut.reasons.includes("retake_loser") ? "retake"
  : cut.reasons.includes("disfluency") ? "disfluency"
  : "silence";

// --- start screen --------------------------------------------------------

$("drop").addEventListener("click", () => $("file").click());
$("drop").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("drop").classList.add("over");
});
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", (e) => {
  e.preventDefault();
  $("drop").classList.remove("over");
  if (e.dataTransfer.files.length) takeFile(e.dataTransfer.files[0]);
});
$("file").addEventListener("change", (e) => {
  if (e.target.files.length) takeFile(e.target.files[0]);
});
$("script").addEventListener("input", refreshStart);

function takeFile(f) {
  file = f;
  const mb = (f.size / 1048576).toFixed(0);
  $("drop-text").textContent = `${f.name} — ${mb} MB`;
  $("drop").classList.add("loaded");
  refreshStart();
}

function refreshStart() {
  $("analyse").disabled = !(file && $("script").value.trim());
}

$("analyse").addEventListener("click", async () => {
  $("analyse").disabled = true;
  hide("start-error");

  const body = new FormData();
  body.append("video", file);
  body.append("script", $("script").value);

  const response = await fetch("/api/jobs", { method: "POST", body });
  if (!response.ok) {
    const detail = (await response.json()).detail || "upload failed";
    $("start-error").textContent = detail;
    show("start-error");
    $("analyse").disabled = false;
    return;
  }

  job = (await response.json()).id;
  hide("start");
  show("working");
  poll();
});

// --- polling -------------------------------------------------------------

const STAGE_TEXT = {
  queued: "Queued", probing: "Reading the file",
  transcribing: "Transcribing", analysing: "Matching your script",
  rendering: "Rendering",
};

async function poll() {
  const state = await (await fetch(`/api/jobs/${job}`)).json();

  if (state.state === "failed") {
    hide("working");
    show("start");
    $("start-error").textContent = state.error;
    show("start-error");
    $("analyse").disabled = false;
    return;
  }

  if (state.state === "ready") {
    report = state.report;
    applied = new Set(report.cuts.map((_, i) => i));
    hide("working");
    drawReview();
    show("review");
    return;
  }

  if (state.state === "done") {
    hide("working");
    $("finished-text").textContent =
      `Done — ${ts(report.duration)} cut down to ${ts(outputSeconds())}.`;
    $("result").src = `/api/jobs/${job}/download`;
    $("download").href = `/api/jobs/${job}/download`;
    show("finished");
    return;
  }

  $("stage").textContent = STAGE_TEXT[state.state] || state.state;
  $("bar-fill").style.width = `${Math.round(state.progress * 100)}%`;
  setTimeout(poll, 500);
}

// --- review --------------------------------------------------------------

function activeCuts() {
  return report.cuts.filter((_, i) => applied.has(i));
}

function outputSeconds() {
  const removed = activeCuts().reduce((sum, c) => sum + (c.end - c.start), 0);
  return report.duration - removed;
}

function drawReview() {
  $("filename").textContent = report.video ? report.video.split(/[\\/]/).pop() : "";
  drawTimeline();
  drawCuts();
  drawAdlibs();
  drawLines();
  refreshTotals();
}

function drawTimeline() {
  const bar = $("timeline");
  bar.innerHTML = "";
  let at = 0;
  const push = (cls, width) => {
    if (width <= 0) return;
    const d = document.createElement("div");
    d.className = cls;
    d.style.flex = String(width);
    bar.appendChild(d);
  };

  report.cuts.forEach((cut, i) => {
    push("kept", cut.start - at);
    push(applied.has(i) ? reasonOf(cut) : "kept", cut.end - cut.start);
    at = cut.end;
  });
  push("kept", report.duration - at);
}

function drawCuts() {
  const judged = report.cuts
    .map((cut, index) => ({ cut, index }))
    .filter(({ cut }) => reasonOf(cut) !== "silence");

  const silence = report.cuts.filter((c) => reasonOf(c) === "silence");
  const total = silence.reduce((s, c) => s + (c.end - c.start), 0);
  $("silence-summary").textContent =
    `Silence: ${silence.length} cuts, ${total.toFixed(1)}s total — removed automatically.`;

  const host = $("cuts");
  host.innerHTML = "";
  if (!judged.length) {
    host.innerHTML = '<p class="empty">None — nothing needed a judgment call.</p>';
    return;
  }

  judged.forEach(({ cut, index }) => {
    const row = document.createElement("div");
    row.className = "row";

    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.addEventListener("change", () => {
      box.checked ? applied.add(index) : applied.delete(index);
      row.classList.toggle("off", !box.checked);
      drawTimeline();
      refreshTotals();
    });

    const play = document.createElement("button");
    play.className = "play";
    play.textContent = "▶";
    play.title = "Play this moment";
    play.addEventListener("click", () => {
      const player = $("player");
      player.currentTime = Math.max(0, cut.start - 1.5);
      player.play();
    });

    row.innerHTML =
      `<span class="time">${ts(cut.start)}</span>` +
      `<span class="badge ${reasonOf(cut)}">${reasonOf(cut)}</span>` +
      `<span class="detail">${cut.detail}</span>`;
    row.prepend(box);
    row.appendChild(play);
    host.appendChild(row);
  });
}

function drawAdlibs() {
  const host = $("adlibs");
  host.innerHTML = "";
  if (!report.adlibs.length) {
    host.innerHTML = '<p class="empty">None.</p>';
    return;
  }
  report.adlibs.forEach((a) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<span class="time">${ts(a.start)}</span><span class="detail">${a.text}</span>`;
    host.appendChild(row);
  });
}

function drawLines() {
  const host = $("lines");
  host.innerHTML = "";
  report.lines.forEach((line) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML =
      `<span class="time">${line.start === null ? "--:--" : ts(line.start)}</span>` +
      `<span class="status ${line.status}">${line.status}</span>` +
      `<span class="detail">${line.matched_text || line.script}</span>`;
    host.appendChild(row);
  });
}

function refreshTotals() {
  $("duration").textContent =
    `${ts(report.duration)} → ${ts(outputSeconds())}`;
  $("render").textContent = `Render ${outputSeconds().toFixed(1)}s`;
}

// --- actions -------------------------------------------------------------

$("player").src = "";
$("render").addEventListener("click", async () => {
  $("render").disabled = true;
  hide("review-error");

  const response = await fetch(`/api/jobs/${job}/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cuts: [...applied] }),
  });

  if (!response.ok) {
    $("review-error").textContent = (await response.json()).detail;
    show("review-error");
    $("render").disabled = false;
    return;
  }

  hide("review");
  $("stage").textContent = "Rendering";
  $("bar-fill").style.width = "100%";
  show("working");
  poll();
});

$("xml").addEventListener("click", async () => {
  const response = await fetch(`/api/jobs/${job}/xml`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cuts: [...applied] }),
  });
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "timeline.xml";
  link.click();
});

$("again").addEventListener("click", () => location.reload());
```

- [ ] **Step 2: Point the player at the job's video once review opens**

In `drawReview()`, the player source must be set. Add this as the first line of
`drawReview()`:

```javascript
  $("player").src = `/api/jobs/${job}/video`;
```

- [ ] **Step 3: Verify the file is served**

Run: `python -c "import urllib.request" && python -m pytest tests/test_web_api.py -q`
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add web/static/app.js
git commit -m "feat: add web UI behaviour - upload, polling, review, render"
```

---

## Task 10: Entry point, smoke test and docs

**Files:**
- Create: `web/__main__.py`
- Test: `tests/test_web_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Write `web/__main__.py`**

```python
"""python -m web — start the server and open a browser."""
from __future__ import annotations

import threading
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"Auto-Editor running at {url}")
    # bind first, then open, so the browser does not race the server
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("web.server:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the end-to-end smoke test**

Create `tests/test_web_smoke.py`:

```python
"""Full round trip through the API against real media.

Transcription is stubbed with a hand-built word list - Whisper's own behaviour
is covered by the engine's tests, and downloading a model here would make the
suite unusable.
"""
import io
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models import Word
from web import server

pytestmark = pytest.mark.smoke

HAS_FFMPEG = shutil.which("ffmpeg") is not None

SCRIPT = "Click the blue button.\n\nThat is everything you need."


def spoken():
    def run(text, start, dur=0.3):
        return [Word(t, round(start + i * dur, 3), round(start + i * dur + dur, 3))
                for i, t in enumerate(text.split())]
    return run("click the the blue button", 0.0) + run("that is everything you need", 6.0)


@pytest.fixture
def clip(tmp_path: Path) -> bytes:
    path = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path),
    ], capture_output=True, check=True)
    return path.read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", server.JobStore(tmp_path / "work"))
    monkeypatch.setattr(
        "web.jobs.transcribe",
        lambda video, cfg, on_progress=None: (spoken(), 10.0))
    return TestClient(server.app)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_upload_review_render_download(client, clip, tmp_path):
    created = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(clip), "video/mp4")},
        data={"script": SCRIPT})
    assert created.status_code == 200
    job_id = created.json()["id"]

    server.WORKER.join()
    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["state"] == "ready", state.get("error")

    report = state["report"]
    assert report["cuts"], "the stutter and the gap should both be found"

    every = list(range(len(report["cuts"])))
    assert client.post(f"/api/jobs/{job_id}/render",
                       json={"cuts": every}).status_code == 200
    server.WORKER.join()

    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "done"

    downloaded = client.get(f"/api/jobs/{job_id}/download")
    assert downloaded.status_code == 200
    assert len(downloaded.content) > 1000

    out = tmp_path / "out.mp4"
    out.write_bytes(downloaded.content)
    from transcribe import probe_duration
    assert probe_duration(out) < 10.0, "the render must be shorter than the source"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_unchecking_every_cut_renders_the_full_length(client, clip, tmp_path):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(clip), "video/mp4")},
        data={"script": SCRIPT}).json()["id"]
    server.WORKER.join()

    client.post(f"/api/jobs/{job_id}/render", json={"cuts": []})
    server.WORKER.join()

    out = tmp_path / "full.mp4"
    out.write_bytes(client.get(f"/api/jobs/{job_id}/download").content)

    from transcribe import probe_duration
    assert probe_duration(out) > 9.0, "rejecting every cut keeps the recording"
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/test_web_smoke.py -v -m smoke`
Expected: 2 passed

- [ ] **Step 4: Add a README section**

Insert immediately after the `## Install` section in `README.md`:

````markdown
## The app

A local web UI. Drop in a recording, paste your script, approve the cuts you
agree with, render.

```bash
python -m web
```

Opens at `http://127.0.0.1:8000`. Everything stays on your machine — no upload
leaves it, and the server binds to localhost only.

The review screen is the point. After analysis you get the source video, a
timeline coloured by what happens to each span, and every judgment-call cut as
a row you can play and uncheck. Silence is summarised rather than listed, since
it is never a judgment call. Output duration updates as you toggle, so the cost
of keeping something is visible before you commit to it.

Nothing renders until you ask. The job state machine has no transition from
`ready` to `rendering` without an explicit approval call, so "report first,
render second" is structural rather than a habit.

The UI runs in subtractive mode with `--verbatim` on. `--reorder`, batch mode
and the tuning knobs stay on the command line.
````

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 299 passed

Run: `python -m pytest tests/ -q -m smoke`
Expected: 19 passed

- [ ] **Step 6: Manual check**

Run: `python -m web`
Expected: a browser opens at `http://127.0.0.1:8000` showing the drop zone and
script box. Stop with Ctrl-C.

- [ ] **Step 7: Commit**

```bash
git add web/__main__.py tests/test_web_smoke.py README.md
git commit -m "feat: add python -m web entry point, smoke tests and docs"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4 architecture / file layout | 2, 3, 4, 10 |
| §5 job lifecycle, review gate | 2, 3 |
| §5 state in memory, files on disk, one at a time | 3 |
| §6 all seven routes | 4, 5, 6, 7 |
| §6 upload streams to disk | 4 |
| §6 range requests | 5 |
| §7 progress callback | 1 |
| §8 review screen (player, timeline, cuts, ad-libs, live total) | 8, 9 |
| §8 approval reuses `keep_segments` | 3 |
| §9 errors surfaced, never tracebacks | 3 (`fail`), 4, 9 |
| §10 all five listed test areas | 2, 3, 5, 6, 10 |
| §11 dependencies | 1 |
| §12 omissions (no waveform, no reorder, no history) | honoured throughout |

No gaps.

**Placeholder scan:** No TBD/TODO. Every code step is complete and runnable.
Task 4 Step 4 creates a deliberate placeholder `index.html`, replaced in Task 8
— called out explicitly rather than left implicit.

**Type consistency verified:**
- `JobState` values match between `jobs.py` (Task 2), the API tests (Tasks 4-7)
  and `STAGE_TEXT` in `app.js` (Task 9).
- `approved_segments(report, keep_applied: set[int])` — defined Task 3, called
  in Tasks 6 and 7 with `set(request.cuts)`.
- `RenderRequest.cuts: list[int]` — defined Task 6, reused by the XML route in
  Task 7. Both endpoints take the same shape deliberately.
- `run_analysis(job, cfg, transcribe_fn, analyse_fn)` and
  `run_render(job, cfg, keep_applied, render_fn)` — defined Task 3, called in
  Tasks 4 and 6.
- `Job.as_dict()` keys (`id`, `state`, `progress`, `report`, `error`,
  `has_output`) — produced Task 2, consumed by `poll()` in Task 9.
- `ProgressFn = Callable[[float], None]` — defined Task 1, used by
  `run_analysis`'s lambda in Task 3.
- Report keys read by the frontend (`duration`, `cuts[].start/end/reasons/detail`,
  `adlibs[].start/text`, `lines[].status/start/matched_text/script`) all match
  what `build_report` in `report.py` already emits.

**One ordering note for the implementer:** Task 4 mounts `StaticFiles` at the
*end* of `server.py`, and Tasks 5, 6 and 7 each insert routes *before* that
mount. A mount registered before a route will shadow it. Keep the mount last.
