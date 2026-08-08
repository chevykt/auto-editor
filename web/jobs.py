"""Job lifecycle for the web UI. Everything here is testable without a server.

The state machine exists to enforce the review gate structurally rather than by
convention: there is no transition from ready to done, so a job cannot render
without an explicit approval call.
"""
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
                 transcribe_fn=None, analyse_fn=None) -> None:
    """Take a job from queued to ready. Never further.

    Stopping at ready is the review gate: rendering is a separate, explicit
    call. The two function arguments exist so tests can drive the pipeline
    without a speech model.

    They default to None rather than to the functions themselves, so the
    module-level names are looked up when this RUNS, not when it is defined.
    A default of `transcribe_fn=transcribe` captures the reference at import
    and silently ignores any later monkeypatch - which means a test that
    thinks it stubbed Whisper actually loads it.
    """
    transcribe_fn = transcribe_fn or transcribe
    analyse_fn = analyse_fn or analyse

    try:
        job.advance(JobState.PROBING)
        job.advance(JobState.TRANSCRIBING)

        words, duration = transcribe_fn(
            job.video, cfg,
            on_progress=lambda fraction: setattr(job, "progress", fraction))

        job.advance(JobState.ANALYSING)
        job.report = analyse_fn(words, job.script, duration, cfg,
                                str(job.video), video_path=job.video)
        job.progress = 1.0
        job.advance(JobState.READY)
    except Exception as exc:                          # noqa: BLE001
        job.fail(str(exc))


def run_render(job: Job, cfg: EditorConfig, keep_applied: set[int],
               render_fn=None) -> None:
    """Render the approved cut list. Only callable from ready.

    render_fn defaults to None for the same reason as run_analysis: a default
    bound at import time cannot be monkeypatched.
    """
    render_fn = render_fn or render_video

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
        # filename cannot write outside its own job folder. It is not quite
        # enough on its own: Path("..").name returns ".." unchanged, which
        # would resolve to the parent of the job folder. "." and "" already
        # come back empty, so the three are rejected together.
        safe = Path(filename).name
        if safe in ("", ".", ".."):
            safe = "upload.mp4"

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
