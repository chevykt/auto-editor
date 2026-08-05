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
