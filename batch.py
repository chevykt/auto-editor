"""Batch mode: pair up recordings with their scripts across a directory."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})

# What a render writes. Excluded from discovery so a second batch run does not
# treat its own output as fresh input.
RENDER_SUFFIX = ".cut"


@dataclass(frozen=True)
class Job:
    video: Path
    script: Path

    @property
    def name(self) -> str:
        return self.video.stem


def is_render_output(video: Path) -> bool:
    """True for files this tool produced, e.g. lesson.cut.mp4."""
    return Path(video.stem).suffix == RENDER_SUFFIX


def find_jobs(directory: str | Path,
              script_suffix: str = ".txt") -> tuple[list[Job], list[Path]]:
    """Pair every recording with the same-stem script beside it.

    Returns (jobs, unpaired). Unpaired recordings are reported rather than
    silently skipped - a missing script is usually a typo, and a batch that
    quietly processes four of five files is worse than one that says so.
    """
    directory = Path(directory)
    jobs: list[Job] = []
    unpaired: list[Path] = []

    for video in sorted(directory.iterdir()):
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if is_render_output(video):
            continue

        script = video.with_suffix(script_suffix)
        if script.exists():
            jobs.append(Job(video=video, script=script))
        else:
            unpaired.append(video)

    return jobs, unpaired


def output_path(job: Job, out_dir: Path | None, suffix: str) -> Path:
    """Where one job's output goes: beside the video, or under --out-dir."""
    name = job.name + suffix
    return (out_dir / name) if out_dir else job.video.with_name(name)
