"""Records the browser as it performs the plan.

Uses Chrome's own screencast rather than Playwright's video recorder, which is
low bitrate: each frame arrives as a lossless PNG with the moment it was drawn,
and ffmpeg turns those into a constant 30 fps video. Chrome only sends a frame
when the page changes, so a still page costs nothing.

Measured 2026-11-06 on supabase.com/docs at 1920x1080: 59 frames/s during
cursor motion, captured frames identical to page.screenshot() apart from the
page's own animation, and text in the encoded MP4 indistinguishable from the
screenshot at 100%. PNG frames are kept; no fallback was needed.
"""
from __future__ import annotations

import base64
import subprocess
import time
from pathlib import Path

from playwright.sync_api import Page

FPS = 30


def concat_lines(frames: list[tuple[float, Path]], end: float) -> list[str]:
    """ffmpeg concat-demuxer lines showing each frame until the next one arrives.

    The last file is listed twice because the demuxer ignores the final
    duration otherwise.
    """
    if not frames:
        raise ValueError("no frames were captured")
    lines: list[str] = []
    for i, (t, path) in enumerate(frames):
        until = frames[i + 1][0] if i + 1 < len(frames) else end
        lines += [f"file '{path.as_posix()}'", f"duration {round(until - t, 6):g}"]
    lines.append(f"file '{frames[-1][1].as_posix()}'")
    return lines


class Capture:
    """Records page from enter to exit; frames land in frames_dir."""

    def __init__(self, page: Page, frames_dir: Path):
        self.page = page
        self.frames_dir = frames_dir
        self.frames: list[tuple[float, Path]] = []
        self.ended = 0.0
        self._cdp = None

    def __enter__(self) -> "Capture":
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._cdp = self.page.context.new_cdp_session(self.page)
        self._cdp.on("Page.screencastFrame", self._frame)
        self._cdp.send("Page.startScreencast", {
            "format": "png", "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})
        return self

    def _frame(self, params: dict) -> None:
        path = self.frames_dir / f"{len(self.frames):06d}.png"
        path.write_bytes(base64.b64decode(params["data"]))
        stamp = params["metadata"].get("timestamp") or time.time()
        self.frames.append((stamp, path))
        self._cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

    def __exit__(self, *exc) -> None:
        self.ended = time.time()
        self._cdp.send("Page.stopScreencast")
        self._cdp.detach()

    def to_video(self, out: Path, fps: int = FPS) -> tuple[Path, float]:
        """The recording as a constant-rate MP4, and the epoch time of its first frame."""
        listing = self.frames_dir / "frames.txt"
        listing.write_text("\n".join(concat_lines(self.frames, self.ended)) + "\n",
                           encoding="utf-8")
        # The demuxer pads after the last frame (measured: +0.13 s on a 2.1 s
        # capture), so the output is cut to the span that was actually recorded.
        span = self.ended - self.frames[0][0]
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-t", f"{span:.6f}", "-vf", f"fps={fps},format=yuv420p",
             "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out)],
            capture_output=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed making the capture:\n{result.stderr[-2000:]}")
        return out, self.frames[0][0]
