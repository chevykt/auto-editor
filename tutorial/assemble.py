"""The finished tutorial: fitted pieces of the capture, under the cleaned voiceover.

Every piece is forced to an exact number of frames, taken from the running total
rather than its own length. Rounding each piece on its own would let a frame of
error build up per piece, and over dozens of pieces the clicks would drift off
the words that describe them.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from models import Span
from render import run_with_graph
from tutorial.fit import Hold, Piece, Play
from xml_export import probe_video_meta, write_fcpxml

FPS = 30


def _seconds(piece: Piece) -> float:
    if isinstance(piece, Hold):
        return piece.seconds
    return (piece.src_end - piece.src_start) / piece.speed


def frame_counts(pieces: list[Piece], fps: int = FPS) -> list[int]:
    counts, t = [], 0.0
    for piece in pieces:
        end = t + _seconds(piece)
        counts.append(round(end * fps) - round(t * fps))
        t = end
    return counts


def pieces_graph(pieces: list[Piece], fps: int = FPS) -> str:
    """A filter graph turning [0:v] into [outv], one chain per piece."""
    if not pieces:
        raise ValueError("nothing to assemble")
    chains = []
    for piece, frames in zip(pieces, frame_counts(pieces, fps)):
        if frames <= 0:
            continue
        if isinstance(piece, Play):
            head = (f"trim=start={piece.src_start:.6f}:end={piece.src_end:.6f},"
                    f"setpts=(PTS-STARTPTS)/{piece.speed}")
        else:
            head = f"trim=start={piece.src_at:.6f},trim=end_frame=1,setpts=PTS-STARTPTS"
        # pad, then cut: whatever the source gave, the piece is exactly `frames` long
        chains.append(f"[0:v]{head},fps={fps},tpad=stop_mode=clone:stop={frames},"
                      f"trim=end_frame={frames},setpts=PTS-STARTPTS[p{len(chains)}]")
    labels = "".join(f"[p{i}]" for i in range(len(chains)))
    return ";\n".join(chains + [f"{labels}concat=n={len(chains)}:v=1:a=0[outv]"])


def _duration(path: Path, stream: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
         "stream=duration", "-of", "json", str(path)],
        capture_output=True, encoding="utf-8", errors="replace", check=True)
    return float(json.loads(out.stdout)["streams"][0]["duration"])


def _within(pieces: list[Piece], length: float, fps: int) -> list[Piece]:
    """Keep every source time inside the capture, so no chain comes up empty."""
    last = max(0.0, length - 1 / fps)
    out: list[Piece] = []
    for p in pieces:
        if isinstance(p, Hold):
            out.append(Hold(min(p.src_at, last), p.seconds))
        else:
            start = min(p.src_start, last)
            out.append(Play(start, max(min(p.src_end, length), start + 1 / fps), p.speed))
    return out


def assemble(capture: Path, pieces: list[Piece], voice_wav: Path, out: Path) -> Path:
    pieces = _within(pieces, _duration(capture, "v:0"), FPS)
    run_with_graph(pieces_graph(pieces), Path(out).stem, lambda graph: [
        "ffmpeg", "-y", "-i", str(capture), "-i", str(voice_wav),
        "-/filter_complex", str(graph), "-map", "[outv]", "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", str(out)])

    video, audio = _duration(out, "v:0"), _duration(out, "a:0")
    if abs(video - audio) > 1 / FPS + 0.03:     # aac pads its last packet slightly
        raise RuntimeError(f"picture ({video:.2f}s) and voice ({audio:.2f}s) came out "
                           "different lengths")
    return Path(out)


def write_timeline(final: Path, windows: tuple[tuple[float, float], ...],
                   out: Path) -> Path:
    """The finished video cut at beat boundaries, for trimming beats in Resolve."""
    return write_fcpxml(final, [Span(a, b) for a, b in windows],
                        probe_video_meta(final), out)
