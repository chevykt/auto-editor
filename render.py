"""ffmpeg rendering of the surviving segments, in original order.

Frame-accurate cutting of video and audio together requires re-encoding;
stream copy cannot do it.

Audio uses a short per-segment afade rather than a true acrossfade. A real
crossfade overlaps adjacent segments, shortening total audio by the fade
length at every cut, which accumulates into audible A/V desync over hundreds
of cuts. Per-segment fades suppress clicks identically while preserving exact
durations.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from config import EditorConfig
from models import Span


def build_filter_graph(segments: list[Span], cfg: EditorConfig) -> str:
    """filter_complex graph selecting and concatenating the surviving segments."""
    if not segments:
        raise ValueError("cannot render: no surviving segments")

    fade = cfg.crossfade_s
    parts: list[str] = []

    for i, seg in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={seg.start:.6f}:end={seg.end:.6f},"
            f"setpts=PTS-STARTPTS[v{i}]")

        achain = (f"[0:a]atrim=start={seg.start:.6f}:end={seg.end:.6f},"
                  f"asetpts=PTS-STARTPTS")
        # a fade needs room at both ends; skip it on very short segments
        if fade > 0 and seg.duration > 2 * fade:
            out_at = round(seg.duration - fade, 6)
            achain += (f",afade=t=in:st=0:d={fade:g}"
                       f",afade=t=out:st={out_at:g}:d={fade:g}")
        parts.append(f"{achain}[a{i}]")

    streams = "".join(f"[v{i}][a{i}]" for i in range(len(segments)))
    parts.append(f"{streams}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    return ";\n".join(parts)


def build_ffmpeg_command(video: str | Path, out: str | Path,
                         graph_path: str | Path,
                         cfg: EditorConfig) -> list[str]:
    """Assemble the ffmpeg argv.

    The graph goes in via -filter_complex_script rather than inline: a few
    hundred cuts produces a filter string far past the Windows 8191-character
    command-line limit, which would fail only on long recordings.
    """
    return [
        "ffmpeg", "-y",
        "-i", str(video),
        "-filter_complex_script", str(graph_path),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", cfg.video_codec,
        "-preset", cfg.video_preset,
        "-crf", str(cfg.video_crf),
        "-c:a", cfg.audio_codec,
        "-b:a", cfg.audio_bitrate,
        str(out),
    ]


def render_video(video: str | Path, out: str | Path, report: dict,
                 cfg: EditorConfig) -> Path:
    """Render the surviving segments of `report` to `out`."""
    segments = [Span(s["start"], s["end"]) for s in report.get("segments", [])]
    if not segments:
        raise ValueError("cannot render: no surviving segments in report")

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg")

    graph = build_filter_graph(segments, cfg)

    tmp = Path(tempfile.gettempdir()) / f"auto_editor_graph_{Path(out).stem}.txt"
    tmp.write_text(graph, encoding="utf-8")

    try:
        cmd = build_ffmpeg_command(video, out, tmp, cfg)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}")
    finally:
        tmp.unlink(missing_ok=True)

    return Path(out)
