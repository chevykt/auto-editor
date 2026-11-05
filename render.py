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


def build_filter_graph(segments: list[Span], cfg: EditorConfig,
                       video: bool = True) -> str:
    """filter_complex graph selecting and concatenating the surviving segments.

    video=False builds the audio half only, for a voiceover with no picture.
    """
    if not segments:
        raise ValueError("cannot render: no surviving segments")

    fade = cfg.crossfade_s
    parts: list[str] = []

    for i, seg in enumerate(segments):
        if video:
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

    n = len(segments)
    if video:
        streams = "".join(f"[v{i}][a{i}]" for i in range(n))
        parts.append(f"{streams}concat=n={n}:v=1:a=1[outv][outa]")
    else:
        streams = "".join(f"[a{i}]" for i in range(n))
        parts.append(f"{streams}concat=n={n}:v=0:a=1[outa]")
    return ";\n".join(parts)


def build_ffmpeg_command(video: str | Path, out: str | Path,
                         graph_path: str | Path,
                         cfg: EditorConfig) -> list[str]:
    """Assemble the ffmpeg argv.

    The graph goes in from a file rather than inline: a few hundred cuts
    produces a filter string far past the Windows 8191-character command-line
    limit, which would fail only on long recordings. The "-/" prefix means "read
    this option's value from a file"; it replaced -filter_complex_script, which
    ffmpeg 9 removed.
    """
    return [
        "ffmpeg", "-y",
        "-i", str(video),
        "-/filter_complex", str(graph_path),
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

    run_with_graph(build_filter_graph(segments, cfg), Path(out).stem,
                   lambda graph_path: build_ffmpeg_command(video, out, graph_path, cfg))
    return Path(out)


def render_audio(src: str | Path, out: str | Path, segments: list[Span],
                 cfg: EditorConfig) -> Path:
    """The kept segments of an audio file, joined, as a 16-bit WAV."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: winget install Gyan.FFmpeg")
    run_with_graph(build_filter_graph(segments, cfg, video=False), Path(out).stem,
                   lambda graph_path: ["ffmpeg", "-y", "-i", str(src),
                                       "-/filter_complex", str(graph_path),
                                       "-map", "[outa]", "-c:a", "pcm_s16le", str(out)])
    return Path(out)


def run_with_graph(graph: str, name: str, command) -> None:
    """Run the ffmpeg argv that command(graph_path) builds, with the graph in a file.

    Raises RuntimeError with ffmpeg's last output on failure.
    """
    tmp = Path(tempfile.gettempdir()) / f"auto_editor_graph_{name}.txt"
    tmp.write_text(graph, encoding="utf-8")

    try:
        # Explicit, because Windows would decode as cp1252 and ffmpeg echoes the
        # filename: one undecodable byte fails a render that actually succeeded.
        result = subprocess.run(command(tmp), capture_output=True, encoding="utf-8",
                                errors="replace")
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n"
                f"{result.stderr[-2000:]}")
    finally:
        tmp.unlink(missing_ok=True)
