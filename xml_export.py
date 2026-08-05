"""Premiere-compatible XML timeline export (FCP7 xmeml).

Useful while alignment and disfluency detection are still being tuned: rather
than trusting the auto-cut, you get a sequence whose cuts can be nudged by hand.

Two things make this format easy to get wrong:

1. Everything is in FRAMES, not seconds. A timeline built from rounded
   per-clip durations drifts against the source unless each clip's length is
   derived from its own rounded in/out points, which is what happens here.
2. NTSC rates are expressed as an integer timebase plus an ntsc flag - 29.97
   is <timebase>30</timebase><ntsc>TRUE</ntsc>, not <timebase>29.97</timebase>.
   Writing the decimal produces a sequence Premiere silently misinterprets.
"""
from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from models import Span

# NTSC rates: (true fps, integer timebase). A rate within tolerance of one of
# these is written as the timebase with the ntsc flag set.
_NTSC_RATES = ((24000 / 1001, 24), (30000 / 1001, 30), (60000 / 1001, 60))
_NTSC_TOLERANCE = 0.01


@dataclass(frozen=True)
class VideoMeta:
    fps: float
    width: int
    height: int
    duration: float


def timebase_and_ntsc(fps: float) -> tuple[int, bool]:
    """FCP7 expresses 29.97 as timebase 30 + ntsc TRUE, never as a decimal."""
    for true_rate, timebase in _NTSC_RATES:
        if abs(fps - true_rate) < _NTSC_TOLERANCE:
            return timebase, True
    return int(round(fps)), False


def probe_video_meta(video: str | Path) -> VideoMeta:
    """Frame rate, dimensions and duration, via ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffprobe not found on PATH. Install ffmpeg: winget install Gyan.FFmpeg")

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(video)],
        capture_output=True, text=True, check=True)

    fields: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()

    num, _, den = fields["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)

    return VideoMeta(fps=fps, width=int(fields["width"]),
                     height=int(fields["height"]),
                     duration=float(fields["duration"]))


def build_clips(segments: list[Span], fps: float) -> list[tuple[int, int, int, int]]:
    """(timeline_in, timeline_out, source_in, source_out), all in frames.

    Each clip's length comes from its own rounded source points, and the next
    clip starts exactly where the last one ended. Rounding a duration
    independently of the in/out points is what makes a timeline drift.
    """
    clips: list[tuple[int, int, int, int]] = []
    position = 0

    for seg in segments:
        src_in = int(round(seg.start * fps))
        src_out = int(round(seg.end * fps))
        length = src_out - src_in
        if length <= 0:
            continue
        clips.append((position, position + length, src_in, src_out))
        position += length

    return clips


def _rate(parent: ET.Element, fps: float) -> None:
    timebase, ntsc = timebase_and_ntsc(fps)
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _file_element(parent: ET.Element, video: Path, meta: VideoMeta,
                  full: bool) -> None:
    """The source file. Defined once, then referenced by id."""
    file_el = ET.SubElement(parent, "file", {"id": "file-1"})
    if not full:
        return

    ET.SubElement(file_el, "name").text = video.name
    # FCP7 convention; Premiere accepts the localhost form on every platform
    ET.SubElement(file_el, "pathurl").text = video.resolve().as_uri().replace(
        "file:///", "file://localhost/", 1)
    _rate(file_el, meta.fps)
    ET.SubElement(file_el, "duration").text = str(
        int(round(meta.duration * meta.fps)))

    media = ET.SubElement(file_el, "media")
    video_el = ET.SubElement(media, "video")
    chars = ET.SubElement(video_el, "samplecharacteristics")
    ET.SubElement(chars, "width").text = str(meta.width)
    ET.SubElement(chars, "height").text = str(meta.height)
    audio_el = ET.SubElement(media, "audio")
    ET.SubElement(audio_el, "channelcount").text = "2"


def _clipitem(track: ET.Element, index: int, clip: tuple[int, int, int, int],
              video: Path, meta: VideoMeta, is_audio: bool) -> None:
    tl_in, tl_out, src_in, src_out = clip
    suffix = "a" if is_audio else "v"

    item = ET.SubElement(track, "clipitem",
                         {"id": f"clipitem-{suffix}{index + 1}"})
    ET.SubElement(item, "name").text = video.name
    ET.SubElement(item, "enabled").text = "TRUE"
    ET.SubElement(item, "duration").text = str(src_out - src_in)
    _rate(item, meta.fps)
    ET.SubElement(item, "start").text = str(tl_in)
    ET.SubElement(item, "end").text = str(tl_out)
    ET.SubElement(item, "in").text = str(src_in)
    ET.SubElement(item, "out").text = str(src_out)

    # the file is spelled out on the first clip only; the rest reference it
    _file_element(item, video, meta, full=(index == 0 and not is_audio))

    if is_audio:
        source = ET.SubElement(item, "sourcetrack")
        ET.SubElement(source, "mediatype").text = "audio"
        ET.SubElement(source, "trackindex").text = "1"


def build_xml(video: str | Path, segments: list[Span], meta: VideoMeta,
              sequence_name: str = "Auto-Editor Cut") -> str:
    """An FCP7 xmeml sequence containing the surviving segments, in order."""
    video = Path(video)
    clips = build_clips(segments, meta.fps)
    if not clips:
        raise ValueError("cannot export XML: no surviving segments")

    total = clips[-1][1]

    xmeml = ET.Element("xmeml", {"version": "4"})
    sequence = ET.SubElement(xmeml, "sequence", {"id": "sequence-1"})
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(total)
    _rate(sequence, meta.fps)

    media = ET.SubElement(sequence, "media")

    video_media = ET.SubElement(media, "video")
    fmt = ET.SubElement(video_media, "format")
    chars = ET.SubElement(fmt, "samplecharacteristics")
    _rate(chars, meta.fps)
    ET.SubElement(chars, "width").text = str(meta.width)
    ET.SubElement(chars, "height").text = str(meta.height)

    video_track = ET.SubElement(video_media, "track")
    for i, clip in enumerate(clips):
        _clipitem(video_track, i, clip, video, meta, is_audio=False)

    audio_media = ET.SubElement(media, "audio")
    audio_track = ET.SubElement(audio_media, "track")
    for i, clip in enumerate(clips):
        _clipitem(audio_track, i, clip, video, meta, is_audio=True)

    ET.indent(xmeml, space="  ")
    body = ET.tostring(xmeml, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE xmeml>\n" + body + "\n")


def write_xml(video: str | Path, segments: list[Span], meta: VideoMeta,
              path: str | Path, sequence_name: str = "Auto-Editor Cut") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_xml(video, segments, meta, sequence_name),
                    encoding="utf-8")
    return path
