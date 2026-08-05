"""Premiere-compatible XML timeline export (FCP7 xmeml)."""
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from models import Span
from xml_export import (VideoMeta, build_clips, build_xml, timebase_and_ntsc,
                        write_xml)

META = VideoMeta(fps=25.0, width=1920, height=1080, duration=100.0)


# --- frame rate handling -------------------------------------------------

def test_integer_rates_are_not_ntsc():
    assert timebase_and_ntsc(25.0) == (25, False)
    assert timebase_and_ntsc(30.0) == (30, False)


def test_ntsc_rates_become_an_integer_timebase_plus_a_flag():
    """29.97 must be written as timebase 30 + ntsc TRUE, never as a decimal."""
    assert timebase_and_ntsc(30000 / 1001) == (30, True)
    assert timebase_and_ntsc(24000 / 1001) == (24, True)
    assert timebase_and_ntsc(60000 / 1001) == (60, True)


def test_a_rate_near_but_not_ntsc_is_treated_as_integer():
    assert timebase_and_ntsc(29.5) == (30, False)


# --- frame conversion ----------------------------------------------------

def test_seconds_convert_to_frames():
    clips = build_clips([Span(1.0, 3.0)], fps=25.0)
    assert clips == [(0, 50, 25, 75)]


def test_clips_are_laid_end_to_end_on_the_timeline():
    clips = build_clips([Span(0.0, 2.0), Span(10.0, 11.0)], fps=25.0)
    assert clips[0][:2] == (0, 50)
    assert clips[1][:2] == (50, 75), "second clip starts where the first ended"


def test_timeline_length_equals_the_sum_of_clip_lengths():
    """Rounding each clip's length from its own in/out points, rather than
    rounding the duration separately, is what keeps a timeline from drifting."""
    segments = [Span(i * 0.7, i * 0.7 + 0.33) for i in range(50)]
    clips = build_clips(segments, fps=29.97)
    assert clips[-1][1] == sum(c[3] - c[2] for c in clips)


def test_out_of_order_segments_keep_their_order():
    """Reorder mode produces segments that jump backwards in the source."""
    clips = build_clips([Span(10.0, 11.0), Span(1.0, 2.0)], fps=25.0)
    assert clips[0][2] == 250, "first clip sources from 10s"
    assert clips[1][2] == 25, "second clip sources from 1s"
    assert clips[1][0] == 25, "but sits second on the timeline"


def test_zero_length_segments_are_skipped():
    assert build_clips([Span(1.0, 1.0)], fps=25.0) == []


# --- document structure --------------------------------------------------

def parse(xml: str) -> ET.Element:
    return ET.fromstring(xml)


def test_document_is_well_formed_xmeml():
    xml = build_xml("raw.mp4", [Span(0.0, 2.0)], META)
    assert xml.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<!DOCTYPE xmeml>" in xml
    root = parse(xml)
    assert root.tag == "xmeml"
    assert root.get("version") == "4"


def test_both_a_video_and_an_audio_track_are_written():
    root = parse(build_xml("raw.mp4", [Span(0.0, 2.0), Span(5.0, 6.0)], META))
    video_clips = root.findall("./sequence/media/video/track/clipitem")
    audio_clips = root.findall("./sequence/media/audio/track/clipitem")
    assert len(video_clips) == 2
    assert len(audio_clips) == 2


def test_audio_clips_declare_their_source_track():
    root = parse(build_xml("raw.mp4", [Span(0.0, 2.0)], META))
    audio = root.find("./sequence/media/audio/track/clipitem")
    assert audio.find("./sourcetrack/mediatype").text == "audio"


def test_the_source_file_is_defined_once_and_referenced_after():
    """Repeating the full file block on every clip is what makes Premiere
    import the same media as several distinct assets."""
    root = parse(build_xml("raw.mp4", [Span(0.0, 1.0), Span(2.0, 3.0),
                                       Span(4.0, 5.0)], META))
    files = root.findall(".//file")
    assert len(files) == 6, "one per clip across both tracks"

    defined = [f for f in files if f.find("pathurl") is not None]
    assert len(defined) == 1, "exactly one carries the definition"
    assert all(f.get("id") == "file-1" for f in files), "all share one id"


def test_pathurl_is_an_absolute_localhost_file_url(tmp_path):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    root = parse(build_xml(video, [Span(0.0, 1.0)], META))
    url = root.find(".//file/pathurl").text
    assert url.startswith("file://localhost/")
    assert url.endswith("raw.mp4")


def test_sequence_duration_matches_the_final_clip():
    root = parse(build_xml("raw.mp4", [Span(0.0, 2.0), Span(5.0, 6.0)], META))
    assert root.find("./sequence/duration").text == "75"


def test_ntsc_flag_reaches_the_document():
    ntsc_meta = VideoMeta(fps=30000 / 1001, width=1920, height=1080,
                          duration=100.0)
    root = parse(build_xml("raw.mp4", [Span(0.0, 2.0)], ntsc_meta))
    rate = root.find("./sequence/rate")
    assert rate.find("timebase").text == "30"
    assert rate.find("ntsc").text == "TRUE"


def test_dimensions_reach_the_sequence_format():
    root = parse(build_xml("raw.mp4", [Span(0.0, 2.0)], META))
    chars = root.find("./sequence/media/video/format/samplecharacteristics")
    assert chars.find("width").text == "1920"
    assert chars.find("height").text == "1080"


def test_a_name_with_xml_special_characters_is_escaped(tmp_path):
    # & is legal in a Windows filename and is the case that breaks XML
    video = tmp_path / "tutorial q&a take 2.mp4"
    xml = build_xml(video, [Span(0.0, 1.0)], META)
    # parsing is the real check: a raw & makes the document malformed
    root = parse(xml)
    assert "q&a" in root.find(".//file/name").text, "round-trips intact"
    assert "&amp;" in xml, "and is escaped on the wire"


def test_no_segments_raises():
    with pytest.raises(ValueError, match="no surviving segments"):
        build_xml("raw.mp4", [], META)


def test_write_xml_creates_the_file(tmp_path):
    path = write_xml("raw.mp4", [Span(0.0, 2.0)], META, tmp_path / "cut.xml")
    assert path.exists()
    parse(path.read_text(encoding="utf-8"))
