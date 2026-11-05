"""DaVinci Resolve timeline export (FCPXML 1.9)."""
from fractions import Fraction
from xml.etree import ElementTree as ET

import pytest

from models import Span
from xml_export import VideoMeta, build_fcpxml, write_fcpxml

META = VideoMeta(fps=60.0, width=1920, height=1080, duration=100.0)
NTSC = VideoMeta(fps=30000 / 1001, width=1920, height=1080, duration=100.0)


def seconds(value: str) -> Fraction:
    return Fraction(value.removesuffix("s"))


def spine(xml: str) -> list[ET.Element]:
    return list(ET.fromstring(xml).find(".//spine"))


def test_document_is_fcpxml_1_9():
    xml = build_fcpxml("raw.mp4", [Span(0.0, 2.0)], META)
    assert "<!DOCTYPE fcpxml>" in xml
    assert ET.fromstring(xml).get("version") == "1.9"


def test_each_segment_is_one_clip_laid_end_to_end():
    clips = spine(build_fcpxml("raw.mp4", [Span(1.0, 3.0), Span(10.0, 11.0)], META))
    assert [c.tag for c in clips] == ["asset-clip", "asset-clip"]
    assert seconds(clips[0].get("offset")) == 0
    assert seconds(clips[0].get("start")) == 1
    assert seconds(clips[0].get("duration")) == 2
    assert seconds(clips[1].get("offset")) == 2, "starts where the first ended"
    assert seconds(clips[1].get("start")) == 10


def test_the_whole_recording_is_the_asset_so_edges_can_be_dragged_out():
    root = ET.fromstring(build_fcpxml("raw.mp4", [Span(40.0, 41.0)], META))
    asset = root.find(".//asset")
    assert seconds(asset.get("duration")) == 100
    assert asset.get("hasVideo") == asset.get("hasAudio") == "1"


def test_every_clip_uses_the_one_asset():
    root = ET.fromstring(build_fcpxml("raw.mp4", [Span(0.0, 1.0), Span(2.0, 3.0)], META))
    refs = {c.get("ref") for c in root.find(".//spine")}
    assert refs == {root.find(".//asset").get("id")}


def test_the_media_path_is_an_absolute_file_uri(tmp_path):
    video = tmp_path / "my take.mp4"
    root = ET.fromstring(build_fcpxml(video, [Span(0.0, 1.0)], META))
    src = root.find(".//media-rep").get("src")
    assert src.startswith("file:///")
    assert src.endswith("my%20take.mp4")


def test_times_sit_on_frame_boundaries_at_an_ntsc_rate():
    clips = spine(build_fcpxml("raw.mp4", [Span(1.0, 3.0), Span(7.3, 9.1)], NTSC))
    frame = Fraction(1001, 30000)
    for clip in clips:
        for attr in ("offset", "start", "duration"):
            assert (seconds(clip.get(attr)) / frame).denominator == 1


def test_the_format_declares_the_rate_exactly():
    root = ET.fromstring(build_fcpxml("raw.mp4", [Span(0.0, 1.0)], NTSC))
    assert seconds(root.find(".//format").get("frameDuration")) == Fraction(1001, 30000)


def test_the_sequence_ends_where_the_last_clip_ends():
    root = ET.fromstring(build_fcpxml("raw.mp4", [Span(0.0, 2.0), Span(5.0, 6.0)], META))
    assert seconds(root.find(".//sequence").get("duration")) == 3


def test_reordered_segments_keep_their_order():
    clips = spine(build_fcpxml("raw.mp4", [Span(10.0, 11.0), Span(1.0, 2.0)], META))
    assert seconds(clips[0].get("start")) == 10
    assert seconds(clips[1].get("start")) == 1


def test_no_segments_raises():
    with pytest.raises(ValueError, match="no surviving segments"):
        build_fcpxml("raw.mp4", [], META)


def test_a_special_character_in_the_name_is_escaped(tmp_path):
    root = ET.fromstring(build_fcpxml(tmp_path / "q&a.mp4", [Span(0.0, 1.0)], META))
    assert root.find(".//asset").get("name") == "q&a.mp4"


def test_write_fcpxml_creates_the_file(tmp_path):
    path = write_fcpxml("raw.mp4", [Span(0.0, 2.0)], META, tmp_path / "cut.fcpxml")
    ET.fromstring(path.read_text(encoding="utf-8"))
