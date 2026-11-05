"""Building the final video from the fitted pieces, and the DaVinci timeline."""
import json
import subprocess
from xml.etree import ElementTree as ET

import pytest

from tutorial.assemble import assemble, frame_counts, pieces_graph, write_timeline
from tutorial.fit import Hold, Play


def test_frame_counts_come_from_the_running_total_so_rounding_never_drifts():
    pieces = [Hold(0, 0.51)] * 30            # 15.3 s in 0.51 s steps
    counts = frame_counts(pieces, fps=30)
    assert sum(counts) == round(15.3 * 30)
    assert set(counts) <= {15, 16}


def test_graph_has_one_chain_per_piece_and_a_video_only_concat():
    g = pieces_graph([Play(0, 2, 1), Hold(2, 1.5), Play(2, 6, 2.0)])
    assert g.count("[0:v]trim=") == 3
    assert "setpts=(PTS-STARTPTS)/2.0" in g
    assert "trim=end_frame=45" in g, "the 1.5 s hold is exactly 45 frames"
    assert "concat=n=3:v=1:a=0[outv]" in g


def test_no_pieces_is_an_error():
    with pytest.raises(ValueError):
        pieces_graph([])


def probe(path, stream):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries",
         "stream=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["streams"][0]["duration"])


def make(tmp_path, video_s, audio_s):
    video, voice = tmp_path / "capture.mp4", tmp_path / "voice.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=1920x1080:rate=30:duration={video_s}",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={audio_s}", str(voice)], check=True)
    return video, voice


@pytest.mark.smoke
def test_assembled_video_matches_the_voice_length(tmp_path):
    video, voice = make(tmp_path, 5, 4)
    out = assemble(video, [Play(0, 2, 1), Hold(2, 1.0), Play(2, 4, 2.0)], voice,
                   tmp_path / "final.mp4")
    assert probe(out, "v:0") == pytest.approx(4.0, abs=1 / 30)
    assert probe(out, "a:0") == pytest.approx(4.0, abs=1 / 30)


@pytest.mark.smoke
def test_a_hold_past_the_end_of_the_capture_shows_its_last_frame(tmp_path):
    video, voice = make(tmp_path, 2, 3)
    out = assemble(video, [Play(0, 2, 1), Hold(2.6, 1.0)], voice, tmp_path / "final.mp4")
    assert probe(out, "v:0") == pytest.approx(3.0, abs=1 / 30)


@pytest.mark.smoke
def test_timeline_has_one_clip_per_beat(tmp_path):
    video, _ = make(tmp_path, 6, 1)
    path = write_timeline(video, ((0, 2), (2, 4.5), (4.5, 6)), tmp_path / "final.fcpxml")
    clips = ET.fromstring(path.read_text(encoding="utf-8")).find(".//spine")
    assert len(clips) == 3
