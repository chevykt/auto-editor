"""Recording the browser: frames with real timestamps, turned into a 30 fps video."""
import json
import subprocess
from pathlib import Path

import pytest

from tutorial.capture import Capture, concat_lines
from tutorial.plan import Beat, Plan, Step
from tutorial.runner import open_browser, run_steps, start

PAGE = (Path(__file__).parent / "fixtures" / "tutorial_page.html").as_uri()


def test_concat_list_gives_each_frame_its_own_duration(tmp_path):
    a, b, c = (tmp_path / n for n in ("a.png", "b.png", "c.png"))
    assert concat_lines([(10.0, a), (10.5, b), (11.25, c)], end=12.0) == [
        f"file '{a.as_posix()}'", "duration 0.5",
        f"file '{b.as_posix()}'", "duration 0.75",
        f"file '{c.as_posix()}'", "duration 0.75",
        f"file '{c.as_posix()}'"]


def test_concat_list_needs_a_frame():
    with pytest.raises(ValueError, match="no frames"):
        concat_lines([], end=1.0)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    return data["streams"][0], float(data["format"]["duration"])


@pytest.mark.smoke
def test_capture_is_1080p_30fps_and_spans_the_recording(tmp_path):
    plan = Plan("t", PAGE, (Beat("b", ()),))
    with open_browser(headless=True, profile=tmp_path / "profile") as page:
        start(page, plan)
        with Capture(page, tmp_path / "frames") as capture:
            run_steps(page, plan, [
                (0, 0, Step("click", "#new", None, None)),
                (0, 1, Step("type", "label=Name", "hello", None)),
                (0, 2, Step("click", "#email", None, None)),
            ], tmp_path / "shots")

    video, t0 = capture.to_video(tmp_path / "capture.mp4")
    stream, duration = probe(video)
    assert (stream["width"], stream["height"]) == (1920, 1080)
    assert stream["r_frame_rate"] == "30/1"
    assert t0 == capture.frames[0][0]
    assert duration == pytest.approx(capture.ended - t0, abs=0.1)
    assert len(capture.frames) > 20, "cursor motion produced frames"
