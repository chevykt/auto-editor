"""End-to-end checks against real media. Slow; excluded from the default run.

Run with: python -m pytest tests/test_smoke_endtoend.py -v -m smoke
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from config import EditorConfig
from render import render_video
from transcribe import probe_duration

pytestmark = pytest.mark.smoke

HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A 10-second test clip: colour bars plus a sine tone."""
    path = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        str(path),
    ], capture_output=True, check=True)
    return path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_probe_duration_reads_real_media(clip):
    assert 9.5 < probe_duration(clip) < 10.5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_removes_the_cut_spans(clip, tmp_path):
    # keep 0-2s and 6-10s: a 6-second output from a 10-second source
    report = {"duration": 10.0,
              "segments": [{"start": 0.0, "end": 2.0},
                           {"start": 6.0, "end": 10.0}]}
    out = tmp_path / "out.mp4"
    render_video(clip, out, report, EditorConfig())

    assert out.exists()
    assert 5.5 < probe_duration(out) < 6.5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_audio_and_video_stay_in_sync_across_many_cuts(clip, tmp_path):
    """The afade-not-acrossfade decision, verified on real media.

    Fifty cuts with fades on. A true crossfade would shorten audio by the
    fade length at every cut; here the audio and video stream durations must
    still agree.
    """
    segments = [{"start": i * 0.2, "end": i * 0.2 + 0.1} for i in range(50)]
    out = tmp_path / "sync.mp4"
    render_video(clip, out, {"duration": 10.0, "segments": segments},
                 EditorConfig(crossfade_ms=75))

    def stream_duration(kind: str) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", kind,
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
            capture_output=True, text=True, check=True)
        return float(r.stdout.strip().splitlines()[0])

    assert abs(stream_duration("v:0") - stream_duration("a:0")) < 0.2


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_survives_a_graph_past_the_windows_arg_limit(clip, tmp_path):
    # 300 segments produces a filter string well over 8191 characters
    segments = [{"start": round(i * 0.03, 3), "end": round(i * 0.03 + 0.02, 3)}
                for i in range(300)]
    out = tmp_path / "many.mp4"
    render_video(clip, out, {"duration": 10.0, "segments": segments},
                 EditorConfig(crossfade_ms=0))
    assert out.exists()
