"""Full round trip through the API against real media.

Transcription is stubbed with a hand-built word list - Whisper's own behaviour
is covered by the engine's tests, and downloading a model here would make the
suite unusable. Everything else is real: real ffmpeg, real analysis, real
render.
"""
import io
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models import Word
from transcribe import probe_duration
from web import server

pytestmark = pytest.mark.smoke

HAS_FFMPEG = shutil.which("ffmpeg") is not None

SCRIPT = "Click the blue button.\n\nThat is everything you need."


def spoken():
    """A take with a stutter and a long pause, both of which should be found."""
    def run(text, start, dur=0.3):
        return [Word(t, round(start + i * dur, 3), round(start + i * dur + dur, 3))
                for i, t in enumerate(text.split())]

    return (run("click the the blue button", 0.0)
            + run("that is everything you need", 6.0))


@pytest.fixture
def clip(tmp_path: Path) -> bytes:
    path = tmp_path / "clip.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path),
    ], capture_output=True, check=True)
    return path.read_bytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STORE", server.JobStore(tmp_path / "work"))
    monkeypatch.setattr(
        "web.jobs.transcribe",
        lambda video, cfg, on_progress=None: (spoken(), 10.0))
    return TestClient(server.app)


def start(client, clip, script=SCRIPT):
    return client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(clip), "video/mp4")},
        data={"script": script}).json()["id"]


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_upload_review_render_download(client, clip, tmp_path):
    job_id = start(client, clip)
    server.WORKER.join()

    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["state"] == "ready", state.get("error")

    report = state["report"]
    assert report["cuts"], "the stutter and the gap should both be found"

    every = list(range(len(report["cuts"])))
    assert client.post(f"/api/jobs/{job_id}/render",
                       json={"cuts": every}).status_code == 200
    server.WORKER.join()

    assert client.get(f"/api/jobs/{job_id}").json()["state"] == "done"

    downloaded = client.get(f"/api/jobs/{job_id}/download")
    assert downloaded.status_code == 200

    out = tmp_path / "out.mp4"
    out.write_bytes(downloaded.content)
    assert probe_duration(out) < 10.0, "the render must be shorter than the source"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_unchecking_every_cut_renders_the_full_length(client, clip, tmp_path):
    """Rejecting the whole cut list is a real choice, not an error."""
    job_id = start(client, clip)
    server.WORKER.join()

    client.post(f"/api/jobs/{job_id}/render", json={"cuts": []})
    server.WORKER.join()

    out = tmp_path / "full.mp4"
    out.write_bytes(client.get(f"/api/jobs/{job_id}/download").content)
    assert probe_duration(out) > 9.0, "rejecting every cut keeps the recording"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_the_stutter_is_found_and_is_a_judgment_call(client, clip):
    job_id = start(client, clip)
    server.WORKER.join()

    report = client.get(f"/api/jobs/{job_id}").json()["report"]
    judged = [c for c in report["cuts"] if c["reasons"] != ["silence"]]
    assert any("the" in c["detail"] for c in judged), \
        "the doubled 'the' must appear as a reviewable cut, not silently"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_xml_export_matches_the_approved_cuts(client, clip):
    from xml.etree import ElementTree as ET

    job_id = start(client, clip)
    server.WORKER.join()

    response = client.post(f"/api/jobs/{job_id}/xml", json={"cuts": []})
    assert response.status_code == 200
    root = ET.fromstring(response.text)
    clips = root.findall("./sequence/media/video/track/clipitem")
    assert len(clips) == 1, "no cuts approved means one unbroken clip"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_deleting_a_job_removes_its_files(client, clip):
    job_id = start(client, clip)
    server.WORKER.join()
    directory = server.STORE.get(job_id).directory

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert not directory.exists()
