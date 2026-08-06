import io
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from web import server
from xml_export import VideoMeta


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "run_analysis", lambda job, cfg: None)
    monkeypatch.setattr(
        server, "probe_video_meta",
        lambda video: VideoMeta(fps=25.0, width=1920, height=1080,
                                duration=10.0))
    return TestClient(server.app)


@pytest.fixture
def ready_job(client):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00" * 64), "video/mp4")},
        data={"script": "hello"}).json()["id"]

    job = server.STORE.get(job_id)
    for state in (server.JobState.PROBING, server.JobState.TRANSCRIBING,
                  server.JobState.ANALYSING, server.JobState.READY):
        job.advance(state)
    job.report = {
        "duration": 10.0,
        "cuts": [{"start": 1.0, "end": 2.0, "reasons": ["silence"],
                  "detail": "gap", "duration": 1.0}],
        "segments": [],
    }
    return job_id


def test_xml_reflects_the_approved_cut_list(client, ready_job):
    response = client.post(f"/api/jobs/{ready_job}/xml", json={"cuts": [0]})
    assert response.status_code == 200

    root = ET.fromstring(response.text)
    clips = root.findall("./sequence/media/video/track/clipitem")
    assert len(clips) == 2, "one cut splits the timeline into two clips"


def test_unchecking_the_cut_yields_a_single_clip(client, ready_job):
    response = client.post(f"/api/jobs/{ready_job}/xml", json={"cuts": []})
    root = ET.fromstring(response.text)
    assert len(root.findall("./sequence/media/video/track/clipitem")) == 1


def test_xml_before_ready_is_rejected(client):
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00"), "video/mp4")},
        data={"script": "hello"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/xml",
                       json={"cuts": []}).status_code == 409
