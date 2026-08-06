import io

import pytest
from fastapi.testclient import TestClient

from web import server

READY_REPORT = {
    "duration": 10.0,
    "mode": "subtractive",
    "summary": {"output_seconds": 8.0},
    "cuts": [
        {"start": 1.0, "end": 2.0, "reasons": ["silence"], "detail": "gap",
         "duration": 1.0},
        {"start": 4.0, "end": 5.0, "reasons": ["disfluency"], "detail": "the",
         "duration": 1.0},
    ],
    "lines": [],
    "adlibs": [],
    "segments": [],
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "run_analysis", lambda job, cfg: None)

    rendered = {}

    def fake_render(video, out, report, cfg):
        rendered["report"] = report
        out.write_bytes(b"rendered video")

    monkeypatch.setattr(server, "render_video", fake_render)
    client = TestClient(server.app)
    client.rendered = rendered
    return client


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
    job.report = dict(READY_REPORT)
    return job_id


def test_approving_every_cut_renders_the_expected_segments(client, ready_job):
    response = client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0, 1]})
    assert response.status_code == 200
    server.WORKER.join()

    segments = client.rendered["report"]["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [
        (0.0, 1.0), (2.0, 4.0), (5.0, 10.0)]


def test_unchecking_a_cut_puts_that_span_back(client, ready_job):
    client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0]})
    server.WORKER.join()

    segments = client.rendered["report"]["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [(0.0, 1.0), (2.0, 10.0)]


def test_the_job_reaches_done(client, ready_job):
    client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0, 1]})
    server.WORKER.join()
    assert client.get(f"/api/jobs/{ready_job}").json()["state"] == "done"


def test_rendering_before_ready_is_rejected(client):
    """The review gate, at the HTTP layer."""
    job_id = client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(b"\x00"), "video/mp4")},
        data={"script": "hello"}).json()["id"]

    response = client.post(f"/api/jobs/{job_id}/render", json={"cuts": []})
    assert response.status_code == 409
    assert "ready" in response.json()["detail"].lower()


def test_download_returns_the_rendered_file(client, ready_job):
    client.post(f"/api/jobs/{ready_job}/render", json={"cuts": [0, 1]})
    server.WORKER.join()

    response = client.get(f"/api/jobs/{ready_job}/download")
    assert response.status_code == 200
    assert response.content == b"rendered video"


def test_download_before_rendering_is_404(client, ready_job):
    assert client.get(f"/api/jobs/{ready_job}/download").status_code == 404


def test_delete_removes_the_job(client, ready_job):
    assert client.delete(f"/api/jobs/{ready_job}").status_code == 200
    assert client.get(f"/api/jobs/{ready_job}").status_code == 404


def test_deleting_an_unknown_job_is_404(client):
    assert client.delete("/api/jobs/nope").status_code == 404
