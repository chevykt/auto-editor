"""Range support is load-bearing, not an optimisation.

Clicking a cut seeks the video element to that timestamp. Without ranges the
browser refetches the whole 200 MB file on every seek, so the central feature
of the review screen degrades silently rather than failing loudly. That is
exactly the kind of thing worth asserting.
"""
import io

import pytest
from fastapi.testclient import TestClient

from web import server

PAYLOAD = bytes(range(256)) * 400          # 102400 bytes


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)
    monkeypatch.setattr(server, "run_analysis", lambda job, cfg: None)
    return TestClient(server.app)


@pytest.fixture
def job_id(client):
    return client.post(
        "/api/jobs",
        files={"video": ("raw.mp4", io.BytesIO(PAYLOAD), "video/mp4")},
        data={"script": "hello"}).json()["id"]


def test_the_whole_file_is_served_without_a_range_header(client, job_id):
    response = client.get(f"/api/jobs/{job_id}/video")
    assert response.status_code == 200
    assert len(response.content) == len(PAYLOAD)


def test_the_server_advertises_range_support(client, job_id):
    response = client.get(f"/api/jobs/{job_id}/video")
    assert response.headers.get("accept-ranges") == "bytes"


def test_a_range_request_returns_206_with_only_that_slice(client, job_id):
    response = client.get(f"/api/jobs/{job_id}/video",
                          headers={"Range": "bytes=1000-1999"})
    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 1000-1999/{len(PAYLOAD)}"
    assert response.content == PAYLOAD[1000:2000]


def test_an_open_ended_range_runs_to_the_end(client, job_id):
    start = len(PAYLOAD) - 500
    response = client.get(f"/api/jobs/{job_id}/video",
                          headers={"Range": f"bytes={start}-"})
    assert response.status_code == 206
    assert response.content == PAYLOAD[start:]


def test_video_for_an_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope/video").status_code == 404
