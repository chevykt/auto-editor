import io

import pytest
from fastapi.testclient import TestClient

from web import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A server whose work directory and pipeline are both under test control."""
    store = server.JobStore(tmp_path / "work")
    monkeypatch.setattr(server, "STORE", store)

    def instant(job, cfg):
        job.advance(server.JobState.PROBING)
        job.advance(server.JobState.TRANSCRIBING)
        job.advance(server.JobState.ANALYSING)
        job.report = {
            "duration": 10.0,
            "mode": "subtractive",
            "summary": {"output_seconds": 8.0},
            "cuts": [
                {"start": 1.0, "end": 2.0, "reasons": ["silence"],
                 "detail": "gap", "duration": 1.0},
                {"start": 4.0, "end": 5.0, "reasons": ["disfluency"],
                 "detail": "the", "duration": 1.0},
            ],
            "lines": [],
            "adlibs": [],
            "segments": [{"start": 0.0, "end": 1.0}],
        }
        job.progress = 1.0
        job.advance(server.JobState.READY)

    monkeypatch.setattr(server, "run_analysis", instant)
    return TestClient(server.app)


def upload(client, name="raw.mp4", data=b"\x00" * 2048, script="hello world"):
    return client.post("/api/jobs",
                       files={"video": (name, io.BytesIO(data), "video/mp4")},
                       data={"script": script})


def test_upload_creates_a_job(client):
    response = upload(client)
    assert response.status_code == 200
    assert response.json()["id"]


def test_the_uploaded_bytes_land_on_disk(client):
    job_id = upload(client, data=b"abcdef").json()["id"]
    job = server.STORE.get(job_id)
    assert job.video.read_bytes() == b"abcdef"


def test_a_large_upload_is_written_in_chunks_not_buffered(client):
    """200 MB recordings must not be held in memory."""
    payload = b"x" * (3 * 1024 * 1024 + 17)
    job_id = upload(client, data=payload).json()["id"]
    assert server.STORE.get(job_id).video.stat().st_size == len(payload)
    assert server.UPLOAD_CHUNK <= 1024 * 1024


def test_an_empty_script_is_rejected_before_a_job_exists(client):
    response = upload(client, script="   ")
    assert response.status_code == 400
    assert "script" in response.json()["detail"].lower()


def test_polling_returns_state_and_report(client):
    job_id = upload(client).json()["id"]
    server.WORKER.join()

    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "ready"
    assert body["progress"] == 1.0
    assert len(body["report"]["cuts"]) == 2


def test_polling_an_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_the_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()


def test_static_files_must_be_revalidated(client):
    """A cached app.js looks exactly like a fix that did not take effect.

    The file is right on disk, the server sends the right bytes, and the
    browser runs the old script anyway. Found by a screenshot showing stale
    UI text after a verified-correct deploy.
    """
    for path in ("/static/app.js", "/static/style.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers.get("cache-control") == "no-cache", path


def test_the_index_page_is_not_cached_either(client):
    """index.html is served by a route, not StaticFiles, and is re-read from
    disk on every request - so it cannot go stale."""
    assert client.get("/").status_code == 200


def test_a_queued_job_reports_how_many_are_ahead(client):
    """Transcription is serial, so a job can sit at "queued" through no fault
    of its own. Without a count there is no way to tell a busy queue from a
    wedged server - which is exactly how it looked when one did wedge."""
    job_id = upload(client).json()["id"]
    body = client.get(f"/api/jobs/{job_id}").json()
    assert "waiting" in body


def test_waiting_is_zero_once_a_job_is_past_the_queue(client):
    job_id = upload(client).json()["id"]
    server.WORKER.join()
    assert client.get(f"/api/jobs/{job_id}").json()["waiting"] == 0
