"""FastAPI routes. Translation only - the logic lives in web.jobs."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import EditorConfig
from web.jobs import JobState, JobStore, Worker, run_analysis, run_render

STATIC = Path(__file__).resolve().parent / "static"
WORK = Path(__file__).resolve().parent / "work"

# 1 MiB at a time: a 200 MB upload must never be buffered whole in memory.
UPLOAD_CHUNK = 1024 * 1024

app = FastAPI(title="Auto-Editor")
STORE = JobStore(WORK)
WORKER = Worker()


def editor_config() -> EditorConfig:
    """Settings for a UI run.

    verbatim is on because Whisper otherwise tidies away the stutters and
    restarts the review screen exists to show.
    """
    return EditorConfig(verbatim=True)


def _job_or_404(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...),
                     script: str = Form(...)) -> dict:
    if not script.strip():
        raise HTTPException(status_code=400,
                            detail="script is empty - paste your script")

    job = STORE.create(video.filename or "upload.mp4", script)

    with job.video.open("wb") as handle:
        while chunk := await video.read(UPLOAD_CHUNK):
            handle.write(chunk)

    cfg = editor_config()
    WORKER.submit(lambda: run_analysis(job, cfg))
    return {"id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_or_404(job_id).as_dict()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
