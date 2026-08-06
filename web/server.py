"""FastAPI routes. Translation only - the logic lives in web.jobs."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import EditorConfig
from render import render_video
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


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    """The source recording.

    FileResponse implements HTTP range requests, which is what makes seeking
    to a cut instant instead of refetching the whole file. Verified by
    tests/test_web_range.py rather than assumed.
    """
    job = _job_or_404(job_id)
    if not job.video.exists():
        raise HTTPException(status_code=404, detail="video not found")
    return FileResponse(job.video, media_type="video/mp4")


class RenderRequest(BaseModel):
    """Indices into report["cuts"] that should STILL be removed.

    Anything absent has been unchecked in the review screen and stays in the
    finished video.
    """
    cuts: list[int]


@app.post("/api/jobs/{job_id}/render")
def start_render(job_id: str, request: RenderRequest) -> dict:
    job = _job_or_404(job_id)
    if job.state is not JobState.READY:
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.state.value}, not ready to render")

    cfg = editor_config()
    keep = set(request.cuts)
    WORKER.submit(lambda: run_render(job, cfg, keep, render_fn=render_video))
    return {"state": "rendering"}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    job = _job_or_404(job_id)
    if job.output is None or not job.output.exists():
        raise HTTPException(status_code=404, detail="nothing rendered yet")
    return FileResponse(job.output, media_type="video/mp4",
                        filename=f"{job.video.stem}.edited.mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not STORE.delete(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    return {"deleted": job_id}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
