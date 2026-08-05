import pytest

from config import EditorConfig
from models import Word
from web.jobs import (JobState, JobStore, Worker, approved_segments,
                      run_analysis)


def fake_transcribe(words, duration):
    def _fn(video, cfg, on_progress=None):
        if on_progress:
            on_progress(1.0)
        return words, duration
    return _fn


def fake_analyse(report):
    def _fn(words, script, duration, cfg, video, split_sentences=False):
        return report
    return _fn


REPORT = {
    "duration": 10.0,
    "cuts": [
        {"start": 1.0, "end": 2.0, "reasons": ["silence"], "detail": "gap"},
        {"start": 4.0, "end": 5.0, "reasons": ["disfluency"], "detail": "the"},
    ],
    "segments": [],
}


def test_the_pipeline_stops_at_ready(tmp_path):
    """The review gate. Analysis must never render on its own."""
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")

    run_analysis(job, EditorConfig(),
                 transcribe_fn=fake_transcribe([Word("hi", 0.0, 0.2)], 10.0),
                 analyse_fn=fake_analyse(REPORT))

    assert job.state is JobState.READY
    assert job.report == REPORT


def test_progress_reaches_the_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")

    def transcribing(video, cfg, on_progress=None):
        on_progress(0.5)
        return [Word("hi", 0.0, 0.2)], 10.0

    run_analysis(job, EditorConfig(), transcribe_fn=transcribing,
                 analyse_fn=fake_analyse(REPORT))
    assert job.progress == 1.0, "completes at 1.0 once analysed"


def test_a_transcription_failure_fails_the_job_with_a_message(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")

    def broken(video, cfg, on_progress=None):
        raise RuntimeError("ffprobe not found on PATH")

    run_analysis(job, EditorConfig(), transcribe_fn=broken,
                 analyse_fn=fake_analyse(REPORT))

    assert job.state is JobState.FAILED
    assert "ffprobe" in job.error


def test_keeping_every_cut_matches_the_engines_own_segments():
    segments = approved_segments(REPORT, {0, 1})
    assert [(s.start, s.end) for s in segments] == [
        (0.0, 1.0), (2.0, 4.0), (5.0, 10.0)]


def test_unchecking_a_cut_puts_that_span_back():
    segments = approved_segments(REPORT, {0})
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.0), (2.0, 10.0)]


def test_unchecking_everything_keeps_the_whole_recording():
    segments = approved_segments(REPORT, set())
    assert [(s.start, s.end) for s in segments] == [(0.0, 10.0)]


def test_an_out_of_range_index_is_ignored():
    segments = approved_segments(REPORT, {0, 99})
    assert [(s.start, s.end) for s in segments] == [(0.0, 1.0), (2.0, 10.0)]


def test_create_makes_a_directory_per_job(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")
    assert job.directory.is_dir()
    assert job.video.name == "raw.mp4"
    assert store.get(job.id) is job


def test_a_path_in_the_filename_cannot_escape_the_job_directory(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("../../evil.mp4", "hello")
    assert job.video.name == "evil.mp4"
    assert job.video.parent.parent == tmp_path


def test_delete_removes_the_job_and_its_files(tmp_path):
    store = JobStore(tmp_path)
    job = store.create("raw.mp4", "hello")
    job.video.write_bytes(b"x")

    assert store.delete(job.id) is True
    assert store.get(job.id) is None
    assert not job.directory.exists()


def test_deleting_an_unknown_job_is_false(tmp_path):
    assert JobStore(tmp_path).delete("nope") is False


def test_the_worker_runs_submitted_work():
    done = []
    worker = Worker()
    worker.submit(lambda: done.append(1))
    worker.join()
    assert done == [1]


def test_the_worker_survives_a_failing_job():
    done = []
    worker = Worker()
    worker.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    worker.submit(lambda: done.append(2))
    worker.join()
    assert done == [2], "a failed job must not stop the queue"
