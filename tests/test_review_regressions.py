"""Defects found by a full code review. Each test fails without its fix."""
import pytest
from fastapi.testclient import TestClient

from aligner import align_script
from config import EditorConfig
from conftest import words
from disfluency import detect_disfluencies
from edit import analyse, config_from_args, main, parse_args
from models import CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine, Span
from render import render_video
from report import build_report, load_report, write_json
from web.jobs import Worker


# --- the script itself decides what counts as a stumble --------------------

@pytest.mark.parametrize("phrase", ["the theory", "and android", "for format",
                                    "add additional", "not nothing"])
def test_a_scripted_word_pair_is_not_a_truncated_restart(phrase):
    """These look like fragment + full word, and are ordinary English."""
    ws = words(f"explain {phrase} now")
    assert detect_disfluencies(ws, 0, len(ws), EditorConfig(),
                               script=f"explain {phrase} now") == []


def test_a_real_restart_is_still_cut_when_the_script_lacks_it():
    ws = words("open the config configuration file")
    cuts = detect_disfluencies(ws, 0, len(ws), EditorConfig(),
                               script="open the configuration file")
    assert len(cuts) == 1


def test_a_scripted_double_word_survives():
    ws = words("I had had enough of it")
    assert detect_disfluencies(ws, 0, len(ws), EditorConfig(),
                               script="I had had enough of it") == []


def test_a_real_double_word_is_still_cut():
    ws = words("from the the top left menu")
    cuts = detect_disfluencies(ws, 0, len(ws), EditorConfig(),
                               script="from the top left menu")
    assert len(cuts) == 1


def test_a_phrase_the_script_says_twice_survives():
    text = "click here and then click here again"
    ws = words(text)
    assert detect_disfluencies(ws, 0, len(ws), EditorConfig(), script=text) == []


def test_analyse_hands_each_line_to_the_detectors():
    text = "the theory is simple and it works every single time"
    ws = words(text)
    report = analyse(ws, text, ws[-1].end, EditorConfig(), "v.mp4")
    assert report["cuts"] == []


# --- retakes must not swallow what was said between attempts ---------------

def test_talk_between_two_takes_is_not_cut_with_the_losing_take():
    ws = words("click the red button here is why that matters a lot "
               "click the blue button",
               gaps=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    _, cuts = align_script(ws, [ScriptLine(0, "click the blue button")],
                           EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].end == ws[3].end, "only the losing take itself goes"


# --- every knob a config file accepts must take effect ---------------------

def test_every_config_key_reaches_the_editor_config(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('video_crf = 23\nretake_max_gap_s = 5.0\n'
                    'suspect_ratio = 9.0\nclean_stutters = false\n',
                    encoding="utf-8")
    cfg = config_from_args(parse_args(["--config", str(toml), "--video", "v"]))
    assert cfg.video_crf == 23
    assert cfg.retake_max_gap_s == 5.0
    assert cfg.suspect_ratio == 9.0
    assert cfg.clean_stutters is False


# --- suspects are not cuts until approved ----------------------------------

def _report_with_suspect():
    ws = words("alpha bravo charlie delta")
    suspect = CutSpan(ws[1].start, ws[1].end, {CutReason.SUSPECT}, "long token")
    silence = CutSpan(ws[3].end, 5.0, {CutReason.SILENCE}, "tail")
    match = LineMatch(0, Span(ws[0].start, ws[0].end), 1.0, (0, 1), MatchStatus.OK)
    return build_report([ScriptLine(0, "alpha")], [match], [suspect, silence],
                        ws, 5.0, EditorConfig(), "v.mp4")


def test_an_unapproved_suspect_is_not_counted_as_a_cut():
    assert _report_with_suspect()["summary"]["cut_count"] == 1


def test_a_word_under_an_unapproved_suspect_is_still_listed_as_kept():
    assert "bravo" in _report_with_suspect()["adlibs"][0]["text"]


# --- a hand-edited report ---------------------------------------------------

@pytest.fixture
def captured_render(monkeypatch):
    seen = {}
    monkeypatch.setattr("render.render_video",
                        lambda video, out, report, cfg: seen.update(report))
    return seen


def test_deleting_a_cut_from_the_report_puts_it_back(tmp_path, captured_render):
    report = _report_with_suspect()
    report["segments"] = [{"start": 0.0, "end": 1.2}]      # stale after the edit
    report["cuts"] = [c for c in report["cuts"] if "silence" not in c["reasons"]]
    path = write_json(report, tmp_path / "r.json")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    assert main(["--video", str(video), "--from-report", str(path),
                 "--out", str(tmp_path / "o.mp4")]) == 0
    assert captured_render["segments"] == [{"start": 0.0, "end": 5.0}]


def test_a_reordered_report_keeps_its_own_segments(tmp_path, captured_render):
    report = _report_with_suspect()
    report["mode"] = "reorder"
    report["segments"] = [{"start": 3.0, "end": 4.0}, {"start": 0.0, "end": 1.0}]
    path = write_json(report, tmp_path / "r.json")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    main(["--video", str(video), "--from-report", str(path),
          "--out", str(tmp_path / "o.mp4")])
    assert captured_render["segments"] == load_report(path)["segments"]


def test_export_fcpxml_flag_writes_beside_the_report(tmp_path, captured_render,
                                                     monkeypatch):
    from xml_export import VideoMeta

    monkeypatch.setattr("edit.probe_video_meta", lambda video: VideoMeta(
        fps=60.0, width=1920, height=1080, duration=5.0))
    path = write_json(_report_with_suspect(), tmp_path / "r.json")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    assert main(["--video", str(video), "--from-report", str(path),
                 "--out", str(tmp_path / "o.mp4"), "--export-fcpxml"]) == 0
    assert (tmp_path / "v.report.fcpxml").exists()
    assert not (tmp_path / "v.report.xml").exists()


# --- Windows consoles decode as cp1252 unless told otherwise ---------------

def test_ffmpeg_output_is_read_as_utf8(monkeypatch, tmp_path):
    """ffmpeg echoes the filename; a byte cp1252 cannot decode killed the render."""
    seen = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return type("Done", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr("render.subprocess.run", run)
    monkeypatch.setattr("render.shutil.which", lambda name: name)
    render_video("v.mp4", tmp_path / "o.mp4",
                 {"segments": [{"start": 0.0, "end": 1.0}]}, EditorConfig())
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"


# --- the queue --------------------------------------------------------------

def test_a_running_job_counts_as_something_to_wait_for():
    """One job running and one queued left qsize at 1, so the UI said nothing."""
    import threading

    worker = Worker()
    release = threading.Event()
    started = threading.Event()

    def running():
        started.set()
        release.wait(5)

    worker.submit(running)
    started.wait(5)
    worker.submit(lambda: None)
    assert worker.waiting() == 2
    release.set()
    worker.join()


# --- exports ----------------------------------------------------------------

def test_export_of_a_non_latin1_filename_does_not_crash(monkeypatch, tmp_path):
    """Starlette encodes headers as latin-1, so a raw filename raised a 500."""
    import web.server as server
    from xml_export import VideoMeta

    meta = VideoMeta(fps=60.0, width=1920, height=1080, duration=5.0)
    monkeypatch.setattr(server, "probe_video_meta", lambda video: meta)
    job = server.STORE.create("教程 — take 2.mp4", "script")
    job.report = {"duration": 5.0, "cuts": []}
    job.state = server.JobState.READY

    client = TestClient(server.app)
    for route in ("xml", "fcpxml"):
        response = client.post(f"/api/jobs/{job.id}/{route}", json={"cuts": []})
        assert response.status_code == 200
        assert "UTF-8''" in response.headers["content-disposition"]
    server.STORE.delete(job.id)
