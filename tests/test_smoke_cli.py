"""Full CLI run against real media, with no Whisper download.

Pre-seeding the transcript cache lets this exercise the entire pipeline -
parse, silence, align, retakes, disfluencies, cut list, both reports, and
the ffmpeg render - against a real video file, in seconds.

Run with: python -m pytest tests/test_smoke_cli.py -v -m smoke
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from config import EditorConfig
from edit import main
from models import Word
from report import load_report
from transcribe import (cache_path_for, probe_duration, save_cache,
                        transcription_key)

pytestmark = pytest.mark.smoke

HAS_FFMPEG = shutil.which("ffmpeg") is not None

SCRIPT = """Welcome to the configuration screen.

Click the blue button to save your changes.

That's everything you need to know.
"""


def spoken_transcript() -> list[Word]:
    """A realistic take: a stutter, a truncated restart, an ad-lib, dead air.

    Timings are hand-built at 0.3s per word so each phenomenon is unambiguous:
      0.0- 1.8  "welcome to the config- configuration screen"     <- restart
      1.8- 6.0  (4.2s of dead air)                                <- silence
      6.0- 8.7  "click the the blue button to save your changes"  <- repeat
      8.7- 9.0  (0.3s gap, below the 600ms floor - kept)
      9.0-11.4  "and by the way this part is improvised"          <- ad-lib
     11.4-12.0  (0.6s gap, exactly at the floor - cut)
     12.0-13.8  "thats everything you need to know"
     13.8-20.0  (6.2s of trailing dead air)                       <- silence
    """
    def run(text, start, dur=0.3):
        return [Word(t, round(start + i * dur, 3), round(start + i * dur + dur, 3))
                for i, t in enumerate(text.split())]

    return (run("welcome to the config- configuration screen", 0.0)
            + run("click the the blue button to save your changes", 6.0)
            + run("and by the way this part is improvised", 9.0)
            + run("thats everything you need to know", 12.0))


def seed(video: Path) -> None:
    """Pre-seed the transcript so the suite never downloads a speech model.

    Must use the same path transcribe() will look up - content hash plus the
    settings key - or the seed is silently ignored and Whisper actually runs.
    """
    cfg = EditorConfig()
    save_cache(cache_path_for(video, transcription_key(cfg)),
               spoken_transcript(), 20.0, transcription_key(cfg))


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never write into the developer's real transcript cache."""
    monkeypatch.setattr("transcribe.CACHE_DIR", tmp_path / "cache")


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path]:
    """A 20-second clip plus a script, with the transcript cache pre-seeded."""
    video = tmp_path / "raw.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=20:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video),
    ], capture_output=True, check=True)

    script = tmp_path / "script.txt"
    script.write_text(SCRIPT, encoding="utf-8")
    seed(video)
    return video, script


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_dry_run_writes_both_reports_and_renders_nothing(project, tmp_path):
    video, script = project
    out = tmp_path / "final.mp4"

    assert main(["--script", str(script), "--video", str(video),
                 "--out", str(out)]) == 0

    assert (tmp_path / "raw.report.json").exists()
    assert (tmp_path / "raw.report.md").exists()
    assert not out.exists(), "dry run must not render"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_full_pipeline_matches_lines_cuts_faults_and_keeps_the_adlib(project, tmp_path):
    video, script = project
    main(["--script", str(script), "--video", str(video)])
    rep = load_report(tmp_path / "raw.report.json")

    # all three script lines found
    assert [l["status"] for l in rep["lines"]] == ["ok", "ok", "ok"]

    # the ad-lib survived and is reported
    assert any("improvised" in a["text"] for a in rep["adlibs"])

    # the stutter and the truncated restart were both caught
    details = " ".join(c["detail"] for c in rep["cuts"])
    assert "immediate repeat" in details
    assert "truncated restart" in details

    # the 3.6s of dead air was cut
    assert any("silence" in c["reasons"] for c in rep["cuts"])
    assert rep["summary"]["cut_seconds"] > 3.0


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_auto_renders_a_shorter_video(project, tmp_path):
    video, script = project
    out = tmp_path / "final.mp4"

    assert main(["--script", str(script), "--video", str(video),
                 "--out", str(out), "--auto"]) == 0

    assert out.exists()
    rendered = probe_duration(out)
    assert rendered < 20.0, "output must be shorter than the source"
    expected = load_report(tmp_path / "raw.report.json")["summary"]["output_seconds"]
    assert abs(rendered - expected) < 0.5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_from_report_renders_a_hand_edited_cut_list(project, tmp_path):
    video, script = project
    main(["--script", str(script), "--video", str(video)])

    # a human disagrees with the cuts and keeps only the first 5 seconds
    report_path = tmp_path / "raw.report.json"
    rep = load_report(report_path)
    rep["segments"] = [{"start": 0.0, "end": 5.0}]
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    out = tmp_path / "approved.mp4"
    assert main(["--script", str(script), "--video", str(video),
                 "--out", str(out), "--from-report", str(report_path)]) == 0

    assert 4.5 < probe_duration(out) < 5.5


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_reorder_renders_out_of_order_segments(project, tmp_path):
    """Reorder mode concatenates windows in script order, not timeline order.

    ffmpeg has to seek backwards between segments here, which the subtractive
    path never asks it to do - segments there are always ascending.
    """
    video, script = project
    out = tmp_path / "reordered.mp4"

    assert main(["--script", str(script), "--video", str(video),
                 "--out", str(out), "--reorder", "--auto"]) == 0

    rep = load_report(tmp_path / "raw.report.json")
    assert rep["mode"] == "reorder"
    assert out.exists()

    expected = rep["summary"]["output_seconds"]
    assert abs(probe_duration(out) - expected) < 0.5
    # the ad-lib is gone: reorder keeps only matched windows
    assert probe_duration(out) < probe_duration(video)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_reorder_report_says_content_was_discarded_not_kept(project, tmp_path):
    video, script = project
    main(["--script", str(script), "--video", str(video), "--reorder"])
    md = (tmp_path / "raw.report.md").read_text(encoding="utf-8")

    assert "DISCARDED" in md
    assert "Kept, but not in the script" not in md


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_export_xml_writes_a_timeline_matching_the_report(project, tmp_path):
    """XML export against real media, where the frame rate comes from the file."""
    from xml.etree import ElementTree as ET

    video, script = project
    assert main(["--script", str(script), "--video", str(video),
                 "--export-xml"]) == 0

    xml_path = tmp_path / "raw.report.xml"
    assert xml_path.exists()

    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    rep = load_report(tmp_path / "raw.report.json")

    clips = root.findall("./sequence/media/video/track/clipitem")
    assert len(clips) == len(rep["segments"])

    # the clip at 25fps must line up with the first segment's timestamps
    fps = 25.0
    first = rep["segments"][0]
    assert clips[0].find("in").text == str(int(round(first["start"] * fps)))
    assert clips[0].find("out").text == str(int(round(first["end"] * fps)))

    # sequence length agrees with the report, within a frame
    total = int(root.find("./sequence/duration").text)
    assert abs(total / fps - rep["summary"]["output_seconds"]) < (1.0 / fps) * 2


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_export_xml_works_alongside_a_render(project, tmp_path):
    """The spec asks for XML instead of OR alongside a rendered MP4."""
    video, script = project
    out = tmp_path / "final.mp4"

    assert main(["--script", str(script), "--video", str(video),
                 "--out", str(out), "--export-xml", "--auto"]) == 0

    assert out.exists()
    assert (tmp_path / "raw.report.xml").exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_batch_processes_every_pair_and_reports_orphans(project, tmp_path,
                                                        capsys):
    video, script = project
    work = tmp_path / "work"
    work.mkdir()

    # two complete pairs plus one recording with no script
    for name in ("lesson_a", "lesson_b"):
        shutil.copy(video, work / f"{name}.mp4")
        shutil.copy(script, work / f"{name}.txt")
        seed(work / f"{name}.mp4")
    shutil.copy(video, work / "orphan.mp4")

    assert main(["--batch", str(work)]) == 0

    err = capsys.readouterr().err
    assert "2 recording(s) to process" in err
    assert "skipping orphan.mp4" in err
    assert "2/2 succeeded" in err

    for name in ("lesson_a", "lesson_b"):
        assert (work / f"{name}.report.json").exists()
        assert (work / f"{name}.report.md").exists()
    assert not (work / "orphan.report.json").exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_batch_renders_one_output_per_recording_into_out_dir(project, tmp_path):
    video, script = project
    work, edited = tmp_path / "work", tmp_path / "edited"
    work.mkdir()

    for name in ("one", "two"):
        shutil.copy(video, work / f"{name}.mp4")
        shutil.copy(script, work / f"{name}.txt")
        seed(work / f"{name}.mp4")

    assert main(["--batch", str(work), "--out-dir", str(edited), "--auto"]) == 0

    for name in ("one", "two"):
        rendered = edited / f"{name}.cut.mp4"
        assert rendered.exists()
        assert probe_duration(rendered) < 20.0
        assert (edited / f"{name}.report.md").exists()

    # a re-run must not treat the renders as fresh input
    assert main(["--batch", str(edited)]) == 2


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_batch_survives_one_bad_recording(project, tmp_path, capsys):
    """One unreadable file must not abandon the rest of the batch."""
    video, script = project
    work = tmp_path / "work"
    work.mkdir()

    shutil.copy(video, work / "good.mp4")
    shutil.copy(script, work / "good.txt")
    seed(work / "good.mp4")

    (work / "broken.mp4").write_bytes(b"not a video")
    shutil.copy(script, work / "broken.txt")

    assert main(["--batch", str(work)]) == 1, "non-zero when any job fails"

    err = capsys.readouterr().err
    assert "1/2 succeeded" in err
    assert (work / "good.report.json").exists(), "the good job still ran"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_config_file_drives_a_real_run(project, tmp_path):
    video, script = project
    conf = tmp_path / "auto-editor.toml"
    conf.write_text("threshold = 0.9\nsilence_ms = 900\nclean_stutters = false\n",
                    encoding="utf-8")

    assert main(["--script", str(script), "--video", str(video),
                 "--config", str(conf)]) == 0

    rep = load_report(tmp_path / "raw.report.json")
    assert rep["config"]["threshold"] == 0.9
    assert rep["config"]["silence_ms"] == 900
    assert rep["config"]["clean_stutters"] is False


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_no_clean_stutters_leaves_the_disfluencies_in(project, tmp_path):
    video, script = project
    main(["--script", str(script), "--video", str(video), "--no-clean-stutters"])
    rep = load_report(tmp_path / "raw.report.json")

    details = " ".join(c["detail"] for c in rep["cuts"])
    assert "immediate repeat" not in details
    assert "truncated restart" not in details
    # silence is still cut - it is not a disfluency
    assert any("silence" in c["reasons"] for c in rep["cuts"])
