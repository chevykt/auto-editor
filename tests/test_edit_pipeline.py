import pytest

from pathlib import Path

from config import EditorConfig
from edit import (analyse, build_parser, config_from_args, main, report_paths,
                  validate_args)
from conftest import words


def test_default_report_paths_append_rather_than_replace_suffixes():
    # "raw.report" looks like it already has a ".report" extension, so a
    # with_suffix(".json") here would silently write "raw.json" instead.
    j, m = report_paths(Path("/tmp/raw.mp4"), None)
    assert j.name == "raw.report.json"
    assert m.name == "raw.report.md"


def test_explicit_report_path_is_used_as_a_stem():
    j, m = report_paths(Path("/tmp/raw.mp4"), "out/review")
    assert j.name == "review.json"
    assert m.name == "review.md"
    assert j.parent == Path("out")


def test_report_paths_survive_dots_in_the_video_stem():
    j, m = report_paths(Path("/tmp/take.2.final.mp4"), None)
    assert j.name == "take.2.final.report.json"
    assert m.name == "take.2.final.report.md"


def test_analyse_produces_matches_and_cuts():
    ws = words("click the the blue button")
    report = analyse(ws, "click the blue button", duration=10.0,
                     cfg=EditorConfig(), video="raw.mp4")
    assert report["lines"][0]["status"] in ("ok", "review")
    assert report["summary"]["cut_count"] >= 1


def test_analyse_flags_a_missing_line():
    ws = words("click the blue button")
    report = analyse(ws, "click the blue button\n\nnothing resembling this",
                     duration=10.0, cfg=EditorConfig(), video="raw.mp4")
    assert report["lines"][1]["status"] == "miss"
    assert report["summary"]["misses"] == 1


def test_analyse_keeps_adlibs():
    ws = words("click the blue button and honestly this part was made up")
    report = analyse(ws, "click the blue button", duration=10.0,
                     cfg=EditorConfig(), video="raw.mp4")
    assert any("made up" in a["text"] for a in report["adlibs"])


def test_analyse_cuts_trailing_dead_air():
    ws = words("click the blue button")          # ends at 1.2s
    report = analyse(ws, "click the blue button", duration=30.0,
                     cfg=EditorConfig(), video="raw.mp4")
    assert report["summary"]["cut_seconds"] > 25.0


def test_parser_defaults_match_config():
    args = build_parser().parse_args(["--script", "s.txt", "--video", "v.mp4"])
    cfg = config_from_args(args)
    assert cfg.threshold == 0.75
    assert cfg.silence_ms == 600
    assert cfg.clean_stutters is True


def test_no_clean_stutters_flag_disables_detection():
    args = build_parser().parse_args(
        ["--script", "s.txt", "--video", "v.mp4", "--no-clean-stutters"])
    assert config_from_args(args).clean_stutters is False


def test_no_cache_flag_disables_cache():
    args = build_parser().parse_args(
        ["--script", "s.txt", "--video", "v.mp4", "--no-cache"])
    assert config_from_args(args).use_cache is False


def test_dry_run_and_auto_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--script", "s.txt", "--video", "v.mp4", "--dry-run", "--auto"])


def test_export_xml_is_still_rejected_as_not_implemented():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--script", "s.txt", "--video", "v.mp4", "--export-xml"])


def test_reorder_is_accepted_and_reaches_the_config():
    args = build_parser().parse_args(
        ["--script", "s.txt", "--video", "v.mp4", "--reorder"])
    assert config_from_args(args).reorder is True


def test_reorder_is_off_by_default():
    args = build_parser().parse_args(["--script", "s.txt", "--video", "v.mp4"])
    assert config_from_args(args).reorder is False


def test_video_is_always_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--script", "s.txt"])


def test_script_is_required_for_a_normal_run():
    args = build_parser().parse_args(["--video", "v.mp4"])
    assert "--script is required" in validate_args(args)


def test_script_is_not_required_when_rendering_from_a_report():
    # the cut list is already decided; the script would never be read
    args = build_parser().parse_args(
        ["--video", "v.mp4", "--from-report", "r.json", "--out", "o.mp4"])
    assert validate_args(args) is None


def test_out_is_required_when_rendering():
    for extra in (["--auto"], ["--from-report", "r.json"]):
        args = build_parser().parse_args(
            ["--script", "s.txt", "--video", "v.mp4"] + extra)
        assert "--out is required" in validate_args(args)


def test_out_is_not_required_for_a_report_only_run():
    args = build_parser().parse_args(["--script", "s.txt", "--video", "v.mp4"])
    assert validate_args(args) is None


def test_missing_out_is_caught_before_any_transcription(monkeypatch):
    """A missing --out must not cost a full Whisper run to discover."""
    def explode(*a, **k):
        raise AssertionError("must not transcribe before validating args")

    monkeypatch.setattr("edit.transcribe", explode)
    assert main(["--script", "s.txt", "--video", "v.mp4", "--auto"]) == 2


def test_missing_input_files_report_cleanly(tmp_path, capsys):
    """A typo'd path must not surface a raw traceback."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    assert main(["--script", str(tmp_path / "nope.txt"),
                 "--video", str(video)]) == 2
    assert "--script not found" in capsys.readouterr().err

    assert main(["--video", str(tmp_path / "nope.mp4"),
                 "--script", "s.txt"]) == 2
    assert "--video not found" in capsys.readouterr().err

    assert main(["--video", str(video), "--out", str(tmp_path / "o.mp4"),
                 "--from-report", str(tmp_path / "nope.json")]) == 2
    assert "--from-report not found" in capsys.readouterr().err


def test_missing_files_are_caught_before_transcription(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not transcribe before checking inputs")

    monkeypatch.setattr("edit.transcribe", explode)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    assert main(["--script", str(tmp_path / "nope.txt"),
                 "--video", str(video), "--auto",
                 "--out", str(tmp_path / "o.mp4")]) == 2


def test_every_tuning_knob_is_exposed():
    args = build_parser().parse_args([
        "--script", "s.txt", "--video", "v.mp4",
        "--threshold", "0.9", "--review-margin", "0.1", "--lookahead-s", "45",
        "--tiebreak", "score", "--silence-ms", "800", "--padding-ms", "50",
        "--min-segment-ms", "200", "--crossfade-ms", "40",
        "--duplicate-window-s", "6", "--max-disfluency-words", "3",
        "--model", "small",
    ])
    cfg = config_from_args(args)
    assert (cfg.threshold, cfg.review_margin, cfg.lookahead_s) == (0.9, 0.1, 45.0)
    assert (cfg.tiebreak, cfg.silence_ms, cfg.padding_ms) == ("score", 800, 50)
    assert (cfg.min_segment_ms, cfg.crossfade_ms) == (200, 40)
    assert (cfg.duplicate_window_s, cfg.max_disfluency_words) == (6.0, 3)
    assert cfg.model == "small"
