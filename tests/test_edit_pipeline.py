import pytest

from pathlib import Path

from config import EditorConfig
from edit import analyse, build_parser, config_from_args, report_paths
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


def test_reorder_and_export_xml_are_rejected_as_not_implemented():
    for flag in ("--reorder", "--export-xml"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["--script", "s.txt", "--video", "v.mp4", flag])


def test_script_and_video_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--video", "v.mp4"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--script", "s.txt"])


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
