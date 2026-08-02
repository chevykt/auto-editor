import json

from config import EditorConfig
from models import (CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine,
                    Span)
from report import build_report, load_report, write_json
from conftest import words


def sample():
    ws = words("click the blue button")
    lines = [ScriptLine(0, "click the blue button")]
    matches = [LineMatch(0, Span(0.0, 1.2), 1.0, (0, 4), MatchStatus.OK,
                         matched_text="click the blue button")]
    cuts = [CutSpan(1.5, 2.5, {CutReason.SILENCE}, "gap 1.00s"),
            CutSpan(3.0, 3.4, {CutReason.DISFLUENCY}, "immediate repeat: 'the'")]
    return ws, lines, matches, cuts


def test_report_round_trips_through_json(tmp_path):
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, duration=10.0,
                       cfg=EditorConfig(), video="raw.mp4")
    path = tmp_path / "r.json"
    write_json(rep, path)

    loaded = load_report(path)
    assert loaded["duration"] == 10.0
    assert len(loaded["cuts"]) == 2
    assert loaded["cuts"][0]["reasons"] == ["silence"]


def test_json_is_valid_and_human_readable(tmp_path):
    ws, lines, matches, cuts = sample()
    path = tmp_path / "r.json"
    write_json(build_report(lines, matches, cuts, ws, 10.0, EditorConfig(),
                            "raw.mp4"), path)
    text = path.read_text(encoding="utf-8")
    json.loads(text)
    assert "\n  " in text          # indented, so it can be hand-edited


def test_report_records_config_used():
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, 10.0,
                       EditorConfig(threshold=0.9), "raw.mp4")
    assert rep["config"]["threshold"] == 0.9


def test_report_counts_and_totals_cut_time():
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, 10.0, EditorConfig(), "raw.mp4")
    assert rep["summary"]["cut_count"] == 2
    assert round(rep["summary"]["cut_seconds"], 3) == 1.4
    assert round(rep["summary"]["output_seconds"], 3) == 8.6


def test_unmatched_spans_are_listed_as_likely_adlibs():
    ws = words("click the blue button and by the way this is improvised")
    lines = [ScriptLine(0, "click the blue button")]
    matches = [LineMatch(0, Span(ws[0].start, ws[3].end), 1.0, (0, 4),
                         MatchStatus.OK, matched_text="click the blue button")]
    rep = build_report(lines, matches, [], ws, duration=5.0,
                       cfg=EditorConfig(), video="raw.mp4")
    assert rep["adlibs"]
    assert "improvised" in rep["adlibs"][0]["text"]


def test_miss_is_recorded_explicitly():
    ws, lines, _, _ = sample()
    matches = [LineMatch(0, None, 0.0, None, MatchStatus.MISS)]
    rep = build_report(lines, matches, [], ws, 5.0, EditorConfig(), "raw.mp4")
    assert rep["lines"][0]["status"] == "miss"
    assert rep["summary"]["misses"] == 1


def test_segments_are_the_complement_of_the_cuts():
    ws, lines, matches, cuts = sample()
    rep = build_report(lines, matches, cuts, ws, 10.0, EditorConfig(), "raw.mp4")
    # cuts at 1.5-2.5 and 3.0-3.4 over a 10s timeline
    assert [(s["start"], s["end"]) for s in rep["segments"]] == [
        (0.0, 1.5), (2.5, 3.0), (3.4, 10.0)]
