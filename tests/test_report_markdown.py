from config import EditorConfig
from models import (CutReason, CutSpan, LineMatch, MatchStatus, ScriptLine,
                    Span)
from report import build_report, render_markdown, write_markdown
from conftest import words


def sample_report():
    ws = words("click the blue button and this bit is improvised")
    lines = [ScriptLine(0, "click the blue button"),
             ScriptLine(1, "a line that never got said")]
    matches = [
        LineMatch(0, Span(0.0, 1.2), 1.0, (0, 4), MatchStatus.OK,
                  matched_text="click the blue button"),
        LineMatch(1, None, 0.0, None, MatchStatus.MISS),
    ]
    cuts = [CutSpan(2.0, 2.6, {CutReason.SILENCE}, "gap 0.60s"),
            CutSpan(3.0, 3.4, {CutReason.DISFLUENCY}, "immediate repeat: 'the'"),
            CutSpan(5.0, 6.0, {CutReason.RETAKE_LOSER}, "losing take of line 0")]
    return build_report(lines, matches, cuts, ws, 20.0, EditorConfig(), "raw.mp4")


def test_markdown_contains_every_script_line():
    md = render_markdown(sample_report())
    assert "click the blue button" in md
    assert "a line that never got said" in md


def test_misses_are_called_out():
    md = render_markdown(sample_report())
    assert "MISS" in md


def test_silence_is_summarised_not_enumerated():
    md = render_markdown(sample_report())
    # the silence cut's detail should not appear line by line
    assert "gap 0.60s" not in md
    assert "silence" in md.lower()


def test_non_silence_cuts_are_listed_individually():
    md = render_markdown(sample_report())
    assert "immediate repeat" in md
    assert "losing take of line 0" in md


def test_adlibs_section_present():
    md = render_markdown(sample_report())
    assert "improvised" in md


def test_write_markdown_creates_the_file(tmp_path):
    path = write_markdown(sample_report(), tmp_path / "r.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Auto-Editor Review")


def test_out_of_order_match_is_flagged():
    lines = [ScriptLine(0, "click the blue button")]
    matches = [LineMatch(0, Span(0.0, 1.2), 1.0, (0, 4), MatchStatus.OK,
                         scoped=False, matched_text="click the blue button")]
    md = render_markdown(build_report(lines, matches, [], words("click the blue button"),
                                      10.0, EditorConfig(), "raw.mp4"))
    assert "out of order" in md
