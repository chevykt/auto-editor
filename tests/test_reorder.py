"""Reorder mode: rebuild from matched windows, in script order."""
from config import EditorConfig
from edit import analyse
from models import CutReason, CutSpan, LineMatch, MatchStatus, Span
from reorder import _subtract, matched_window, reorder_segments
from conftest import words


def match(idx, i, j):
    return LineMatch(idx, None, 1.0, (i, j), MatchStatus.OK)


NO_PAD = EditorConfig(reorder_pad_ms=0)


# --- window extraction ---------------------------------------------------

def test_window_spans_the_matched_words():
    ws = words("alpha beta gamma delta")
    w = matched_window(ws, match(0, 1, 3), NO_PAD)
    assert (w.start, w.end) == (ws[1].start, ws[2].end)


def test_window_breathes_outward_into_surrounding_silence():
    ws = words("alpha beta gamma delta", gaps=[0.5, 0, 0.5])
    w = matched_window(ws, match(0, 1, 3), EditorConfig(reorder_pad_ms=60))
    assert w.start == round(ws[1].start - 0.06, 6)
    assert w.end == round(ws[2].end + 0.06, 6)


def test_pad_never_reaches_into_a_neighbouring_word():
    # neighbours are flush against the window, so there is no room to breathe
    ws = words("alpha beta gamma delta")
    w = matched_window(ws, match(0, 1, 3), EditorConfig(reorder_pad_ms=200))
    assert w.start == ws[0].end
    assert w.end == ws[3].start


def test_pad_at_the_very_start_cannot_go_negative():
    ws = words("alpha beta")
    w = matched_window(ws, match(0, 0, 1), EditorConfig(reorder_pad_ms=500))
    assert w.start == 0.0


def test_a_missed_line_has_no_window():
    ws = words("alpha beta")
    assert matched_window(ws, LineMatch(0, None, 0.0, None, MatchStatus.MISS),
                          NO_PAD) is None


# --- subtracting disfluencies from inside a window -----------------------

def test_subtract_splits_a_segment_around_an_interior_cut():
    out = _subtract(Span(0.0, 10.0), [CutSpan(4.0, 5.0, {CutReason.DISFLUENCY})])
    assert [(s.start, s.end) for s in out] == [(0.0, 4.0), (5.0, 10.0)]


def test_subtract_ignores_cuts_outside_the_segment():
    out = _subtract(Span(4.0, 6.0), [CutSpan(0.0, 1.0, {CutReason.DISFLUENCY}),
                                     CutSpan(8.0, 9.0, {CutReason.DISFLUENCY})])
    assert [(s.start, s.end) for s in out] == [(4.0, 6.0)]


def test_subtract_clips_a_cut_overlapping_the_edge():
    out = _subtract(Span(4.0, 6.0), [CutSpan(3.0, 4.5, {CutReason.DISFLUENCY})])
    assert [(s.start, s.end) for s in out] == [(4.5, 6.0)]


def test_a_cut_covering_the_whole_segment_leaves_nothing():
    assert _subtract(Span(4.0, 6.0),
                     [CutSpan(3.0, 7.0, {CutReason.DISFLUENCY})]) == []


# --- assembling in script order ------------------------------------------

def test_segments_come_back_in_script_order_not_timeline_order():
    """The whole point of the mode.

    Line 0 was recorded second and line 1 first; the output must put them
    back the way the script reads.
    """
    ws = words("second line here first line here")
    matches = [match(0, 3, 6), match(1, 0, 3)]
    segs = reorder_segments(ws, matches, [], NO_PAD)

    assert [(s.start, s.end) for s in segs] == [
        (ws[3].start, ws[5].end),   # line 0, spoken later
        (ws[0].start, ws[2].end),   # line 1, spoken earlier
    ]


def test_missed_lines_contribute_nothing():
    ws = words("alpha beta gamma delta")
    matches = [match(0, 0, 2),
               LineMatch(1, None, 0.0, None, MatchStatus.MISS),
               match(2, 2, 4)]
    assert len(reorder_segments(ws, matches, [], NO_PAD)) == 2


def test_a_disfluency_inside_a_window_is_removed_from_it():
    ws = words("click the the blue button")
    stutter = [CutSpan(ws[1].start, ws[2].start, {CutReason.DISFLUENCY})]
    segs = reorder_segments(ws, [match(0, 0, 5)], stutter, NO_PAD)

    assert len(segs) == 2, "the window splits around the stutter"
    assert segs[0].end == ws[1].start
    assert segs[1].start == ws[2].start


def test_no_matches_yields_no_segments():
    assert reorder_segments(words("alpha beta"), [], [], NO_PAD) == []


# --- through the full pipeline -------------------------------------------

def test_analyse_reorder_puts_lines_back_in_script_order():
    ws = words("close the window first open the window")
    report = analyse(ws, "open the window\n\nclose the window", duration=10.0,
                     cfg=EditorConfig(reorder=True, reorder_pad_ms=0),
                     video="raw.mp4")

    assert report["mode"] == "reorder"
    segs = report["segments"]
    # "open the window" was spoken second but is script line 0
    assert segs[0]["start"] > segs[1]["start"], "script order, not timeline order"


def test_reorder_discards_adlibs_while_default_keeps_them():
    ws = words("open the window and honestly this bit was improvised")
    script = "open the window"

    default = analyse(ws, script, 10.0, EditorConfig(), "raw.mp4")
    reordered = analyse(ws, script, 10.0,
                        EditorConfig(reorder=True), "raw.mp4")

    kept = sum(s["end"] - s["start"] for s in default["segments"])
    rebuilt = sum(s["end"] - s["start"] for s in reordered["segments"])
    assert rebuilt < kept, "reorder must drop the ad-lib the default keeps"
    assert reordered["adlibs"], "and must still report what it discarded"


def test_reorder_summary_reflects_the_rebuilt_length():
    ws = words("open the window and honestly this bit was improvised")
    report = analyse(ws, "open the window", 10.0,
                     EditorConfig(reorder=True), "raw.mp4")
    total = sum(s["end"] - s["start"] for s in report["segments"])
    assert abs(report["summary"]["output_seconds"] - total) < 1e-6
    assert abs(report["summary"]["cut_seconds"] - (10.0 - total)) < 1e-6
