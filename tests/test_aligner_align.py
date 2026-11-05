from aligner import align_script
from config import EditorConfig
from models import CutReason, MatchStatus, ScriptLine
from conftest import words


def lines(*texts):
    return [ScriptLine(i, t) for i, t in enumerate(texts)]


def test_matches_lines_in_order():
    ws = words("open the settings panel then click the blue button")
    matches, cuts = align_script(ws, lines("open the settings panel",
                                           "click the blue button"),
                                 EditorConfig())
    assert [m.status for m in matches] == [MatchStatus.OK, MatchStatus.OK]
    assert matches[0].word_range == (0, 4)
    assert matches[1].word_range[0] >= 4
    assert cuts == []


def test_similar_lines_do_not_steal_each_others_occurrence():
    ws = words("click the blue button click the blue button")
    matches, _ = align_script(ws, lines("click the blue button",
                                        "click the blue button"),
                              EditorConfig())
    assert matches[0].word_range == (0, 4)
    assert matches[1].word_range == (4, 8)


def test_retake_marks_loser_for_cut():
    ws = words("click the red button click the blue button")
    matches, cuts = align_script(ws, lines("click the blue button"),
                                 EditorConfig())
    assert matches[0].word_range == (4, 8)
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.RETAKE_LOSER}
    assert cuts[0].start == ws[0].start
    assert cuts[0].end == ws[4].start


def test_unmatchable_line_is_a_miss_and_does_not_move_cursor():
    ws = words("open the settings panel then click the blue button")
    matches, _ = align_script(ws, lines("open the settings panel",
                                        "absolutely nothing like this exists",
                                        "click the blue button"),
                              EditorConfig())
    assert matches[1].status == MatchStatus.MISS
    assert matches[1].span is None
    assert matches[2].status == MatchStatus.OK
    assert matches[2].word_range[0] >= 4


def test_low_score_match_is_flagged_for_review():
    ws = words("click the blue button")
    cfg = EditorConfig(threshold=0.5, review_margin=0.45)
    matches, _ = align_script(ws, lines("click a blue widget"), cfg)
    assert matches[0].status == MatchStatus.REVIEW


def test_out_of_scope_line_found_via_full_recording_fallback():
    ws = words("click the blue button then open the settings panel")
    matches, _ = align_script(ws, lines("open the settings panel",
                                        "click the blue button"),
                              EditorConfig())
    assert matches[1].status != MatchStatus.MISS
    assert matches[1].scoped is False


def test_matched_text_is_populated():
    ws = words("click the blue button")
    matches, _ = align_script(ws, lines("click the blue button"), EditorConfig())
    assert matches[0].matched_text == "click the blue button"


def test_a_distant_second_match_is_not_a_retake():
    """The bug that deleted 29 seconds of a real tutorial.

    A line matched at 101s and again at 174s. Both cleared the threshold, so
    the later one was called a losing retake and cut - but it was the speaker
    covering the same point again while demonstrating, which is what a tutorial
    does. A retake follows the flub within seconds, not a minute.
    """
    ws = words("click the blue button " + "filler " * 40 + "click the blue button",
               dur=0.3)
    matches, cuts = align_script(ws, lines("click the blue button"),
                                 EditorConfig(retake_max_gap_s=5.0))

    assert [c for c in cuts if CutReason.RETAKE_LOSER in c.reasons] == [], \
        "a match a minute later is not a discarded take"
    assert matches[0].status is not MatchStatus.MISS


def test_an_adjacent_second_match_is_still_a_retake():
    """The genuine case must survive the distance guard."""
    ws = words("click the red button click the blue button")
    _, cuts = align_script(ws, lines("click the blue button"),
                           EditorConfig(retake_max_gap_s=5.0))
    assert [c for c in cuts if CutReason.RETAKE_LOSER in c.reasons]


def test_a_line_beyond_the_lookahead_is_found_without_searching_backwards():
    """A tutorial demonstrates between script lines, routinely overrunning the
    lookahead. Falling straight to a whole-recording search let a line match
    BEFORE the cursor and overlap the previous line, stealing its audio.
    """
    ws = words("open the settings panel " + "demo " * 60 + "click the blue button",
               dur=0.3)
    cfg = EditorConfig(lookahead_s=2.0)     # far too small on purpose
    matches, _ = align_script(ws, lines("open the settings panel",
                                        "click the blue button"), cfg)

    assert matches[0].word_range == (0, 4)
    assert matches[1].word_range[0] > matches[0].word_range[1], "no overlap"
    assert matches[1].scoped is True, "found going forward, not by searching back"


def test_matches_never_overlap_each_other():
    ws = words("open the settings panel " + "demo " * 30 + "click the blue button",
               dur=0.3)
    matches, _ = align_script(ws, lines("open the settings panel",
                                        "click the blue button"),
                              EditorConfig(lookahead_s=1.0))
    spans = [m.word_range for m in matches if m.word_range]
    for earlier, later in zip(spans, spans[1:]):
        assert later[0] >= earlier[1], f"{earlier} overlaps {later}"
