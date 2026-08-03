"""Mid-line restarts: an abandoned attempt matching the START of a line.

Normal retake resolution only fires when a line produces two windows that each
clear the alignment threshold. A speaker who gets partway in, stops, and starts
over leaves a fragment far too short for that - and without this detector it is
reported as a probable ad-lib, i.e. discarded content labelled as intentional.
"""
from aligner import align_script, find_abandoned_prefix
from config import EditorConfig
from models import CutReason, ScriptLine
from conftest import words

LINE = ScriptLine(0, "turn on automatic sync then pick how often you want it to run")


def test_finds_an_abandoned_opening():
    # "turn on automatic sync then" spoken, abandoned, then restarted at idx 5
    ws = words("turn on automatic sync then turn on automatic sync then pick how")
    assert find_abandoned_prefix(ws, 0, 5, LINE, EditorConfig()) == 0


def test_prefers_the_longest_qualifying_fragment():
    ws = words("turn on automatic sync then turn on automatic sync then pick how")
    # the full 5-word opening beats the 2-word one
    assert find_abandoned_prefix(ws, 0, 5, LINE, EditorConfig()) == 0


def test_ignores_a_fragment_that_is_not_the_line_opening():
    # real ad-lib sharing vocabulary with the line, but not its opening
    ws = words("you want it to run honestly turn on automatic sync then pick")
    assert find_abandoned_prefix(ws, 0, 5, LINE, EditorConfig()) is None


def test_ignores_a_fragment_separated_by_a_long_pause():
    # 5s gap: a new thought, not a restart
    ws = words("turn on automatic sync then turn on automatic sync then pick how",
               gaps=[0, 0, 0, 0, 5.0])
    assert find_abandoned_prefix(ws, 0, 5, LINE, EditorConfig()) is None


def test_ignores_a_single_word_fragment():
    ws = words("turn turn on automatic sync then pick how")
    assert find_abandoned_prefix(ws, 0, 1, LINE, EditorConfig()) is None


def test_empty_region_yields_nothing():
    ws = words("turn on automatic sync then pick how")
    assert find_abandoned_prefix(ws, 3, 3, LINE, EditorConfig()) is None


def test_disabled_by_config():
    ws = words("turn on automatic sync then turn on automatic sync then pick how")
    cfg = EditorConfig(partial_retakes=False)
    assert find_abandoned_prefix(ws, 0, 5, LINE, cfg) is None


def test_out_of_order_words_do_not_count_as_a_prefix():
    # same bag of words, wrong order - a restart repeats the opening in sequence
    ws = words("sync automatic on turn then turn on automatic sync then pick how")
    assert find_abandoned_prefix(ws, 0, 5, LINE, EditorConfig()) is None


# --- end to end through align_script -------------------------------------

def test_abandoned_opening_is_cut_as_a_retake_loser():
    ws = words("turn on automatic sync then "
               "turn on automatic sync then pick how often you want it to run")
    matches, cuts = align_script(ws, [LINE], EditorConfig())

    assert matches[0].word_range[0] == 5, "winner is the complete restart"
    losers = [c for c in cuts if CutReason.RETAKE_LOSER in c.reasons]
    assert len(losers) == 1
    assert losers[0].start == ws[0].start
    assert losers[0].end == ws[5].start
    assert "mid-line restart" in losers[0].detail


def test_a_clean_take_produces_no_partial_retake_cut():
    ws = words("turn on automatic sync then pick how often you want it to run")
    _, cuts = align_script(ws, [LINE], EditorConfig())
    assert cuts == []


def test_a_full_retake_does_not_also_fire_the_partial_detector():
    """A losing take is a losing take once, not twice.

    A whole-line retake looks like an abandoned prefix too, since it repeats
    the line's opening. Both detectors firing would double-count the same
    words and duplicate the entry in the report.
    """
    ws = words("click the red button click the blue button")
    _, cuts = align_script(ws, [ScriptLine(0, "click the blue button")],
                           EditorConfig())
    assert len(cuts) == 1
    assert "mid-line restart" not in cuts[0].detail


def test_an_adlib_before_a_line_is_not_cut():
    """The property that matters: ad-libs must survive this detector."""
    ws = words("honestly this next bit is the part people get wrong "
               "turn on automatic sync then pick how often you want it to run")
    matches, cuts = align_script(ws, [LINE], EditorConfig())
    assert [c for c in cuts if CutReason.RETAKE_LOSER in c.reasons] == []
    assert matches[0].status.value == "ok"
