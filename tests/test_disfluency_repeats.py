from config import EditorConfig
from disfluency import detect_immediate_repeats
from models import CutReason, Word
from conftest import words


def test_adjacent_duplicate_word_is_cut():
    ws = words("click the the button")
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.DISFLUENCY}
    assert cuts[0].start == ws[1].start
    assert cuts[0].end == ws[2].start


def test_repeat_with_filler_between_is_cut_including_filler():
    ws = words("click uh click the button")
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].start == ws[0].start
    assert cuts[0].end == ws[2].start


def test_repeat_separated_by_long_pause_is_left_alone():
    # 2s gap - likely deliberate, not a stutter
    ws = words("the the button", gaps=[2.0, 0.0])
    assert detect_immediate_repeats(ws, 0, len(ws), EditorConfig()) == []


def test_punctuation_and_case_differences_still_count_as_repeat():
    ws = words("The the button")
    assert len(detect_immediate_repeats(ws, 0, len(ws), EditorConfig())) == 1


def test_non_repeat_is_untouched():
    ws = words("click the blue button")
    assert detect_immediate_repeats(ws, 0, len(ws), EditorConfig()) == []


def test_triple_repeat_cuts_both_earlier_copies():
    ws = words("the the the button")
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 2


def test_standalone_punctuation_token_does_not_break_a_repeat():
    ws = words("click the the button")
    ws = ws[:2] + [Word("-", ws[1].end, ws[1].end)] + ws[2:]
    cuts = detect_immediate_repeats(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].detail == "immediate repeat: 'the'"


def test_range_is_respected():
    ws = words("the the button and the the widget")
    cuts = detect_immediate_repeats(ws, 3, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].start == ws[4].start
