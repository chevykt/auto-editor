from config import EditorConfig
from disfluency import detect_truncated_restarts
from models import CutReason
from conftest import words


def test_fragment_followed_by_full_word_is_cut():
    ws = words("the config- configuration screen")
    cuts = detect_truncated_restarts(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.DISFLUENCY}
    assert cuts[0].start == ws[1].start
    assert cuts[0].end == ws[2].start


def test_short_fragment_below_min_chars_is_ignored():
    # "in" is a prefix of "international" but too short to be a safe signal
    ws = words("in international waters")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_small_growth_is_ignored():
    # "the" -> "there" grows by only 2 chars; likely real speech
    ws = words("the there button")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_identical_words_are_not_a_restart():
    # that is a repeat, handled by the other detector
    ws = words("config config screen")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_long_pause_between_breaks_the_signal():
    ws = words("config- configuration screen", gaps=[1.5, 0.0])
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_non_prefix_is_ignored():
    ws = words("setup configuration screen")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []
