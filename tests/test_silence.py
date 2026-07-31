from config import EditorConfig
from models import CutReason
from silence import detect_silence
from conftest import words


def test_gap_at_or_above_threshold_is_cut():
    # three words, 0.7s gap after the first, 0.1s after the second
    ws = words("alpha beta gamma", dur=0.3, gaps=[0.7, 0.1])
    cuts = detect_silence(ws, EditorConfig(silence_ms=600))
    assert len(cuts) == 1
    assert cuts[0].start == 0.3
    assert cuts[0].end == 1.0
    assert cuts[0].reasons == {CutReason.SILENCE}


def test_gap_below_threshold_is_kept():
    ws = words("alpha beta", dur=0.3, gaps=[0.5])
    assert detect_silence(ws, EditorConfig(silence_ms=600)) == []


def test_gap_exactly_at_threshold_is_cut():
    ws = words("alpha beta", dur=0.3, gaps=[0.6])
    assert len(detect_silence(ws, EditorConfig(silence_ms=600))) == 1


def test_leading_silence_is_cut_when_long_enough():
    ws = words("alpha beta", start=2.0, dur=0.3)
    cuts = detect_silence(ws, EditorConfig(silence_ms=600))
    assert cuts[0].start == 0.0
    assert cuts[0].end == 2.0


def test_trailing_silence_is_cut_when_duration_known():
    ws = words("alpha beta", dur=0.3)          # ends at 0.6
    cuts = detect_silence(ws, EditorConfig(silence_ms=600), duration=5.0)
    assert cuts[-1].start == 0.6
    assert cuts[-1].end == 5.0


def test_no_trailing_cut_without_duration():
    ws = words("alpha beta", dur=0.3)
    assert detect_silence(ws, EditorConfig(silence_ms=600)) == []


def test_empty_word_list():
    assert detect_silence([], EditorConfig()) == []
