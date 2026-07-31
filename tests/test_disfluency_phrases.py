from config import EditorConfig
from disfluency import detect_duplicate_phrases, detect_disfluencies
from models import CutReason
from conftest import words


def test_repeated_phrase_within_window_cuts_earlier_copy():
    ws = words("open the settings open the settings panel")
    cuts = detect_duplicate_phrases(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].start == ws[0].start
    assert cuts[0].end == ws[3].start


def test_repeat_outside_window_is_kept():
    # 6s apart with a 4s window - intentional callback, not a stutter
    ws = words("open the settings open the settings panel",
               gaps=[0, 0, 6.0, 0, 0, 0])
    assert detect_duplicate_phrases(ws, 0, len(ws), EditorConfig()) == []


def test_longer_repeat_preferred_over_shorter():
    ws = words("click the blue button click the blue button now")
    cuts = detect_duplicate_phrases(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].end == ws[4].start


def test_phrase_longer_than_cap_is_not_cut():
    cfg = EditorConfig(max_disfluency_words=2)
    ws = words("one two three one two three")
    cuts = detect_duplicate_phrases(ws, 0, len(ws), cfg)
    # the cap bounds the PHRASE that may be matched, not the span removed:
    # the removed span also covers the gap before the clean continuation.
    # So the 3-word clause must never be matched as a unit.
    assert cuts
    assert all("one two three" not in c.detail for c in cuts)
    assert "one two" in cuts[0].detail


def test_single_word_repeat_is_left_to_the_repeat_detector():
    ws = words("the the button")
    assert detect_duplicate_phrases(ws, 0, len(ws), EditorConfig()) == []


def test_detect_disfluencies_runs_all_three_and_sorts():
    ws = words("the the config- configuration open the settings open the settings now")
    cuts = detect_disfluencies(ws, 0, len(ws), EditorConfig())
    assert len(cuts) >= 3
    assert cuts == sorted(cuts, key=lambda c: c.start)
    assert all(c.reasons == {CutReason.DISFLUENCY} for c in cuts)


def test_detect_disfluencies_disabled_by_flag():
    ws = words("the the button")
    cfg = EditorConfig(clean_stutters=False)
    assert detect_disfluencies(ws, 0, len(ws), cfg) == []


def test_emphasis_repeat_with_pause_survives():
    # "no, no, don't click that" spoken deliberately, with a beat between
    ws = words("no no dont click that", gaps=[1.8, 0, 0, 0])
    cuts = detect_disfluencies(ws, 0, len(ws), EditorConfig())
    assert cuts == []
