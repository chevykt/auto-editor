"""Tokens long enough to be hiding a false start Whisper absorbed.

These are reported, never applied. A long token has innocent explanations, and
an earlier attempt at cutting this class automatically produced six wrong cuts
on a real recording. Flagging costs nothing when wrong.
"""
from config import EditorConfig
from models import CutReason
from suspects import find_suspects, suspect_cuts
from conftest import words


def normal(count: int = 30):
    """A recording of ordinary 0.3s words."""
    return words(" ".join(f"w{i}" for i in range(count)), dur=0.3)


def with_outlier(duration: float, at: int = 10, count: int = 30):
    """The same, with one token stretched - a false start absorbed into it."""
    ws = normal(count)
    stretched = ws[at]
    shifted = [w for w in ws[:at]]
    shifted.append(type(stretched)(stretched.text, stretched.start,
                                   stretched.start + duration))
    offset = duration - 0.3
    for w in ws[at + 1:]:
        shifted.append(type(w)(w.text, round(w.start + offset, 3),
                               round(w.end + offset, 3)))
    return shifted


# --- finding them --------------------------------------------------------

def test_an_extreme_outlier_is_flagged():
    """The real case: 'and' ran 2.06s against a 0.32s median, 6.4x."""
    found = find_suspects(with_outlier(2.0), EditorConfig())
    assert len(found) == 1
    assert found[0].word_index == 10
    assert found[0].ratio > 5.0


def test_an_ordinary_recording_flags_nothing():
    assert find_suspects(normal(), EditorConfig()) == []


def test_a_mild_outlier_is_left_alone():
    """4.3x was the next candidate on the real recording and was innocent.

    The threshold is deliberately above it: missing a mild false start costs
    nothing, and flagging every drawn-out word would make the section useless.
    """
    assert find_suspects(with_outlier(1.4), EditorConfig()) == []


def test_the_absolute_floor_also_has_to_be_cleared():
    """A recording of very fast words must not flag ordinary ones just because
    the median is tiny."""
    fast = words(" ".join(f"w{i}" for i in range(30)), dur=0.05)
    # 0.5s is 10x that median, but still not a suspiciously long word
    stretched = list(fast[:10]) + [type(fast[10])(fast[10].text, fast[10].start,
                                                  fast[10].start + 0.5)]
    assert find_suspects(stretched + list(fast[11:]),
                         EditorConfig(suspect_min_duration_s=0.9)) == []


def test_too_few_words_to_know_what_normal_is():
    assert find_suspects(with_outlier(2.0, at=2, count=6), EditorConfig()) == []


def test_disabled_by_config():
    assert find_suspects(with_outlier(2.0),
                         EditorConfig(flag_suspects=False)) == []


# --- what would be removed -----------------------------------------------

def test_the_proposed_cut_leaves_the_last_attempt():
    ws = with_outlier(2.0)
    cuts = suspect_cuts(find_suspects(ws, EditorConfig()), ws, EditorConfig())

    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.SUSPECT}
    assert cuts[0].start == ws[10].start
    assert cuts[0].end < ws[10].end, "the surviving attempt is at the end"
    assert "possible hidden false start" in cuts[0].detail


def test_the_detail_names_the_evidence():
    ws = with_outlier(2.0)
    detail = suspect_cuts(find_suspects(ws, EditorConfig()), ws,
                          EditorConfig())[0].detail
    assert "2.00s" in detail
    assert "x the median" in detail


def test_no_suspects_means_no_cuts():
    assert suspect_cuts([], normal(), EditorConfig()) == []


def test_a_cut_that_would_invert_is_dropped():
    """If one median's width is the whole token there is nothing to remove."""
    ws = words(" ".join(f"w{i}" for i in range(30)), dur=0.3)
    from suspects import Suspect
    assert suspect_cuts([Suspect(0, "w0", 0.0, 0.2, 9.9)], ws,
                        EditorConfig()) == []
