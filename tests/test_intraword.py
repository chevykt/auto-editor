"""Repeats Whisper collapsed into one token, found from the audio.

Every test here builds an envelope by hand, so the detection logic is exercised
with no media, no ffmpeg and no model - the same property that makes the rest
of the pipeline debuggable.
"""
from config import EditorConfig
from intraword import (Envelope, Gap, detect_collapsed_repeats, find_gaps,
                       silence_threshold)
from models import CutReason, Word

LOUD, QUIET = 3000.0, 20.0

# The pass is opt-in, so every detector test has to turn it on.
ON = EditorConfig(intraword_repeats=True)


def envelope(pattern: str, frame_ms: int = 10) -> Envelope:
    """'##..##' -> loud, loud, quiet, quiet, loud, loud. One char per frame."""
    return Envelope(frames=[LOUD if c == "#" else QUIET for c in pattern],
                    frame_s=frame_ms / 1000.0)


def word(text: str, start: float, end: float) -> Word:
    return Word(text, start, end)


# --- threshold -----------------------------------------------------------

def test_threshold_sits_between_the_noise_floor_and_speech():
    env = envelope("#" * 80 + "." * 20)
    t = silence_threshold(env)
    assert QUIET < t < LOUD


def test_threshold_survives_an_empty_envelope():
    assert silence_threshold(Envelope(frames=[], frame_s=0.01)) == 0.0


# --- finding gaps --------------------------------------------------------

def test_an_interior_silence_run_is_found():
    """The real case: 70ms of speech, 150ms quiet, then the second attempt."""
    env = envelope("#" * 7 + "." * 15 + "#" * 18 + "#" * 60)
    words = [word("the", 0.0, 0.40)]

    gaps = find_gaps(env, words, EditorConfig(intraword_min_gap_ms=120))
    assert len(gaps) == 1
    assert gaps[0].word_index == 0
    assert round(gaps[0].start, 3) == 0.07
    assert round(gaps[0].duration, 3) == 0.15


def test_a_gap_shorter_than_the_floor_is_ignored():
    """A plosive closure runs 50-80ms and must not read as two attempts."""
    env = envelope("#" * 10 + "." * 7 + "#" * 23 + "#" * 60)
    gaps = find_gaps(env, [word("button", 0.0, 0.40)],
                     EditorConfig(intraword_min_gap_ms=120))
    assert gaps == []


def test_silence_at_the_start_of_a_token_is_edge_slop_not_a_gap():
    """Whisper's word boundaries are approximate; leading quiet is not a pause
    between two attempts."""
    env = envelope("." * 20 + "#" * 20 + "#" * 60)
    gaps = find_gaps(env, [word("the", 0.0, 0.40)], EditorConfig())
    assert gaps == []


def test_silence_at_the_end_of_a_token_is_edge_slop_too():
    env = envelope("#" * 20 + "." * 20 + "#" * 60)
    gaps = find_gaps(env, [word("the", 0.0, 0.40)], EditorConfig())
    assert gaps == []


def test_a_clean_word_has_no_gaps():
    env = envelope("#" * 40 + "#" * 60)
    assert find_gaps(env, [word("the", 0.0, 0.40)], EditorConfig()) == []


def test_two_gaps_in_one_token_are_both_reported():
    env = envelope("#" * 5 + "." * 14 + "#" * 5 + "." * 14 + "#" * 12 + "#" * 60)
    gaps = find_gaps(env, [word("the", 0.0, 0.50)],
                     EditorConfig(intraword_min_gap_ms=120))
    assert len(gaps) == 2


def test_a_token_too_short_to_hold_a_gap_is_skipped():
    env = envelope("#" * 100)
    assert find_gaps(env, [word("a", 0.0, 0.02)], EditorConfig()) == []


def test_no_words_or_no_frames():
    assert find_gaps(envelope("###"), [], EditorConfig()) == []
    assert find_gaps(Envelope([], 0.01), [word("a", 0.0, 0.4)],
                     EditorConfig()) == []


# --- turning gaps into cuts ----------------------------------------------

def test_the_cut_removes_the_first_attempt_and_the_gap():
    words = [word("the", 10.00, 10.40)]
    gaps = [Gap(0, 10.08, 10.23)]

    cuts = detect_collapsed_repeats(words, gaps, ON)
    assert len(cuts) == 1
    assert (cuts[0].start, cuts[0].end) == (10.00, 10.23)
    assert cuts[0].reasons == {CutReason.DISFLUENCY}
    assert "collapsed repeat" in cuts[0].detail


def test_three_attempts_leave_only_the_last():
    """'the the the' in one token: cut to the end of the LAST gap."""
    words = [word("the", 10.00, 10.60)]
    gaps = [Gap(0, 10.08, 10.23), Gap(0, 10.30, 10.45)]

    cuts = detect_collapsed_repeats(words, gaps, ON)
    assert len(cuts) == 1
    assert cuts[0].end == 10.45


def test_an_implausibly_long_cut_is_refused():
    """More likely a mis-timestamped token than a stutter."""
    words = [word("the", 10.0, 14.0)]
    gaps = [Gap(0, 12.0, 13.5)]
    assert detect_collapsed_repeats(
        words, gaps,
        EditorConfig(intraword_repeats=True, intraword_max_cut_s=1.0)) == []


def test_disabled_by_config():
    words = [word("the", 10.00, 10.40)]
    gaps = [Gap(0, 10.08, 10.23)]
    assert detect_collapsed_repeats(
        words, gaps, EditorConfig(intraword_repeats=False)) == []


def test_no_gaps_means_no_cuts():
    assert detect_collapsed_repeats([word("the", 0.0, 0.4)], [], ON) == []


def test_cuts_come_back_in_timeline_order():
    words = [word("a", 1.0, 1.5), word("b", 5.0, 5.5)]
    gaps = [Gap(1, 5.1, 5.3), Gap(0, 1.1, 1.3)]
    cuts = detect_collapsed_repeats(words, gaps, ON)
    assert [c.start for c in cuts] == [1.0, 5.0]


def test_a_token_overhanging_a_pause_is_not_a_stutter():
    """Whisper word tokens routinely run into the following pause.

    Without a flank check on BOTH sides, 'minutes.', 'from', 'part.' and 'you'
    were all flagged as stutters on a real recording, cutting three seconds of
    ordinary speech. Speech, then a long quiet tail with only a blip after it,
    is a token overhanging a pause - not two attempts.
    """
    env = envelope("#" * 20 + "." * 25 + "#" + "#" * 60)
    gaps = find_gaps(env, [word("minutes.", 0.0, 0.46)],
                     EditorConfig(intraword_min_gap_ms=120,
                                  intraword_flank_ms=40))
    assert gaps == []


def test_real_speech_either_side_still_counts():
    """The genuine case must survive the flank guard."""
    env = envelope("#" * 7 + "." * 15 + "#" * 18 + "#" * 60)
    gaps = find_gaps(env, [word("the", 0.0, 0.40)],
                     EditorConfig(intraword_min_gap_ms=120,
                                  intraword_flank_ms=40))
    assert len(gaps) == 1


def test_a_lone_blip_before_the_gap_is_not_enough():
    env = envelope("#" + "." * 25 + "#" * 20 + "#" * 60)
    gaps = find_gaps(env, [word("x", 0.0, 0.46)],
                     EditorConfig(intraword_min_gap_ms=120,
                                  intraword_flank_ms=40))
    assert gaps == []


def test_the_pass_is_off_by_default():
    """It finds real collapsed repeats, but on the one real recording available
    it also flagged six ordinary words - removing 2.6s of speech to recover a
    70ms stutter. That is the wrong side of this tool's bias, so it is opt-in.
    """
    assert EditorConfig().intraword_repeats is False

    words = [word("the", 10.00, 10.40)]
    gaps = [Gap(0, 10.08, 10.23)]
    assert detect_collapsed_repeats(words, gaps, EditorConfig()) == []
    assert detect_collapsed_repeats(words, gaps, ON), "still works when asked for"
