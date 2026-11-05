"""Fitting the recorded footage to the voiceover, beat by beat."""
import pytest

from tutorial.fit import FitError, Hold, Play, fit_all, fit_beat
from tutorial.plan import Beat, Plan, Step
from tutorial.runner import SETTLE_S, StepEvent
from tutorial.voice import VoiceTiming


def total(pieces):
    return sum(p.seconds if isinstance(p, Hold) else (p.src_end - p.src_start) / p.speed
               for p in pieces)


def test_short_footage_holds_before_the_anchor_so_the_click_lands_on_the_word():
    pieces = fit_beat((0, 3), [], [(2.0, 5.0)], (0, 6))
    assert pieces[0] == Hold(0, 3.0)            # wait first
    assert pieces[1] == Play(0, 2.0, 1)         # then move and click on the word
    assert total(pieces) == pytest.approx(6.0)


def test_short_footage_with_no_anchor_plays_then_holds_the_result():
    assert fit_beat((0, 2), [], [], (0, 5)) == [Play(0, 2, 1), Hold(2, 3)]


def test_long_footage_drops_waits_before_speeding_up():
    pieces = fit_beat((0, 10), [(2, 7)], [], (0, 6))
    assert all(isinstance(p, Hold) or p.speed == 1 for p in pieces)
    assert total(pieces) == pytest.approx(6.0)
    assert Play(0, 3, 1) in pieces, "the start of the wait stays visible"


def test_footage_too_long_even_without_waits_is_sped_up():
    pieces = fit_beat((0, 9), [], [], (0, 6))
    assert pieces == [Play(0, 9, 1.5)]


def test_speed_up_is_capped():
    with pytest.raises(FitError):
        fit_beat((0, 10), [], [], (0, 4))


# --- the whole video --------------------------------------------------------

T0 = 1000.0


def ev(beat, step, kind, start, action, end):
    return StepEvent(beat, step, kind, T0 + start, T0 + action, T0 + end)


def plan_of(*steps_per_beat):
    return Plan("t", "https://x", tuple(Beat(f"beat {i}", tuple(steps))
                                        for i, steps in enumerate(steps_per_beat)))


CLICK = Step("click", "#a", None, None)
ANCHORED = Step("click", "#a", None, "new project")


def test_whole_video_matches_the_voiceover_length():
    plan = plan_of([CLICK], [ANCHORED], [CLICK])
    events = [ev(0, 0, "click", 0.0, 0.5, 0.6), ev(1, 0, "click", 1.2, 1.7, 1.8),
              ev(2, 0, "click", 2.4, 2.9, 3.0)]
    timing = VoiceTiming(12.0, ((0, 4), (4, 9), (9, 12)), {(1, 0): 6.0})
    pieces, warnings = fit_all(events, T0, timing, plan)
    assert total(pieces) == pytest.approx(12.0)
    assert warnings == []


def test_the_anchored_click_lands_on_its_word():
    plan = plan_of([ANCHORED])
    events = [ev(0, 0, "click", 0.0, 0.5, 0.6)]
    timing = VoiceTiming(5.0, ((0, 5),), {(0, 0): 3.0})
    pieces, _ = fit_all(events, T0, timing, plan)
    # output time at which source 0.5 (the click) is shown
    out = 0.0
    for p in pieces:
        if isinstance(p, Hold):
            out += p.seconds
        elif p.src_start <= 0.5 <= p.src_end:
            out += (0.5 - p.src_start) / p.speed
            break
        else:
            out += (p.src_end - p.src_start) / p.speed
    assert out == pytest.approx(3.0)


def test_a_beat_without_steps_is_one_hold():
    plan = plan_of([CLICK], [])
    events = [ev(0, 0, "click", 0.0, 0.5, 0.6)]
    timing = VoiceTiming(8.0, ((0, 3), (3, 8)), {})
    pieces, _ = fit_all(events, T0, timing, plan)
    assert isinstance(pieces[-1], Hold)
    assert pieces[-1].src_at == pytest.approx(0.6 + SETTLE_S)
    assert pieces[-1].seconds == pytest.approx(5.0)


def test_a_first_beat_without_steps_holds_the_first_frame():
    plan = plan_of([], [CLICK])
    events = [ev(1, 0, "click", 0.0, 0.5, 0.6)]
    pieces, _ = fit_all(events, T0, VoiceTiming(6.0, ((0, 2), (2, 6)), {}), plan)
    assert pieces[0] == Hold(0.0, 2.0)


def test_an_anchor_outside_its_beat_is_dropped_with_a_warning():
    plan = plan_of([ANCHORED])
    events = [ev(0, 0, "click", 0.0, 0.5, 0.6)]
    timing = VoiceTiming(5.0, ((0, 5),), {(0, 0): 7.0})
    pieces, warnings = fit_all(events, T0, timing, plan)
    assert total(pieces) == pytest.approx(5.0)
    assert len(warnings) == 1 and "beat 1, step 1" in warnings[0]


def test_an_anchor_on_the_beats_first_words_is_not_a_warning():
    """'Scroll down to step three…' anchored on 'Scroll down': the word opens the
    beat, so the step already starts with it - nothing to pin, nothing to warn."""
    plan = plan_of([ANCHORED])
    events = [ev(0, 0, "scroll", 0.0, 0.01, 0.8)]
    pieces, warnings = fit_all(events, T0, VoiceTiming(3.0, ((0, 3),), {(0, 0): 0.0}), plan)
    assert warnings == []
    assert total(pieces) == pytest.approx(3.0)


def test_a_fit_error_names_the_beat():
    plan = plan_of([CLICK], [CLICK])
    events = [ev(0, 0, "click", 0.0, 0.5, 0.6), ev(1, 0, "click", 1.2, 1.7, 12.0)]
    with pytest.raises(FitError, match="beat 2"):
        fit_all(events, T0, VoiceTiming(4.0, ((0, 2), (2, 4)), {}), plan)


def test_setup_events_are_ignored():
    plan = plan_of([CLICK])
    events = [ev(-1, 0, "goto", -5.0, -5.0, -4.0), ev(0, 0, "click", 0.0, 0.5, 0.6)]
    pieces, _ = fit_all(events, T0, VoiceTiming(3.0, ((0, 3),), {}), plan)
    assert total(pieces) == pytest.approx(3.0)
