"""Turning Keon's voiceover into beat windows and anchor times."""
import pytest

from config import EditorConfig
from conftest import words
from models import Span
from tutorial.plan import Beat, Plan, Step
from tutorial.voice import (VoiceError, beat_windows, clean_voiceover, find_anchor,
                            to_clean_time)


def test_clean_time_shifts_by_removed_spans():
    kept = [Span(0, 2), Span(5, 8)]
    assert to_clean_time(1.0, kept) == 1.0
    assert to_clean_time(6.0, kept) == 3.0
    assert to_clean_time(3.0, kept) is None


def test_windows_tile_the_whole_voiceover_including_an_intro():
    assert beat_windows([1.5, 4.0, 9.0], 12.0) == ((0.0, 4.0), (4.0, 9.0), (9.0, 12.0))


def test_anchor_is_the_start_of_the_phrase():
    ws = words("start on your dashboard and click new project")
    assert find_anchor(ws, (0, len(ws)), "New project") == ws[6].start


def test_a_paraphrased_anchor_is_none():
    ws = words("then press the create button")
    assert find_anchor(ws, (0, len(ws)), "New project") is None


def test_anchor_search_stays_inside_the_beat():
    ws = words("click new project then later new project again")
    assert find_anchor(ws, (3, len(ws)), "new project") == ws[5].start


def plan_of(*beats):
    return Plan("t", "https://example.com", tuple(beats))


@pytest.fixture
def voice(monkeypatch, tmp_path):
    """Stub the speech model and the audio render; the timing logic is real."""
    spoken = {}

    def transcribe(path, cfg, on_progress=None):
        return spoken["words"], spoken["duration"]

    rendered = {}
    monkeypatch.setattr("tutorial.voice.transcribe", transcribe)
    monkeypatch.setattr("tutorial.voice.render_audio",
                        lambda src, out, segments, cfg: rendered.update(segments=segments))

    def run(ws, duration, plan):
        spoken.update(words=ws, duration=duration)
        timing, warnings = clean_voiceover(tmp_path / "v.wav", plan,
                                           EditorConfig(verbatim=True), tmp_path / "c.wav")
        return timing, warnings, rendered["segments"]
    return run


def test_windows_and_anchors_come_out_in_cleaned_time(voice):
    # a 2 s pause after "button" is silence the cleaning removes
    ws = words("open the dashboard and click the new project button "
               "then type a name for it", gaps=[0] * 8 + [2.0])
    plan = plan_of(
        Beat("open the dashboard and click the new project button",
             (Step("click", "x", None, "new project"),)),
        Beat("then type a name for it", (Step("type", "y", "app", "name"),)))
    timing, warnings, kept = voice(ws, ws[-1].end, plan)

    assert warnings == []
    assert timing.windows[0][0] == 0.0
    assert timing.windows[-1][1] == pytest.approx(timing.duration)
    assert timing.duration < ws[-1].end, "the pause was cut"
    assert timing.anchors[(0, 0)] == pytest.approx(ws[6].start)
    assert timing.anchors[(1, 0)] < timing.windows[1][1]
    assert timing.anchors[(1, 0)] > timing.windows[1][0]


def test_a_missed_line_stops_with_the_report(voice):
    ws = words("something else entirely was said here today")
    plan = plan_of(Beat("open the dashboard and click the new project button", ()))
    with pytest.raises(VoiceError) as err:
        voice(ws, ws[-1].end, plan)
    assert err.value.report["lines"][0]["status"] == "miss"


def test_missing_anchor_becomes_a_warning_not_an_error(voice):
    ws = words("then press the create button to make it")
    plan = plan_of(Beat("then press the create button to make it",
                        (Step("click", "x", None, "New project"),)))
    timing, warnings, _ = voice(ws, ws[-1].end, plan)
    assert timing.anchors == {}
    assert len(warnings) == 1 and "New project" in warnings[0]
