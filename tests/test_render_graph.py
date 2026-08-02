import pytest

from config import EditorConfig
from models import Span
from render import build_filter_graph


def test_single_segment_graph():
    graph = build_filter_graph([Span(1.0, 5.0)], EditorConfig(crossfade_ms=0))
    assert "trim=start=1.000000:end=5.000000" in graph
    assert "atrim=start=1.000000:end=5.000000" in graph
    assert "concat=n=1:v=1:a=1[outv][outa]" in graph


def test_multi_segment_graph_labels_every_segment():
    segs = [Span(0.0, 2.0), Span(4.0, 6.0), Span(8.0, 10.0)]
    graph = build_filter_graph(segs, EditorConfig(crossfade_ms=0))
    for i in range(3):
        assert f"[v{i}]" in graph
        assert f"[a{i}]" in graph
    assert "concat=n=3:v=1:a=1[outv][outa]" in graph


def test_timestamps_are_reset_per_segment():
    graph = build_filter_graph([Span(5.0, 7.0)], EditorConfig(crossfade_ms=0))
    assert "setpts=PTS-STARTPTS" in graph
    assert "asetpts=PTS-STARTPTS" in graph


def test_fades_are_applied_when_crossfade_enabled():
    graph = build_filter_graph([Span(0.0, 4.0)], EditorConfig(crossfade_ms=75))
    assert "afade=t=in:st=0:d=0.075" in graph
    assert "afade=t=out:st=3.925:d=0.075" in graph


def test_fade_is_skipped_on_segments_too_short_to_hold_one():
    graph = build_filter_graph([Span(0.0, 0.1)], EditorConfig(crossfade_ms=75))
    assert "afade" not in graph


def test_no_fade_when_crossfade_is_zero():
    graph = build_filter_graph([Span(0.0, 4.0)], EditorConfig(crossfade_ms=0))
    assert "afade" not in graph


def test_empty_segments_raises():
    with pytest.raises(ValueError):
        build_filter_graph([], EditorConfig())


def test_fade_never_overlaps_segments_so_durations_are_preserved():
    """The whole reason we use afade rather than acrossfade.

    Every segment's atrim covers exactly its own span, with no overlap into
    the neighbour. Total audio duration therefore equals the sum of segment
    durations, so audio cannot drift out of sync with video across many cuts.
    """
    segs = [Span(0.0, 2.0), Span(4.0, 6.0), Span(8.0, 10.0)]
    graph = build_filter_graph(segs, EditorConfig(crossfade_ms=75))
    assert "acrossfade" not in graph
    for seg in segs:
        assert f"atrim=start={seg.start:.6f}:end={seg.end:.6f}" in graph
