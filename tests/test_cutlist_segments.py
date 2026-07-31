from config import EditorConfig
from cutlist import build_cutlist, keep_segments
from models import CutReason, CutSpan


def test_keep_segments_are_the_complement_of_cuts():
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(4.0, 5.0, {CutReason.SILENCE})]
    segs = keep_segments(cuts, duration=6.0)
    assert [(s.start, s.end) for s in segs] == [(0.0, 1.0), (2.0, 4.0), (5.0, 6.0)]


def test_cut_at_the_very_start_produces_no_leading_segment():
    cuts = [CutSpan(0.0, 2.0, {CutReason.SILENCE})]
    segs = keep_segments(cuts, duration=5.0)
    assert [(s.start, s.end) for s in segs] == [(2.0, 5.0)]


def test_cut_running_to_the_end_produces_no_trailing_segment():
    cuts = [CutSpan(3.0, 5.0, {CutReason.SILENCE})]
    segs = keep_segments(cuts, duration=5.0)
    assert [(s.start, s.end) for s in segs] == [(0.0, 3.0)]


def test_no_cuts_yields_the_whole_timeline():
    assert [(s.start, s.end) for s in keep_segments([], duration=5.0)] == [(0.0, 5.0)]


def test_sliver_between_two_cuts_is_absorbed():
    # 0.05s survivor between two cuts, below the 150ms floor
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(2.05, 3.0, {CutReason.DISFLUENCY})]
    final = build_cutlist(cuts, duration=5.0, cfg=EditorConfig(min_segment_ms=150,
                                                              padding_ms=0))
    assert len(final) == 1
    assert (final[0].start, final[0].end) == (1.0, 3.0)


def test_segment_above_the_floor_survives():
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(2.5, 3.0, {CutReason.DISFLUENCY})]
    final = build_cutlist(cuts, duration=5.0, cfg=EditorConfig(min_segment_ms=150,
                                                              padding_ms=0))
    assert len(final) == 2


def test_absorption_does_not_cascade_into_deleting_everything():
    # a long survivor sits after two slivers; it must not be swallowed
    cuts = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(2.05, 3.0, {CutReason.SILENCE}),
            CutSpan(3.05, 4.0, {CutReason.SILENCE})]
    final = build_cutlist(cuts, duration=20.0, cfg=EditorConfig(min_segment_ms=150,
                                                               padding_ms=0))
    segs = keep_segments(final, duration=20.0)
    assert (0.0, 1.0) in [(s.start, s.end) for s in segs]
    assert (4.0, 20.0) in [(s.start, s.end) for s in segs]


def test_build_cutlist_merges_pads_and_absorbs_in_order():
    cuts = [CutSpan(4.0, 5.0, {CutReason.SILENCE}),
            CutSpan(1.0, 2.0, {CutReason.SILENCE}),
            CutSpan(1.5, 2.5, {CutReason.DISFLUENCY})]
    final = build_cutlist(cuts, duration=10.0, cfg=EditorConfig(padding_ms=50))
    assert [round(c.start, 3) for c in final] == [1.05, 4.05]
    assert final[0].reasons == {CutReason.SILENCE, CutReason.DISFLUENCY}
