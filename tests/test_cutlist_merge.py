from cutlist import merge_spans
from models import CutReason, CutSpan


def test_non_overlapping_spans_are_untouched():
    cuts = [CutSpan(0.0, 1.0, {CutReason.SILENCE}),
            CutSpan(2.0, 3.0, {CutReason.DISFLUENCY})]
    assert len(merge_spans(cuts)) == 2


def test_overlapping_spans_merge_and_union_reasons():
    cuts = [CutSpan(0.0, 2.0, {CutReason.RETAKE_LOSER}, "loser"),
            CutSpan(1.0, 3.0, {CutReason.DISFLUENCY}, "stutter")]
    merged = merge_spans(cuts)
    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0.0, 3.0)
    assert merged[0].reasons == {CutReason.RETAKE_LOSER, CutReason.DISFLUENCY}
    assert "loser" in merged[0].detail and "stutter" in merged[0].detail


def test_touching_spans_merge():
    cuts = [CutSpan(0.0, 1.0, {CutReason.SILENCE}),
            CutSpan(1.0, 2.0, {CutReason.SILENCE})]
    assert len(merge_spans(cuts)) == 1


def test_fully_contained_span_does_not_shrink_the_outer_one():
    cuts = [CutSpan(0.0, 5.0, {CutReason.RETAKE_LOSER}),
            CutSpan(1.0, 2.0, {CutReason.DISFLUENCY})]
    merged = merge_spans(cuts)
    assert (merged[0].start, merged[0].end) == (0.0, 5.0)


def test_unsorted_input_is_handled():
    cuts = [CutSpan(5.0, 6.0, {CutReason.SILENCE}),
            CutSpan(0.0, 1.0, {CutReason.SILENCE})]
    merged = merge_spans(cuts)
    assert [c.start for c in merged] == [0.0, 5.0]


def test_does_not_mutate_input():
    original = CutSpan(0.0, 2.0, {CutReason.SILENCE}, "a")
    merge_spans([original, CutSpan(1.0, 3.0, {CutReason.DISFLUENCY}, "b")])
    assert original.end == 2.0
    assert original.reasons == {CutReason.SILENCE}


def test_empty_input():
    assert merge_spans([]) == []
