from config import EditorConfig
from cutlist import apply_padding
from models import CutReason, CutSpan


def test_boundaries_are_pulled_inward():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert out[0].start == 1.075
    assert out[0].end == 2.925


def test_span_shorter_than_twice_padding_is_clamped_not_inverted():
    cuts = [CutSpan(1.0, 1.05, {CutReason.DISFLUENCY})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert not out or out[0].end >= out[0].start


def test_span_exactly_twice_padding_collapses_to_nothing_and_is_dropped():
    cuts = [CutSpan(1.0, 1.15, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert out == []


def test_zero_padding_is_a_passthrough():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=0))
    assert (out[0].start, out[0].end) == (1.0, 3.0)


def test_reasons_and_detail_survive_padding():
    cuts = [CutSpan(1.0, 3.0, {CutReason.RETAKE_LOSER}, "loser")]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert out[0].reasons == {CutReason.RETAKE_LOSER}
    assert out[0].detail == "loser"


def test_applies_to_every_reason_type_alike():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE}),
            CutSpan(5.0, 7.0, {CutReason.DISFLUENCY}),
            CutSpan(9.0, 11.0, {CutReason.RETAKE_LOSER})]
    out = apply_padding(cuts, EditorConfig(padding_ms=100))
    assert [round(c.start, 3) for c in out] == [1.1, 5.1, 9.1]
