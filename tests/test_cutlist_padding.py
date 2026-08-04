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


def test_default_padding_leaves_a_usable_cut_on_a_short_stutter():
    """Why the default is 30ms and not 75ms.

    Measured on a real recording: a duplicated "the" ran 220ms. At 75ms the
    pull-in took 150ms of that, leaving only 70ms actually removed - two
    thirds of the stutter survived into the render. Padding sized for silence
    gaps is disproportionate against a single short word.

    If someone raises this default again, this is the case to re-measure.
    """
    stutter = [CutSpan(4.020, 4.240, {CutReason.DISFLUENCY})]

    kept = apply_padding(stutter, EditorConfig())[0]
    assert kept.duration > 0.15, "default must remove most of a 220ms stutter"

    old = apply_padding(stutter, EditorConfig(padding_ms=75))[0]
    assert old.duration < 0.08, "75ms left most of the stutter in"


def test_applies_to_every_reason_type_alike():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE}),
            CutSpan(5.0, 7.0, {CutReason.DISFLUENCY}),
            CutSpan(9.0, 11.0, {CutReason.RETAKE_LOSER})]
    out = apply_padding(cuts, EditorConfig(padding_ms=100))
    assert [round(c.start, 3) for c in out] == [1.1, 5.1, 9.1]
