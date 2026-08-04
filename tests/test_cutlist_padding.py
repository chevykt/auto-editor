from config import EditorConfig
from cutlist import apply_padding, merge_spans
from models import CutReason, CutSpan


def test_boundaries_are_pulled_inward():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(silence_padding_ms=75))
    assert out[0].start == 1.075
    assert out[0].end == 2.925


def test_span_shorter_than_twice_padding_is_clamped_not_inverted():
    cuts = [CutSpan(1.0, 1.05, {CutReason.DISFLUENCY})]
    out = apply_padding(cuts, EditorConfig(padding_ms=75))
    assert not out or out[0].end >= out[0].start


def test_span_exactly_twice_padding_collapses_to_nothing_and_is_dropped():
    cuts = [CutSpan(1.0, 1.15, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(silence_padding_ms=75))
    assert out == []


def test_zero_padding_is_a_passthrough():
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(cuts, EditorConfig(padding_ms=0, silence_padding_ms=0))
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


def test_silence_is_padded_more_than_a_disfluency():
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE}),
            CutSpan(5.0, 7.0, {CutReason.DISFLUENCY}),
            CutSpan(9.0, 11.0, {CutReason.RETAKE_LOSER})]
    out = apply_padding(cuts, cfg)
    assert [round(c.start, 3) for c in out] == [1.1, 5.03, 9.03]


def test_a_mixed_span_is_treated_as_non_silence():
    """The disfluency is what needs protecting from over-trimming."""
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE, CutReason.DISFLUENCY})]
    assert round(apply_padding(cuts, cfg)[0].start, 3) == 1.03


def test_each_boundary_of_a_merged_span_uses_its_own_origin():
    """A silence gap merged with an adjacent stutter has one boundary of each.

    Padding both ends the same way is wrong at whichever end loses: too tight
    on the silence risks clipping a breath, too generous on the disfluency
    leaves most of the stutter in.
    """
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    sources = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
               CutSpan(2.0, 2.5, {CutReason.DISFLUENCY})]
    merged = merge_spans(sources)
    assert len(merged) == 1

    out = apply_padding(merged, cfg, sources=sources)[0]
    assert round(out.start, 3) == 1.1, "silence start gets the generous value"
    assert round(out.end, 3) == 2.47, "disfluency end gets the tight value"


def test_unattributable_boundary_falls_back_to_the_tight_value():
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    out = apply_padding([CutSpan(1.0, 3.0, {CutReason.SILENCE})], cfg,
                        sources=[CutSpan(50.0, 60.0, {CutReason.SILENCE})])
    assert round(out[0].start, 3) == 1.03
