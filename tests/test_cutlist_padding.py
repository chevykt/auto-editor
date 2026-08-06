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


def test_a_speech_cut_grows_outward_to_swallow_the_whole_fragment():
    """Reported by ear: you could still hear the front of the cut word.

    A disfluency span IS the speech being removed, so shrinking it inward
    leaves the edges of the fumble audible. Measured on a real recording: a
    'config' fragment ran 15.30-15.76, and pulling in 30ms each side left
    60ms of it in the render, which is plainly audible.
    """
    fragment = [CutSpan(15.30, 15.76, {CutReason.DISFLUENCY})]
    out = apply_padding(fragment, EditorConfig(padding_ms=30))[0]

    assert out.start < 15.30, "must start before the fragment does"
    assert out.end > 15.76, "must end after the fragment does"
    assert out.duration > 0.46, "the whole fragment, plus margin"


def test_a_silence_cut_still_shrinks_inward():
    """The opposite case, and why direction has to depend on content.

    A silence span contains no speech, so a margin of quiet on each side is
    what stops a cut clipping a breath or a plosive.
    """
    gap = [CutSpan(1.0, 3.0, {CutReason.SILENCE})]
    out = apply_padding(gap, EditorConfig(silence_padding_ms=75))[0]

    assert out.start > 1.0 and out.end < 3.0
    assert out.duration < 2.0


def test_silence_shrinks_while_speech_grows():
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE}),
            CutSpan(5.0, 7.0, {CutReason.DISFLUENCY}),
            CutSpan(9.0, 11.0, {CutReason.RETAKE_LOSER})]
    out = apply_padding(cuts, cfg)
    assert [round(c.start, 3) for c in out] == [1.1, 4.97, 8.97]
    assert [round(c.end, 3) for c in out] == [2.9, 7.03, 11.03]


def test_a_mixed_span_is_treated_as_speech():
    """It contains speech, so it grows to remove all of it."""
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    cuts = [CutSpan(1.0, 3.0, {CutReason.SILENCE, CutReason.DISFLUENCY})]
    assert round(apply_padding(cuts, cfg)[0].start, 3) == 0.97


def test_each_boundary_of_a_merged_span_uses_its_own_origin():
    """A silence gap merged with an adjacent stutter has one boundary of each.

    They want opposite treatment: the silence end shrinks to protect a breath,
    the disfluency end grows to swallow the fumble. One direction for both is
    wrong at whichever end loses.
    """
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    sources = [CutSpan(1.0, 2.0, {CutReason.SILENCE}),
               CutSpan(2.0, 2.5, {CutReason.DISFLUENCY})]
    merged = merge_spans(sources)
    assert len(merged) == 1

    out = apply_padding(merged, cfg, sources=sources)[0]
    assert round(out.start, 3) == 1.1, "silence start shrinks inward"
    assert round(out.end, 3) == 2.53, "disfluency end grows outward"


def test_a_cut_can_never_start_before_zero():
    cfg = EditorConfig(padding_ms=100)
    out = apply_padding([CutSpan(0.02, 1.0, {CutReason.DISFLUENCY})], cfg)
    assert out[0].start == 0.0


def test_unattributable_boundary_falls_back_to_speech_treatment():
    """Unknown provenance is treated as speech: better to over-remove a
    fragment than to leave a fumble audible."""
    cfg = EditorConfig(padding_ms=30, silence_padding_ms=100)
    out = apply_padding([CutSpan(1.0, 3.0, {CutReason.SILENCE})], cfg,
                        sources=[CutSpan(50.0, 60.0, {CutReason.SILENCE})])
    assert round(out[0].start, 3) == 0.97
