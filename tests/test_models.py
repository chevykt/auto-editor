from models import Word, Span, CutSpan, CutReason, MatchStatus, normalize


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Config-,") == "config"
    assert normalize("  The  ") == "the"
    assert normalize("!!!") == ""


def test_span_duration():
    assert Span(1.0, 2.5).duration == 1.5


def test_cutspan_duration_and_reasons_are_mutable_set():
    c = CutSpan(1.0, 2.0, {CutReason.SILENCE}, "gap")
    c.reasons.add(CutReason.DISFLUENCY)
    assert c.duration == 1.0
    assert c.reasons == {CutReason.SILENCE, CutReason.DISFLUENCY}


def test_word_is_hashable_and_frozen():
    w = Word("hello", 0.0, 0.4)
    assert {w}
    assert w.text == "hello"


def test_enums_serialise_as_plain_strings():
    assert CutReason.SILENCE.value == "silence"
    assert MatchStatus.REVIEW.value == "review"
