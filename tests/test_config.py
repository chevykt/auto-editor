from config import EditorConfig


def test_defaults_match_spec():
    c = EditorConfig()
    assert c.threshold == 0.75
    assert c.silence_ms == 600
    assert c.padding_ms == 75
    assert c.model == "medium"
    assert c.clean_stutters is True
    assert c.tiebreak == "latest"
    assert c.duplicate_window_s == 4.0
    assert c.max_disfluency_words == 5
    assert c.min_segment_ms == 150
    assert c.crossfade_ms == 75
    assert c.review_margin == 0.05
    assert c.lookahead_s == 30.0


def test_second_derived_properties():
    c = EditorConfig(silence_ms=600, padding_ms=75, min_segment_ms=150,
                     crossfade_ms=75)
    assert c.silence_s == 0.6
    assert c.padding_s == 0.075
    assert c.min_segment_s == 0.15
    assert c.crossfade_s == 0.075
