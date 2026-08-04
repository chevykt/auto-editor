from config import EditorConfig
from disfluency import detect_truncated_restarts
from models import CutReason, Word
from conftest import words


def test_fragment_followed_by_full_word_is_cut():
    ws = words("the config- configuration screen")
    cuts = detect_truncated_restarts(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].reasons == {CutReason.DISFLUENCY}
    assert cuts[0].start == ws[1].start
    assert cuts[0].end == ws[2].start


def test_short_fragment_below_min_chars_is_ignored():
    # "in" is a prefix of "international" but too short to be a safe signal
    ws = words("in international waters")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_small_growth_is_ignored():
    # "the" -> "there" grows by only 2 chars; likely real speech
    ws = words("the there button")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_identical_words_are_not_a_restart():
    # that is a repeat, handled by the other detector
    ws = words("config config screen")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_long_pause_between_breaks_the_signal():
    ws = words("config- configuration screen", gaps=[1.5, 0.0])
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_non_prefix_is_ignored():
    ws = words("setup configuration screen")
    assert detect_truncated_restarts(ws, 0, len(ws), EditorConfig()) == []


def test_standalone_punctuation_token_does_not_break_adjacency():
    """Found on a real recording.

    Whisper transcribed a mid-word restart as three tokens - 'config', '-',
    'configuration' - with the hyphen a separate zero-duration entry. It
    normalises to nothing, so treating it as a word put a gap between the
    fragment and the full word and the restart went undetected.
    """
    ws = words("the config configuration panel")
    # splice in a zero-width '-' exactly where Whisper put it
    ws = ws[:2] + [Word("-", ws[1].end, ws[1].end)] + ws[2:]

    cuts = detect_truncated_restarts(ws, 0, len(ws), EditorConfig())
    assert len(cuts) == 1
    assert cuts[0].detail == "truncated restart: 'config' -> 'configuration'"
    assert cuts[0].start == ws[1].start
    assert cuts[0].end == ws[3].start


def test_punctuation_token_does_not_itself_look_like_a_fragment():
    ws = words("the config configuration panel")
    ws = ws[:2] + [Word("...", ws[1].end, ws[1].end)] + ws[2:]
    cuts = detect_truncated_restarts(ws, 0, len(ws), EditorConfig())
    assert all("'...'" not in c.detail for c in cuts)
