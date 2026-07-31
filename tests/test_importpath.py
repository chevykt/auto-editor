"""Guards the test-import contract the whole suite depends on.

Every other test file does `from conftest import words`. That only works
because pytest.ini puts both the project root and tests/ on pythonpath.
If this file fails, no other test file can import its fixtures.
"""
from conftest import words
from models import Word


def test_conftest_is_importable_as_a_top_level_module():
    assert words("hello world")


def test_words_builds_contiguous_timestamps():
    ws = words("a b c", dur=0.3)
    assert [(w.start, w.end) for w in ws] == [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9)]


def test_words_applies_gaps_after_each_token():
    ws = words("a b c", dur=0.3, gaps=[0.5, 0.0])
    assert ws[1].start == 0.8
    assert ws[2].start == 1.1


def test_words_gaps_shorter_than_token_list_are_padded():
    ws = words("a b c", dur=0.3, gaps=[0.5])
    assert ws[2].start == 1.1


def test_words_returns_model_instances():
    assert all(isinstance(w, Word) for w in words("a b"))
