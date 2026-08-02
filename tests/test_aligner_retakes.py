import pytest

from aligner import Candidate, resolve_retakes
from config import EditorConfig


def test_highest_score_wins():
    cands = [Candidate(0.80, 0, 4), Candidate(0.95, 10, 14)]
    winner, losers = resolve_retakes(cands, EditorConfig())
    assert winner.start_idx == 10
    assert [l.start_idx for l in losers] == [0]


def test_tie_breaks_toward_later_occurrence_by_default():
    cands = [Candidate(0.90, 0, 4), Candidate(0.90, 10, 14)]
    winner, losers = resolve_retakes(cands, EditorConfig(tiebreak="latest"))
    assert winner.start_idx == 10
    assert [l.start_idx for l in losers] == [0]


def test_near_tie_within_rounding_also_breaks_later():
    cands = [Candidate(0.9004, 0, 4), Candidate(0.9001, 10, 14)]
    winner, _ = resolve_retakes(cands, EditorConfig(tiebreak="latest"))
    assert winner.start_idx == 10


def test_score_tiebreak_prefers_earlier_on_exact_tie():
    cands = [Candidate(0.90, 0, 4), Candidate(0.90, 10, 14)]
    winner, losers = resolve_retakes(cands, EditorConfig(tiebreak="score"))
    assert winner.start_idx == 0
    assert [l.start_idx for l in losers] == [10]


def test_single_candidate_has_no_losers():
    winner, losers = resolve_retakes([Candidate(0.9, 0, 4)], EditorConfig())
    assert winner.start_idx == 0
    assert losers == []


def test_empty_candidates_raises():
    with pytest.raises(ValueError):
        resolve_retakes([], EditorConfig())
