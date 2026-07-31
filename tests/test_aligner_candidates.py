from aligner import find_candidates, suppress_overlaps
from config import EditorConfig
from conftest import words


def test_exact_match_scores_one():
    ws = words("click the blue button now")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    assert cands
    assert max(c.score for c in cands) == 1.0


def test_paraphrase_still_matches_above_threshold():
    ws = words("now click on the blue button")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    assert cands
    assert max(c.score for c in cands) >= 0.75


def test_unrelated_text_produces_no_candidates():
    ws = words("completely different words entirely")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    assert cands == []


def test_search_is_confined_to_word_range():
    ws = words("click the blue button and then click the blue button")
    cands = find_candidates(ws, 6, len(ws), "click the blue button", EditorConfig())
    assert cands
    assert all(c.start_idx >= 6 for c in cands)


def test_suppress_overlaps_keeps_best_of_each_cluster():
    ws = words("click the blue button pause click the blue button")
    cands = find_candidates(ws, 0, len(ws), "click the blue button", EditorConfig())
    kept = suppress_overlaps(cands)
    assert len(kept) == 2
    assert kept[0].start_idx == 0
    assert kept[1].start_idx == 5


def test_suppress_overlaps_on_empty_list():
    assert suppress_overlaps([]) == []
