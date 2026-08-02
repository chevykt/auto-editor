from script_parser import parse_script


def test_splits_on_blank_lines_by_default():
    text = "First beat here.\nStill first beat.\n\nSecond beat.\n\n\nThird beat."
    lines = parse_script(text)
    assert [l.text for l in lines] == [
        "First beat here. Still first beat.",
        "Second beat.",
        "Third beat.",
    ]
    assert [l.index for l in lines] == [0, 1, 2]


def test_split_sentences_mode():
    text = "Click the button. Then wait! Does it work? Yes."
    lines = parse_script(text, split_sentences=True)
    assert [l.text for l in lines] == [
        "Click the button.",
        "Then wait!",
        "Does it work?",
        "Yes.",
    ]


def test_ignores_empty_and_whitespace_only_blocks():
    lines = parse_script("\n\n   \n\nOnly beat.\n\n   \n")
    assert [l.text for l in lines] == ["Only beat."]
    assert [l.index for l in lines] == [0]


def test_empty_script_returns_empty_list():
    assert parse_script("   \n\n  ") == []


def test_word_count_property():
    assert parse_script("one two three")[0].word_count == 3
