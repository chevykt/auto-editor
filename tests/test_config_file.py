"""Config file: tuning knobs in TOML, with the command line still winning."""
import pytest

from config import EditorConfig, load_config_file, tunable_names
from edit import config_from_args, parse_args


def write(tmp_path, text: str):
    path = tmp_path / "auto-editor.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_values_load_from_toml(tmp_path):
    path = write(tmp_path, "threshold = 0.9\nsilence_ms = 800\n")
    assert load_config_file(path) == {"threshold": 0.9, "silence_ms": 800}


def test_negative_flags_are_written_positively_in_the_file(tmp_path):
    """The file says clean_stutters = false; argparse wants no_clean_stutters."""
    path = write(tmp_path, "clean_stutters = false\nuse_cache = false\n")
    assert load_config_file(path) == {"no_clean_stutters": True,
                                      "no_cache": True}


def test_a_typo_is_an_error_not_a_silent_default(tmp_path):
    path = write(tmp_path, "treshold = 0.9\n")
    with pytest.raises(ValueError, match="treshold"):
        load_config_file(path)


def test_the_error_lists_the_valid_keys(tmp_path):
    path = write(tmp_path, "nonsense = 1\n")
    with pytest.raises(ValueError, match="threshold"):
        load_config_file(path)


def test_every_editorconfig_field_is_settable(tmp_path):
    """The spec asks for a config file instead of CLI flags for ALL knobs."""
    names = tunable_names()
    assert "threshold" in names and "crossfade_ms" in names
    assert "verbatim" in names and "reorder" in names


def test_an_empty_file_is_valid(tmp_path):
    assert load_config_file(write(tmp_path, "")) == {}


# --- precedence ----------------------------------------------------------

def test_config_file_overrides_code_defaults(tmp_path):
    path = write(tmp_path, "threshold = 0.9\n")
    args = parse_args(["--script", "s.txt", "--video", "v.mp4",
                       "--config", str(path)])
    assert config_from_args(args).threshold == 0.9


def test_command_line_overrides_the_config_file(tmp_path):
    path = write(tmp_path, "threshold = 0.9\nsilence_ms = 800\n")
    args = parse_args(["--script", "s.txt", "--video", "v.mp4",
                       "--config", str(path), "--threshold", "0.6"])
    cfg = config_from_args(args)
    assert cfg.threshold == 0.6, "explicit flag wins"
    assert cfg.silence_ms == 800, "untouched key still comes from the file"


def test_defaults_survive_where_the_file_is_silent(tmp_path):
    path = write(tmp_path, "threshold = 0.9\n")
    args = parse_args(["--script", "s.txt", "--video", "v.mp4",
                       "--config", str(path)])
    assert config_from_args(args).padding_ms == EditorConfig().padding_ms


def test_boolean_knobs_round_trip_through_the_file(tmp_path):
    path = write(tmp_path, "verbatim = true\nclean_stutters = false\n")
    args = parse_args(["--script", "s.txt", "--video", "v.mp4",
                       "--config", str(path)])
    cfg = config_from_args(args)
    assert cfg.verbatim is True
    assert cfg.clean_stutters is False


def test_no_config_flag_leaves_everything_at_defaults():
    args = parse_args(["--script", "s.txt", "--video", "v.mp4"])
    assert config_from_args(args) == EditorConfig()


def test_the_shipped_example_config_is_valid():
    """It would otherwise rot silently the first time a knob is renamed."""
    from pathlib import Path
    example = Path(__file__).resolve().parent.parent / "auto-editor.example.toml"
    assert example.exists()
    loaded = load_config_file(example)
    assert loaded, "the example should actually set something"
