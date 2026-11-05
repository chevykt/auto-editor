"""py -3.14 -m tutorial: login, script, check, make."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import tutorial.__main__ as cli
from tutorial.voice import VoiceTiming

PLAN = '''
title = "t"
start_url = "https://example.com"
cleanup = [{ click = "text=Delete" }]

[[beat]]
say = "Click New project."
steps = [{ click = "text=New project" }]

[[beat]]
say = "Then name it."
steps = []
'''


@pytest.fixture
def plan(tmp_path):
    path = tmp_path / "create-project.toml"
    path.write_text(PLAN, encoding="utf-8")
    return path


def test_make_without_a_voiceover_exits_2(plan, capsys):
    assert cli.main(["make", str(plan)]) == 2
    assert "voice" in capsys.readouterr().err


def test_a_missing_plan_exits_2(tmp_path, capsys):
    assert cli.main(["script", str(tmp_path / "nope.toml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_a_broken_plan_exits_2_and_says_where(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('start_url = "https://x"\n[[beat]]\nsay = "a"\nsteps = [{ fly = 1 }]\n',
                   encoding="utf-8")
    assert cli.main(["check", str(bad)]) == 2
    assert "beat 1, step 1" in capsys.readouterr().err


def test_script_writes_one_beat_per_paragraph(plan):
    assert cli.main(["script", str(plan)]) == 0
    text = plan.with_name("create-project.script.txt").read_text(encoding="utf-8")
    assert text == "Click New project.\n\nThen name it.\n"


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Every stage replaced; records what ran, in order."""
    ran = []

    @contextmanager
    def browser(headless):
        ran.append("browser")
        yield SimpleNamespace()

    def steps(page, plan, steps, shots):
        ran.append("cleanup" if steps and steps[0][0] == -1 else "steps")
        return []

    class Capture:
        def __init__(self, page, frames_dir):
            pass

        def __enter__(self):
            raise RuntimeError("capture broke")

        def __exit__(self, *exc):
            pass

    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"")
    monkeypatch.setattr(cli, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(cli, "clean_voiceover", lambda *a: (
        VoiceTiming(4.0, ((0, 2), (2, 4)), {}), []))
    monkeypatch.setattr(cli, "open_browser", browser)
    monkeypatch.setattr(cli, "start", lambda page, plan: ran.append("start"))
    monkeypatch.setattr(cli, "run_steps", steps)
    monkeypatch.setattr(cli, "Capture", Capture)
    return ran, voice


def test_cleanup_runs_even_when_the_recording_fails(plan, stubbed, capsys):
    ran, voice = stubbed
    assert cli.main(["make", str(plan), str(voice)]) == 1
    assert ran[-1] == "cleanup"
    assert "capture broke" in capsys.readouterr().err


def test_a_bad_read_stops_before_the_browser_opens(plan, stubbed, monkeypatch, capsys):
    ran, voice = stubbed
    from tutorial.voice import VoiceError

    def bad(*args):
        raise VoiceError({"lines": [{"index": 0, "status": "miss", "score": 0.0,
                                      "script": "Click New project.", "start": None,
                                      "end": None, "scoped": False, "matched_text": ""}],
                          "summary": {}, "cuts": [], "adlibs": [], "segments": [],
                          "video": "v", "duration": 1.0})

    monkeypatch.setattr(cli, "clean_voiceover", bad)
    monkeypatch.setattr(cli, "write_markdown", lambda report, path: path.write_text("r"))
    assert cli.main(["make", str(plan), str(voice)]) == 1
    assert ran == []
    assert plan.with_name("create-project.voice-report.md").exists()
