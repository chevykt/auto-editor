import pytest

from config import EditorConfig
from render import build_ffmpeg_command, render_video


def test_command_uses_filter_complex_script_not_inline():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", "graph.txt", EditorConfig())
    assert "-filter_complex_script" in cmd
    assert "-filter_complex" not in cmd
    assert cmd[cmd.index("-filter_complex_script") + 1] == "graph.txt"


def test_command_maps_the_concat_outputs():
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", "graph.txt", EditorConfig())
    assert "[outv]" in cmd and "[outa]" in cmd


def test_command_honours_codec_config():
    cfg = EditorConfig(video_codec="libx265", video_crf=22, audio_bitrate="128k")
    cmd = build_ffmpeg_command("in.mp4", "out.mp4", "graph.txt", cfg)
    assert "libx265" in cmd
    assert "22" in cmd
    assert "128k" in cmd


def test_command_overwrites_without_prompting():
    assert "-y" in build_ffmpeg_command("in.mp4", "out.mp4", "g.txt", EditorConfig())


def test_render_refuses_a_report_with_no_segments(tmp_path):
    report = {"segments": [], "duration": 10.0}
    with pytest.raises(ValueError, match="no surviving segments"):
        render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                     EditorConfig())


def test_render_writes_the_graph_to_a_file(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        graph_path = cmd[cmd.index("-filter_complex_script") + 1]
        seen["graph"] = open(graph_path, encoding="utf-8").read()

        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr("render.subprocess.run", fake_run)
    monkeypatch.setattr("render.shutil.which", lambda n: "ffmpeg")

    report = {"segments": [{"start": 0.0, "end": 2.0},
                           {"start": 4.0, "end": 6.0}], "duration": 10.0}
    render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                 EditorConfig())
    assert "concat=n=2" in seen["graph"]


def test_graph_file_is_cleaned_up_even_when_ffmpeg_fails(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["path"] = cmd[cmd.index("-filter_complex_script") + 1]

        class R:
            returncode = 1
            stderr = "boom"
        return R()

    monkeypatch.setattr("render.subprocess.run", fake_run)
    monkeypatch.setattr("render.shutil.which", lambda n: "ffmpeg")

    report = {"segments": [{"start": 0.0, "end": 2.0}], "duration": 10.0}
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                     EditorConfig())
    from pathlib import Path
    assert not Path(seen["path"]).exists()


def test_missing_ffmpeg_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr("render.shutil.which", lambda n: None)
    report = {"segments": [{"start": 0.0, "end": 2.0}], "duration": 10.0}
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        render_video(tmp_path / "in.mp4", tmp_path / "out.mp4", report,
                     EditorConfig())


def test_graph_for_many_cuts_exceeds_the_windows_arg_limit(tmp_path, monkeypatch):
    """Proves -filter_complex_script is load-bearing, not stylistic."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["graph"] = open(
            cmd[cmd.index("-filter_complex_script") + 1], encoding="utf-8").read()

        class R:
            returncode = 0
            stderr = ""
        return R()

    monkeypatch.setattr("render.subprocess.run", fake_run)
    monkeypatch.setattr("render.shutil.which", lambda n: "ffmpeg")

    segments = [{"start": i * 2.0, "end": i * 2.0 + 1.0} for i in range(300)]
    render_video(tmp_path / "in.mp4", tmp_path / "out.mp4",
                 {"segments": segments, "duration": 600.0}, EditorConfig())
    assert len(seen["graph"]) > 8191
