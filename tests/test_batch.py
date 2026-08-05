"""Batch mode: pairing recordings with scripts across a directory."""
from pathlib import Path

from batch import Job, find_jobs, is_render_output, output_path


def make(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_bytes(b"")


def test_pairs_recordings_with_same_stem_scripts(tmp_path):
    make(tmp_path, "one.mp4", "one.txt", "two.mp4", "two.txt")
    jobs, unpaired = find_jobs(tmp_path)

    assert [j.name for j in jobs] == ["one", "two"]
    assert unpaired == []


def test_recordings_without_a_script_are_reported_not_silently_skipped(tmp_path):
    """A batch that quietly processes four of five files is worse than one
    that says which it missed."""
    make(tmp_path, "one.mp4", "one.txt", "orphan.mp4")
    jobs, unpaired = find_jobs(tmp_path)

    assert [j.name for j in jobs] == ["one"]
    assert [p.name for p in unpaired] == ["orphan.mp4"]


def test_every_common_container_is_recognised(tmp_path):
    for ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
        make(tmp_path, f"clip{ext}", "clip.txt")
        jobs, _ = find_jobs(tmp_path)
        assert jobs, f"{ext} should be recognised"
        (tmp_path / f"clip{ext}").unlink()


def test_extension_matching_is_case_insensitive(tmp_path):
    make(tmp_path, "one.MP4", "one.txt")
    jobs, _ = find_jobs(tmp_path)
    assert len(jobs) == 1


def test_rendered_output_is_not_picked_up_as_fresh_input(tmp_path):
    """Otherwise a second batch run edits its own output."""
    make(tmp_path, "one.mp4", "one.txt", "one.cut.mp4")
    jobs, unpaired = find_jobs(tmp_path)

    assert [j.name for j in jobs] == ["one"]
    assert unpaired == [], "the render must not even be reported as unpaired"


def test_is_render_output_identifies_our_own_files():
    assert is_render_output(Path("lesson.cut.mp4"))
    assert not is_render_output(Path("lesson.mp4"))
    assert not is_render_output(Path("lesson.take2.mp4"))


def test_non_media_files_are_ignored(tmp_path):
    make(tmp_path, "one.mp4", "one.txt", "notes.md", "one.report.json")
    jobs, unpaired = find_jobs(tmp_path)
    assert len(jobs) == 1
    assert unpaired == []


def test_subdirectories_are_ignored(tmp_path):
    (tmp_path / "nested").mkdir()
    make(tmp_path / "nested", "deep.mp4", "deep.txt")
    make(tmp_path, "one.mp4", "one.txt")

    jobs, _ = find_jobs(tmp_path)
    assert [j.name for j in jobs] == ["one"]


def test_dots_in_a_stem_still_pair_correctly(tmp_path):
    make(tmp_path, "take.2.final.mp4", "take.2.final.txt")
    jobs, _ = find_jobs(tmp_path)
    assert [j.name for j in jobs] == ["take.2.final"]


def test_empty_directory(tmp_path):
    assert find_jobs(tmp_path) == ([], [])


def test_jobs_come_back_in_a_stable_order(tmp_path):
    make(tmp_path, "c.mp4", "c.txt", "a.mp4", "a.txt", "b.mp4", "b.txt")
    jobs, _ = find_jobs(tmp_path)
    assert [j.name for j in jobs] == ["a", "b", "c"]


# --- output placement ----------------------------------------------------

def test_output_lands_beside_the_video_without_an_out_dir(tmp_path):
    job = Job(video=tmp_path / "one.mp4", script=tmp_path / "one.txt")
    assert output_path(job, None, ".cut.mp4") == tmp_path / "one.cut.mp4"


def test_out_dir_redirects_every_output(tmp_path):
    job = Job(video=tmp_path / "one.mp4", script=tmp_path / "one.txt")
    out = output_path(job, Path("edited"), ".cut.mp4")
    assert out == Path("edited") / "one.cut.mp4"


def test_report_stem_follows_the_same_rule(tmp_path):
    job = Job(video=tmp_path / "one.mp4", script=tmp_path / "one.txt")
    assert output_path(job, Path("edited"), ".report").name == "one.report"
