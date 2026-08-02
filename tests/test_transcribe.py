from config import EditorConfig
from models import Word
from transcribe import (cache_path_for, load_cached, save_cache, transcribe)


def test_cache_path_sits_beside_the_video(tmp_path):
    video = tmp_path / "raw.mp4"
    assert cache_path_for(video).name == "raw.transcript.json"
    assert cache_path_for(video).parent == tmp_path


def test_cache_path_handles_dots_in_the_stem(tmp_path):
    video = tmp_path / "take.2.final.mp4"
    assert cache_path_for(video).name == "take.2.final.transcript.json"


def test_cache_round_trip(tmp_path):
    ws = [Word("hello", 0.0, 0.4), Word("world", 0.5, 0.9)]
    path = tmp_path / "raw.transcript.json"
    save_cache(path, ws, duration=12.5, model="medium")

    loaded, duration = load_cached(path, model="medium")
    assert loaded == ws
    assert duration == 12.5


def test_cache_miss_on_different_model(tmp_path):
    path = tmp_path / "raw.transcript.json"
    save_cache(path, [Word("hi", 0.0, 0.2)], duration=1.0, model="small")
    assert load_cached(path, model="medium") == (None, None)


def test_cache_miss_on_absent_file(tmp_path):
    assert load_cached(tmp_path / "nope.json", model="medium") == (None, None)


def test_corrupt_cache_is_treated_as_a_miss(tmp_path):
    path = tmp_path / "raw.transcript.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_cached(path, model="medium") == (None, None)


def test_transcribe_uses_cache_without_loading_the_model(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    save_cache(cache_path_for(video), [Word("cached", 0.0, 0.5)],
               duration=3.0, model="medium")

    def explode(*a, **k):
        raise AssertionError("model must not be loaded on a cache hit")

    monkeypatch.setattr("transcribe._run_whisper", explode)
    ws, duration = transcribe(video, EditorConfig(model="medium"))
    assert [w.text for w in ws] == ["cached"]
    assert duration == 3.0


def test_no_cache_flag_forces_a_fresh_transcription(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    save_cache(cache_path_for(video), [Word("stale", 0.0, 0.5)],
               duration=3.0, model="medium")

    monkeypatch.setattr("transcribe.probe_duration", lambda v: 9.0)
    monkeypatch.setattr("transcribe._run_whisper",
                        lambda v, c: [Word("fresh", 0.0, 0.5)])

    ws, duration = transcribe(video, EditorConfig(model="medium", use_cache=False))
    assert [w.text for w in ws] == ["fresh"]
    assert duration == 9.0
    # the fresh result must overwrite the stale cache
    assert load_cached(cache_path_for(video), "medium")[0][0].text == "fresh"
