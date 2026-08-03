import pytest

from config import EditorConfig
from models import Word
from transcribe import (_device_attempts, _run_whisper, cache_path_for,
                        load_cached, save_cache, transcribe)

CUBLAS_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


def _fake_whisper(monkeypatch, fail_on_construct=(), fail_on_decode=()):
    """Stub faster_whisper. Returns the list of devices actually attempted."""
    import faster_whisper

    tried: list[str] = []

    class FakeModel:
        def __init__(self, name, device, compute_type):
            tried.append(device)
            self.device = device
            if device in fail_on_construct:
                raise RuntimeError(CUBLAS_ERROR)

    def fake_decode(model, video, cfg):
        if model.device in fail_on_decode:
            raise RuntimeError(CUBLAS_ERROR)
        return [Word("ok", 0.0, 0.3)]

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr("transcribe._decode", fake_decode)
    return tried


def test_auto_tries_cuda_then_cpu():
    assert _device_attempts(EditorConfig(device="auto")) == [
        ("cuda", "float16"), ("cpu", "int8")]


def test_explicit_device_is_the_only_attempt():
    assert _device_attempts(EditorConfig(device="cpu", compute_type="int8")) == [
        ("cpu", "int8")]


def test_falls_back_to_cpu_when_cuda_fails_during_decoding(monkeypatch, tmp_path):
    """The real failure mode: CUDA loads, then dies encoding.

    faster-whisper returns a lazy generator, so a missing CUDA runtime
    surfaces during iteration - well after the constructor succeeded.
    """
    tried = _fake_whisper(monkeypatch, fail_on_decode=("cuda",))
    ws = _run_whisper(tmp_path / "v.mp4", EditorConfig(device="auto"))
    assert tried == ["cuda", "cpu"]
    assert [w.text for w in ws] == ["ok"]


def test_falls_back_to_cpu_when_cuda_fails_at_construction(monkeypatch, tmp_path):
    tried = _fake_whisper(monkeypatch, fail_on_construct=("cuda",))
    ws = _run_whisper(tmp_path / "v.mp4", EditorConfig(device="auto"))
    assert tried == ["cuda", "cpu"]
    assert [w.text for w in ws] == ["ok"]


def test_cuda_is_used_when_it_works(monkeypatch, tmp_path):
    tried = _fake_whisper(monkeypatch)
    _run_whisper(tmp_path / "v.mp4", EditorConfig(device="auto"))
    assert tried == ["cuda"], "must not fall back when CUDA works"


def test_explicit_device_failure_is_raised_not_swallowed(monkeypatch, tmp_path):
    """Asking for CPU explicitly and having it fail is a real error."""
    _fake_whisper(monkeypatch, fail_on_decode=("cpu",))
    with pytest.raises(RuntimeError, match="cublas"):
        _run_whisper(tmp_path / "v.mp4", EditorConfig(device="cpu"))


def test_failure_on_the_last_attempt_is_raised(monkeypatch, tmp_path):
    _fake_whisper(monkeypatch, fail_on_decode=("cuda", "cpu"))
    with pytest.raises(RuntimeError, match="cublas"):
        _run_whisper(tmp_path / "v.mp4", EditorConfig(device="auto"))


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
