import pytest

from config import EditorConfig
from models import Word
from transcribe import (VERBATIM_PROMPT, _device_attempts, _run_whisper,
                        cache_path_for, effective_prompt, load_cached,
                        save_cache, transcribe, transcription_key)

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

    def fake_decode(model, video, cfg, on_progress=None, duration=None):
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
    save_cache(path, ws, duration=12.5, key="k1")

    loaded, duration = load_cached(path, key="k1")
    assert loaded == ws
    assert duration == 12.5


def test_cache_miss_on_different_key(tmp_path):
    path = tmp_path / "raw.transcript.json"
    save_cache(path, [Word("hi", 0.0, 0.2)], duration=1.0, key="k1")
    assert load_cached(path, key="k2") == (None, None)


def test_cache_miss_on_absent_file(tmp_path):
    assert load_cached(tmp_path / "nope.json", key="k1") == (None, None)


def test_corrupt_cache_is_treated_as_a_miss(tmp_path):
    path = tmp_path / "raw.transcript.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_cached(path, key="k1") == (None, None)


def test_decode_drops_punctuation_only_tokens(monkeypatch, tmp_path):
    """Whisper emits standalone punctuation as its own zero-duration token.

    Keeping them puts a non-speech entry between two words, which breaks the
    adjacency every disfluency detector relies on. Dropping them at the source
    means no downstream stage has to know about it.
    """
    from transcribe import _decode

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSegment:
        words = [FakeWord(" config", 15.3, 15.76),
                 FakeWord("-", 15.76, 15.76),
                 FakeWord(" configuration", 15.76, 16.36),
                 FakeWord(" ...", 16.36, 16.36)]

    class FakeModel:
        def transcribe(self, *a, **k):
            return [FakeSegment()], None

    out = _decode(FakeModel(), tmp_path / "v.mp4", EditorConfig())
    assert [w.text for w in out] == ["config", "configuration"]


# --- verbatim mode -------------------------------------------------------

def test_no_prompt_by_default():
    assert effective_prompt(EditorConfig()) is None


def test_verbatim_uses_the_builtin_prompt():
    assert effective_prompt(EditorConfig(verbatim=True)) == VERBATIM_PROMPT


def test_builtin_prompt_actually_demonstrates_the_disfluencies():
    """The prompt only biases Whisper if it models what we want back."""
    p = VERBATIM_PROMPT.lower()
    assert "the the" in p, "must demonstrate an immediate repeat"
    assert "config-" in p, "must demonstrate a truncated restart"
    assert "uh" in p and "um" in p, "must demonstrate fillers"


def test_explicit_prompt_overrides_verbatim():
    cfg = EditorConfig(verbatim=True, initial_prompt="my own prompt")
    assert effective_prompt(cfg) == "my own prompt"


def test_explicit_empty_prompt_is_honoured_not_treated_as_unset():
    assert effective_prompt(EditorConfig(verbatim=True, initial_prompt="")) == ""


# --- cache keying --------------------------------------------------------

def test_verbatim_invalidates_the_cache():
    """A verbatim transcript is a different transcript; it must not be served
    from a cache built without it."""
    assert transcription_key(EditorConfig()) != \
        transcription_key(EditorConfig(verbatim=True))


def test_model_size_invalidates_the_cache():
    assert transcription_key(EditorConfig(model="small")) != \
        transcription_key(EditorConfig(model="medium"))


def test_language_invalidates_the_cache():
    assert transcription_key(EditorConfig(language="en")) != \
        transcription_key(EditorConfig(language="fr"))


def test_custom_prompt_invalidates_the_cache():
    assert transcription_key(EditorConfig(initial_prompt="a")) != \
        transcription_key(EditorConfig(initial_prompt="b"))


def test_identical_settings_share_a_key():
    assert transcription_key(EditorConfig(model="small", verbatim=True)) == \
        transcription_key(EditorConfig(model="small", verbatim=True))


def test_settings_that_do_not_affect_the_words_share_a_key():
    """Thresholds and padding change the cut list, never the transcript."""
    assert transcription_key(EditorConfig(threshold=0.9, padding_ms=200)) == \
        transcription_key(EditorConfig(threshold=0.5, padding_ms=10))


def test_verbatim_run_does_not_reuse_a_plain_cache(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    save_cache(cache_path_for(video), [Word("tidy", 0.0, 0.5)], 3.0,
               transcription_key(EditorConfig()))

    monkeypatch.setattr("transcribe.probe_duration", lambda v: 3.0)
    monkeypatch.setattr(
        "transcribe._run_whisper",
        lambda v, c, p=None, d=None: [Word("the", 0.0, 0.2),
                                      Word("the", 0.2, 0.4)])

    ws, _ = transcribe(video, EditorConfig(verbatim=True))
    assert [w.text for w in ws] == ["the", "the"], "must re-transcribe, not reuse"


def test_transcribe_uses_cache_without_loading_the_model(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    cfg = EditorConfig(model="medium")
    save_cache(cache_path_for(video), [Word("cached", 0.0, 0.5)],
               duration=3.0, key=transcription_key(cfg))

    def explode(*a, **k):
        raise AssertionError("model must not be loaded on a cache hit")

    monkeypatch.setattr("transcribe._run_whisper", explode)
    ws, duration = transcribe(video, cfg)
    assert [w.text for w in ws] == ["cached"]
    assert duration == 3.0


def test_no_cache_flag_forces_a_fresh_transcription(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    cfg = EditorConfig(model="medium", use_cache=False)
    save_cache(cache_path_for(video), [Word("stale", 0.0, 0.5)],
               duration=3.0, key=transcription_key(cfg))

    monkeypatch.setattr("transcribe.probe_duration", lambda v: 9.0)
    monkeypatch.setattr("transcribe._run_whisper",
                        lambda v, c, p=None, d=None: [Word("fresh", 0.0, 0.5)])

    ws, duration = transcribe(video, cfg)
    assert [w.text for w in ws] == ["fresh"]
    assert duration == 9.0
    # the fresh result must overwrite the stale cache
    assert load_cached(cache_path_for(video),
                       transcription_key(cfg))[0][0].text == "fresh"


def test_progress_is_reported_as_a_fraction(monkeypatch, tmp_path):
    """A two-minute wait with no feedback reads as a hang."""
    import faster_whisper

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSegment:
        def __init__(self, end, words):
            self.end, self.words = end, words

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return [FakeSegment(5.0, [FakeWord(" one", 0.0, 1.0)]),
                    FakeSegment(10.0, [FakeWord(" two", 5.0, 6.0)])], None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr("transcribe.probe_duration", lambda v: 10.0)

    seen = []
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    transcribe(video, EditorConfig(use_cache=False), on_progress=seen.append)

    assert seen == [0.5, 1.0]


def test_progress_never_exceeds_one(monkeypatch, tmp_path):
    """Whisper's final segment can overshoot the probed duration."""
    import faster_whisper

    class FakeWord:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    class FakeSegment:
        end = 12.0
        words = [FakeWord(" over", 0.0, 1.0)]

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, *a, **k):
            return [FakeSegment()], None

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setattr("transcribe.probe_duration", lambda v: 10.0)

    seen = []
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    transcribe(video, EditorConfig(use_cache=False), on_progress=seen.append)

    assert seen == [1.0]


def test_a_cache_hit_reports_complete_without_loading_a_model(tmp_path,
                                                              monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    cfg = EditorConfig()
    save_cache(cache_path_for(video), [Word("hi", 0.0, 0.2)], 3.0,
               transcription_key(cfg))

    def explode(*a, **k):
        raise AssertionError("must not load a model on a cache hit")

    monkeypatch.setattr("transcribe._run_whisper", explode)

    seen = []
    transcribe(video, cfg, on_progress=seen.append)
    assert seen == [1.0]


def test_omitting_the_callback_keeps_current_behaviour(tmp_path, monkeypatch):
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"")
    cfg = EditorConfig()
    save_cache(cache_path_for(video), [Word("hi", 0.0, 0.2)], 3.0,
               transcription_key(cfg))
    ws, duration = transcribe(video, cfg)
    assert [w.text for w in ws] == ["hi"]
    assert duration == 3.0
