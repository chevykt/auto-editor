# Web Tutorial Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a reviewed `plan.toml` plus Keon's recorded voiceover into a finished tutorial MP4 and a DaVinci timeline, with the browser steps performed and recorded automatically and fitted to his voice.

**Architecture:** A new `tutorial/` package in the auto-editor repo. Pure modules (plan parsing, voice timing, fitting, ffmpeg graph building) carry the logic and are unit tested; thin Playwright modules (runner, capture) drive Chrome and are smoke tested against a local HTML page. The existing engine cleans and aligns the voiceover; the existing `build_fcpxml` writes the timeline.

**Tech Stack:** Python 3.14 (`py -3.14`), Playwright for Python + Chromium, faster-whisper / rapidfuzz via the existing engine, ffmpeg 7+ (`-/filter_complex`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-web-tutorial-recorder-design.md`

## Global Constraints

- Run everything with `py -3.14` — `python` is not on PATH on this machine.
- Output: 1920×1080, 30 fps MP4, H.264 + AAC, audio = the cleaned voiceover.
- Picture is fitted to voice, never the reverse; the cleaned voiceover's length is the video's length.
- No credentials in plans, code, logs or tests. Login happens by hand via `login` in the profile at `~/.auto-editor/browser-profile`.
- Never complete a real financial action: a click whose target text matches a blocked phrase raises `BlockedAction` unless `demo_account = true`.
- Graph files go to ffmpeg via `-/filter_complex <file>`, as `render.py` does.
- New dependency: `playwright` only. After `pip install playwright`, run `py -3.14 -m playwright install chromium`.
- Commits: no Co-Authored-By, "Generated with", emoji, or AI mentions. `git add` only the files a task names — the tree has unrelated uncommitted work.
- Smoke tests (real Chromium / ffmpeg) are marked `@pytest.mark.smoke`, matching `pytest.ini`.

## Review Focus

1. **A selector matches nothing, or the site is slow.** `check` and `make` stop within `STEP_TIMEOUT_S` (15 s) naming beat and step, and save a screenshot — never hang. Test in Task 3.
2. **The saved login has expired.** The start page redirects to a sign-in page; the run stops with "run `py -3.14 -m tutorial login`", not a screenshot of a login form mid-video. Test in Task 3.
3. **Keon talks before the first beat or after the last** (an intro ad-lib, a sign-off). Beat windows still tile the whole cleaned voiceover from 0 to its end; the first beat's first frame holds through the intro. Test in Task 2.
4. **Keon paraphrases an anchor word** so it is never spoken. That step falls back to even spreading with a warning in the run log; the run does not fail. Test in Task 2.
5. **A beat has no steps** (pure narration). Its window is one hold of the current frame. Test in Task 5.

---

### Task 1: Plan file

**Files:**
- Create: `tutorial/__init__.py` (empty), `tutorial/plan.py`
- Test: `tests/test_tutorial_plan.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Step(kind: str, target: str | None, value: str | None, at: str | None)` — `kind` ∈ `STEP_KINDS = ("goto","click","type","press","hover","scroll","wait_for","pause")`. For `type`, `value` is the text and `target` the `into` locator; for `goto`, `target` is the URL; for `pause`, `value` is seconds as a string; for `press`, `value` is the key.
  - `@dataclass(frozen=True) Beat(say: str, steps: tuple[Step, ...])`
  - `@dataclass(frozen=True) Plan(title: str, start_url: str, beats: tuple[Beat, ...], blur: tuple[str, ...] = (), demo_account: bool = False, setup: tuple[Step, ...] = (), cleanup: tuple[Step, ...] = ())`
  - `load_plan(path: str | Path) -> Plan` — raises `PlanError(ValueError)` with the beat number and step index in the message.
  - `script_text(plan: Plan) -> str` — beats' `say` joined by `"\n\n"`, the format `parse_script` reads.

- [ ] **Step 1: Write the failing tests**

```python
def test_loads_the_spec_example(tmp_path):        # the Plan format block from the spec, verbatim
    plan = load_plan(write(tmp_path, SPEC_EXAMPLE))
    assert plan.start_url == "https://supabase.com/dashboard/projects"
    assert plan.blur == ("[data-testid=user-email]",)
    assert plan.beats[1].steps[0] == Step("type", "label=Name", "my-first-app", "name")

def test_script_text_is_one_beat_per_paragraph(tmp_path):
    assert script_text(load_plan(write(tmp_path, SPEC_EXAMPLE))).split("\n\n")[0] \
        == "Start on your Supabase dashboard and click New project."

def test_unknown_step_kind_names_beat_and_step(tmp_path):   # beat 2, step 1 has {drag = "x"}
    with pytest.raises(PlanError, match=r"beat 2, step 1.*drag"): load_plan(...)

def test_a_beat_without_say_is_rejected(tmp_path): ...       # PlanError "beat 1"
def test_a_step_with_two_kinds_is_rejected(tmp_path): ...    # {click=..., goto=...} -> PlanError
def test_pause_must_be_a_number(tmp_path): ...               # {pause = "soon"} -> PlanError
def test_setup_and_cleanup_are_optional_step_lists(tmp_path): ...
def test_a_beat_with_no_steps_is_allowed(tmp_path): ...      # pure narration
```

- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_plan.py -v` — expect FAIL (module missing).
- [ ] **Step 3: Implement `tutorial/plan.py`** with `tomllib`. Each step table must have exactly one key from `STEP_KINDS`; `into` and `at` are the only other keys allowed (`into` only with `type`).
- [ ] **Step 4: Run** the same command — expect PASS.
- [ ] **Step 5: Commit** `git add tutorial/__init__.py tutorial/plan.py tests/test_tutorial_plan.py` — `feat(tutorial): plan file format`.

---

### Task 2: Voice timing

**Files:**
- Create: `tutorial/voice.py`
- Modify: `render.py` — `build_filter_graph(segments, cfg, video: bool = True)`; with `video=False` it emits only the audio chains and `concat=n=N:v=0:a=1[outa]`. Add `render_audio(src, out, segments, cfg) -> Path` (maps `[outa]`, writes WAV `pcm_s16le`).
- Test: `tests/test_tutorial_voice.py`, `tests/test_render_graph.py` (append)

**Interfaces:**
- Consumes: `Plan`, `script_text` (Task 1); `transcribe(video, cfg)`, `analyse(...)`, `align_script(words, lines, cfg)`, `parse_script(text)`, `Span`.
- Produces:
  - `@dataclass(frozen=True) VoiceTiming(duration: float, windows: tuple[tuple[float, float], ...], anchors: dict[tuple[int, int], float])` — all in cleaned-voiceover seconds; `windows[i]` is beat `i`; `anchors[(beat, step)]` is the start of the anchor phrase's first word.
  - `to_clean_time(t: float, kept: list[Span]) -> float | None` — None when `t` falls in a removed span.
  - `beat_windows(starts: list[float], duration: float) -> tuple[tuple[float, float], ...]` — tiles `[0, duration]`: window `i` runs from `starts[i]` to `starts[i+1]`, the first starts at `0.0`, the last ends at `duration`.
  - `find_anchor(words: list[Word], word_range: tuple[int, int], phrase: str) -> float | None` — best `fuzz.ratio` window of the phrase's word count inside the range, at or above 0.8; else None.
  - `clean_voiceover(voice: Path, plan: Plan, cfg: EditorConfig, out_wav: Path) -> tuple[VoiceTiming, list[str]]` — the list is warnings. Raises `VoiceError` carrying the analysis report when any line is `miss` or `review`.

- [ ] **Step 1: Write the failing tests**

```python
def test_clean_time_shifts_by_removed_spans():
    kept = [Span(0, 2), Span(5, 8)]
    assert to_clean_time(1.0, kept) == 1.0
    assert to_clean_time(6.0, kept) == 3.0
    assert to_clean_time(3.0, kept) is None

def test_windows_tile_the_whole_voiceover_including_an_intro():   # Review Focus 3
    assert beat_windows([1.5, 4.0, 9.0], 12.0) == ((0.0, 4.0), (4.0, 9.0), (9.0, 12.0))

def test_anchor_is_the_start_of_the_phrase():
    ws = words("start on your dashboard and click new project")
    assert find_anchor(ws, (0, len(ws)), "New project") == ws[6].start

def test_a_paraphrased_anchor_is_none():                           # Review Focus 4
    ws = words("then press the create button")
    assert find_anchor(ws, (0, len(ws)), "New project") is None

def test_a_missed_line_stops_with_the_report(monkeypatch, tmp_path): ...  # stub transcribe; VoiceError
def test_missing_anchor_becomes_a_warning_not_an_error(monkeypatch, tmp_path): ...
def test_audio_only_graph_has_no_video_chains():
    graph = build_filter_graph([Span(0, 1), Span(2, 3)], EditorConfig(), video=False)
    assert "[0:v]" not in graph and "v=0:a=1[outa]" in graph
```

- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_voice.py tests/test_render_graph.py -v` — expect FAIL.
- [ ] **Step 3: Implement.** `clean_voiceover`: `transcribe` → `analyse(words, script_text(plan), duration, cfg, str(voice))` → stop on miss/review → `render_audio` the report's segments → `align_script(words, parse_script(script_text(plan)), cfg)` for each beat's word range → map line starts and anchors through `to_clean_time` → `beat_windows`. A line start that maps to None uses the next kept instant. `cfg` comes from `EditorConfig(verbatim=True)` as the web UI uses.
- [ ] **Step 4: Run** the same command plus `py -3.14 -m pytest -q` — expect all PASS.
- [ ] **Step 5: Commit** `git add tutorial/voice.py tests/test_tutorial_voice.py tests/test_render_graph.py` and `git add -p render.py` (only the audio-only hunks) — `feat(tutorial): clean and time the voiceover`.

---

### Task 3: Browser runner (steps, cursor, blur, safety)

**Files:**
- Create: `tutorial/runner.py`, `tutorial/overlay.js`
- Create: `tests/fixtures/tutorial_page.html` — a static page with a "New project" button, a labelled "Name" input, a "Confirm payment" button, an element `#email` holding `someone@example.com`, and a 2-second delayed element `#late`; plus `tests/fixtures/login.html` and `tests/fixtures/to_login.html` (redirects to `login.html`).
- Test: `tests/test_tutorial_runner.py` (pure guard tests + smoke tests on the fixtures)

**Interfaces:**
- Consumes: `Plan`, `Step` (Task 1).
- Produces:
  - `@dataclass(frozen=True) StepEvent(beat: int, step: int, kind: str, start: float, action: float, end: float)` — `time.time()` seconds; `action` is when the click / first keystroke happens; `beat == -1` for setup and cleanup.
  - `BLOCKED_PHRASES: tuple[str, ...] = ("confirm payment", "place order", "submit return", "deposit", "withdraw", "pay now", "buy", "sell", "transfer")`
  - `is_blocked(text: str, demo_account: bool) -> bool` — case-insensitive whole-word match.
  - `class BlockedAction(RuntimeError)`, `class StepFailed(RuntimeError)` (`beat`, `step`, `screenshot: Path`), `class SessionExpired(RuntimeError)`.
  - `open_browser(headless: bool, profile: Path = PROFILE_DIR) -> tuple[BrowserContext, Page]` — `launch_persistent_context(profile, viewport 1920×1080, device_scale_factor=1)`, `PROFILE_DIR = Path.home() / ".auto-editor" / "browser-profile"`.
  - `start(page: Page, plan: Plan) -> None` — installs `overlay.js` (cursor + `blur` CSS) via `add_init_script`, goes to `start_url`, raises `SessionExpired` if the final URL's path contains `login`, `signin`, `sign-in` or `auth` and `start_url`'s does not.
  - `run_steps(page: Page, plan: Plan, steps: list[tuple[int, int, Step]], shots: Path) -> list[StepEvent]`
- Constants: `STEP_TIMEOUT_S = 15`, `CURSOR_TRAVEL_S = 0.5`, `TYPE_DELAY_S = 0.06`, `SETTLE_S = 0.6` (wait after each beat's last step).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("text", ["Confirm payment", "PLACE ORDER", "Deposit"])
def test_financial_buttons_are_blocked(text):
    assert is_blocked(text, demo_account=False)

def test_a_demo_account_may_click_them():
    assert not is_blocked("Place order", demo_account=True)

def test_ordinary_words_containing_a_phrase_are_not_blocked():
    assert not is_blocked("Buyer guide", demo_account=False)

@pytest.mark.smoke
def test_steps_run_and_are_timed(page_fixture): ...       # click New project, type into Name; events ordered, start<=action<=end
@pytest.mark.smoke
def test_blocked_click_raises_before_clicking(page_fixture): ...   # page records clicks in window.__clicks; it stays empty
@pytest.mark.smoke
def test_missing_selector_fails_fast_with_a_screenshot(page_fixture): ...   # Review Focus 1: < STEP_TIMEOUT_S + 2 s, screenshot exists
@pytest.mark.smoke
def test_redirect_to_login_is_session_expired(page_fixture): ...   # Review Focus 2
@pytest.mark.smoke
def test_blurred_element_is_blurred(page_fixture): ...    # getComputedStyle(#email).filter contains "blur"
@pytest.mark.smoke
def test_the_drawn_cursor_ends_on_the_clicked_element(page_fixture): ...
```

The smoke fixture opens the fixture page through `file://` with `profile=tmp_path / "profile"` (never the real profile).

- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_runner.py -v -m "smoke or not smoke"` — expect FAIL.
- [ ] **Step 3: Implement.** Before each `click`, read the target's visible text and raise `BlockedAction` if blocked. The cursor is a fixed-position element in `overlay.js` exposing `window.__cursor.moveTo(x, y, ms)` and `.click()` (ripple); the runner moves it to the target's centre over `CURSOR_TRAVEL_S`, waits for it, then clicks with Playwright. On any Playwright timeout, `page.screenshot(path=shots / f"beat{b}-step{s}.png")` then raise `StepFailed`.
- [ ] **Step 4: Run** the same command — expect PASS.
- [ ] **Step 5: Commit** the runner, overlay, fixtures and test — `feat(tutorial): run plan steps in the browser`.

---

### Task 4: Screen capture

**Files:**
- Create: `tutorial/capture.py`
- Test: `tests/test_tutorial_capture.py`

**Interfaces:**
- Consumes: `Page`, `open_browser` (Task 3).
- Produces:
  - `concat_lines(frames: list[tuple[float, Path]], end: float) -> list[str]` — the ffmpeg concat-demuxer list.
  - `class Capture` — `Capture(page, frames_dir: Path)` used as a context manager; `.frames: list[tuple[float, Path]]` (epoch seconds from CDP `metadata.timestamp`, frame file); `.to_video(out: Path, fps: int = 30) -> tuple[Path, float]` returns the CFR MP4 and `t0` (epoch time of its first frame).

- [ ] **Step 1: Write the failing tests**

```python
def test_concat_list_gives_each_frame_its_own_duration(tmp_path):
    a, b, c = (tmp_path / n for n in ("a.png", "b.png", "c.png"))
    assert concat_lines([(10.0, a), (10.5, b), (11.25, c)], end=12.0) == [
        f"file '{a.as_posix()}'", "duration 0.5",
        f"file '{b.as_posix()}'", "duration 0.75",
        f"file '{c.as_posix()}'", "duration 0.75",
        f"file '{c.as_posix()}'"]

@pytest.mark.smoke
def test_capture_is_1080p_30fps_and_spans_the_recording(page_fixture, tmp_path): ...
    # ffprobe: 1920x1080, r_frame_rate 30/1, duration within 0.1 s of the capture's span
```

- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_capture.py -v -m "smoke or not smoke"` — expect FAIL.
- [ ] **Step 3: Implement** with a CDP session: `Page.startScreencast(format="png", maxWidth=1920, maxHeight=1080, everyNthFrame=1)`, write each frame, `Page.screencastFrameAck`. `to_video` runs `ffmpeg -f concat -safe 0 -i list.txt -vf fps=30,format=yuv420p -c:v libx264 -crf 18`. Screencast only sends frames on change, so a still page costs nothing.
- [ ] **Step 4: Measure by hand on a real page.** Record 5 s of `https://supabase.com/docs` with a cursor sweep. Pass when (a) body text is as sharp as `page.screenshot()` at 100 % zoom, and (b) at least 24 frames/s arrive during cursor motion. If (b) fails, switch to `format="jpeg", quality=95` and re-measure; if text then blurs, fall back to ffmpeg `gdigrab` of a headed window. Record the outcome as a comment at the top of `capture.py`.
- [ ] **Step 5: Run** the tests — expect PASS. **Commit** `tutorial/capture.py tests/test_tutorial_capture.py` — `feat(tutorial): capture the browser at 1080p`.

---

### Task 5: Fitting

**Files:**
- Create: `tutorial/fit.py`
- Test: `tests/test_tutorial_fit.py`

**Interfaces:**
- Consumes: `StepEvent`, `SETTLE_S` (Task 3), `VoiceTiming` (Task 2), `Plan` (Task 1).
- Produces:
  - `@dataclass(frozen=True) Play(src_start: float, src_end: float, speed: float)` and `@dataclass(frozen=True) Hold(src_at: float, seconds: float)` — `Piece = Play | Hold`, source times in capture seconds (event time − `t0`).
  - `MAX_SPEED = 2.0`
  - `class FitError(ValueError)` — message names the beat.
  - `fit_beat(span: tuple[float, float], waits: list[tuple[float, float]], marks: list[tuple[float, float]], window: tuple[float, float]) -> list[Piece]` — `marks` are `(source action time, target voice time)` pairs, sorted.
  - `fit_all(events: list[StepEvent], t0: float, timing: VoiceTiming, plan: Plan) -> tuple[list[Piece], list[str]]` — pieces for the whole video plus warnings.

Algorithm (the tests below do not fully determine it):

```
fixed points: (span.start, window.start), *marks, (span.end, window.end)
for each consecutive pair (s0,t0) -> (s1,t1):
    S, T = s1 - s0, t1 - t0
    if S <= T:
        if it's the last interval: Play(s0, s1, 1) then Hold(s1, T - S)   # settle, then hold
        else:                      Hold(s0, T - S) then Play(s0, s1, 1)   # wait, then move and click on the word
    else:
        drop the parts of `waits` inside (s0, s1), longest first, until S' <= T (keep 0.2 s of each)
        if still S' > T: speed = S' / T; if speed > MAX_SPEED: raise FitError(beat)
        emit Play pieces for the kept parts at that speed
```

`fit_all`: per beat, `span` = first event `start` → last event `end` + `SETTLE_S`; `waits` = events of kind `wait_for` or `pause`, plus each `goto`'s start→end; `marks` = steps whose `(beat, step)` is in `timing.anchors`. Steps with an `at` whose anchor is missing are left unmarked and named in the warnings. A beat with no events emits one `Hold` of the previous beat's last source frame (or the first frame, for beat 0) for its whole window.

- [ ] **Step 1: Write the failing tests**

```python
def total(pieces):
    return sum(p.seconds if isinstance(p, Hold) else (p.src_end - p.src_start) / p.speed for p in pieces)

def test_short_footage_holds_before_the_anchor_so_the_click_lands_on_the_word():
    pieces = fit_beat((0, 3), [], [(2.0, 5.0)], (0, 6))
    assert pieces[0] == Hold(0, 3.0)            # wait first
    assert total(pieces) == pytest.approx(6.0)

def test_long_footage_drops_waits_before_speeding_up():
    pieces = fit_beat((0, 10), [(2, 7)], [], (0, 6))
    assert all(isinstance(p, Hold) or p.speed == 1 for p in pieces)
    assert total(pieces) == pytest.approx(6.0)

def test_speed_up_is_capped():
    with pytest.raises(FitError):
        fit_beat((0, 10), [], [], (0, 4))

def test_whole_video_matches_the_voiceover_length(): ...    # fit_all over 3 beats: total == timing.duration
def test_a_beat_without_steps_is_one_hold(): ...             # Review Focus 5
def test_an_unmarked_step_is_a_warning(): ...                # Review Focus 4
```

- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_fit.py -v` — expect FAIL.
- [ ] **Step 3: Implement** `tutorial/fit.py`.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `tutorial/fit.py tests/test_tutorial_fit.py` — `feat(tutorial): fit footage to the voiceover`.

---

### Task 6: Assembly and timeline

**Files:**
- Create: `tutorial/assemble.py`
- Test: `tests/test_tutorial_assemble.py`

**Interfaces:**
- Consumes: `Play`, `Hold` (Task 5); `write_fcpxml`, `probe_video_meta` (`xml_export.py`); `Span` (`models.py`).
- Produces:
  - `pieces_graph(pieces: list[Piece], fps: int = 30) -> str` — `Play` → `trim=start:end,setpts=(PTS-STARTPTS)/speed`; `Hold` → one-frame `trim` + `tpad=stop_mode=clone:stop_duration=seconds`; each chain ends `fps=30`; then `concat=n=N:v=1:a=0[outv]`.
  - `assemble(capture: Path, pieces: list[Piece], voice_wav: Path, out: Path) -> Path` — maps `[outv]` and the WAV; raises if `|video − audio| > 1/30 s` on the result.
  - `write_timeline(final: Path, windows: tuple[tuple[float, float], ...], out: Path) -> Path` — `write_fcpxml(final, [Span(a, b) for a, b in windows], probe_video_meta(final), out)`: the finished video cut at beat boundaries, so Keon can trim or slide beats in Resolve. *(Plan's call: no new FCPXML code.)*

- [ ] **Step 1: Write the failing tests**

```python
def test_graph_has_one_chain_per_piece_and_a_video_only_concat():
    g = pieces_graph([Play(0, 2, 1), Hold(2, 1.5), Play(2, 6, 2.0)])
    assert g.count("trim=") == 3 and "tpad=stop_mode=clone:stop_duration=1.5" in g
    assert "setpts=(PTS-STARTPTS)/2.0" in g and "concat=n=3:v=1:a=0[outv]" in g

@pytest.mark.smoke
def test_assembled_video_matches_the_voice_length(tmp_path): ...   # lavfi testsrc + sine inputs, within 1/30 s
@pytest.mark.smoke
def test_timeline_has_one_clip_per_beat(tmp_path): ...
```

- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_assemble.py -v -m "smoke or not smoke"` — expect FAIL.
- [ ] **Step 3: Implement**, following `render.py`: graph written to a temp file, ffmpeg output decoded as UTF-8 with `errors="replace"`.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `tutorial/assemble.py tests/test_tutorial_assemble.py` — `feat(tutorial): assemble the video and timeline`.

---

### Task 7: Command line, README, pilot

**Files:**
- Create: `tutorial/__main__.py`
- Modify: `README.md` (new "Tutorial recorder" section), `.gitignore` (add `tutorial-out/`)
- Test: `tests/test_tutorial_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int` with subcommands:
  - `login [URL]` — headed browser on the profile; returns when the window closes.
  - `script PLAN` — writes `<plan stem>.script.txt` beside the plan.
  - `check PLAN` — headless: `start`, setup, all beat steps, cleanup; prints `ok` or the `StepFailed` / `BlockedAction` / `SessionExpired` message; exit 1 on failure.
  - `make PLAN VOICE` — in order: `clean_voiceover` (stop early on `VoiceError`, writing the report beside the plan), `start`, setup, `Capture` around the beat steps, cleanup (always, in `finally`), `fit_all`, `assemble`, `write_timeline`. Output folder `tutorial-out/<plan stem>/` with `final.mp4`, `final.fcpxml`, `voice.clean.wav`, `run.log` (warnings + step events). Exit 2 on input errors, 1 on run failures.

- [ ] **Step 1: Write the failing tests** — `make` without VOICE exits 2 with a message; a missing plan file exits 2; `script` writes the expected text; `make` with every stage stubbed (`monkeypatch` on `tutorial.__main__` names) still runs cleanup when capture raises.
- [ ] **Step 2: Run** `py -3.14 -m pytest tests/test_tutorial_cli.py -v` — expect FAIL.
- [ ] **Step 3: Implement** with `argparse` subparsers.
- [ ] **Step 4: Run** `py -3.14 -m pytest -q` and `py -3.14 -m pytest -q -m smoke` — expect all PASS.
- [ ] **Step 5: README** — the four commands, the sign-in-once rule, the blocked-phrase rule and `demo_account`, and that Claude drafts plans which Keon reviews.
- [ ] **Step 6: Commit** `tutorial/__main__.py tests/test_tutorial_cli.py README.md .gitignore` — `feat(tutorial): command line`. (README and .gitignore carry earlier uncommitted edits; commit them only once Keon has said to commit the pending review work.)
- [ ] **Step 7: Pilot (with Keon).** Claude explores the Supabase dashboard in the built-in browser and writes `tutorials/supabase-create-project.toml` (one keyword, `blur` on the account email, a `cleanup` that deletes the project). Keon runs `login`, then `check`; Claude fixes steps until `ok`. Keon records the voiceover from the `script` output and runs `make`. Keon watches `final.mp4`; his notes become the next round of work. Not done until he has watched it.
