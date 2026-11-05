# Web Tutorial Recorder — Design

## Intent

Keon makes paid tutorial videos for Answer Asap: each topic (Supabase, Kaggle,
H&R Block, OpenTable, broker sites, BandLab's web studio…) has a list of
keywords, and each keyword is a video. The backlog is 39 topics and 1,147
keywords. The only platform rule is that the voice must be his own.

Today every video means reading a script **and** performing and recording the
on-screen steps, then cutting the two together. This tool removes everything
except the reading: Keon records the voiceover, the tool performs the steps in a
browser, records them, and fits the picture to his voice.

Success: for a typical 1–3 minute web tutorial, Keon reads the script once (with
as many fumbles and retakes as he likes) and gets a finished MP4 plus a DaVinci
timeline, with no manual editing needed.

## Goals

- Web-based tutorials only, in Chrome.
- Steps are written ahead of time, reviewed, and replayed identically every run
  (approach 1, chosen by Keon).
- Picture is fitted to voice, never the reverse: the voiceover is the fixed
  timeline.
- Output: final MP4 (1920×1080, 30 fps, voiceover audio) and an FCPXML timeline
  for DaVinci Resolve.
- Re-recording after a site change means fixing a step and rerunning, not
  re-reading the script.

## Non-goals

- Mobile apps and desktop programs (Godot, Houdini, Revit…). Keon decided to
  stay web-only.
- An AI choosing clicks at record time (approach 2, rejected).
- AI or synthetic voice of any kind — against the platform's one rule.
- Recording the site's own sound (would matter for BandLab effects). Later.
- A web UI. The CLI is enough while Claude writes the plans; Keon's part is one
  command.
- Intros, outros, title cards, music. Keon adds those in DaVinci if he wants
  them; the platform sets no format rules. *(Claude's call.)*

## Anti-goals (failure even if it technically works)

- **The tool must never complete a real financial action.** Broker and tax
  tutorials stop at the confirmation screen: no deposits, trades, filings or
  payments are submitted. Plans are reviewed for this, and the step runner
  refuses a click whose target text matches a configurable list ("Confirm
  payment", "Place order", "Submit return", "Deposit"…) unless the plan marks
  that site as a demo account.
- **No credentials anywhere in plans or code.** Keon signs in by hand once in the
  tool's own browser profile; recordings reuse that session.
- **No private details on screen in a published video.** Account emails, names
  and balances are blurred (see Plan format: `blur`).
- Footage that visibly glitches: a cursor jumping, a page flashing half-loaded,
  a click landing seconds before or after the words that describe it.

## Approaches considered

1. **Written steps, replayed** (picked). Per beat, a short list of browser steps
   drafted by Claude from exploring the live site, reviewed by Keon, replayed by
   Playwright. Deterministic and cheap to re-record; costs writing steps per
   video, which Claude does.
2. **AI clicks live.** An AI reads each sentence and decides clicks during the
   recording. Little prep, but different every run, can misclick on camera,
   costs per run.
3. **Keon clicks, tool syncs.** He records himself silently, the tool fits it to
   the voiceover. Simplest, saves the least time.

Keon picked 1 because the AI's judgment happens at prep time, where it can be
read, not at record time, where mistakes are filmed.

## How it works

```
plan.toml ──► script.txt ──► Keon reads it ──► voiceover.wav
    │                                              │
    │                        clean + align (existing engine)
    ▼                                              ▼
 browser run ──► raw capture + step log ──► fit to beat windows ──► final.mp4
                                                                └─► final.fcpxml
```

1. **Plan.** One `plan.toml` per video (Claude drafts it, Keon reviews). It holds
   the narration, split into beats, and each beat's steps.
2. **Script.** `script` command writes the narration as a `.txt`, one beat per
   paragraph — the same format the auto-editor already reads.
3. **Check.** `check` runs every step without recording and reports the first
   failing step (beat, step, screenshot). Run before Keon records, so he never
   reads a script for a plan that doesn't work.
4. **Voiceover.** Keon records with any recorder.
5. **Make.** `make plan.toml voiceover.wav`:
   - Clean the voiceover with the existing engine — `transcribe` + `analyse`
     drop silences, losing retakes and stutters — and render the cleaned audio.
     The cleaned voiceover is the timeline. Lines the aligner marks `miss` or
     `review` stop the run with the report, so a bad read is caught before
     recording.
   - Align each beat to its words in the cleaned voiceover (existing
     `align_script`; beats are the script lines). This gives every beat a time
     window and every word a timestamp. Word times are carried through the
     cleaning by the kept segments' offsets, not by transcribing a second time.
     *(Claude's call.)* Unscripted speech between beats (an ad-lib) belongs to
     the beat before it, whose last frame holds through it. *(Claude's call.)*
   - Run the plan once, start to finish, in one browser session (page state
     carries between beats), capturing the screen and logging each step's
     start and end time.
   - Fit: for each beat, map its stretch of capture onto its voiceover window
     (rules below), then join the beats.
   - Write `final.mp4` and `final.fcpxml`.

## Fitting rules

- A step with an `at` anchor lands on its anchor word: the click (or the first
  keystroke) happens at the moment that word is spoken.
- Dead time in the capture (page loads, waits) is cut first, then shortened, to
  make room. Typing may be sped up. Clicks and the cursor's travel to them are
  never sped up beyond what still looks natural.
- If the capture is shorter than the voiceover window, the last frame of the
  beat holds until the window ends.
- If a beat's capture cannot be made to fit (more action than narration even
  after cutting waits), the run stops and names the beat — the fix is fewer
  steps or more narration, not a frantic video.

## Plan format

TOML, read with the standard library.

```toml
title = "How to create a project in Supabase"
start_url = "https://supabase.com/dashboard/projects"
blur = ["[data-testid=user-email]"]   # CSS selectors blurred for the whole run
demo_account = false                  # true only for broker demo accounts

[[beat]]
say = "Start on your Supabase dashboard and click New project."
steps = [
  { click = "role=button[name='New project']", at = "New project" },
]

[[beat]]
say = "Give it a name, then choose a strong database password."
steps = [
  { type = "my-first-app", into = "label=Name", at = "name" },
  { click = "text=Generate a password", at = "password" },
]
```

Optional `setup` and `cleanup` step lists run before and after recording, off
camera. Many tutorials create something (a Supabase project, a Kaggle notebook);
`cleanup` deletes it so reruns don't pile up resources or hit free-tier limits
(Supabase allows two free projects), and `setup` reaches the starting state.
*(Claude's call.)*

```toml
cleanup = [
  { goto = "https://supabase.com/dashboard/project/_/settings/general" },
  # …delete the project created above
]
```

Step vocabulary: `goto`, `click`, `type` (+ `into`), `press`, `hover`,
`scroll`, `wait_for` (text or selector), `pause` (seconds). Selectors are
Playwright locators, preferring role and visible text over CSS so they survive
redesigns. `at` is optional; without it, steps are spread across the beat in
order.

## On-screen look

- 1920×1080 browser viewport, no browser chrome or bookmarks bar in frame.
- A drawn cursor (the real one is not captured) that travels smoothly to each
  target and shows a brief ripple on click.
- Human-paced typing.
- `blur` selectors get a CSS blur for the whole run.

## Where it lives

In the auto-editor repo, as a new package `tutorial/` with
`py -3.14 -m tutorial login|script|check|make`. It reuses `transcribe.py`,
`aligner.py`, `edit.analyse`, `render.py` and `xml_export.py` rather than
copying them. `login` opens the tool's browser profile so Keon can sign in to
each site by hand.

New dependency: Playwright for Python (the Chromium build is already on this
machine from another project).

## Pilot

One Supabase keyword, end to end: Claude writes the plan, Keon records, the tool
produces the video, Keon judges it by watching. Supabase is first because it is
web-only, the dashboard is ordinary page controls, and it is the largest
web-only topic ($146, 46 keywords). BandLab's web studio comes after, since its
timeline is drawn as an image and clicks there are by position.

## Left to the builder

- **How the screen is captured.** It must give crisp 1080p text with accurate
  timestamps per frame. Playwright's built-in video recording is low bitrate and
  likely not good enough; candidates are the browser's screencast frames or
  ffmpeg capturing the window. Decide by recording a sample and comparing.
- The exact retiming mechanics within the fitting rules.
- The cursor's look and the ripple.
- Default list of blocked confirmation phrases.
- Layout of the output folder.
