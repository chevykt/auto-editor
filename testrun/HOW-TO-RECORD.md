# Calibration recording — directions

Goal: one ~60-second recording that deliberately contains every phenomenon the
tool is supposed to handle, so you can check each one in the report.

Read `script.txt` aloud while screen-recording anything (a browser, a folder —
the picture doesn't matter, only the audio does).

## What to do on each line

Read line 1 **cleanly**. This is your control — if this one doesn't match, the
problem is transcription or threshold, not the detectors.

> Welcome back. Today I want to show you how to get set up in about two minutes.

Pause **~2 seconds**. This gives the silence detector something obvious to find.

Read line 2, but **stutter on one word** — say a word twice in a row, quickly,
the way you actually would. "from the the top left menu."

> Start by opening the dashboard from the top left menu.

Read line 3, but **cut yourself off mid-word on "configuration"** and restart
it: "head into the config— configuration panel". Say it fast and immediately,
without a pause — that adjacency is what the detector keys on.

> From there, head into the configuration panel and find the sync settings.

Read line 4, **get it wrong, stop, and retake the whole line**. Say something
noticeably different the first time, then say the real line. Don't restart the
recording — the whole point is that the tool finds and drops the bad take.

> Turn on automatic sync, then pick how often you want it to run.

Read line 5 cleanly, then **ad-lib for ~10 seconds** about anything — something
that appears nowhere in the script. "And honestly, that's the part most people
get wrong, so it's worth double checking." This is the most important test:
that content must survive.

> Save your changes, and that's it. You're done.

Pause ~3 seconds before you stop recording, so there's trailing dead air to cut.

## Recording on Windows

No install needed: press **Win + Alt + R** to start and stop the Xbox Game Bar
recorder. Output lands in `Videos\Captures`. Make sure your mic is on — Game
Bar sometimes defaults to system audio only.

If you already have OBS, use that. Any format ffmpeg reads is fine (mp4, mkv,
mov).

## Then

Copy the recording to this folder as `raw.mp4` and follow the commands your
assistant gave you, or see the README.
