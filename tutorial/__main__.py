"""py -3.14 -m tutorial login|script|check|make — the web tutorial recorder.

Keon's part is two commands: `script` gives him what to read, and `make` turns
his recording into the finished video. `login` and `check` keep a plan honest
before he spends a read on it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from config import EditorConfig
from report import write_markdown
from tutorial.assemble import assemble, write_timeline
from tutorial.capture import Capture
from tutorial.fit import FitError, fit_all
from tutorial.plan import Plan, PlanError, load_plan, script_text
from tutorial.runner import (BlockedAction, SessionExpired, StepFailed, open_browser,
                             run_steps, start)
from tutorial.voice import VoiceError, clean_voiceover

OUT_ROOT = Path("tutorial-out")
_RUN_ERRORS = (StepFailed, BlockedAction, SessionExpired, FitError, VoiceError,
               RuntimeError)


def _beat_steps(plan: Plan) -> list:
    return [(b, s, step) for b, beat in enumerate(plan.beats)
            for s, step in enumerate(beat.steps)]


def _side_steps(steps) -> list:
    return [(-1, s, step) for s, step in enumerate(steps)]


def _login(url: str) -> int:
    with open_browser(headless=False) as page:
        page.goto(url)
        print("Sign in to the site in the window that opened, then close it.")
        page.wait_for_event("close", timeout=0)
    return 0


def _check(plan: Plan, out: Path) -> int:
    shots = out / "shots"
    with open_browser(headless=True) as page:
        start(page, plan)
        try:
            run_steps(page, plan, _side_steps(plan.setup), shots)
            run_steps(page, plan, _beat_steps(plan), shots)
        finally:
            run_steps(page, plan, _side_steps(plan.cleanup), shots)
    print("ok")
    return 0


def _make(plan: Plan, plan_path: Path, voice: Path, out: Path) -> int:
    clean = out / "voice.clean.wav"
    try:
        timing, warnings = clean_voiceover(voice, plan, EditorConfig(verbatim=True), clean)
    except VoiceError as exc:
        report = plan_path.with_name(plan_path.stem + ".voice-report.md")
        write_markdown(exc.report, report)
        raise VoiceError(exc.report) from exc

    shots, frames = out / "shots", out / "frames"
    with open_browser(headless=True) as page:
        start(page, plan)
        run_steps(page, plan, _side_steps(plan.setup), shots)
        try:
            with Capture(page, frames) as capture:
                events = run_steps(page, plan, _beat_steps(plan), shots)
        finally:
            # whatever happened, what the tutorial created must not pile up
            try:
                run_steps(page, plan, _side_steps(plan.cleanup), shots)
            except _RUN_ERRORS as exc:
                warnings.append(f"cleanup did not finish: {exc}")

    recording, t0 = capture.to_video(out / "capture.mp4")
    shutil.rmtree(frames, ignore_errors=True)
    pieces, fit_warnings = fit_all(events, t0, timing, plan)
    final = assemble(recording, pieces, clean, out / "final.mp4")
    write_timeline(final, timing.windows, out / "final.fcpxml")

    lines = warnings + fit_warnings + [
        f"beat {e.beat + 1} step {e.step + 1} {e.kind}: {e.start - t0:.2f}s "
        f"action {e.action - t0:.2f}s end {e.end - t0:.2f}s" for e in events]
    (out / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for warning in warnings + fit_warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"done: {final}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -3.14 -m tutorial")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("login", help="sign in to a site once, by hand").add_argument(
        "url", nargs="?", default="about:blank")
    for name, help_text in (("script", "write the narration to read"),
                            ("check", "run every step without recording")):
        sub.add_parser(name, help=help_text).add_argument("plan")
    make = sub.add_parser("make", help="voiceover in, finished video out")
    make.add_argument("plan")
    make.add_argument("voice", help="your recorded voiceover")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command == "login":
        return _login(args.url)

    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"error: plan not found: {plan_path}", file=sys.stderr)
        return 2
    try:
        plan = load_plan(plan_path)
    except PlanError as exc:
        print(f"error: {plan_path.name}: {exc}", file=sys.stderr)
        return 2

    if args.command == "script":
        target = plan_path.with_name(plan_path.stem + ".script.txt")
        target.write_text(script_text(plan) + "\n", encoding="utf-8")
        print(f"read this: {target}")
        return 0

    out = OUT_ROOT / plan_path.stem
    out.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "check":
            return _check(plan, out)
        voice = Path(args.voice)
        if not voice.exists():
            print(f"error: voiceover not found: {voice}", file=sys.stderr)
            return 2
        return _make(plan, plan_path, voice, out)
    except _RUN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
