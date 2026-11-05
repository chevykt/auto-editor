"""Performs a plan's steps in Chrome, timing each one for the fitter.

Every action moves a drawn cursor to its target first, so the viewer sees where
the click is about to land. The timing log is what lets the fitter put each
click on the word that describes it.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from tutorial.plan import Plan, Step

PROFILE_DIR = Path.home() / ".auto-editor" / "browser-profile"
OVERLAY = Path(__file__).with_name("overlay.js")

STEP_TIMEOUT_S = 15
CURSOR_TRAVEL_S = 0.5
TYPE_DELAY_S = 0.06
SETTLE_S = 0.6          # held after each beat's last step, so the result is seen

# Whole words, so "Buyer guide" is not caught by "buy". A demo broker account
# may click them; nothing else may, because the click would be real.
BLOCKED_PHRASES = ("confirm payment", "place order", "submit return", "deposit",
                   "withdraw", "pay now", "buy", "sell", "transfer")
_LOGIN_PATHS = ("login", "signin", "sign-in", "auth")


class BlockedAction(RuntimeError):
    """A click that would complete a real financial action."""


class SessionExpired(RuntimeError):
    """The saved sign-in no longer works."""


class StepFailed(RuntimeError):
    def __init__(self, beat: int, step: int, screenshot: Path, reason: str):
        where = "setup/cleanup" if beat < 0 else f"beat {beat + 1}"
        super().__init__(f"{where}, step {step + 1} failed: {reason}\n"
                         f"screenshot: {screenshot}")
        self.beat, self.step, self.screenshot = beat, step, screenshot


@dataclass(frozen=True)
class StepEvent:
    beat: int           # -1 for setup and cleanup
    step: int
    kind: str
    start: float        # time.time() seconds
    action: float       # when the click or first keystroke happens
    end: float


def is_blocked(text: str, demo_account: bool) -> bool:
    if demo_account:
        return False
    return any(re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE)
               for p in BLOCKED_PHRASES)


@contextmanager
def open_browser(headless: bool, profile: Path = PROFILE_DIR):
    """A page in Chrome on the saved profile, 1920x1080, closed on exit."""
    profile.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            str(profile), headless=headless, device_scale_factor=1,
            viewport={"width": 1920, "height": 1080})
        try:
            yield context.pages[0] if context.pages else context.new_page()
        finally:
            context.close()


def start(page: Page, plan: Plan) -> None:
    """Install the cursor and blur, open start_url, and check we are still signed in."""
    script = OVERLAY.read_text(encoding="utf-8").replace(
        "__TUTORIAL_BLUR__", json.dumps(list(plan.blur)))
    page.add_init_script(script=script)
    page.goto(plan.start_url, timeout=STEP_TIMEOUT_S * 1000)
    page.wait_for_load_state("load", timeout=STEP_TIMEOUT_S * 1000)

    wanted = urlparse(plan.start_url).path.lower()
    landed = urlparse(page.url).path.lower()
    if any(p in landed and p not in wanted for p in _LOGIN_PATHS):
        raise SessionExpired(
            f"opening {plan.start_url} went to a login page ({page.url}). "
            "Sign in again with: py -3.14 -m tutorial login")


def _locate(page: Page, target: str):
    # "label=" is this format's own shorthand; everything else is a Playwright selector
    if target.startswith("label="):
        return page.get_by_label(target[len("label="):]).first
    return page.locator(target).first


def _move_to(page: Page, locator) -> None:
    locator.scroll_into_view_if_needed(timeout=STEP_TIMEOUT_S * 1000)
    box = locator.bounding_box(timeout=STEP_TIMEOUT_S * 1000)
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.evaluate("([x, y, ms]) => window.__cursor.moveTo(x, y, ms)",
                  [x, y, CURSOR_TRAVEL_S * 1000])


def _perform(page: Page, plan: Plan, step: Step) -> float:
    """Do one step; returns the moment of its action."""
    timeout = STEP_TIMEOUT_S * 1000
    if step.kind == "goto":
        action = time.time()
        page.goto(step.target, timeout=timeout)
        return action
    if step.kind == "press":
        action = time.time()
        page.keyboard.press(step.value)
        return action
    if step.kind == "pause":
        page.wait_for_timeout(float(step.value) * 1000)
        return time.time()
    if step.kind == "wait_for":
        _locate(page, step.target).wait_for(state="visible", timeout=timeout)
        return time.time()

    locator = _locate(page, step.target)
    locator.wait_for(state="visible", timeout=timeout)
    if step.kind == "scroll":
        action = time.time()
        locator.evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'center'})")
        page.wait_for_timeout(800)
        return action
    if step.kind == "click":
        label = locator.evaluate(
            "e => e.innerText || e.value || e.getAttribute('aria-label') || ''")
        if is_blocked(label, plan.demo_account):
            raise BlockedAction(f"refusing to click {label!r}: it would complete a "
                                "real financial action (set demo_account only for "
                                "a demo account)")
    _move_to(page, locator)
    action = time.time()
    if step.kind == "hover":
        locator.hover(timeout=timeout)
    elif step.kind == "click":
        page.evaluate("window.__cursor.click()")
        locator.click(timeout=timeout)
    elif step.kind == "type":
        page.evaluate("window.__cursor.click()")
        locator.click(timeout=timeout)
        locator.press_sequentially(step.value, delay=TYPE_DELAY_S * 1000)
    return action


def run_steps(page: Page, plan: Plan, steps: list[tuple[int, int, Step]],
              shots: Path) -> list[StepEvent]:
    """Perform (beat, step index, step) in order and log when each happened."""
    events: list[StepEvent] = []
    for n, (beat, index, step) in enumerate(steps):
        began = time.time()
        try:
            action = _perform(page, plan, step)
        except PlaywrightError as exc:
            shots.mkdir(parents=True, exist_ok=True)
            shot = shots / f"beat{beat + 1}-step{index + 1}.png"
            page.screenshot(path=str(shot))
            raise StepFailed(beat, index, shot, str(exc).splitlines()[0]) from exc
        events.append(StepEvent(beat, index, step.kind, began, action, time.time()))

        last_of_beat = n + 1 == len(steps) or steps[n + 1][0] != beat
        if beat >= 0 and last_of_beat:
            page.wait_for_timeout(SETTLE_S * 1000)
    return events
