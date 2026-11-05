"""Performing a plan's steps in the browser: timing, cursor, blur, and the safety stops."""
import time
from pathlib import Path

import pytest

from tutorial import runner
from tutorial.plan import Beat, Plan, Step
from tutorial.runner import (BlockedAction, SessionExpired, StepFailed, is_blocked,
                             open_browser, run_steps, start)

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = (FIXTURES / "tutorial_page.html").as_uri()


@pytest.mark.parametrize("text", ["Confirm payment", "PLACE ORDER", "Deposit", "Buy now"])
def test_financial_buttons_are_blocked(text):
    assert is_blocked(text, demo_account=False)


def test_a_demo_account_may_click_them():
    assert not is_blocked("Place order", demo_account=True)


def test_ordinary_words_containing_a_phrase_are_not_blocked():
    assert not is_blocked("Buyer guide", demo_account=False)
    assert not is_blocked("New project", demo_account=False)


def plan_for(url=PAGE, blur=(), demo_account=False):
    return Plan("t", url, (Beat("b", ()),), blur=tuple(blur), demo_account=demo_account)


@pytest.fixture
def browser(tmp_path):
    with open_browser(headless=True, profile=tmp_path / "profile") as page:
        yield page


@pytest.mark.smoke
def test_steps_run_and_are_timed(browser, tmp_path):
    plan = plan_for()
    start(browser, plan)
    events = run_steps(browser, plan, [
        (0, 0, Step("click", "role=button[name='New project']", None, None)),
        (0, 1, Step("type", "label=Name", "my-app", None)),
    ], tmp_path / "shots")

    assert browser.input_value("#name") == "my-app"
    assert browser.evaluate("window.__clicks").count("new") == 1
    assert [(e.beat, e.step, e.kind) for e in events] == [(0, 0, "click"), (0, 1, "type")]
    for e in events:
        assert e.start <= e.action <= e.end
    assert events[0].end <= events[1].start


@pytest.mark.smoke
def test_blocked_click_raises_before_clicking(browser, tmp_path):
    plan = plan_for()
    start(browser, plan)
    with pytest.raises(BlockedAction):
        run_steps(browser, plan, [(0, 0, Step("click", "#pay", None, None))],
                  tmp_path / "shots")
    assert "pay" not in browser.evaluate("window.__clicks")


@pytest.mark.smoke
def test_missing_selector_fails_fast_with_a_screenshot(browser, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STEP_TIMEOUT_S", 1)
    plan = plan_for()
    start(browser, plan)
    began = time.time()
    with pytest.raises(StepFailed) as err:
        run_steps(browser, plan, [(2, 3, Step("click", "text=Nope", None, None))],
                  tmp_path / "shots")
    assert time.time() - began < 3
    assert (err.value.beat, err.value.step) == (2, 3)
    assert err.value.screenshot.exists()


@pytest.mark.smoke
def test_wait_for_waits_for_a_late_element(browser, tmp_path):
    plan = plan_for()
    start(browser, plan)
    events = run_steps(browser, plan, [(0, 0, Step("wait_for", "#late", None, None))],
                       tmp_path / "shots")
    assert events[0].end - events[0].start > 1.0


@pytest.mark.smoke
def test_redirect_to_login_is_session_expired(browser):
    with pytest.raises(SessionExpired, match="login"):
        start(browser, plan_for((FIXTURES / "redirect.html").as_uri()))


@pytest.mark.smoke
def test_blurred_element_is_blurred(browser):
    start(browser, plan_for(blur=["#email"]))
    assert "blur" in browser.evaluate(
        "getComputedStyle(document.querySelector('#email')).filter")


@pytest.mark.smoke
def test_the_drawn_cursor_ends_on_the_clicked_element(browser, tmp_path):
    plan = plan_for()
    start(browser, plan)
    run_steps(browser, plan, [(0, 0, Step("click", "#new", None, None))], tmp_path / "s")
    box = browser.locator("#new").bounding_box()
    x, y = browser.evaluate("[window.__cursor.x, window.__cursor.y]")
    assert abs(x - (box["x"] + box["width"] / 2)) < 2
    assert abs(y - (box["y"] + box["height"] / 2)) < 2
