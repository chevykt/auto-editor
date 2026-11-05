"""The plan.toml format: a video's narration split into beats, and each beat's steps."""
import pytest

from tutorial.plan import PlanError, Step, load_plan, script_text

# The Plan format block from the design spec, verbatim.
SPEC_EXAMPLE = '''
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
'''

HEADER = 'title = "t"\nstart_url = "https://example.com"\n'


def write(tmp_path, text):
    path = tmp_path / "plan.toml"
    path.write_text(text, encoding="utf-8")
    return path


def beat(say, steps="[]"):
    return f'\n[[beat]]\nsay = "{say}"\nsteps = {steps}\n'


def test_loads_the_spec_example(tmp_path):
    plan = load_plan(write(tmp_path, SPEC_EXAMPLE))
    assert plan.start_url == "https://supabase.com/dashboard/projects"
    assert plan.blur == ("[data-testid=user-email]",)
    assert plan.demo_account is False
    assert plan.beats[0].steps[0] == Step(
        "click", "role=button[name='New project']", None, "New project")
    assert plan.beats[1].steps[0] == Step("type", "label=Name", "my-first-app", "name")


def test_script_text_is_one_beat_per_paragraph(tmp_path):
    paragraphs = script_text(load_plan(write(tmp_path, SPEC_EXAMPLE))).split("\n\n")
    assert paragraphs == [
        "Start on your Supabase dashboard and click New project.",
        "Give it a name, then choose a strong database password."]


def test_unknown_step_kind_names_beat_and_step(tmp_path):
    text = HEADER + beat("one") + beat("two", '[{ drag = "x" }]')
    with pytest.raises(PlanError, match=r"beat 2, step 1.*drag"):
        load_plan(write(tmp_path, text))


def test_a_beat_without_say_is_rejected(tmp_path):
    with pytest.raises(PlanError, match="beat 1"):
        load_plan(write(tmp_path, HEADER + "\n[[beat]]\nsteps = []\n"))


def test_a_step_with_two_kinds_is_rejected(tmp_path):
    text = HEADER + beat("one", '[{ click = "a", goto = "https://x" }]')
    with pytest.raises(PlanError, match=r"beat 1, step 1"):
        load_plan(write(tmp_path, text))


def test_into_is_only_allowed_with_type(tmp_path):
    text = HEADER + beat("one", '[{ click = "a", into = "b" }]')
    with pytest.raises(PlanError, match="into"):
        load_plan(write(tmp_path, text))


def test_pause_must_be_a_number(tmp_path):
    with pytest.raises(PlanError, match="pause"):
        load_plan(write(tmp_path, HEADER + beat("one", '[{ pause = "soon" }]')))


def test_a_numeric_pause_is_kept_as_its_value(tmp_path):
    plan = load_plan(write(tmp_path, HEADER + beat("one", "[{ pause = 1.5 }]")))
    assert plan.beats[0].steps[0] == Step("pause", None, "1.5", None)


def test_goto_and_press_put_their_value_in_the_right_field(tmp_path):
    text = HEADER + beat("one", '[{ goto = "https://x" }, { press = "Enter" }]')
    steps = load_plan(write(tmp_path, text)).beats[0].steps
    assert steps == (Step("goto", "https://x", None, None),
                     Step("press", None, "Enter", None))


def test_setup_and_cleanup_are_optional_step_lists(tmp_path):
    text = (HEADER + 'setup = [{ goto = "https://x/new" }]\n'
            'cleanup = [{ click = "text=Delete" }]\n' + beat("one"))
    plan = load_plan(write(tmp_path, text))
    assert plan.setup == (Step("goto", "https://x/new", None, None),)
    assert plan.cleanup == (Step("click", "text=Delete", None, None),)
    bare = load_plan(write(tmp_path, HEADER + beat("one")))
    assert bare.setup == () and bare.cleanup == ()


def test_a_cleanup_error_says_cleanup(tmp_path):
    text = HEADER + 'cleanup = [{ fly = "x" }]\n' + beat("one")
    with pytest.raises(PlanError, match=r"cleanup, step 1"):
        load_plan(write(tmp_path, text))


def test_a_beat_with_no_steps_is_allowed(tmp_path):
    plan = load_plan(write(tmp_path, HEADER + beat("just talking")))
    assert plan.beats[0].steps == ()


def test_a_plan_needs_at_least_one_beat(tmp_path):
    with pytest.raises(PlanError, match="beat"):
        load_plan(write(tmp_path, HEADER))


def test_missing_start_url_is_rejected(tmp_path):
    with pytest.raises(PlanError, match="start_url"):
        load_plan(write(tmp_path, 'title = "t"\n' + beat("one")))


# --- the reading sheet and Keon's script rules -------------------------------

from tutorial.plan import Beat, Plan, format_problems, reading_sheet


def sheet_plan(*says, needs=()):
    return Plan("How to make a project", "https://x",
                tuple(Beat(s, ()) for s in says), needs=tuple(needs))


LONG = " ".join(["word"] * 140)


def test_needs_is_an_optional_list(tmp_path):
    plan = load_plan(write(tmp_path, HEADER + 'needs = ["A free account", "Node.js"]\n'
                           + beat("one")))
    assert plan.needs == ("A free account", "Node.js")
    assert load_plan(write(tmp_path, HEADER + beat("one"))).needs == ()


def test_reading_sheet_follows_the_terraria_batch_layout():
    plan = sheet_plan("How to make a project. " + LONG,
                      "And that's how you basically make a project.",
                      needs=["A free account"])
    lines = reading_sheet(plan).splitlines()
    assert lines[0] == "How to make a project"
    assert lines[1] == "153 words, about 58 seconds"
    assert lines[3:5] == ["Needed equipment and requirements", "- A free account"]
    assert lines[-3:] == [f"How to make a project. {LONG}", "",
                          "And that's how you basically make a project."]


def test_a_plan_without_needs_has_no_requirements_heading():
    plan = sheet_plan("How to x. " + LONG, "And that's how you basically x.")
    assert "Needed equipment" not in reading_sheet(plan)


def test_a_plan_in_keons_format_has_no_problems():
    plan = sheet_plan("How to make a project. " + LONG,
                      "And that’s how you basically make a project.")
    assert format_problems(plan) == []


def test_the_opening_must_be_the_how_to_title():
    plan = sheet_plan("Here is how to make a project. " + LONG,
                      "And that's how you basically make a project.")
    assert any("How to" in p for p in format_problems(plan))


def test_the_closing_must_be_and_thats_how_you_basically():
    plan = sheet_plan("How to make a project. " + LONG, "That is all.")
    assert any("And that's how you basically" in p for p in format_problems(plan))


def test_a_script_under_50_seconds_is_flagged():
    plan = sheet_plan("How to make a project.", "And that's how you basically do it.")
    assert any("50 seconds" in p for p in format_problems(plan))
