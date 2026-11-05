"""plan.toml: a tutorial's narration, split into beats, and the browser steps for each.

Claude drafts a plan by exploring the live site; Keon reviews it before he
records a word. Errors name the beat and step, because a plan is edited by
hand and "invalid plan" alone would send him searching.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

STEP_KINDS = ("goto", "click", "type", "press", "hover", "scroll", "wait_for", "pause")


class PlanError(ValueError):
    """A plan file that cannot be run as written."""


@dataclass(frozen=True)
class Step:
    kind: str
    target: str | None      # locator, or the URL for goto
    value: str | None       # text for type, key for press, seconds for pause
    at: str | None          # anchor phrase in the narration


@dataclass(frozen=True)
class Beat:
    say: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Plan:
    title: str
    start_url: str
    beats: tuple[Beat, ...]
    blur: tuple[str, ...] = ()
    demo_account: bool = False
    setup: tuple[Step, ...] = ()
    cleanup: tuple[Step, ...] = ()
    needs: tuple[str, ...] = ()     # "Needed equipment and requirements"


def _step(raw: dict, where: str) -> Step:
    kinds = [k for k in raw if k in STEP_KINDS]
    if len(kinds) != 1:
        raise PlanError(f"{where}: needs exactly one of {', '.join(STEP_KINDS)}, "
                        f"got {', '.join(raw) or 'nothing'}")
    kind = kinds[0]
    extra = set(raw) - {kind, "into", "at"}
    if extra:
        raise PlanError(f"{where}: unknown key {', '.join(sorted(extra))}")
    if "into" in raw and kind != "type":
        raise PlanError(f"{where}: 'into' only goes with 'type'")

    value = raw[kind]
    at = raw.get("at")
    if kind == "pause":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(f"{where}: pause needs a number of seconds, got {value!r}")
        return Step(kind, None, str(value), at)
    if kind == "type":
        return Step(kind, raw.get("into"), str(value), at)
    if kind == "press":
        return Step(kind, None, str(value), at)
    return Step(kind, str(value), None, at)


def _steps(raw: list, where: str) -> tuple[Step, ...]:
    return tuple(_step(s, f"{where}, step {i}") for i, s in enumerate(raw, 1))


def load_plan(path: str | Path) -> Plan:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("start_url"):
        raise PlanError("plan needs a start_url")
    raw_beats = data.get("beat") or []
    if not raw_beats:
        raise PlanError("plan needs at least one [[beat]]")

    beats = []
    for i, raw in enumerate(raw_beats, 1):
        if not str(raw.get("say", "")).strip():
            raise PlanError(f"beat {i}: needs a 'say' line")
        beats.append(Beat(raw["say"].strip(), _steps(raw.get("steps", []), f"beat {i}")))

    return Plan(title=data.get("title", ""), start_url=data["start_url"],
                beats=tuple(beats), blur=tuple(data.get("blur", [])),
                demo_account=bool(data.get("demo_account", False)),
                setup=_steps(data.get("setup", []), "setup"),
                cleanup=_steps(data.get("cleanup", []), "cleanup"),
                needs=tuple(data.get("needs", [])))


def script_text(plan: Plan) -> str:
    """The narration as the auto-editor reads scripts: one beat per paragraph."""
    return "\n\n".join(b.say for b in plan.beats)


# Keon's own pace, from his Terraria batch: 151 words read in about 57 seconds.
WORDS_PER_SECOND = 151 / 57
MIN_SECONDS = 50
OPENING = "how to"
CLOSING = "and that's how you basically"


def _words(plan: Plan) -> int:
    return sum(len(b.say.split()) for b in plan.beats)


def reading_sheet(plan: Plan) -> str:
    """The script laid out like Keon's Terraria batch: title, length, needs, script."""
    words = _words(plan)
    parts = [plan.title, f"{words} words, about {round(words / WORDS_PER_SECOND)} seconds"]
    if plan.needs:
        parts.append("\nNeeded equipment and requirements\n"
                     + "\n".join(f"- {need}" for need in plan.needs))
    return "\n".join(parts) + "\n\n" + script_text(plan) + "\n"


def format_problems(plan: Plan) -> list[str]:
    """Where a script breaks Keon's rules for these videos."""
    problems = []
    if not plan.beats[0].say.lower().startswith(OPENING):
        problems.append('the script must open with its title, "How to ..."')
    closing = plan.beats[-1].say.lower().replace("’", "'")
    if not closing.startswith(CLOSING):
        problems.append("the last line must start \"And that's how you basically ...\"")
    seconds = _words(plan) / WORDS_PER_SECOND
    if seconds < MIN_SECONDS:
        problems.append(f"about {round(seconds)} seconds to read; videos need at least "
                        f"{MIN_SECONDS} seconds (roughly "
                        f"{round(MIN_SECONDS * WORDS_PER_SECOND)} words)")
    return problems
