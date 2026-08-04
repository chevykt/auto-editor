"""Review artifacts.

The JSON report is the editable, re-loadable one: it is what --from-report
consumes, so a human can delete a cut they disagree with and render from the
result. The Markdown report is for reading.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from config import EditorConfig
from cutlist import keep_segments
from models import CutSpan, LineMatch, ScriptLine, Span, Word


def _adlib_spans(matches: list[LineMatch], cuts: list[CutSpan],
                 words: list[Word]) -> list[dict]:
    """Words that matched no script line and were not cut - probable ad-libs."""
    claimed: set[int] = set()
    for m in matches:
        if m.word_range:
            claimed.update(range(*m.word_range))

    def is_cut(w: Word) -> bool:
        # Midpoint, not strict containment: cut boundaries are pulled inward by
        # the padding, so a word the render removes is often not fully enclosed
        # by the padded span. Strict containment would list it as a surviving
        # ad-lib when it is actually being cut.
        midpoint = (w.start + w.end) / 2
        return any(c.start <= midpoint <= c.end for c in cuts)

    out: list[dict] = []
    run: list[Word] = []
    for i, w in enumerate(words):
        if i not in claimed and not is_cut(w):
            run.append(w)
        elif run:
            out.append({"start": run[0].start, "end": run[-1].end,
                        "text": " ".join(x.text for x in run)})
            run = []
    if run:
        out.append({"start": run[0].start, "end": run[-1].end,
                    "text": " ".join(x.text for x in run)})
    return out


def build_report(lines: list[ScriptLine], matches: list[LineMatch],
                 cuts: list[CutSpan], words: list[Word], duration: float,
                 cfg: EditorConfig, video: str,
                 segments: list[Span] | None = None,
                 mode: str = "subtractive") -> dict:
    """Assemble the full review report as a plain dict.

    segments: overrides the surviving segments. Reorder mode supplies its own,
    in script order, since they are not the complement of a cut list.
    mode: "subtractive" or "reorder" - changes what the report can honestly
    say about unmatched content, which one mode keeps and the other discards.
    """
    if segments is None:
        segments = keep_segments(cuts, duration)

    # Derived from the segments rather than summed from the cuts, so the number
    # is right in both modes. In reorder mode the segments are not a complement
    # of the cut list at all.
    output_seconds = sum(s.duration for s in segments)
    by_line = {m.line_idx: m for m in matches}

    return {
        "video": video,
        "duration": duration,
        "mode": mode,
        "config": asdict(cfg),
        "summary": {
            "line_count": len(lines),
            "misses": sum(1 for m in matches if m.status.value == "miss"),
            "reviews": sum(1 for m in matches if m.status.value == "review"),
            "cut_count": len(cuts),
            "cut_seconds": duration - output_seconds,
            "output_seconds": output_seconds,
        },
        "lines": [
            {
                "index": l.index,
                "script": l.text,
                "status": by_line[l.index].status.value,
                "score": by_line[l.index].score,
                "scoped": by_line[l.index].scoped,
                "matched_text": by_line[l.index].matched_text,
                "start": by_line[l.index].span.start if by_line[l.index].span else None,
                "end": by_line[l.index].span.end if by_line[l.index].span else None,
            }
            for l in lines if l.index in by_line
        ],
        "cuts": [
            {
                "start": c.start,
                "end": c.end,
                "duration": c.duration,
                "reasons": sorted(r.value for r in c.reasons),
                "detail": c.detail,
            }
            for c in cuts
        ],
        "adlibs": _adlib_spans(matches, cuts, words),
        "segments": [{"start": s.start, "end": s.end} for s in segments],
    }


def write_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ts(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    m, s = divmod(float(seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def render_markdown(report: dict) -> str:
    """Human review document.

    Silence is summarised rather than enumerated - it is never a judgment
    call, and listing hundreds of gaps would bury the cuts that are.
    """
    s = report["summary"]
    reordering = report.get("mode") == "reorder"
    out: list[str] = [
        "# Auto-Editor Review",
        "",
        f"**Source:** `{report['video']}`  ",
        ("**Mode:** reorder - rebuilt from matched windows in script order. "
         "Anything that did not match a script line has been DISCARDED.  "
         if reordering else
         "**Mode:** subtractive - original order preserved, unmatched content kept.  "),
        f"**Duration:** {_ts(report['duration'])} -> "
        f"{_ts(s['output_seconds'])} "
        f"({s['cut_seconds']:.1f}s removed across {s['cut_count']} cuts)  ",
        f"**Script lines:** {s['line_count']}  "
        f"({s['misses']} missed, {s['reviews']} needing review)",
        "",
        "## Script lines",
        "",
        "| # | Status | Score | Time | Matched text |",
        "|---|---|---|---|---|",
    ]

    for line in report["lines"]:
        flag = {"ok": "ok", "review": "**REVIEW**", "miss": "**MISS**"}[line["status"]]
        if not line["scoped"] and line["status"] != "miss":
            flag += " (out of order)"
        text = line["matched_text"] or f"_{line['script']}_"
        out.append(f"| {line['index']} | {flag} | {line['score']:.2f} | "
                   f"{_ts(line['start'])} | {text} |")

    silence = [c for c in report["cuts"] if c["reasons"] == ["silence"]]
    judged = [c for c in report["cuts"] if c["reasons"] != ["silence"]]

    out += [
        "",
        "## Cuts",
        "",
        f"Silence: **{len(silence)} cuts**, "
        f"{sum(c['duration'] for c in silence):.1f}s total.",
        "",
        "Everything below is a judgment call. Check that none of these ate an "
        "intentional ad-lib.",
        "",
        "| Time | Length | Reason | Detail |",
        "|---|---|---|---|",
    ]
    for c in judged:
        out.append(f"| {_ts(c['start'])} | {c['duration']:.2f}s | "
                   f"{', '.join(c['reasons'])} | {c['detail']} |")
    if not judged:
        out.append("| _none_ | | | |")

    if reordering:
        out += ["", "## Discarded (not in the script)", "",
                "Reorder mode keeps only what matched a script line. "
                "**Everything below was removed**, including any ad-lib you "
                "meant to keep. Use the default subtractive mode to preserve it.",
                ""]
    else:
        out += ["", "## Kept, but not in the script", "",
                "Probable ad-libs. These were left in deliberately.", ""]

    if report["adlibs"]:
        for a in report["adlibs"]:
            out.append(f"- `{_ts(a['start'])}` {a['text']}")
    else:
        out.append("_none_")

    if reordering:
        missed = [l for l in report["lines"] if l["status"] == "miss"]
        out += ["", "## Holes in the reassembly", ""]
        if missed:
            out.append("These lines had no usable take, so the script order "
                       "below is incomplete:")
            out += [f"- line {l['index']}: {l['script']}" for l in missed]
        else:
            out.append("_none - every script line contributed a segment._")

    return "\n".join(out) + "\n"


def write_markdown(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
