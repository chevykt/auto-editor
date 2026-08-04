"""CLI entrypoint and orchestration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aligner import align_script
from config import EditorConfig
from cutlist import build_cutlist
from disfluency import detect_disfluencies
from models import Word
from report import build_report, load_report, write_json, write_markdown
from script_parser import parse_script
from silence import detect_silence
from transcribe import transcribe


class _NotImplementedFlag(argparse.Action):
    """Reject a specced-but-unbuilt flag loudly instead of ignoring it."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            f"{option_string} is not implemented in this build (phases 1-2 only). "
            "See docs/superpowers/specs for the deferred scope.")


def build_parser() -> argparse.ArgumentParser:
    d = EditorConfig()
    p = argparse.ArgumentParser(
        prog="edit.py",
        description="Remove dead air, false starts, and stutters from a "
                    "script-matched tutorial recording, keeping ad-libs.")

    p.add_argument("--script",
                   help="Voiceover script (.txt). Required unless --from-report")
    p.add_argument("--video", required=True, help="Raw screen recording")
    p.add_argument("--out", help="Rendered output (required with --auto/--from-report)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Report only, no render (default)")
    mode.add_argument("--auto", action="store_true",
                      help="Render immediately, skipping the review gate")

    p.add_argument("--from-report", help="Render from an approved JSON report")
    p.add_argument("--report-path", help="Stem for the .json/.md report pair")

    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--review-margin", type=float, default=d.review_margin)
    p.add_argument("--lookahead-s", type=float, default=d.lookahead_s)
    p.add_argument("--tiebreak", choices=["latest", "score"], default=d.tiebreak)
    p.add_argument("--silence-ms", type=int, default=d.silence_ms)
    p.add_argument("--padding-ms", type=int, default=d.padding_ms,
                   help="Pull-in at disfluency and retake cut boundaries")
    p.add_argument("--silence-padding-ms", type=int, default=d.silence_padding_ms,
                   help="Pull-in at silence cut boundaries; can afford to be "
                        "larger since a silence cut is bounded by speech")
    p.add_argument("--min-segment-ms", type=int, default=d.min_segment_ms)
    p.add_argument("--crossfade-ms", type=int, default=d.crossfade_ms)
    p.add_argument("--duplicate-window-s", type=float, default=d.duplicate_window_s)
    p.add_argument("--max-disfluency-words", type=int, default=d.max_disfluency_words)
    p.add_argument("--model", default=d.model)
    p.add_argument("--language", default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=d.device,
                   help="Transcription device. auto tries CUDA, falls back to CPU")
    p.add_argument("--compute-type", default=d.compute_type,
                   help="Only used with an explicit --device")
    p.add_argument("--verbatim", action="store_true",
                   help="Bias Whisper toward transcribing stutters and fillers "
                        "instead of tidying them away. Needed for disfluency "
                        "detection to have anything to find")
    p.add_argument("--initial-prompt", default=None,
                   help="Style prompt for Whisper; overrides --verbatim's built-in")
    p.add_argument("--split-sentences", action="store_true")
    p.add_argument("--no-clean-stutters", action="store_true",
                   help="Disable disfluency detection entirely")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore any cached transcript")

    p.add_argument("--reorder", nargs=0, action=_NotImplementedFlag,
                   help=argparse.SUPPRESS)
    p.add_argument("--export-xml", nargs=0, action=_NotImplementedFlag,
                   help=argparse.SUPPRESS)
    return p


def config_from_args(args: argparse.Namespace) -> EditorConfig:
    return EditorConfig(
        threshold=args.threshold,
        review_margin=args.review_margin,
        lookahead_s=args.lookahead_s,
        tiebreak=args.tiebreak,
        silence_ms=args.silence_ms,
        padding_ms=args.padding_ms,
        silence_padding_ms=args.silence_padding_ms,
        min_segment_ms=args.min_segment_ms,
        crossfade_ms=args.crossfade_ms,
        duplicate_window_s=args.duplicate_window_s,
        max_disfluency_words=args.max_disfluency_words,
        clean_stutters=not args.no_clean_stutters,
        model=args.model,
        language=args.language,
        use_cache=not args.no_cache,
        device=args.device,
        compute_type=args.compute_type,
        verbatim=args.verbatim,
        initial_prompt=args.initial_prompt,
    )


def validate_args(args: argparse.Namespace) -> str | None:
    """Cross-argument rules argparse cannot express. Returns an error, or None.

    Checked before any work starts. Transcribing a 20-minute recording and
    only then reporting a missing --out would waste several minutes.
    """
    if not args.from_report and not args.script:
        return "--script is required unless rendering with --from-report"
    if (args.auto or args.from_report) and not args.out:
        return "--out is required when rendering"
    return None


def missing_inputs(args: argparse.Namespace) -> str | None:
    """First input file that does not exist, as an error message, or None.

    Checked up front so a typo'd path reports cleanly instead of surfacing a
    raw traceback part-way through a run.
    """
    candidates = [("--video", args.video), ("--from-report", args.from_report)]
    if not args.from_report:
        candidates.append(("--script", args.script))

    for label, path in candidates:
        if path and not Path(path).exists():
            return f"{label} not found: {path}"
    return None


def report_paths(video: Path, report_path: str | None) -> tuple[Path, Path]:
    """Where the .json / .md report pair goes.

    Suffixes are appended by name, never via Path.with_suffix: the default
    stem "raw.report" already looks like it has a ".report" extension, so
    with_suffix(".json") would REPLACE it and write "raw.json".
    """
    stem = Path(report_path) if report_path \
        else video.with_name(video.stem + ".report")
    return (stem.with_name(stem.name + ".json"),
            stem.with_name(stem.name + ".md"))


def analyse(words: list[Word], script_text: str, duration: float,
            cfg: EditorConfig, video: str,
            split_sentences: bool = False) -> dict:
    """The whole of phase 1: transcript + script in, review report out."""
    lines = parse_script(script_text, split_sentences=split_sentences)

    raw = detect_silence(words, cfg, duration=duration)
    matches, retake_cuts = align_script(words, lines, cfg)
    raw += retake_cuts

    for m in matches:
        if m.word_range:
            raw += detect_disfluencies(words, m.word_range[0], m.word_range[1], cfg)

    cuts = build_cutlist(raw, duration, cfg)
    return build_report(lines, matches, cuts, words, duration, cfg, video)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    problem = validate_args(args)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    missing = missing_inputs(args)
    if missing:
        print(f"error: {missing}", file=sys.stderr)
        return 2

    cfg = config_from_args(args)
    video = Path(args.video)

    if args.from_report:
        report = load_report(args.from_report)
    else:
        script_text = Path(args.script).read_text(encoding="utf-8")

        print(f"Transcribing {video.name} (model={cfg.model})...", file=sys.stderr)
        words, duration = transcribe(video, cfg)
        print(f"  {len(words)} words, {duration:.1f}s", file=sys.stderr)

        report = analyse(words, script_text, duration, cfg, str(video),
                         split_sentences=args.split_sentences)

        json_path, md_path = report_paths(video, args.report_path)
        write_json(report, json_path)
        write_markdown(report, md_path)

        s = report["summary"]
        print(f"\nReport: {md_path}", file=sys.stderr)
        print(f"  {s['cut_count']} cuts, {s['cut_seconds']:.1f}s removed "
              f"-> {s['output_seconds']:.1f}s output", file=sys.stderr)
        print(f"  {s['misses']} missed lines, {s['reviews']} needing review",
              file=sys.stderr)

    if not (args.auto or args.from_report):
        print("\nReview the report, then re-run with --auto, or render the "
              "approved report with --from-report.", file=sys.stderr)
        return 0

    from render import render_video
    render_video(video, Path(args.out), report, cfg)
    print(f"\nRendered: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
