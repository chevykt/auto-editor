"""CLI entrypoint and orchestration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aligner import align_script
from batch import find_jobs, output_path
from config import EditorConfig, load_config_file
from cutlist import build_cutlist
from disfluency import detect_disfluencies
from models import Span, Word
from reorder import reorder_segments
from report import build_report, load_report, write_json, write_markdown
from script_parser import parse_script
from suspects import find_suspects, suspect_cuts
from silence import detect_silence
from transcribe import transcribe
from xml_export import probe_video_meta, write_xml


def build_parser() -> argparse.ArgumentParser:
    d = EditorConfig()
    p = argparse.ArgumentParser(
        prog="edit.py",
        description="Remove dead air, false starts, and stutters from a "
                    "script-matched tutorial recording, keeping ad-libs.")

    p.add_argument("--script",
                   help="Voiceover script (.txt). Required unless --from-report")
    p.add_argument("--video", help="Raw screen recording. Required unless --batch")
    p.add_argument("--batch", metavar="DIR",
                   help="Process every recording in DIR against the same-stem "
                        ".txt script beside it")
    p.add_argument("--out-dir", metavar="DIR",
                   help="Where batch outputs go; defaults to beside each video")
    p.add_argument("--config", metavar="FILE",
                   help="TOML file of tuning knobs. Command-line flags win")
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

    p.add_argument("--reorder", action="store_true",
                   help="Rebuild from matched windows in SCRIPT order, "
                        "discarding everything unmatched. Loses ad-libs and "
                        "natural flow; only for lines recorded out of sequence")
    p.add_argument("--reorder-pad-ms", type=int, default=d.reorder_pad_ms,
                   help="Outward breathing room on each reordered window")

    p.add_argument("--export-xml", action="store_true",
                   help="Also write a Premiere-compatible XML timeline, so "
                        "cuts can be nudged by hand instead of trusted blindly")
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
        reorder=args.reorder,
        reorder_pad_ms=args.reorder_pad_ms,
    )


def validate_args(args: argparse.Namespace) -> str | None:
    """Cross-argument rules argparse cannot express. Returns an error, or None.

    Checked before any work starts. Transcribing a 20-minute recording and
    only then reporting a missing --out would waste several minutes.
    """
    if args.batch:
        if args.video or args.script:
            return "--batch pairs recordings with scripts itself; drop --video/--script"
        if args.out:
            return "--batch writes one output per recording; use --out-dir"
        if args.from_report:
            return "--from-report renders a single approved report, not a batch"
        return None

    if not args.video:
        return "--video is required unless using --batch"
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
    if args.config and not Path(args.config).exists():
        return f"--config not found: {args.config}"

    if args.batch:
        directory = Path(args.batch)
        if not directory.exists():
            return f"--batch directory not found: {args.batch}"
        if not directory.is_dir():
            return f"--batch expects a directory: {args.batch}"
        return None

    candidates = [("--video", args.video), ("--from-report", args.from_report)]
    if not args.from_report:
        candidates.append(("--script", args.script))

    for label, path in candidates:
        if path and not Path(path).exists():
            return f"{label} not found: {path}"
    return None


def report_paths(video: Path,
                 report_path: str | None) -> tuple[Path, Path, Path]:
    """Where the .json / .md / .xml outputs go.

    Suffixes are appended by name, never via Path.with_suffix: the default
    stem "raw.report" already looks like it has a ".report" extension, so
    with_suffix(".json") would REPLACE it and write "raw.json".
    """
    stem = Path(report_path) if report_path \
        else video.with_name(video.stem + ".report")
    return (stem.with_name(stem.name + ".json"),
            stem.with_name(stem.name + ".md"),
            stem.with_name(stem.name + ".xml"))


def analyse(words: list[Word], script_text: str, duration: float,
            cfg: EditorConfig, video: str,
            split_sentences: bool = False) -> dict:
    """The whole of phase 1: transcript + script in, review report out."""
    lines = parse_script(script_text, split_sentences=split_sentences)

    raw = detect_silence(words, cfg, duration=duration)
    matches, retake_cuts = align_script(words, lines, cfg)
    raw += retake_cuts

    disfluencies: list = []
    for m in matches:
        if m.word_range:
            disfluencies += detect_disfluencies(
                words, m.word_range[0], m.word_range[1], cfg)

    raw += disfluencies

    cuts = build_cutlist(raw, duration, cfg)

    # Reported, never applied. Appended after build_cutlist so they cannot
    # merge with a real cut and quietly become one.
    cuts = cuts + suspect_cuts(find_suspects(words, cfg), words, cfg)
    cuts.sort(key=lambda c: c.start)

    if cfg.reorder:
        # Reconstructive: keep only matched windows, in script order. Silence
        # and retake cuts are irrelevant here - anything outside a matched
        # window is already gone - but disfluencies still sit INSIDE the
        # windows we are keeping, so those are subtracted from each one.
        segments = reorder_segments(
            words, matches, build_cutlist(disfluencies, duration, cfg), cfg)
        return build_report(lines, matches, cuts, words, duration, cfg, video,
                            segments=segments, mode="reorder")

    return build_report(lines, matches, cuts, words, duration, cfg, video)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line, with a --config file supplying the defaults.

    Precedence is code defaults < config file < command line. Loading the file
    into argparse defaults, rather than merging afterwards, is what lets an
    explicitly-typed flag win without having to detect which flags were typed.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)

    parser = build_parser()
    if known.config and Path(known.config).exists():
        parser.set_defaults(**load_config_file(known.config))

    return parser.parse_args(argv)


def run_job(video: Path, script: Path | None, args: argparse.Namespace,
            cfg: EditorConfig, out: Path | None,
            report_stem: str | None) -> None:
    """One recording, end to end. Shared by the single and batch paths."""
    if args.from_report:
        report = load_report(args.from_report)
    else:
        script_text = script.read_text(encoding="utf-8")

        print(f"  Transcribing {video.name} (model={cfg.model})...",
              file=sys.stderr)
        words, duration = transcribe(video, cfg)
        print(f"  {len(words)} words, {duration:.1f}s", file=sys.stderr)

        report = analyse(words, script_text, duration, cfg, str(video),
                         split_sentences=args.split_sentences)

        json_path, md_path, _ = report_paths(video, report_stem)
        write_json(report, json_path)
        write_markdown(report, md_path)

        s = report["summary"]
        print(f"  Report: {md_path}", file=sys.stderr)
        print(f"  {s['cut_count']} cuts, {s['cut_seconds']:.1f}s removed "
              f"-> {s['output_seconds']:.1f}s output", file=sys.stderr)
        print(f"  {s['misses']} missed lines, {s['reviews']} needing review",
              file=sys.stderr)

    if args.export_xml:
        _, _, xml_path = report_paths(video, report_stem)
        segments = [Span(s["start"], s["end"]) for s in report["segments"]]
        write_xml(video, segments, probe_video_meta(video), xml_path)
        print(f"  Timeline: {xml_path}", file=sys.stderr)

    if not (args.auto or args.from_report):
        return

    from render import render_video
    render_video(video, out, report, cfg)
    print(f"  Rendered: {out}", file=sys.stderr)


def run_batch(args: argparse.Namespace, cfg: EditorConfig) -> int:
    """Every recording in a directory, paired with its same-stem script."""
    jobs, unpaired = find_jobs(args.batch)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for orphan in unpaired:
        print(f"skipping {orphan.name}: no {orphan.stem}.txt beside it",
              file=sys.stderr)

    if not jobs:
        print(f"error: no script/recording pairs found in {args.batch}",
              file=sys.stderr)
        return 2

    print(f"\n{len(jobs)} recording(s) to process\n", file=sys.stderr)
    failures: list[tuple[str, str]] = []

    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {job.video.name}", file=sys.stderr)
        out = output_path(job, out_dir, ".cut.mp4")
        stem = str(output_path(job, out_dir, ".report")) if out_dir else None
        try:
            # one bad recording must not abandon the rest of the batch
            run_job(job.video, job.script, args, cfg, out, stem)
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            failures.append((job.video.name, str(exc)))
        print("", file=sys.stderr)

    done = len(jobs) - len(failures)
    print(f"Batch complete: {done}/{len(jobs)} succeeded", file=sys.stderr)
    for name, reason in failures:
        print(f"  {name}: {reason}", file=sys.stderr)

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    problem = validate_args(args)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2

    missing = missing_inputs(args)
    if missing:
        print(f"error: {missing}", file=sys.stderr)
        return 2

    cfg = config_from_args(args)

    if args.batch:
        return run_batch(args, cfg)

    run_job(Path(args.video),
            Path(args.script) if args.script else None,
            args, cfg,
            Path(args.out) if args.out else None,
            args.report_path)

    if not (args.auto or args.from_report):
        print("\nReview the report, then re-run with --auto, or render the "
              "approved report with --from-report.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
