"""Command-line interface for HackDev-LogSentinel."""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import sys
import zlib
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from . import __version__
from .alerts import build_webhook_payload, post_webhook
from .features import extract_features
from .model import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONTAMINATION,
    DEFAULT_MAX_FIT_ROWS,
    DEFAULT_RANGE_FACTOR,
    EmptyBaselineError,
    ModelError,
    fit_incremental,
    fit_model,
    load_bundle,
    save_bundle,
    scan_stream,
)
from .parsers import JSON_FIELD_NAMES, SUPPORTED_FORMATS, ParserStats, stream_log_file
from .sanitize import printable

LOG = logging.getLogger("logsentinel")

WEBHOOK_URL_ENV = "LOGSENTINEL_WEBHOOK_URL"


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# argparse `type=` validators: a bad value is rejected as a usage error up
# front, instead of surfacing only after a (possibly huge) log has been read.


def _contamination(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {raw!r}") from None
    if not 0 < value <= 0.5:
        raise argparse.ArgumentTypeError(f"must be in the range (0, 0.5], got {raw}")
    return value


def _range_factor(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {raw!r}") from None
    if not (value == 0 or 1 < value < float("inf")):
        raise argparse.ArgumentTypeError(f"must be greater than 1, or 0 to turn the range check off; got {raw}")
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {raw}")
    return value


def _field_map(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from None
    if not isinstance(value, dict) or not all(isinstance(json_key, str) and json_key for json_key in value.values()):
        raise argparse.ArgumentTypeError(
            'must be a JSON object mapping field names to JSON keys, e.g. {"ip": "clientAddr"}'
        )
    unknown = sorted(set(value) - set(JSON_FIELD_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown field(s): {', '.join(unknown)} (valid fields: {', '.join(JSON_FIELD_NAMES)})"
        )
    return value


def _webhook_url(raw: str) -> str:
    # The URL is a credential: never echo it back in the error message.
    try:
        parts = urlsplit(raw)
        valid = parts.scheme in ("http", "https") and bool(parts.hostname)
    except ValueError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError(f"must be an http:// or https:// URL (checked for ${WEBHOOK_URL_ENV} too)")
    return raw


def _log_nothing_parsed(stats: ParserStats, path: Path) -> None:
    LOG.error(
        "None of the %d lines in %s could be parsed as %s format, so nothing was analyzed. "
        "Check --format-override (and --field-map for JSON logs).",
        stats.total_lines, path, stats.detected_format,
    )


def _warn_unused_field_map(args: argparse.Namespace, stats: ParserStats) -> None:
    if args.field_map and stats.detected_format != "json":
        LOG.warning("--field-map only applies to JSON logs; ignored for %s format.", stats.detected_format)


def cmd_fit(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline_log_file)
    if not baseline_path.is_file():
        LOG.error("Baseline log file not found: %s", baseline_path)
        return 1

    existing = None
    if args.update:
        model_path = Path(args.update)
        if not model_path.is_file():
            LOG.error("--update model file not found: %s", model_path)
            return 1
        existing = load_bundle(model_path)

    stats = ParserStats()
    entries = stream_log_file(baseline_path, stats, format_override=args.format_override, field_map=args.field_map)
    features = (extract_features(e) for e in entries)

    try:
        if existing is not None:
            bundle = fit_incremental(existing, features, contamination=args.contamination,
                                     max_fit_rows=args.max_fit_rows, range_factor=args.range_factor)
        else:
            contamination = DEFAULT_CONTAMINATION if args.contamination is None else args.contamination
            range_factor = DEFAULT_RANGE_FACTOR if args.range_factor is None else args.range_factor
            bundle = fit_model(features, contamination=contamination, max_fit_rows=args.max_fit_rows,
                               range_factor=range_factor)
    except EmptyBaselineError:
        if stats.total_lines:
            _log_nothing_parsed(stats, baseline_path)
        else:
            LOG.error("Baseline log file contains no log lines: %s", baseline_path)
        return 1
    _warn_unused_field_map(args, stats)

    output_path = Path(args.output)
    save_bundle(bundle, output_path)

    summary = {
        "status": "updated" if existing is not None else "fitted",
        "baseline_log_file": str(baseline_path),
        "detected_format": stats.detected_format,
        "model_output": str(output_path),
        "contamination": bundle.contamination,
        "total_lines": stats.total_lines,
        "parsed_lines": stats.parsed_lines,
        "unparsed_lines": stats.unparsed_lines,
        "total_baseline_rows": bundle.n_samples_seen,
        "cached_sample_size": len(bundle.cached_sample),
        "range_factor": bundle.range_factor,
        "range_bounds": bundle.range_bounds,
    }

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"  {key}: {value}")

    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.is_file():
        LOG.error("Model file not found: %s", model_path)
        return 1

    target_path = Path(args.target_log_file)
    if not target_path.is_file():
        LOG.error("Target log file not found: %s", target_path)
        return 1

    bundle = load_bundle(model_path)

    stats = ParserStats()
    entries = stream_log_file(target_path, stats, format_override=args.format_override, field_map=args.field_map)
    pairs = ((entry, extract_features(entry)) for entry in entries)

    found = 0

    def counted(flagged):
        nonlocal found
        for anomaly in flagged:
            found += 1
            yield anomaly

    range_factor = bundle.range_factor if args.range_factor is None else args.range_factor
    flagged = counted(scan_stream(bundle, pairs, chunk_size=args.chunk_size, range_factor=range_factor))

    def by_score(anomaly: dict) -> float:
        return anomaly["anomaly_score"]

    # Highest score first, ties in log order. With --top, only N anomalies are
    # ever held in memory, however many a huge log produces.
    if args.top is None:
        anomalies = sorted(flagged, key=by_score, reverse=True)
    else:
        anomalies = heapq.nlargest(args.top, flagged, key=by_score)

    if stats.total_lines and not stats.parsed_lines:
        # Reporting "0 anomalies" here would be a false all-clear.
        _log_nothing_parsed(stats, target_path)
        return 1
    _warn_unused_field_map(args, stats)

    result_stats = {
        "total_lines": stats.total_lines,
        "parsed": stats.parsed_lines,
        "unparsed": stats.unparsed_lines,
        "detected_format": stats.detected_format,
        "anomalies_found": found,
        "anomalies_reported": len(anomalies),
        "range_factor": range_factor,
    }

    if args.webhook_url:
        payload = build_webhook_payload(str(target_path), result_stats, anomalies)
        post_webhook(args.webhook_url, payload)

    if args.format == "json":
        rendered = json.dumps({"stats": result_stats, "anomalies": anomalies}, indent=2)
    else:
        lines = [f"LogSentinel scan results (format: {result_stats['detected_format']})", ""]
        lines.append(
            f"Total: {result_stats['total_lines']}  Parsed: {result_stats['parsed']}  "
            f"Unparsed: {result_stats['unparsed']}  Anomalies: {result_stats['anomalies_found']}"
        )
        if len(anomalies) < found:
            lines.append(f"(showing the top {len(anomalies)} by score)")
        lines.append("")
        # Request data is attacker-controlled: escape control characters so a
        # logged request can't smuggle ANSI escape sequences into the terminal.
        for a in anomalies:
            line = (f"[line {a['line_number']}] score={a['anomaly_score']:.4f} ip={printable(a['ip'])} "
                    f"status={a['status']} path={printable(a['path'])}")
            range_reasons = [reason for reason in a["reasons"] if reason != "isolation forest"]
            if range_reasons:
                line += "  <- " + "; ".join(range_reasons)
            lines.append(line)
        rendered = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        LOG.info("Results written to %s", args.output)
    print(rendered)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logsentinel",
        description="ML-based web server log anomaly detector using Isolation Forest.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Options shared by both subcommands. -v is accepted after the subcommand
    # too; its SUPPRESS default keeps it from clobbering a -v given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="Enable verbose (DEBUG) logging.")
    common.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text).")
    common.add_argument("--format-override", choices=SUPPORTED_FORMATS, default=None,
                        help="Force a log format instead of auto-detecting.")
    common.add_argument("--field-map", type=_field_map, default=None, metavar="JSON",
                        help="JSON object mapping field names to keys in a JSON log; dotted keys reach into "
                             f"nested objects. Fields: {', '.join(JSON_FIELD_NAMES)}.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", parents=[common],
                                       help="Fit (or incrementally update) a model on a baseline log file.")
    fit_parser.add_argument("baseline_log_file", help="Log of normal traffic to learn from (plain or gzip).")
    fit_parser.add_argument("-o", "--output", default="logsentinel_model.joblib",
                            help="Where to save the model (default: %(default)s).")
    fit_parser.add_argument("--contamination", type=_contamination, default=None,
                            help=f"Expected proportion of anomalies in the baseline, in (0, 0.5] "
                                 f"(default: {DEFAULT_CONTAMINATION}, or the existing model's value with --update).")
    fit_parser.add_argument("--update", metavar="EXISTING_MODEL", default=None,
                            help="Incrementally update an existing model instead of fitting from scratch.")
    fit_parser.add_argument("--max-fit-rows", type=_positive_int, default=DEFAULT_MAX_FIT_ROWS, metavar="N",
                            help="Train on a uniform random sample of at most N baseline rows; bounds memory "
                                 "on huge baselines (default: %(default)s).")
    fit_parser.add_argument("--range-factor", type=_range_factor, default=None, metavar="X",
                            help="Also flag requests whose URL length, parameter count or response size is more "
                                 "than X times the baseline's maximum (its 99.9th percentile) - the forest alone "
                                 "can't tell how far beyond the baseline a value is. Stored in the model "
                                 f"(default: {DEFAULT_RANGE_FACTOR:g}, or the existing model's value with --update; "
                                 "0 turns the check off).")
    fit_parser.set_defaults(func=cmd_fit)

    scan_parser = subparsers.add_parser("scan", parents=[common], help="Scan a target log file for anomalies.")
    scan_parser.add_argument("target_log_file", help="Log to scan (plain or gzip).")
    scan_parser.add_argument("--model", required=True, help="Model file saved by `fit` (only load trusted files).")
    scan_parser.add_argument("-o", "--output", default=None, help="Also write the results to this file.")
    scan_parser.add_argument("--chunk-size", type=_positive_int, default=DEFAULT_CHUNK_SIZE, metavar="N",
                             help="Rows scored per batch; bounds memory on huge logs (default: %(default)s).")
    scan_parser.add_argument("--top", type=_positive_int, default=None, metavar="N",
                             help="Report only the N highest-scoring anomalies (all are still counted).")
    scan_parser.add_argument("--range-factor", type=_range_factor, default=None, metavar="X",
                             help="Override the model's range factor for this scan (0 turns the check off).")
    scan_parser.add_argument("--webhook-url", type=_webhook_url, default=os.environ.get(WEBHOOK_URL_ENV) or None,
                             metavar="URL",
                             help="POST a Slack-compatible summary here on completion. Prefer setting "
                                  f"${WEBHOOK_URL_ENV}: command-line arguments are visible to other local users.")
    scan_parser.set_defaults(func=cmd_scan)

    return parser


def _discard_stdout() -> None:
    # Python flushes stdout at exit; point the real stdout at /dev/null so that
    # final flush doesn't raise a second BrokenPipeError. Leave file descriptors
    # alone when stdout was redirected in-process (e.g. by an embedding program).
    if sys.stdout is not sys.__stdout__:
        return
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)
    except (OSError, ValueError):
        pass


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        LOG.error("Interrupted by user.")
        return 130
    except BrokenPipeError:  # output piped into e.g. `head`, which exited early
        _discard_stdout()
        return 1
    except (ModelError, OSError, RuntimeError, ValueError, zlib.error) as exc:
        LOG.error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
