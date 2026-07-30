"""Command-line interface for HackDev-LogSentinel."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .alerts import build_webhook_payload, post_webhook
from .features import extract_features
from .model import fit_incremental, fit_model, load_bundle, save_bundle, scan_stream
from .parsers import ParserStats, stream_log_file

LOG = logging.getLogger("logsentinel")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_field_map(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --field-map JSON: {exc}")


def cmd_fit(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline_log_file)
    if not baseline_path.is_file():
        LOG.error("Baseline log file not found: %s", baseline_path)
        return 1

    field_map = _parse_field_map(args.field_map)
    stats = ParserStats()
    entries = stream_log_file(baseline_path, stats, format_override=args.format_override, field_map=field_map)
    features = (extract_features(e) for e in entries)

    try:
        if args.update:
            model_path = Path(args.update)
            if not model_path.is_file():
                LOG.error("--update model file not found: %s", model_path)
                return 1
            existing = load_bundle(model_path)
            bundle = fit_incremental(existing, features, contamination=args.contamination)
        else:
            bundle = fit_model(features, contamination=args.contamination)
    except (ValueError, RuntimeError) as exc:
        LOG.error(str(exc))
        return 1

    output_path = Path(args.output)
    save_bundle(bundle, output_path)

    summary = {
        "status": "updated" if args.update else "fitted",
        "baseline_log_file": str(baseline_path),
        "detected_format": stats.detected_format,
        "model_output": str(output_path),
        "contamination": args.contamination,
        "total_lines": stats.total_lines,
        "unparsed_lines": stats.unparsed_lines,
        "cached_sample_size": len(bundle.cached_sample),
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

    try:
        bundle = load_bundle(model_path)
    except RuntimeError as exc:
        LOG.error(str(exc))
        return 1

    field_map = _parse_field_map(args.field_map)
    stats = ParserStats()
    entries = stream_log_file(target_path, stats, format_override=args.format_override, field_map=field_map)

    def entries_and_features():
        for entry in entries:
            yield entry, extract_features(entry)

    try:
        anomalies = list(scan_stream(bundle, entries_and_features(), chunk_size=args.chunk_size))
    except RuntimeError as exc:
        LOG.error(str(exc))
        return 1

    anomalies.sort(key=lambda a: a["anomaly_score"], reverse=True)

    result_stats = {
        "total_lines": stats.total_lines,
        "parsed": stats.total_lines - stats.unparsed_lines,
        "unparsed": stats.unparsed_lines,
        "detected_format": stats.detected_format,
        "anomalies_found": len(anomalies),
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
        lines.append("")
        for a in anomalies:
            lines.append(f"[line {a['line_number']}] score={a['anomaly_score']:.4f} ip={a['ip']} status={a['status']} path={a['path']}")
        rendered = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit (or incrementally update) a model on a baseline log file.")
    fit_parser.add_argument("baseline_log_file")
    fit_parser.add_argument("-o", "--output", default="logsentinel_model.joblib")
    fit_parser.add_argument("--format", choices=["text", "json"], default="text")
    fit_parser.add_argument("--format-override", choices=["combined", "json", "w3c"], default=None, help="Force a log format instead of auto-detecting.")
    fit_parser.add_argument("--field-map", default=None, help="JSON field-name mapping for the json log format.")
    fit_parser.add_argument("--contamination", type=float, default=0.05)
    fit_parser.add_argument("--update", metavar="EXISTING_MODEL", default=None, help="Incrementally update an existing model instead of fitting from scratch.")
    fit_parser.set_defaults(func=cmd_fit)

    scan_parser = subparsers.add_parser("scan", help="Scan a target log file for anomalies.")
    scan_parser.add_argument("target_log_file")
    scan_parser.add_argument("--model", required=True)
    scan_parser.add_argument("-o", "--output", default=None)
    scan_parser.add_argument("--format", choices=["text", "json"], default="text")
    scan_parser.add_argument("--format-override", choices=["combined", "json", "w3c"], default=None)
    scan_parser.add_argument("--field-map", default=None)
    scan_parser.add_argument("--chunk-size", type=int, default=2000, help="Rows scored per batch (bounds memory on huge logs).")
    scan_parser.add_argument("--webhook-url", default=None, help="POST a Slack-compatible summary here on completion.")
    scan_parser.set_defaults(func=cmd_scan)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        LOG.error("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
