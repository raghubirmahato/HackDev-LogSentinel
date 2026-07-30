#!/usr/bin/env python3
"""
LogSentinel - ML-based web server log anomaly detector using Isolation Forest.

Part of the HackDev cybersecurity toolkit.

Parses Apache/Nginx "combined" format access logs, extracts numeric features
from each request, and uses an unsupervised Isolation Forest model to flag
anomalous requests (e.g. SQL injection attempts, path traversal, unusually
large payloads, malformed/rare requests) relative to a baseline of "normal"
traffic.

Subcommands:
    fit   <baseline_log_file>   Train a model on a baseline (known-good) log file.
    scan  <target_log_file>     Score a target log file against a trained model.

This is a defensive (blue-team) tool intended for analyzing logs you own or
are authorized to access.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, parse_qsl

# ---------------------------------------------------------------------------
# Optional sklearn / joblib / numpy imports.
#
# The parser and feature extractor work without these libraries so that
# `--help`, log parsing, and JSON structure can always be exercised. The
# `fit` and `scan` subcommands require sklearn/joblib/numpy at runtime and
# will fail with a clear, non-zero-exit error if they are unavailable.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib

    _SKLEARN_AVAILABLE = True
    _SKLEARN_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only w/o sklearn installed
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = exc

LOG = logging.getLogger("logsentinel")

# ---------------------------------------------------------------------------
# Combined Log Format parser
# ---------------------------------------------------------------------------
#
# Example line:
# 127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html HTTP/1.1" 200 2326 "-" "Mozilla/5.0"
COMBINED_LOG_PATTERN = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/(?P<http_version>[\d.]+)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)\s+'
    r'"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"\s*$'
)

METHOD_ENCODING = {
    "GET": 0,
    "POST": 1,
    "PUT": 2,
    "DELETE": 3,
    "HEAD": 4,
    "OPTIONS": 5,
}


@dataclass
class LogEntry:
    """A single parsed access-log line."""

    line_number: int
    ip: str
    timestamp: str
    method: str
    path: str
    http_version: str
    status: int
    size: int
    referer: str
    user_agent: str
    raw_line: str


@dataclass
class RequestFeatures:
    """Numeric features extracted from a single request, used for ML input."""

    url_length: int
    param_count: int
    special_char_ratio: float
    entropy: float
    payload_length: int
    method_encoded: int

    def to_vector(self) -> list[float]:
        return [
            float(self.url_length),
            float(self.param_count),
            float(self.special_char_ratio),
            float(self.entropy),
            float(self.payload_length),
            float(self.method_encoded),
        ]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ParseResult:
    """Outcome of parsing a full log file."""

    entries: list[LogEntry] = field(default_factory=list)
    total_lines: int = 0
    unparsed_lines: int = 0
    unparsed_samples: list[str] = field(default_factory=list)


def shannon_entropy(s: str) -> float:
    """Compute the Shannon entropy (bits/char) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _parse_size(raw_size: str) -> int:
    """Parse the response size field, which is '-' for zero/absent bodies."""
    if raw_size == "-":
        return 0
    try:
        return int(raw_size)
    except ValueError:
        return 0


def parse_log_line(line: str, line_number: int) -> LogEntry | None:
    """Parse a single combined-format log line. Returns None if it doesn't match."""
    match = COMBINED_LOG_PATTERN.match(line.strip())
    if match is None:
        return None

    groups = match.groupdict()
    return LogEntry(
        line_number=line_number,
        ip=groups["ip"],
        timestamp=groups["timestamp"],
        method=groups["method"],
        path=groups["path"],
        http_version=groups["http_version"],
        status=int(groups["status"]),
        size=_parse_size(groups["size"]),
        referer=groups["referer"],
        user_agent=groups["user_agent"],
        raw_line=line.rstrip("\n"),
    )


def parse_log_file(file_path: Path) -> ParseResult:
    """Parse an entire log file, skipping and counting unparsable lines."""
    result = ParseResult()

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                result.total_lines += 1
                entry = parse_log_line(raw_line, idx)
                if entry is None:
                    result.unparsed_lines += 1
                    if len(result.unparsed_samples) < 5:
                        result.unparsed_samples.append(stripped[:200])
                    continue
                result.entries.append(entry)
    except OSError as exc:
        LOG.error("Failed to read log file %s: %s", file_path, exc)
        raise

    if result.unparsed_lines:
        LOG.warning(
            "%d of %d lines could not be parsed and were skipped.",
            result.unparsed_lines,
            result.total_lines,
        )
        for sample in result.unparsed_samples:
            LOG.warning("  unparsed sample: %s", sample)

    return result


def extract_features(entry: LogEntry) -> RequestFeatures:
    """Extract numeric ML features from a single parsed log entry."""
    split = urlsplit(entry.path)
    full_path_and_query = entry.path
    query_string = split.query

    url_length = len(entry.path)

    try:
        params = parse_qsl(query_string, keep_blank_values=True)
    except ValueError:
        params = []
    param_count = len(params)

    if full_path_and_query:
        special_chars = sum(1 for ch in full_path_and_query if not ch.isalnum())
        special_char_ratio = special_chars / len(full_path_and_query)
    else:
        special_char_ratio = 0.0

    entropy = shannon_entropy(query_string)
    payload_length = entry.size
    method_encoded = METHOD_ENCODING.get(entry.method.upper(), 6)

    return RequestFeatures(
        url_length=url_length,
        param_count=param_count,
        special_char_ratio=round(special_char_ratio, 6),
        entropy=round(entropy, 6),
        payload_length=payload_length,
        method_encoded=method_encoded,
    )


# ---------------------------------------------------------------------------
# Model persistence bundle
# ---------------------------------------------------------------------------

FEATURE_ORDER = (
    "url_length",
    "param_count",
    "special_char_ratio",
    "entropy",
    "payload_length",
    "method_encoded",
)


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        LOG.error(
            "scikit-learn (and/or joblib/numpy) is not installed in this environment. "
            "Install dependencies with `pip install -r requirements.txt` to use "
            "the fit/scan subcommands. Original import error: %s",
            _SKLEARN_IMPORT_ERROR,
        )
        raise SystemExit(1)


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit an IsolationForest + StandardScaler on a baseline log file."""
    _require_sklearn()

    baseline_path = Path(args.baseline_log_file)
    if not baseline_path.is_file():
        LOG.error("Baseline log file not found: %s", baseline_path)
        return 1

    LOG.info("Parsing baseline log file: %s", baseline_path)
    parse_result = parse_log_file(baseline_path)

    if not parse_result.entries:
        LOG.error("No parsable log entries found in baseline file. Aborting.")
        return 1

    LOG.info(
        "Parsed %d entries (%d total lines, %d unparsed).",
        len(parse_result.entries),
        parse_result.total_lines,
        parse_result.unparsed_lines,
    )

    features = [extract_features(entry) for entry in parse_result.entries]
    matrix = np.array([f.to_vector() for f in features])

    LOG.info(
        "Fitting IsolationForest (contamination=%.3f) on %d samples with %d features.",
        args.contamination,
        matrix.shape[0],
        matrix.shape[1],
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    model = IsolationForest(contamination=args.contamination, random_state=42)
    model.fit(scaled)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_order": FEATURE_ORDER}, output_path)

    LOG.info("Saved fitted model and scaler to %s", output_path)

    summary = {
        "status": "fitted",
        "baseline_log_file": str(baseline_path),
        "model_output": str(output_path),
        "contamination": args.contamination,
        "total_lines": parse_result.total_lines,
        "parsed_entries": len(parse_result.entries),
        "unparsed_lines": parse_result.unparsed_lines,
    }

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print("LogSentinel fit summary")
        print(f"  Baseline file:    {summary['baseline_log_file']}")
        print(f"  Model saved to:   {summary['model_output']}")
        print(f"  Contamination:    {summary['contamination']}")
        print(f"  Total lines:      {summary['total_lines']}")
        print(f"  Parsed entries:   {summary['parsed_entries']}")
        print(f"  Unparsed lines:   {summary['unparsed_lines']}")

    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan a target log file for anomalies using a previously fitted model."""
    _require_sklearn()

    model_path = Path(args.model)
    if not model_path.is_file():
        LOG.error("Model file not found: %s", model_path)
        return 1

    target_path = Path(args.target_log_file)
    if not target_path.is_file():
        LOG.error("Target log file not found: %s", target_path)
        return 1

    LOG.info("Loading model from %s", model_path)
    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        scaler = bundle["scaler"]
    except (KeyError, OSError, ValueError) as exc:
        LOG.error("Failed to load model bundle from %s: %s", model_path, exc)
        return 1

    LOG.info("Parsing target log file: %s", target_path)
    parse_result = parse_log_file(target_path)

    if not parse_result.entries:
        LOG.error("No parsable log entries found in target file. Aborting.")
        return 1

    LOG.info(
        "Parsed %d entries (%d total lines, %d unparsed).",
        len(parse_result.entries),
        parse_result.total_lines,
        parse_result.unparsed_lines,
    )

    features = [extract_features(entry) for entry in parse_result.entries]
    matrix = np.array([f.to_vector() for f in features])
    scaled = scaler.transform(matrix)

    predictions = model.predict(scaled)
    scores = model.decision_function(scaled)

    anomalies: list[dict[str, Any]] = []
    for entry, feat, prediction, score in zip(
        parse_result.entries, features, predictions, scores
    ):
        if prediction == -1:
            anomalies.append(
                {
                    "line_number": entry.line_number,
                    "ip": entry.ip,
                    "path": entry.path,
                    "status": entry.status,
                    "anomaly_score": round(float(-score), 6),
                    "features": feat.to_dict(),
                    "raw_line": entry.raw_line,
                }
            )

    anomalies.sort(key=lambda a: a["anomaly_score"], reverse=True)

    stats = {
        "total_lines": parse_result.total_lines,
        "parsed": len(parse_result.entries),
        "unparsed": parse_result.unparsed_lines,
        "anomalies_found": len(anomalies),
    }

    if args.format == "json":
        output = {"stats": stats, "anomalies": anomalies}
        rendered = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            LOG.info("JSON results written to %s", args.output)
        print(rendered)
    else:
        lines = ["LogSentinel scan results", ""]
        lines.append(
            f"Total lines: {stats['total_lines']}  Parsed: {stats['parsed']}  "
            f"Unparsed: {stats['unparsed']}  Anomalies: {stats['anomalies_found']}"
        )
        lines.append("")
        for anomaly in anomalies:
            lines.append(
                f"[line {anomaly['line_number']}] score={anomaly['anomaly_score']:.4f} "
                f"ip={anomaly['ip']} status={anomaly['status']} path={anomaly['path']}"
            )
            lines.append(f"    raw: {anomaly['raw_line']}")
        rendered = "\n".join(lines)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            LOG.info("Text results written to %s", args.output)
        print(rendered)

    return 0


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logsentinel",
        description=(
            "ML-based web server log anomaly detector using Isolation Forest. "
            "Parses Apache/Nginx combined-format access logs and flags "
            "anomalous requests relative to a trained baseline."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser(
        "fit", help="Fit an anomaly-detection model on a baseline (known-good) log file."
    )
    fit_parser.add_argument(
        "baseline_log_file", help="Path to a baseline access log file in combined format."
    )
    fit_parser.add_argument(
        "-o",
        "--output",
        default="logsentinel_model.joblib",
        help="Path to save the fitted model+scaler bundle (default: logsentinel_model.joblib).",
    )
    fit_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the fit summary (default: text).",
    )
    fit_parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected proportion of anomalies in the baseline data (default: 0.05).",
    )
    fit_parser.set_defaults(func=cmd_fit)

    scan_parser = subparsers.add_parser(
        "scan", help="Scan a target log file for anomalies using a fitted model."
    )
    scan_parser.add_argument(
        "target_log_file", help="Path to the target access log file to scan."
    )
    scan_parser.add_argument(
        "--model",
        required=True,
        help="Path to a model+scaler bundle previously saved by the `fit` subcommand.",
    )
    scan_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to write scan results to, in addition to stdout.",
    )
    scan_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for scan results (default: text).",
    )
    scan_parser.set_defaults(func=cmd_scan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        return args.func(args)
    except FileNotFoundError as exc:
        LOG.error("File not found: %s", exc)
        return 1
    except SystemExit:
        raise
    except (OSError, ValueError, KeyError) as exc:
        LOG.error("Unexpected error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
