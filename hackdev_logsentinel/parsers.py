"""
Multi-format, streaming log parsers.

Supports Apache/Nginx "combined" format, JSON-lines (one JSON object per
line, with a configurable field-name mapping), and generic W3C extended
log format (a "#Fields:" header line defines the columns).

Every parser is a generator - lines are processed and yielded one at a
time, so even multi-gigabyte log files can be handled without loading the
whole file into memory. A mutable `ParserStats` object, passed in by the
caller, is updated as a side effect so streaming and stats-reporting can
coexist without materializing a full entry list.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("logsentinel.parsers")

# ---------------------------------------------------------------------------
# Shared data model
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
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
class ParserStats:
    total_lines: int = 0
    unparsed_lines: int = 0
    unparsed_samples: list[str] = field(default_factory=list)
    detected_format: Optional[str] = None

    def note_unparsed(self, raw: str) -> None:
        self.unparsed_lines += 1
        if len(self.unparsed_samples) < 5:
            self.unparsed_samples.append(raw[:200])


# ---------------------------------------------------------------------------
# Combined Log Format
# ---------------------------------------------------------------------------

COMBINED_LOG_PATTERN = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/(?P<http_version>[\d.]+)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)\s+'
    r'"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"\s*$'
)


def _parse_size(raw_size: str) -> int:
    if raw_size == "-" or not raw_size:
        return 0
    try:
        return int(raw_size)
    except ValueError:
        return 0


def parse_combined_line(line: str, line_number: int) -> Optional[LogEntry]:
    match = COMBINED_LOG_PATTERN.match(line.strip())
    if match is None:
        return None
    g = match.groupdict()
    return LogEntry(
        line_number=line_number, ip=g["ip"], timestamp=g["timestamp"], method=g["method"],
        path=g["path"], http_version=g["http_version"], status=int(g["status"]),
        size=_parse_size(g["size"]), referer=g["referer"], user_agent=g["user_agent"],
        raw_line=line.rstrip("\n"),
    )


# ---------------------------------------------------------------------------
# JSON-lines
# ---------------------------------------------------------------------------

DEFAULT_JSON_FIELD_MAP: dict[str, str] = {
    "ip": "ip", "timestamp": "timestamp", "method": "method", "path": "path",
    "status": "status", "size": "size", "referer": "referer", "user_agent": "user_agent",
}

_JSON_FIELD_ALIASES: dict[str, list[str]] = {
    "ip": ["ip", "remote_addr", "client_ip", "src_ip"],
    "timestamp": ["timestamp", "time", "@timestamp", "date"],
    "method": ["method", "http_method", "verb"],
    "path": ["path", "url", "uri", "request_uri", "request_path"],
    "status": ["status", "status_code", "response_code", "response_status"],
    "size": ["size", "bytes", "response_size", "content_length"],
    "referer": ["referer", "referrer"],
    "user_agent": ["user_agent", "useragent", "ua"],
}


def _resolve_json_field(obj: dict, field_map: dict[str, str], canonical: str) -> object:
    mapped_key = field_map.get(canonical)
    if mapped_key and mapped_key in obj:
        return obj[mapped_key]
    for alias in _JSON_FIELD_ALIASES.get(canonical, []):
        if alias in obj:
            return obj[alias]
    return None


def parse_json_line(line: str, line_number: int, field_map: Optional[dict[str, str]] = None) -> Optional[LogEntry]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    field_map = field_map or {}

    status_raw = _resolve_json_field(obj, field_map, "status")
    try:
        status = int(status_raw) if status_raw is not None else 0
    except (TypeError, ValueError):
        status = 0

    size_raw = _resolve_json_field(obj, field_map, "size")
    try:
        size = int(size_raw) if size_raw is not None else 0
    except (TypeError, ValueError):
        size = 0

    return LogEntry(
        line_number=line_number,
        ip=str(_resolve_json_field(obj, field_map, "ip") or ""),
        timestamp=str(_resolve_json_field(obj, field_map, "timestamp") or ""),
        method=str(_resolve_json_field(obj, field_map, "method") or "GET").upper(),
        path=str(_resolve_json_field(obj, field_map, "path") or "/"),
        http_version="1.1",
        status=status,
        size=size,
        referer=str(_resolve_json_field(obj, field_map, "referer") or ""),
        user_agent=str(_resolve_json_field(obj, field_map, "user_agent") or ""),
        raw_line=stripped,
    )


# ---------------------------------------------------------------------------
# W3C extended log format
# ---------------------------------------------------------------------------

_W3C_FIELD_ALIASES: dict[str, list[str]] = {
    "ip": ["c-ip", "client-ip"],
    "method": ["cs-method"],
    "path": ["cs-uri-stem"],
    "query": ["cs-uri-query"],
    "status": ["sc-status"],
    "size": ["sc-bytes"],
    "user_agent": ["cs(User-Agent)"],
    "referer": ["cs(Referer)"],
    "date": ["date"],
    "time": ["time"],
}


def parse_w3c_fields_header(line: str) -> Optional[list[str]]:
    stripped = line.strip()
    if not stripped.startswith("#Fields:"):
        return None
    return stripped[len("#Fields:"):].strip().split()


def _w3c_index(fields: list[str], canonical: str) -> Optional[int]:
    for alias in _W3C_FIELD_ALIASES.get(canonical, []):
        if alias in fields:
            return fields.index(alias)
    return None


def parse_w3c_line(line: str, line_number: int, fields: list[str]) -> Optional[LogEntry]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    values = stripped.split()
    if len(values) != len(fields):
        return None

    def get(canonical: str, default: str = "") -> str:
        idx = _w3c_index(fields, canonical)
        return values[idx] if idx is not None and idx < len(values) else default

    date_part = get("date")
    time_part = get("time")
    timestamp = f"{date_part} {time_part}".strip()

    path = get("path", "/")
    query = get("query", "")
    if query and query != "-":
        path = f"{path}?{query}"

    try:
        status = int(get("status", "0") or "0")
    except ValueError:
        status = 0
    size_raw = get("size", "0")
    try:
        size = int(size_raw) if size_raw and size_raw != "-" else 0
    except ValueError:
        size = 0

    return LogEntry(
        line_number=line_number,
        ip=get("ip", "-"),
        timestamp=timestamp,
        method=get("method", "GET").upper(),
        path=path,
        http_version="1.1",
        status=status,
        size=size,
        referer=get("referer", "-").replace("+", " "),
        user_agent=get("user_agent", "-").replace("+", " "),
        raw_line=stripped,
    )


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> str:
    """Peek at the first few non-empty lines to guess the log format.

    Returns "json", "w3c", or "combined".
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(20):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#Fields:"):
                    return "w3c"
                if stripped.startswith("#"):
                    continue
                if stripped.startswith("{"):
                    try:
                        json.loads(stripped)
                        return "json"
                    except json.JSONDecodeError:
                        pass
                return "combined"
    except OSError:
        pass
    return "combined"


# ---------------------------------------------------------------------------
# Unified streaming entrypoint
# ---------------------------------------------------------------------------


def stream_log_file(
    path: Path,
    stats: ParserStats,
    format_override: Optional[str] = None,
    field_map: Optional[dict[str, str]] = None,
) -> Iterator[LogEntry]:
    """Stream-parse a log file line by line, yielding LogEntry objects and
    updating `stats` in-place as a side effect (so the caller gets both
    streaming behavior and aggregate counts without a second pass)."""
    fmt = format_override or detect_format(path)
    stats.detected_format = fmt

    w3c_fields: Optional[list[str]] = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            if fmt == "w3c":
                header_fields = parse_w3c_fields_header(raw_line)
                if header_fields is not None:
                    w3c_fields = header_fields
                    continue
                if stripped.startswith("#"):
                    continue
                if w3c_fields is None:
                    stats.total_lines += 1
                    stats.note_unparsed(stripped)
                    continue
                stats.total_lines += 1
                entry = parse_w3c_line(raw_line, idx, w3c_fields)
            elif fmt == "json":
                stats.total_lines += 1
                entry = parse_json_line(raw_line, idx, field_map)
            else:
                stats.total_lines += 1
                entry = parse_combined_line(raw_line, idx)

            if entry is None:
                stats.note_unparsed(stripped)
                continue
            yield entry

    if stats.unparsed_lines:
        LOG.warning(
            "%d of %d lines could not be parsed and were skipped (format=%s).",
            stats.unparsed_lines, stats.total_lines, fmt,
        )
