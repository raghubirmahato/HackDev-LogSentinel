"""
Multi-format, streaming log parsers.

Supports Apache/Nginx "combined" format (plus plain Common Log Format and
nginx's default "main" format with a trailing X-Forwarded-For field),
JSON-lines (one JSON object per line, with a configurable field-name
mapping that also accepts dotted paths into nested objects), and generic
W3C extended log format (a "#Fields:" header line defines the columns).
Gzip-compressed logs (e.g. logrotate's access.log.2.gz) are decompressed
transparently.

Every parser is a generator - lines are processed and yielded one at a
time, so even multi-gigabyte log files can be handled without loading the
whole file into memory. A mutable `ParserStats` object, passed in by the
caller, is updated as a side effect so streaming and stats-reporting can
coexist without materializing a full entry list.

Log lines are attacker-influenced input. Parsers never raise on malformed
content, and they are deliberately lenient about request lines: a line the
parser rejects is a request the model never gets to score, so strict
parsing would let an attacker hide a request just by making it malformed.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional

LOG = logging.getLogger("logsentinel.parsers")

SUPPORTED_FORMATS = ("combined", "json", "w3c")

# Numeric fields are clamped to this range so a corrupt or hostile value can
# never overflow the float conversion in the feature vector.
_MAX_INT_FIELD = 2**53

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

    @property
    def parsed_lines(self) -> int:
        return self.total_lines - self.unparsed_lines

    def note_unparsed(self, raw: str) -> None:
        self.unparsed_lines += 1
        if len(self.unparsed_samples) < 5:
            self.unparsed_samples.append(raw[:200])


def _to_int(raw: object, default: int = 0) -> int:
    """Best-effort int conversion for an untrusted log field. Never raises:
    "-", NaN, Infinity, booleans and absurdly large values all fall back to
    `default` or get clamped to [0, _MAX_INT_FIELD]."""
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        try:
            value = int(float(raw))  # e.g. "200.0"
        except (TypeError, ValueError, OverflowError):
            return default
    return min(max(value, 0), _MAX_INT_FIELD)


def open_log(path: Path) -> IO[str]:
    """Open a log file for streaming text reads. Gzip is detected by its
    magic bytes (not the file extension) and decompressed on the fly; a
    UTF-8 byte-order mark is dropped; undecodable bytes are replaced rather
    than raising."""
    with path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    if is_gzip:
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


# ---------------------------------------------------------------------------
# Combined Log Format
# ---------------------------------------------------------------------------

# Quoted fields allow backslash escapes: Apache writes an embedded quote as
# \" - so an attacker-supplied quote in the User-Agent or request line can't
# terminate the field early and make the whole line unparsable. Referer and
# User-Agent are optional (Common Log Format has neither), and anything after
# them - such as the "$http_x_forwarded_for" field of nginx's default "main"
# log_format - is ignored.
COMBINED_LOG_PATTERN = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>(?:[^"\\]|\\.)*)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)'
    r'(?:\s+"(?P<referer>(?:[^"\\]|\\.)*)"\s+"(?P<user_agent>(?:[^"\\]|\\.)*)")?'
)


def _split_request_line(request: str) -> tuple[str, str, str]:
    """Split a logged request line into (method, target, http_version).

    Tolerates what attack traffic actually looks like: raw spaces inside the
    target, a missing protocol, non-standard methods, and "-" (the server got
    no request line at all, e.g. a 400 or 408)."""
    parts = request.split(None, 1)
    if not parts or parts == ["-"]:
        return "", "", ""
    method = parts[0]
    target = parts[1].strip() if len(parts) == 2 else ""
    version = ""
    pieces = target.rsplit(None, 1)
    if len(pieces) == 2 and pieces[1][:5].upper() == "HTTP/":
        target, version = pieces[0], pieces[1][5:]
    return method, target, version


def parse_combined_line(line: str, line_number: int) -> Optional[LogEntry]:
    match = COMBINED_LOG_PATTERN.match(line.strip())
    if match is None:
        return None
    g = match.groupdict()
    method, path, http_version = _split_request_line(g["request"])
    return LogEntry(
        line_number=line_number, ip=g["ip"], timestamp=g["timestamp"], method=method,
        path=path, http_version=http_version, status=int(g["status"]),
        size=_to_int(g["size"]), referer=g["referer"] or "", user_agent=g["user_agent"] or "",
        raw_line=line.rstrip("\n"),
    )


# ---------------------------------------------------------------------------
# JSON-lines
# ---------------------------------------------------------------------------

DEFAULT_JSON_FIELD_MAP: dict[str, str] = {
    "ip": "ip", "timestamp": "timestamp", "method": "method", "path": "path",
    "status": "status", "size": "size", "referer": "referer", "user_agent": "user_agent",
}

# Tried in order when --field-map doesn't name a field. Covers the usual flat
# names plus nginx variable names (e.g. an `escape=json` log_format), Caddy's
# nested access log, Traefik, and Elastic Common Schema. Dotted names are
# paths into nested objects.
_JSON_FIELD_ALIASES: dict[str, list[str]] = {
    "ip": ["ip", "remote_addr", "client_ip", "src_ip", "ClientHost", "source.ip", "client.ip",
           "request.client_ip", "request.remote_ip"],
    "timestamp": ["timestamp", "time", "@timestamp", "date", "time_iso8601", "time_local", "ts", "StartUTC"],
    "method": ["method", "http_method", "verb", "request_method", "RequestMethod", "http.request.method",
               "request.method"],
    "path": ["path", "url", "uri", "request_uri", "request_path", "RequestPath", "url.original", "request.uri"],
    "status": ["status", "status_code", "response_code", "response_status", "DownstreamStatus",
               "http.response.status_code"],
    "size": ["size", "bytes", "response_size", "content_length", "body_bytes_sent", "bytes_sent",
             "DownstreamContentSize", "http.response.body.bytes"],
    "referer": ["referer", "referrer", "http_referer", "request_Referer", "http.request.referrer",
                "request.headers.Referer"],
    "user_agent": ["user_agent", "useragent", "ua", "http_user_agent", "request_User-Agent",
                   "user_agent.original", "request.headers.User-Agent"],
    # A full "GET /path HTTP/1.1" line (nginx's $request); used only when no
    # separate method/path field is present.
    "request": ["request", "request_line"],
}

JSON_FIELD_NAMES: tuple[str, ...] = tuple(_JSON_FIELD_ALIASES)


def _json_lookup(obj: dict, key: str) -> object:
    """obj[key], or - for a dotted key like "request.headers.User-Agent" that
    isn't a literal top-level key - a walk down nested objects."""
    if key in obj:
        return obj[key]
    if "." not in key:
        return None
    node: object = obj
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _json_scalar(value: object) -> object:
    """Unwrap a list to its first item (Caddy logs header values as lists)
    and reject nested objects, so a dict is never str()-ed into a field."""
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, (dict, list)):
        return None
    return value


def _resolve_json_field(obj: dict, field_map: dict[str, str], canonical: str) -> object:
    mapped_key = field_map.get(canonical)
    if mapped_key:
        value = _json_scalar(_json_lookup(obj, mapped_key))
        if value is not None:
            return value
    for alias in _JSON_FIELD_ALIASES.get(canonical, []):
        value = _json_scalar(_json_lookup(obj, alias))
        if value is not None:
            return value
    return None


def parse_json_line(line: str, line_number: int, field_map: Optional[dict[str, str]] = None) -> Optional[LogEntry]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except (ValueError, RecursionError):  # malformed, oversized integer, or absurdly deep nesting
        return None
    if not isinstance(obj, dict):
        return None

    field_map = field_map or {}

    def text(canonical: str) -> str:
        value = _resolve_json_field(obj, field_map, canonical)
        return "" if value is None else str(value)

    method, path, http_version = text("method"), text("path"), "1.1"
    if not method or not path:
        request_line = _resolve_json_field(obj, field_map, "request")
        if isinstance(request_line, str) and len(request_line.split(None, 1)) == 2:
            req_method, req_path, req_version = _split_request_line(request_line)
            method, path = method or req_method, path or req_path
            http_version = req_version or http_version
    if not path:
        # No request target at all: not an HTTP access record (e.g. a server
        # startup message mixed into the same file), or a --field-map that
        # doesn't match this log. Defaulting to "/" would silently train the
        # model on fake requests, so count it as unparsed instead.
        return None

    return LogEntry(
        line_number=line_number,
        ip=text("ip"),
        timestamp=text("timestamp"),
        method=(method or "GET").upper(),
        path=path,
        http_version=http_version,
        status=_to_int(_resolve_json_field(obj, field_map, "status")),
        size=_to_int(_resolve_json_field(obj, field_map, "size")),
        referer=text("referer"),
        user_agent=text("user_agent"),
        raw_line=stripped,
    )


# ---------------------------------------------------------------------------
# W3C extended log format
# ---------------------------------------------------------------------------

_W3C_FIELD_ALIASES: dict[str, list[str]] = {
    "ip": ["c-ip", "client-ip"],
    "method": ["cs-method"],
    "path": ["cs-uri-stem", "cs-uri"],
    "query": ["cs-uri-query"],
    "status": ["sc-status"],
    "size": ["sc-bytes"],
    "user_agent": ["cs(user-agent)"],
    "referer": ["cs(referer)"],
    "date": ["date"],
    "time": ["time"],
}


def parse_w3c_fields_header(line: str) -> Optional[list[str]]:
    stripped = line.strip()
    if not stripped.startswith("#Fields:"):
        return None
    return stripped[len("#Fields:"):].strip().split()


def _w3c_columns(fields: list[str]) -> dict[str, int]:
    """Map canonical names to column indices (field names are matched
    case-insensitively). Computed once per #Fields header, not per line."""
    lowered = [f.lower() for f in fields]
    columns: dict[str, int] = {}
    for canonical, aliases in _W3C_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                columns[canonical] = lowered.index(alias)
                break
    return columns


def _parse_w3c(stripped: str, line_number: int, n_fields: int, columns: dict[str, int]) -> Optional[LogEntry]:
    if not stripped or stripped.startswith("#") or "path" not in columns:
        return None
    values = stripped.split()
    if len(values) != n_fields:
        return None

    def get(canonical: str, default: str = "") -> str:
        idx = columns.get(canonical)
        return values[idx] if idx is not None else default

    timestamp = f"{get('date')} {get('time')}".strip()

    path = get("path", "/")
    query = get("query", "")
    if query and query != "-":
        path = f"{path}?{query}"

    return LogEntry(
        line_number=line_number,
        ip=get("ip", "-"),
        timestamp=timestamp,
        method=get("method", "GET").upper(),
        path=path,
        http_version="1.1",
        status=_to_int(get("status")),
        size=_to_int(get("size")),
        referer=get("referer", "-").replace("+", " "),
        user_agent=get("user_agent", "-").replace("+", " "),
        raw_line=stripped,
    )


def parse_w3c_line(line: str, line_number: int, fields: list[str]) -> Optional[LogEntry]:
    return _parse_w3c(line.strip(), line_number, len(fields), _w3c_columns(fields))


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------


def detect_format(path: Path) -> str:
    """Peek at the first few non-empty lines to guess the log format.

    Returns "json", "w3c", or "combined". JSON and combined are decided by a
    vote over up to 20 lines rather than by the first line alone, so a file
    that starts mid-record (cut by rotation or `tail -c`) is still detected.
    """
    votes = {"json": 0, "combined": 0}
    try:
        with open_log(Path(path)) as fh:
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
                        if isinstance(json.loads(stripped), dict):
                            votes["json"] += 1
                    except (ValueError, RecursionError):
                        pass
                elif COMBINED_LOG_PATTERN.match(stripped):
                    votes["combined"] += 1
    except (OSError, EOFError, zlib.error):
        pass
    return "json" if votes["json"] > votes["combined"] else "combined"


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
    path = Path(path)
    fmt = format_override or detect_format(path)
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported log format {fmt!r}; expected one of: {', '.join(SUPPORTED_FORMATS)}")
    stats.detected_format = fmt
    LOG.debug("Reading %s as %s format (%s).", path, fmt, "forced" if format_override else "auto-detected")

    w3c_fields: Optional[list[str]] = None
    w3c_columns: dict[str, int] = {}
    line_number = 0

    with open_log(path) as fh:
        try:
            for raw_line in fh:
                line_number += 1
                stripped = raw_line.strip()
                if not stripped:
                    continue

                if fmt == "w3c":
                    header_fields = parse_w3c_fields_header(stripped)
                    if header_fields is not None:
                        w3c_fields, w3c_columns = header_fields, _w3c_columns(header_fields)
                        continue
                    if stripped.startswith("#"):
                        continue
                    stats.total_lines += 1
                    if w3c_fields is None:
                        stats.note_unparsed(stripped)
                        continue
                    entry = _parse_w3c(stripped, line_number, len(w3c_fields), w3c_columns)
                elif fmt == "json":
                    stats.total_lines += 1
                    entry = parse_json_line(stripped, line_number, field_map)
                else:
                    stats.total_lines += 1
                    entry = parse_combined_line(raw_line, line_number)

                if entry is None:
                    stats.note_unparsed(stripped)
                    continue
                yield entry
        except EOFError:
            # A gzip file still being written (or cut short) ends without its
            # trailer; keep what was read instead of discarding the whole run.
            LOG.warning("%s ended unexpectedly after line %d (truncated gzip?); remaining data skipped.",
                        path, line_number)

    if stats.unparsed_lines:
        LOG.warning(
            "%d of %d lines could not be parsed as %s and were skipped (first: %r).",
            stats.unparsed_lines, stats.total_lines, fmt, stats.unparsed_samples[0][:120],
        )
        for sample in stats.unparsed_samples:
            LOG.debug("Unparsed line sample: %r", sample)
