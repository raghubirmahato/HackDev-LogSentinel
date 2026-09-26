import gzip
import json
import logging
import tempfile
import time
from pathlib import Path

import pytest

from hackdev_logsentinel.parsers import (
    ParserStats,
    _to_int,
    detect_format,
    parse_combined_line,
    parse_json_line,
    parse_w3c_fields_header,
    parse_w3c_line,
    stream_log_file,
)


def test_parse_combined_line_valid():
    line = '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html?x=1 HTTP/1.1" 200 2326 "-" "Mozilla/5.0"'
    entry = parse_combined_line(line, 1)
    assert entry is not None
    assert entry.ip == "127.0.0.1"
    assert entry.method == "GET"
    assert entry.path == "/index.html?x=1"
    assert entry.status == 200
    assert entry.size == 2326


def test_parse_combined_line_malformed_returns_none():
    assert parse_combined_line("this is not a log line", 1) is None


def test_parse_json_line_default_aliases():
    line = json.dumps({"remote_addr": "1.2.3.4", "method": "POST", "path": "/api/login",
                       "status_code": 401, "bytes": 512})
    entry = parse_json_line(line, 1)
    assert entry is not None
    assert entry.ip == "1.2.3.4"
    assert entry.method == "POST"
    assert entry.status == 401
    assert entry.size == 512


def test_parse_json_line_custom_field_map():
    line = json.dumps({"clientAddr": "9.9.9.9", "verb": "GET", "requestPath": "/x"})
    field_map = {"ip": "clientAddr", "method": "verb", "path": "requestPath"}
    entry = parse_json_line(line, 1, field_map=field_map)
    assert entry.ip == "9.9.9.9"
    assert entry.method == "GET"
    assert entry.path == "/x"


def test_parse_json_line_malformed_returns_none():
    assert parse_json_line("not json at all", 1) is None
    assert parse_json_line("[1,2,3]", 1) is None  # valid JSON but not an object


def test_parse_w3c_fields_header():
    fields = parse_w3c_fields_header("#Fields: date time c-ip cs-method cs-uri-stem sc-status sc-bytes")
    assert fields == ["date", "time", "c-ip", "cs-method", "cs-uri-stem", "sc-status", "sc-bytes"]


def test_parse_w3c_line():
    fields = ["date", "time", "c-ip", "cs-method", "cs-uri-stem", "sc-status", "sc-bytes"]
    line = "2024-01-01 12:00:00 10.0.0.1 GET /home 200 1500"
    entry = parse_w3c_line(line, 1, fields)
    assert entry.ip == "10.0.0.1"
    assert entry.method == "GET"
    assert entry.path == "/home"
    assert entry.status == 200
    assert entry.size == 1500


def test_detect_format_json():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.json"
        path.write_text('{"ip": "1.2.3.4", "path": "/"}\n')
        assert detect_format(path) == "json"


def test_detect_format_w3c():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.w3c"
        path.write_text("#Software: TestServer\n#Fields: date time c-ip\n2024-01-01 00:00:00 1.2.3.4\n")
        assert detect_format(path) == "w3c"


def test_detect_format_combined():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.txt"
        path.write_text('127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET / HTTP/1.1" 200 100 "-" "curl"\n')
        assert detect_format(path) == "combined"


def test_stream_log_file_combined_and_stats():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.txt"
        good_line = '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET / HTTP/1.1" 200 100 "-" "curl"\n'
        path.write_text(good_line + "garbage line\n" + good_line)
        stats = ParserStats()
        entries = list(stream_log_file(path, stats))
        assert len(entries) == 2
        assert stats.total_lines == 3
        assert stats.unparsed_lines == 1
        assert stats.detected_format == "combined"


def test_stream_log_file_json():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        lines = [json.dumps({"ip": f"1.2.3.{i}", "method": "GET", "path": "/", "status": 200}) for i in range(5)]
        path.write_text("\n".join(lines) + "\n")
        stats = ParserStats()
        entries = list(stream_log_file(path, stats))
        assert len(entries) == 5
        assert stats.detected_format == "json"


def test_stream_log_file_w3c():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.w3c"
        content = (
            "#Software: TestServer\n"
            "#Fields: date time c-ip cs-method cs-uri-stem sc-status sc-bytes\n"
            "2024-01-01 00:00:00 1.2.3.4 GET /home 200 100\n"
            "2024-01-01 00:00:01 1.2.3.5 POST /login 401 50\n"
        )
        path.write_text(content)
        stats = ParserStats()
        entries = list(stream_log_file(path, stats))
        assert len(entries) == 2
        assert entries[0].ip == "1.2.3.4"
        assert entries[1].method == "POST"


# ---------------------------------------------------------------------------
# Regression tests: real-world formats and hostile input
# ---------------------------------------------------------------------------

PREFIX = '203.0.113.9 - - [10/Oct/2023:14:04:00 -0700] '


def test_combined_nginx_main_format_with_trailing_xff_field():
    # nginx's default `log_format main` appends "$http_x_forwarded_for".
    entry = parse_combined_line(PREFIX + '"GET /a HTTP/1.1" 200 5 "-" "Mozilla/5.0" "10.1.1.1"', 1)
    assert entry is not None
    assert (entry.path, entry.user_agent) == ("/a", "Mozilla/5.0")


def test_combined_common_log_format_without_referer_or_agent():
    entry = parse_combined_line(PREFIX + '"GET /index.html HTTP/1.1" 200 2326', 1)
    assert entry is not None
    assert (entry.path, entry.size, entry.referer, entry.user_agent) == ("/index.html", 2326, "", "")


def test_combined_escaped_quote_in_user_agent_is_not_an_evasion():
    # Apache escapes " as \" - an attacker-chosen quote must not make the line unparsable.
    entry = parse_combined_line(PREFIX + r'"GET /x HTTP/1.1" 200 1 "-" "sqlmap \" OR \"1\"=\"1"', 1)
    assert entry is not None
    assert entry.user_agent == r'sqlmap \" OR \"1\"=\"1'


def test_combined_raw_spaces_in_request_target_are_kept():
    entry = parse_combined_line(PREFIX + '"GET /p?id=1 UNION SELECT pass FROM users-- HTTP/1.1" 500 1 "-" "x"', 1)
    assert entry is not None
    assert (entry.method, entry.path, entry.http_version) == ("GET", "/p?id=1 UNION SELECT pass FROM users--", "1.1")


@pytest.mark.parametrize("request_line, expected", [
    ("-", ("", "", "")),                                  # no request received (400/408)
    ("GET /old", ("GET", "/old", "")),                    # HTTP/0.9 style, no protocol
    ("M-SEARCH * HTTP/1.1", ("M-SEARCH", "*", "1.1")),    # non-standard method
    ("\\x16\\x03\\x01\\x02", ("\\x16\\x03\\x01\\x02", "", "")),  # TLS handshake sent to an HTTP port
])
def test_combined_unusual_request_lines_still_parse(request_line, expected):
    entry = parse_combined_line(PREFIX + f'"{request_line}" 400 0 "-" "-"', 1)
    assert entry is not None
    assert (entry.method, entry.path, entry.http_version) == expected


def test_combined_regex_is_linear_on_adversarial_lines():
    hostile = [
        PREFIX + '"GET /' + '\\"' * 200_000 + ' HTTP/1.1" 200 1 "-" "x"',
        PREFIX + '"GET /' + ' ' * 200_000 + 'x HTTP/1.1" 200 1',
        PREFIX + '"' + 'a\\' * 200_000,  # never-terminated quoted field
    ]
    start = time.perf_counter()
    for line in hostile:
        parse_combined_line(line, 1)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize("raw, expected", [
    ("2326", 2326), ("-", 0), ("", 0), (None, 0), (True, 0), ("200.0", 200), (512.9, 512),
    (float("inf"), 0), (float("nan"), 0), ("9" * 5000, 0), (10**400, 2**53), (-5, 0), ([1], 0),
])
def test_to_int_never_raises_and_clamps(raw, expected):
    assert _to_int(raw) == expected


@pytest.mark.parametrize("line", [
    '{"path": "/", "size": Infinity}',
    '{"path": "/", "size": NaN, "status": -Infinity}',
    '{"path": "/", "size": 1' + "0" * 400 + '}',
])
def test_parse_json_line_hostile_numbers(line):
    entry = parse_json_line(line, 1)
    assert entry is not None
    assert 0 <= entry.size <= 2**53
    float(entry.size)  # must be convertible for the feature vector


@pytest.mark.parametrize("line", ["[" * 100_000, '{"a": 1' + "0" * 5000 + "}"])
def test_parse_json_line_pathological_json_is_unparsed_not_a_crash(line):
    assert parse_json_line(line, 1) is None


def test_parse_json_line_without_request_target_is_unparsed():
    # Not an access record (e.g. a startup message sharing the log file); defaulting
    # to "/" would train the model on requests that never happened.
    assert parse_json_line('{"level": "info", "msg": "server started"}', 1) is None


def test_parse_json_line_nginx_variable_names_and_request_line():
    line = json.dumps({"time_local": "10/Oct/2023:13:55:36 +0000", "remote_addr": "1.1.1.1",
                       "request": "POST /login?next=/ HTTP/2.0", "status": "401", "body_bytes_sent": "57",
                       "http_referer": "-", "http_user_agent": "Mozilla"})
    entry = parse_json_line(line, 1)
    assert (entry.ip, entry.method, entry.path, entry.http_version) == ("1.1.1.1", "POST", "/login?next=/", "2.0")
    assert (entry.status, entry.size, entry.user_agent) == (401, 57, "Mozilla")


def test_parse_json_line_caddy_nested_record():
    line = json.dumps({"level": "info", "ts": 1646861401.52, "logger": "http.log.access",
                       "request": {"remote_ip": "10.0.0.9", "client_ip": "10.0.0.8", "method": "GET",
                                   "uri": "/a?b=1", "headers": {"User-Agent": ["curl/7.82.0"]}},
                       "size": 10900, "status": 200})
    entry = parse_json_line(line, 1)
    assert (entry.ip, entry.method, entry.path, entry.user_agent) == ("10.0.0.8", "GET", "/a?b=1", "curl/7.82.0")
    assert (entry.status, entry.size) == (200, 10900)


def test_parse_json_line_ecs_nested_record_skips_object_valued_aliases():
    # "url" and "user_agent" are objects in ECS; they must not be str()-ed into fields.
    line = json.dumps({"source": {"ip": "2.2.2.2"}, "url": {"original": "/api/x?q=1"},
                       "http": {"request": {"method": "put"}, "response": {"status_code": 201, "body": {"bytes": 12}}},
                       "user_agent": {"original": "UA"}})
    entry = parse_json_line(line, 1)
    assert (entry.ip, entry.method, entry.path, entry.status, entry.size, entry.user_agent) == (
        "2.2.2.2", "PUT", "/api/x?q=1", 201, 12, "UA")


def test_parse_json_line_traefik_record():
    line = json.dumps({"ClientHost": "4.4.4.4", "RequestMethod": "PATCH", "RequestPath": "/v1/items/7?x=1",
                       "DownstreamStatus": 204, "DownstreamContentSize": 0, "request_User-Agent": "Go-http-client/2.0",
                       "StartUTC": "2023-10-10T13:55:36Z"})
    entry = parse_json_line(line, 1)
    assert (entry.ip, entry.method, entry.path, entry.status, entry.user_agent) == (
        "4.4.4.4", "PATCH", "/v1/items/7?x=1", 204, "Go-http-client/2.0")


def test_parse_json_line_field_map_accepts_dotted_paths():
    line = json.dumps({"req": {"target": "/deep", "verb": "DELETE"}, "peer": {"addr": "3.3.3.3"}})
    entry = parse_json_line(line, 1, field_map={"path": "req.target", "method": "req.verb", "ip": "peer.addr"})
    assert (entry.ip, entry.method, entry.path) == ("3.3.3.3", "DELETE", "/deep")


def test_parse_w3c_line_case_insensitive_fields_and_query():
    fields = ["date", "time", "C-IP", "CS-METHOD", "cs-uri-stem", "cs-uri-query", "sc-status", "sc-bytes",
              "cs(user-agent)"]
    entry = parse_w3c_line("2024-01-01 12:00:00 10.0.0.1 get /s q=a+b 200 15 Mozilla/5.0+(X11)", 1, fields)
    assert (entry.ip, entry.method, entry.path) == ("10.0.0.1", "GET", "/s?q=a+b")
    assert entry.user_agent == "Mozilla/5.0 (X11)"
    assert entry.timestamp == "2024-01-01 12:00:00"


def test_parse_w3c_line_without_path_column_is_unparsed():
    assert parse_w3c_line("2024-01-01 12:00:00 10.0.0.1", 1, ["date", "time", "c-ip"]) is None


def test_detect_format_json_with_utf8_bom(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('﻿{"ip": "1.2.3.4", "path": "/"}\n', encoding="utf-8")
    assert detect_format(path) == "json"


@pytest.mark.parametrize("first_line", [
    'me": "x", "path": "/cut-off"}',  # file starts mid-record (e.g. cut with `tail -c`)
    '{"ip": "1.2.3.4", "pa',           # record truncated at the end
])
def test_detect_format_json_when_first_record_is_cut(tmp_path, first_line):
    path = tmp_path / "log.jsonl"
    path.write_text(first_line + "\n" + '{"ip": "1.2.3.4", "path": "/"}\n' * 3)
    assert detect_format(path) == "json"


def test_stream_log_file_reads_gzip_transparently(tmp_path):
    path = tmp_path / "access.log.2.gz"  # detected by magic bytes, not the name
    good_line = '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET / HTTP/1.1" 200 100 "-" "curl"\n'
    with gzip.open(path, "wt") as fh:
        fh.write(good_line * 3)
    stats = ParserStats()
    assert len(list(stream_log_file(path, stats))) == 3
    assert stats.detected_format == "combined"


def test_stream_log_file_truncated_gzip_keeps_what_was_read(tmp_path, caplog):
    good_line = '127.0.0.1 - - [10/Oct/2023:13:55:36 -0700] "GET / HTTP/1.1" 200 100 "-" "curl"\n'
    data = gzip.compress((good_line * 2000).encode())
    path = tmp_path / "rotating.log.gz"
    path.write_bytes(data[: len(data) // 2])
    stats = ParserStats()
    with caplog.at_level(logging.WARNING):
        entries = list(stream_log_file(path, stats))
    assert 0 < len(entries) < 2000
    assert stats.unparsed_lines == 0
    assert "ended unexpectedly" in caplog.text


def test_stream_log_file_w3c_lines_before_header_and_header_change(tmp_path):
    path = tmp_path / "log.w3c"
    path.write_text(
        "#Fields: date time c-ip cs-method cs-uri-stem sc-status sc-bytes\n"
        "2024-01-01 00:00:00 1.2.3.4 GET /a 200 100\n"
        "#Fields: date time cs-uri-stem c-ip sc-status\n"  # server restarted with a different layout
        "2024-01-01 00:00:02 /b 1.2.3.6 404\n"
    )
    stats = ParserStats()
    entries = list(stream_log_file(path, stats))
    assert [(e.path, e.ip, e.status) for e in entries] == [("/a", "1.2.3.4", 200), ("/b", "1.2.3.6", 404)]

    headerless = tmp_path / "headerless.w3c"
    headerless.write_text("2024-01-01 00:00:00 1.2.3.4 GET /a 200 100\n")
    stats = ParserStats()
    assert list(stream_log_file(headerless, stats, format_override="w3c")) == []
    assert (stats.total_lines, stats.unparsed_lines) == (1, 1)


def test_stream_log_file_rejects_unknown_format(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("x\n")
    with pytest.raises(ValueError, match="Unsupported log format"):
        list(stream_log_file(path, ParserStats(), format_override="syslog"))
