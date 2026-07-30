import json
import tempfile
from pathlib import Path

from hackdev_logsentinel.parsers import (
    ParserStats,
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
    line = json.dumps({"remote_addr": "1.2.3.4", "method": "POST", "path": "/api/login", "status_code": 401, "bytes": 512})
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
