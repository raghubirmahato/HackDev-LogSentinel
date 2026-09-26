from urllib.parse import urlsplit

import pytest

from hackdev_logsentinel.features import FEATURE_ORDER, extract_features, query_string, shannon_entropy
from hackdev_logsentinel.parsers import LogEntry, parse_combined_line, parse_json_line
from hackdev_logsentinel.sanitize import printable


def _entry(path: str, method: str = "GET", size: int = 100) -> LogEntry:
    return LogEntry(line_number=1, ip="1.2.3.4", timestamp="", method=method, path=path, http_version="1.1",
                    status=200, size=size, referer="", user_agent="", raw_line="")


@pytest.mark.parametrize("path", [
    "/index.html", "/search?q=a+b&page=2", "/a?x=1#frag", "/a#frag?notquery", "http://h/p?q=1#f",
    "//host/p?x", "/p??double", "/?", "", "javascript:alert(1)?x", "/a;b?c=d",
])
def test_query_string_matches_urlsplit(path):
    assert query_string(path) == urlsplit(path).query


@pytest.mark.parametrize("path", ["//[x", "//[::1/a?b=1", "http://[/", "//]evil?x=1", "//[not-an-ip]/p?q"])
def test_crafted_urls_do_not_crash_feature_extraction(path):
    # urlsplit raises ValueError("Invalid IPv6 URL") on these; one such request
    # used to abort an entire scan.
    features = extract_features(_entry(path))
    assert features.url_length == len(path)


def test_crafted_url_in_a_log_line_is_scored_not_fatal():
    entry = parse_combined_line('6.6.6.6 - - [10/Oct/2023:13:55:36 -0700] "GET //[x?id=1 HTTP/1.1" 404 10 "-" "c"', 1)
    features = extract_features(entry)
    assert (features.param_count, features.url_length) == (1, len("//[x?id=1"))


def test_feature_values_for_a_known_request():
    features = extract_features(_entry("/products?id=1&cat=2", method="post", size=480))
    assert features.to_dict() == {
        "url_length": 20, "param_count": 2, "special_char_ratio": round(5 / 20, 6),
        "entropy": round(shannon_entropy("id=1&cat=2"), 6), "payload_length": 480, "method_encoded": 1,
    }


def test_to_vector_follows_feature_order():
    features = extract_features(_entry("/a?b=1", method="DELETE", size=7))
    assert features.to_vector() == [float(features.to_dict()[name]) for name in FEATURE_ORDER]


def test_unknown_and_empty_methods_share_the_other_bucket():
    assert extract_features(_entry("/", method="PROPFIND")).method_encoded == 6
    assert extract_features(_entry("", method="")).method_encoded == 6


def test_hostile_json_sizes_still_produce_a_float_vector():
    entry = parse_json_line('{"path": "/", "size": 1' + "0" * 400 + '}', 1)
    assert all(isinstance(value, float) for value in extract_features(entry).to_vector())


@pytest.mark.parametrize("raw, expected", [
    ("\x1b]0;pwned\x07", "\\x1b]0;pwned\\x07"),         # OSC title-set + BEL
    ("line1\nline2\r", "line1\\x0aline2\\x0d"),         # forged output lines
    ("\x9b31m", "\\x9b31m"),                            # 8-bit CSI
    ("/admin‮txt.exe", "/admin\\u202etxt.exe"),    # bidi override
    ("/café/日本", "/café/日本"),  # ordinary non-ASCII untouched
])
def test_printable_escapes_control_and_bidi_characters(raw, expected):
    assert printable(raw) == expected


def test_printable_truncates():
    assert printable("a" * 50, max_length=10) == "a" * 9 + "…"
    assert printable("short", max_length=10) == "short"
