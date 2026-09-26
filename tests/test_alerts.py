import logging
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("requests")

from hackdev_logsentinel.alerts import build_webhook_payload, post_webhook, redact_url  # noqa: E402

SECRET_URL = "https://hooks.slack.com/services/T000/B000/SECRETTOKEN"


def test_build_webhook_payload_shape():
    stats = {"parsed": 10, "anomalies_found": 2}
    anomalies = [
        {"line_number": 5, "anomaly_score": 0.9, "ip": "1.2.3.4", "path": "/x"},
        {"line_number": 8, "anomaly_score": 0.7, "ip": "5.6.7.8", "path": "/y"},
    ]
    payload = build_webhook_payload("target.log", stats, anomalies)
    assert "text" in payload
    assert "target.log" in payload["text"]
    assert "line 5" in payload["text"]


def test_post_webhook_success():
    fake_response = MagicMock()
    fake_response.status_code = 200
    with patch("hackdev_logsentinel.alerts.requests.post", return_value=fake_response) as mock_post:
        result = post_webhook("https://hooks.example.com/x", {"text": "hi"})
    assert result is True
    mock_post.assert_called_once()


def test_post_webhook_http_failure_does_not_raise():
    fake_response = MagicMock()
    fake_response.status_code = 500
    with patch("hackdev_logsentinel.alerts.requests.post", return_value=fake_response):
        result = post_webhook("https://hooks.example.com/x", {"text": "hi"})
    assert result is False


def test_post_webhook_network_exception_does_not_raise():
    with patch("hackdev_logsentinel.alerts.requests.post", side_effect=ConnectionError("boom")):
        result = post_webhook("https://hooks.example.com/x", {"text": "hi"})
    assert result is False


def _anomaly(path="/x", ip="1.2.3.4", line=1, score=0.5):
    return {"line_number": line, "anomaly_score": score, "ip": ip, "path": path}


def test_payload_neutralizes_slack_control_sequences():
    payload = build_webhook_payload("t.log", {"parsed": 1, "anomalies_found": 1},
                                    [_anomaly(path="/<!channel> <https://evil.example|Reset password> & co")])
    text = payload["text"]
    assert "<!channel>" not in text and "<https://" not in text
    assert "&lt;!channel&gt; &lt;https://evil.example|Reset password&gt; &amp; co" in text


def test_payload_cannot_be_forged_with_newlines_or_escapes():
    payload = build_webhook_payload("t.log", {"parsed": 1, "anomalies_found": 1},
                                    [_anomaly(ip="1.2.3.4\n*LogSentinel scan complete*", path="/x\x1b[2J")])
    assert payload["text"].count("\n") == 2  # header, summary, one anomaly line
    assert "\x1b" not in payload["text"]


def test_payload_truncates_long_paths():
    payload = build_webhook_payload("t.log", {"parsed": 1, "anomalies_found": 1}, [_anomaly(path="/" + "A" * 5000)])
    assert len(payload["text"]) < 500


def test_payload_more_count_uses_total_found_when_list_is_truncated():
    anomalies = [_anomaly(line=i) for i in range(12)]
    payload = build_webhook_payload("t.log", {"parsed": 100, "anomalies_found": 40}, anomalies)
    assert payload["text"].endswith("...and 30 more")


@pytest.mark.parametrize("url, expected", [
    (SECRET_URL, "https://hooks.slack.com/***"),
    ("https://user:pw@example.com:8443/hook?token=abc", "https://example.com/***"),
    ("not a url", "<webhook URL>"),
    ("http://[::1/", "<webhook URL>"),
])
def test_redact_url(url, expected):
    assert redact_url(url) == expected


def test_webhook_secret_never_logged_on_http_error(caplog):
    with patch("hackdev_logsentinel.alerts.requests.post", return_value=MagicMock(status_code=403)):
        with caplog.at_level(logging.WARNING):
            assert post_webhook(SECRET_URL, {"text": "hi"}) is False
    assert "SECRETTOKEN" not in caplog.text
    assert "HTTP 403" in caplog.text


def test_webhook_secret_scrubbed_from_exception_messages(caplog):
    error = ConnectionError(f"Max retries exceeded with url: /services/T000/B000/SECRETTOKEN ({SECRET_URL})")
    with patch("hackdev_logsentinel.alerts.requests.post", side_effect=error):
        with caplog.at_level(logging.WARNING):
            assert post_webhook(SECRET_URL, {"text": "hi"}) is False
    assert "SECRETTOKEN" not in caplog.text
    assert "ConnectionError" in caplog.text


def test_webhook_without_requests_installed_is_skipped(caplog):
    with patch("hackdev_logsentinel.alerts.requests", None):
        with caplog.at_level(logging.WARNING):
            assert post_webhook(SECRET_URL, {"text": "hi"}) is False
    assert "SECRETTOKEN" not in caplog.text


def test_payload_explains_range_hits():
    anomalies = [
        {**_anomaly(line=1), "reasons": ["url_length=311 is 4.6x its baseline maximum (67.91)"]},
        {**_anomaly(line=2), "reasons": ["isolation forest"]},
    ]
    lines = build_webhook_payload("t.log", {"parsed": 2, "anomalies_found": 2}, anomalies)["text"].split("\n")
    assert lines[2].endswith("(url_length=311 is 4.6x its baseline maximum (67.91))")
    assert lines[3].endswith("path=/x")  # forest-only hits keep the plain format
