from unittest.mock import MagicMock, patch

from hackdev_logsentinel.alerts import build_webhook_payload, post_webhook


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
