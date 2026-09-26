"""Webhook alerting for scan results. Never allowed to crash a scan."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlsplit

from .sanitize import printable

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

LOG = logging.getLogger("logsentinel.alerts")

MAX_LISTED_ANOMALIES = 10
_MAX_FIELD_CHARS = 200


def _slack_text(value: object) -> str:
    """Log-derived text made safe for a Slack message: control characters
    escaped (no forged extra lines), long values truncated, and &, <, >
    escaped as Slack requires - otherwise a request for "/<!channel>" would
    ping the whole channel and "<https://evil|text>" would render as a
    disguised link inside a trusted alert."""
    text = printable(value, _MAX_FIELD_CHARS)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_webhook_payload(target_log_file: str, stats: dict, anomalies: list[dict]) -> dict[str, Any]:
    """Slack-compatible payload shape (a simple {"text": ...} message)."""
    lines = [
        f"*LogSentinel scan complete* for `{_slack_text(target_log_file)}`",
        f"Parsed: {stats.get('parsed', 0)} | Anomalies: {stats.get('anomalies_found', 0)}",
    ]
    shown = anomalies[:MAX_LISTED_ANOMALIES]
    for anomaly in shown:
        lines.append(
            f"- line {anomaly['line_number']}: score={anomaly['anomaly_score']:.3f} "
            f"ip={_slack_text(anomaly['ip'])} path={_slack_text(anomaly['path'])}"
        )
    # `anomalies` may be only the top-N of a larger set (scan --top).
    remaining = max(len(anomalies), stats.get("anomalies_found", 0)) - len(shown)
    if remaining > 0:
        lines.append(f"...and {remaining} more")
    return {"text": "\n".join(lines)}


def redact_url(url: str) -> str:
    """Webhook URLs carry their credential in the path (Slack, Discord and
    Teams all do), so only ever log scheme://host."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<webhook URL>"
    return f"{parts.scheme}://{parts.hostname}/***" if parts.hostname else "<webhook URL>"


def _scrub(message: str, url: str) -> str:
    """Remove the secret parts of `url` from an exception message (requests
    errors quote the request path, e.g. "Max retries exceeded with url: ...")."""
    try:
        parts = urlsplit(url)
        sensitive = [url, parts.path, parts.query, parts.password or ""]
    except ValueError:
        sensitive = [url]
    for fragment in sensitive:
        if len(fragment) > 1:
            message = message.replace(fragment, "***")
    return message


def post_webhook(url: str, payload: dict, timeout: float = 10.0) -> bool:
    """POST the payload to `url`. Returns True on success, False on any
    failure - a failed webhook must never abort the scan itself."""
    safe_url = redact_url(url)
    if requests is None:
        LOG.warning("requests not installed; skipping webhook post to %s", safe_url)
        return False
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code >= 400:
            LOG.warning("Webhook POST to %s returned HTTP %d", safe_url, response.status_code)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - alerting must never crash the scan
        LOG.warning("Webhook POST to %s failed: %s", safe_url, _scrub(f"{type(exc).__name__}: {exc}", url))
        return False
