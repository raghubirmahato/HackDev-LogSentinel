"""Webhook alerting for scan results. Never allowed to crash a scan."""

from __future__ import annotations

import logging
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

LOG = logging.getLogger("logsentinel.alerts")


def build_webhook_payload(target_log_file: str, stats: dict, anomalies: list[dict]) -> dict[str, Any]:
    """Slack-compatible payload shape (a simple {"text": ...} message)."""
    lines = [
        f"*LogSentinel scan complete* for `{target_log_file}`",
        f"Parsed: {stats.get('parsed', 0)} | Anomalies: {stats.get('anomalies_found', 0)}",
    ]
    for anomaly in anomalies[:10]:
        lines.append(
            f"- line {anomaly['line_number']}: score={anomaly['anomaly_score']:.3f} "
            f"ip={anomaly['ip']} path={anomaly['path']}"
        )
    if len(anomalies) > 10:
        lines.append(f"...and {len(anomalies) - 10} more")
    return {"text": "\n".join(lines)}


def post_webhook(url: str, payload: dict, timeout: float = 10.0) -> bool:
    """POST the payload to `url`. Returns True on success, False on any
    failure - a failed webhook must never abort the scan itself."""
    if requests is None:
        LOG.warning("requests not installed; skipping webhook post to %s", url)
        return False
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        if response.status_code >= 400:
            LOG.warning("Webhook POST to %s returned HTTP %d", url, response.status_code)
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - alerting must never crash the scan
        LOG.warning("Webhook POST to %s failed: %s", url, exc)
        return False
