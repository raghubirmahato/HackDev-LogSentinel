"""Numeric feature extraction from a parsed log entry."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from urllib.parse import parse_qsl

from .parsers import LogEntry

METHOD_ENCODING: dict[str, int] = {
    "GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "HEAD": 4, "OPTIONS": 5,
}

FEATURE_ORDER = (
    "url_length", "param_count", "special_char_ratio", "entropy",
    "payload_length", "method_encoded",
)

# Non-negative, unbounded features for which "far larger than anything in the
# baseline" is itself suspicious. IsolationForest can't express that - see
# scan_stream() in model.py - so the detector range-checks these explicitly.
MAGNITUDE_FEATURES = ("url_length", "param_count", "payload_length")


@dataclass
class RequestFeatures:
    url_length: int
    param_count: int
    special_char_ratio: float
    entropy: float
    payload_length: int
    method_encoded: int

    def to_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEATURE_ORDER]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def query_string(target: str) -> str:
    """The query component of a request target, split the same way as
    urllib.parse.urlsplit (drop the #fragment, then take what follows the
    first "?"). urlsplit itself isn't used because it raises ValueError on
    inputs like "//[x" ("Invalid IPv6 URL") - request targets are
    attacker-controlled, and one crafted request must not abort a scan."""
    target = target.split("#", 1)[0]
    return target.split("?", 1)[1] if "?" in target else ""


def extract_features(entry: LogEntry) -> RequestFeatures:
    query = query_string(entry.path)
    url_length = len(entry.path)

    try:
        params = parse_qsl(query, keep_blank_values=True)
    except ValueError:
        params = []
    param_count = len(params)

    if entry.path:
        special_chars = sum(1 for ch in entry.path if not ch.isalnum())
        special_char_ratio = special_chars / len(entry.path)
    else:
        special_char_ratio = 0.0

    entropy = shannon_entropy(query)
    method_encoded = METHOD_ENCODING.get(entry.method.upper(), 6)

    return RequestFeatures(
        url_length=url_length,
        param_count=param_count,
        special_char_ratio=round(special_char_ratio, 6),
        entropy=round(entropy, 6),
        payload_length=entry.size,
        method_encoded=method_encoded,
    )
