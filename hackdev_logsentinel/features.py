"""Numeric feature extraction from a parsed log entry."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from urllib.parse import parse_qsl, urlsplit

from .parsers import LogEntry

METHOD_ENCODING: dict[str, int] = {
    "GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "HEAD": 4, "OPTIONS": 5,
}

FEATURE_ORDER = (
    "url_length", "param_count", "special_char_ratio", "entropy",
    "payload_length", "method_encoded",
)


@dataclass
class RequestFeatures:
    url_length: int
    param_count: int
    special_char_ratio: float
    entropy: float
    payload_length: int
    method_encoded: int

    def to_vector(self) -> list[float]:
        return [
            float(self.url_length), float(self.param_count), float(self.special_char_ratio),
            float(self.entropy), float(self.payload_length), float(self.method_encoded),
        ]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(entry: LogEntry) -> RequestFeatures:
    split = urlsplit(entry.path)
    url_length = len(entry.path)

    try:
        params = parse_qsl(split.query, keep_blank_values=True)
    except ValueError:
        params = []
    param_count = len(params)

    if entry.path:
        special_chars = sum(1 for ch in entry.path if not ch.isalnum())
        special_char_ratio = special_chars / len(entry.path)
    else:
        special_char_ratio = 0.0

    entropy = shannon_entropy(split.query)
    method_encoded = METHOD_ENCODING.get(entry.method.upper(), 6)

    return RequestFeatures(
        url_length=url_length,
        param_count=param_count,
        special_char_ratio=round(special_char_ratio, 6),
        entropy=round(entropy, 6),
        payload_length=entry.size,
        method_encoded=method_encoded,
    )
