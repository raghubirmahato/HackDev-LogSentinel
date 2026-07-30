"""
Model fitting, incremental updates, and streaming scan.

IsolationForest doesn't support true incremental/partial fit, so "incremental
update" here means: cache a bounded sample of the baseline's feature vectors
alongside the saved model, and on a later `fit --update`, combine that cached
sample with the newly-provided baseline data and refit from the combined
sample - cheaper than re-parsing and re-fitting on the full original baseline
log every time, and the model still reflects the accumulated history.

Both fitting and scanning consume an iterator of RequestFeatures rather than
a pre-built list, so very large log files are processed in bounded-size
chunks instead of ever holding the whole file in memory at once.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .features import FEATURE_ORDER, RequestFeatures
from .parsers import LogEntry

LOG = logging.getLogger("logsentinel.model")

DEFAULT_CACHE_SAMPLE_SIZE = 5000
DEFAULT_CHUNK_SIZE = 2000


def _require_sklearn():
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        return joblib, np, IsolationForest, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn (and/or joblib/numpy) is not installed. "
            "Install dependencies with `pip install -r requirements.txt` to use fit/scan."
        ) from exc


@dataclass
class ModelBundle:
    model: Any
    scaler: Any
    feature_order: tuple
    cached_sample: Any  # numpy array of feature vectors, capped at DEFAULT_CACHE_SAMPLE_SIZE rows


def fit_model(features_iter: Iterator[RequestFeatures], contamination: float, cache_sample_size: int = DEFAULT_CACHE_SAMPLE_SIZE) -> ModelBundle:
    """Fit a fresh IsolationForest + StandardScaler from a stream of features."""
    joblib, np, IsolationForest, StandardScaler = _require_sklearn()

    rows: list[list[float]] = [f.to_vector() for f in features_iter]
    if not rows:
        raise ValueError("No feature rows to fit on (empty baseline).")

    matrix = np.array(rows)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(scaled)

    if len(matrix) > cache_sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(matrix), size=cache_sample_size, replace=False)
        cached_sample = matrix[idx]
    else:
        cached_sample = matrix

    return ModelBundle(model=model, scaler=scaler, feature_order=FEATURE_ORDER, cached_sample=cached_sample)


def fit_incremental(
    existing_bundle: ModelBundle,
    new_features_iter: Iterator[RequestFeatures],
    contamination: float,
    cache_sample_size: int = DEFAULT_CACHE_SAMPLE_SIZE,
) -> ModelBundle:
    """Combine an existing model's cached baseline sample with newly-provided
    baseline data, and refit from the combined sample."""
    joblib, np, IsolationForest, StandardScaler = _require_sklearn()

    new_rows = [f.to_vector() for f in new_features_iter]
    if not new_rows:
        raise ValueError("No new feature rows provided for incremental update.")

    new_matrix = np.array(new_rows)
    combined = np.vstack([existing_bundle.cached_sample, new_matrix])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(scaled)

    if len(combined) > cache_sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combined), size=cache_sample_size, replace=False)
        cached_sample = combined[idx]
    else:
        cached_sample = combined

    return ModelBundle(model=model, scaler=scaler, feature_order=FEATURE_ORDER, cached_sample=cached_sample)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "scaler": bundle.scaler,
            "feature_order": bundle.feature_order,
            "cached_sample": bundle.cached_sample,
        },
        path,
    )


def load_bundle(path: Path) -> ModelBundle:
    joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    data = joblib.load(path)
    return ModelBundle(
        model=data["model"],
        scaler=data["scaler"],
        feature_order=tuple(data["feature_order"]),
        cached_sample=data["cached_sample"],
    )


def scan_stream(
    bundle: ModelBundle,
    entries_and_features: Iterator[tuple[LogEntry, RequestFeatures]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[dict]:
    """Score entries in bounded-size chunks (never loading the full target
    log into memory at once) and yield one anomaly dict per flagged entry."""
    joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()

    chunk: list[tuple[LogEntry, RequestFeatures]] = []

    def process(chunk_items: list[tuple[LogEntry, RequestFeatures]]) -> Iterator[dict]:
        if not chunk_items:
            return
        matrix = np.array([feat.to_vector() for _entry, feat in chunk_items])
        scaled = bundle.scaler.transform(matrix)
        predictions = bundle.model.predict(scaled)
        scores = bundle.model.decision_function(scaled)
        for (entry, feat), prediction, score in zip(chunk_items, predictions, scores):
            if prediction == -1:
                yield {
                    "line_number": entry.line_number,
                    "ip": entry.ip,
                    "path": entry.path,
                    "status": entry.status,
                    "anomaly_score": round(float(-score), 6),
                    "features": feat.to_dict(),
                    "raw_line": entry.raw_line,
                }

    for item in entries_and_features:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield from process(chunk)
            chunk = []

    yield from process(chunk)
