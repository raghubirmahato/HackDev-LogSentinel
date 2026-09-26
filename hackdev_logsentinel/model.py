"""
Model fitting, incremental updates, and streaming scan.

Fitting uses bounded memory: feature rows stream through a uniform random
sample of at most `max_fit_rows` rows (100k by default) and the forest is
trained on that sample. IsolationForest only draws 256 rows per tree anyway,
so a large uniform sample costs no accuracy while keeping memory flat no
matter how big the baseline log is. Baselines under the cap are used whole.

IsolationForest doesn't support true incremental/partial fit, so "incremental
update" works from a cache: a bounded uniform sample of every baseline row the
model has been fitted on is saved alongside the model, together with how many
rows it stands for. `fit --update` merges that cache with a sample of the new
baseline data *in proportion to how many rows each side represents* and
refits - so the model reflects the whole accumulated history instead of
over-weighting whichever batch arrived last, and the new cache is again a
uniform sample of everything seen so far.

Detection combines the forest with a range check. IsolationForest can't tell
how far a value lies beyond the baseline - every split is drawn inside the
training data's range, so a 300,000-character URL takes exactly the same path
through the trees as the longest baseline URL and gets the same score. So the
model also records each magnitude feature's robust baseline maximum (99.9th
percentile), and a request is flagged when it exceeds `range_factor` times
that bound (2x by default), whatever the forest thinks.

Scanning consumes an iterator and scores rows in bounded-size chunks, so very
large target logs are never held in memory at once.

Bundle compatibility: fields added over time are optional keys, so bundles
from every release load here (range bounds missing from pre-2.2 bundles are
derived from their cached sample), and bundles written here still load in
2.0 and 2.1, which just skip the range check.
"""

from __future__ import annotations

import contextlib
import logging
import math
import numbers
import os
import secrets
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .features import FEATURE_ORDER, MAGNITUDE_FEATURES, RequestFeatures
from .parsers import LogEntry

LOG = logging.getLogger("logsentinel.model")

BUNDLE_FORMAT_VERSION = 2
DEFAULT_CACHE_SAMPLE_SIZE = 5000
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CONTAMINATION = 0.05
DEFAULT_MAX_FIT_ROWS = 100_000
DEFAULT_RANGE_FACTOR = 2.0
RANGE_QUANTILE = 0.999
# Range-check hits are scored on the forest's scale: 0.1 per doubling beyond
# the limit (4x the baseline bound scores 0.1, 16x scores 0.3).
_RANGE_SCORE_PER_DOUBLING = 0.1
_SEED = 42
_SAMPLE_BLOCK_ROWS = 10_000


class ModelError(ValueError):
    """A model file is unreadable, not a LogSentinel bundle, or incompatible
    with this version of LogSentinel."""


class EmptyBaselineError(ValueError):
    """There were no feature rows to fit on."""


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
    feature_order: tuple[str, ...]
    cached_sample: Any  # numpy array: uniform sample (<= cache size rows) of every baseline row seen
    n_samples_seen: int = 0  # number of baseline rows `cached_sample` stands for
    # Robust baseline maximum of each MAGNITUDE_FEATURES entry (None: no range
    # check for that feature); None here means no range check at all.
    range_bounds: Optional[dict[str, Optional[float]]] = None
    range_factor: float = DEFAULT_RANGE_FACTOR  # flag magnitudes beyond factor x bound; 0 turns it off

    @property
    def contamination(self) -> float:
        return self.model.contamination


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_contamination(contamination: object) -> None:
    if not _is_number(contamination) or not 0 < contamination <= 0.5:
        raise ValueError(f"contamination must be a number in (0, 0.5], got {contamination!r}")


def _check_range_factor(range_factor: object) -> None:
    if not _is_number(range_factor) or not (range_factor == 0 or 1 < range_factor < math.inf):
        raise ValueError(f"range_factor must be greater than 1, or 0 to turn the range check off; got {range_factor!r}")


def _range_bounds(sample) -> dict[str, Optional[float]]:
    """Each magnitude feature's robust baseline maximum: the 99.9th
    percentile, so a few attacks hidden in the baseline don't raise the bar.
    It is at least 1, which keeps the check meaningful for a feature that is
    always 0 in the baseline (a site without query strings). The exception is
    an all-zero payload_length: that means the log format doesn't record
    response sizes, so there is nothing to compare against (None)."""
    _joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    bounds: dict[str, Optional[float]] = {}
    for name in MAGNITUDE_FEATURES:
        column = sample[:, FEATURE_ORDER.index(name)]
        if name == "payload_length" and not column.any():
            bounds[name] = None
        else:
            bounds[name] = max(float(np.quantile(column, RANGE_QUANTILE)), 1.0)
    return bounds


def _uniform_sample(rows: Iterable[list[float]], capacity: int, rng) -> tuple[Any, int]:
    """Stream `rows` into a uniform random sample of at most `capacity` rows.
    Returns (sample, rows_seen); memory is bounded by `capacity`, not by the
    length of the stream.

    Every row gets an i.i.d. uniform random key and the rows holding the
    `capacity` smallest keys are kept, which is a uniformly random subset.
    Rows are handled in blocks so the bookkeeping is vectorized, and the
    sample stays in input order, so a stream shorter than `capacity` comes
    back whole and unchanged."""
    _joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    width = len(FEATURE_ORDER)
    sample = np.empty((0, width))
    keys = np.empty(0)
    seen = 0
    block: list[list[float]] = []

    def absorb() -> None:
        nonlocal sample, keys
        block_rows = np.asarray(block, dtype=float).reshape(-1, width)
        block_keys = rng.random(len(block_rows))
        block.clear()
        if len(keys) >= capacity:
            # Sample is full: only rows beating its current worst key can get in.
            fresh = block_keys < keys.max()
            block_rows, block_keys = block_rows[fresh], block_keys[fresh]
        sample = np.concatenate([sample, block_rows])
        keys = np.concatenate([keys, block_keys])
        if len(keys) > capacity:
            keep = np.sort(np.argpartition(keys, capacity - 1)[:capacity])
            sample, keys = sample[keep], keys[keep]

    for row in rows:
        block.append(row)
        seen += 1
        if len(block) >= _SAMPLE_BLOCK_ROWS:
            absorb()
    if block:
        absorb()
    return sample, seen


def _subsample(matrix, size: int, rng):
    """A uniform random subset of `size` rows (all of them if there are fewer)."""
    if len(matrix) <= size:
        return matrix
    return matrix[rng.choice(len(matrix), size=size, replace=False)]


def _merge_samples(a, n_a: int, b, n_b: int, capacity: int, rng):
    """Merge `a` (a uniform sample of n_a rows) with `b` (a uniform sample of a
    disjoint n_b rows) into a sample of the union where each side appears in
    proportion to the population it stands for. The result is as large as
    the inputs allow - neither side can supply more rows than it holds - and
    at most `capacity` rows."""
    _joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    if not n_a or not len(a):
        return _subsample(b, capacity, rng)
    total = n_a + n_b
    size = min(capacity, len(a) * total // n_a, len(b) * total // n_b)
    take_a = min(len(a), round(size * n_a / total))
    take_b = min(len(b), size - take_a)
    return np.concatenate([_subsample(a, take_a, rng), _subsample(b, take_b, rng)])


def _fit_bundle(sample, n_samples_seen: int, contamination: float, cache_sample_size: int,
                range_factor: float) -> ModelBundle:
    _joblib, np, IsolationForest, StandardScaler = _require_sklearn()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample)

    model = IsolationForest(contamination=contamination, random_state=_SEED)
    model.fit(scaled)

    cached_sample = _subsample(sample, cache_sample_size, np.random.default_rng(_SEED))
    return ModelBundle(model=model, scaler=scaler, feature_order=FEATURE_ORDER,
                       cached_sample=cached_sample, n_samples_seen=n_samples_seen,
                       range_bounds=_range_bounds(sample), range_factor=range_factor)


def fit_model(
    features_iter: Iterable[RequestFeatures],
    contamination: float,
    cache_sample_size: int = DEFAULT_CACHE_SAMPLE_SIZE,
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS,
    range_factor: float = DEFAULT_RANGE_FACTOR,
) -> ModelBundle:
    """Fit a fresh IsolationForest + StandardScaler from a stream of features,
    training on a uniform sample of at most `max_fit_rows` rows, and record
    the baseline's range bounds for the range check."""
    _joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    _check_contamination(contamination)
    _check_range_factor(range_factor)

    sample, seen = _uniform_sample((f.to_vector() for f in features_iter), max_fit_rows, np.random.default_rng(_SEED))
    if not seen:
        raise EmptyBaselineError("No feature rows to fit on (empty baseline).")
    if seen > len(sample):
        LOG.info("Baseline has %d rows; fitting on a uniform sample of %d (--max-fit-rows).", seen, len(sample))
    return _fit_bundle(sample, seen, contamination, cache_sample_size, range_factor)


def fit_incremental(
    existing_bundle: ModelBundle,
    new_features_iter: Iterable[RequestFeatures],
    contamination: Optional[float] = None,
    cache_sample_size: int = DEFAULT_CACHE_SAMPLE_SIZE,
    max_fit_rows: int = DEFAULT_MAX_FIT_ROWS,
    range_factor: Optional[float] = None,
) -> ModelBundle:
    """Fold newly-provided baseline data into an existing model and refit.

    The training set merges the existing model's cached sample with a uniform
    sample of the new data, each in proportion to the number of baseline rows
    it represents; the range bounds are recomputed from that merged sample.
    `contamination` and `range_factor` default to the existing model's."""
    _joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()
    if contamination is None:
        contamination = existing_bundle.contamination
    if range_factor is None:
        range_factor = existing_bundle.range_factor
    _check_contamination(contamination)
    _check_range_factor(range_factor)

    rng = np.random.default_rng(_SEED)
    new_sample, n_new = _uniform_sample((f.to_vector() for f in new_features_iter), max_fit_rows, rng)
    if not n_new:
        raise EmptyBaselineError("No new feature rows provided for incremental update.")

    old_sample = np.asarray(existing_bundle.cached_sample, dtype=float)
    # Bundles saved before n_samples_seen was recorded only know their cache size.
    n_old = max(int(existing_bundle.n_samples_seen), len(old_sample))
    combined = _merge_samples(old_sample, n_old, new_sample, n_new, max_fit_rows, rng)
    LOG.info(
        "Update: %d rows of history + %d new rows; refitting on %d rows drawn in proportion.",
        n_old, n_new, len(combined),
    )
    return _fit_bundle(combined, n_old + n_new, contamination, cache_sample_size, range_factor)


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    """Write the bundle atomically: dump to a temp file beside `path`, fsync,
    then rename it over `path`. A crash or Ctrl-C mid-write can never leave a
    truncated model behind - which matters because `fit --update m.joblib
    -o m.joblib` rewrites the only copy of the model in place."""
    joblib, _np, _IsolationForest, _StandardScaler = _require_sklearn()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "logsentinel_version": __version__,
        "model": bundle.model,
        "scaler": bundle.scaler,
        "feature_order": tuple(bundle.feature_order),
        "cached_sample": bundle.cached_sample,
        "n_samples_seen": int(bundle.n_samples_seen),
        # Plain Python values only, so any LogSentinel version can unpickle them.
        "range_bounds": None if bundle.range_bounds is None else {
            name: None if bound is None else float(bound) for name, bound in bundle.range_bounds.items()
        },
        "range_factor": float(bundle.range_factor),
    }
    tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with open(tmp_path, "xb") as fh:
            joblib.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def load_bundle(path: Path) -> ModelBundle:
    """Load and validate a bundle written by save_bundle.

    Model files are pickles, and unpickling can execute arbitrary code: only
    load model files you created yourself or otherwise trust."""
    joblib, np, IsolationForest, _StandardScaler = _require_sklearn()
    path = Path(path)
    try:
        data = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt file can fail in many different ways
        raise ModelError(f"Could not read model file {path}: {type(exc).__name__}: {exc}") from exc

    required = ("model", "scaler", "feature_order", "cached_sample")
    if not isinstance(data, dict) or not all(key in data for key in required):
        raise ModelError(f"{path} is not a LogSentinel model bundle.")
    version = data.get("format_version", 1)
    if not isinstance(version, int) or version > BUNDLE_FORMAT_VERSION:
        raise ModelError(
            f"{path} uses model bundle format {version!r}, which is newer than this version of "
            "LogSentinel supports. Upgrade LogSentinel to load it."
        )
    try:
        feature_order = tuple(data["feature_order"])
        cached_sample = np.asarray(data["cached_sample"], dtype=float)
        n_samples_seen = int(data.get("n_samples_seen") or len(cached_sample))
    except (TypeError, ValueError) as exc:
        raise ModelError(f"{path} is not a valid LogSentinel model bundle: {exc}") from exc

    if feature_order != FEATURE_ORDER:
        raise ModelError(
            f"{path} was trained on features {list(feature_order)}, but this version of LogSentinel "
            f"extracts {list(FEATURE_ORDER)}. Re-run `fit` to rebuild the model."
        )
    if not isinstance(data["model"], IsolationForest) or not hasattr(data["scaler"], "transform"):
        raise ModelError(f"{path} is not a LogSentinel model bundle.")
    if cached_sample.ndim != 2 or cached_sample.shape[1] != len(FEATURE_ORDER) or len(cached_sample) == 0:
        raise ModelError(f"{path} has a malformed cached baseline sample.")
    range_factor = data.get("range_factor", DEFAULT_RANGE_FACTOR)
    try:
        _check_range_factor(range_factor)
    except ValueError as exc:
        raise ModelError(f"{path} has an invalid range factor: {exc}") from exc

    return ModelBundle(
        model=data["model"],
        scaler=data["scaler"],
        feature_order=feature_order,
        cached_sample=cached_sample,
        n_samples_seen=n_samples_seen,
        range_bounds=_stored_range_bounds(data.get("range_bounds"), cached_sample, path),
        range_factor=float(range_factor),
    )


def _stored_range_bounds(stored: object, cached_sample, path: Path) -> dict[str, Optional[float]]:
    """The range bounds saved in a bundle. Any that are missing - all of them
    in bundles saved before LogSentinel 2.2 - are derived from the cached
    sample, which is a uniform sample of the model's baseline."""
    if stored is not None and not isinstance(stored, dict):
        raise ModelError(f"{path} has malformed range bounds.")
    stored = stored or {}
    derived = _range_bounds(cached_sample) if any(name not in stored for name in MAGNITUDE_FEATURES) else {}
    bounds: dict[str, Optional[float]] = {}
    for name in MAGNITUDE_FEATURES:
        if name not in stored:
            bounds[name] = derived[name]
            continue
        bound = stored[name]
        if bound is not None and not (_is_number(bound) and 0 <= bound < math.inf):
            raise ModelError(f"{path} has a malformed range bound for {name}: {bound!r}")
        bounds[name] = None if bound is None else float(bound)
    return bounds


def scan_stream(
    bundle: ModelBundle,
    entries_and_features: Iterable[tuple[LogEntry, RequestFeatures]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    range_factor: Optional[float] = None,
) -> Iterator[dict]:
    """Score entries in bounded-size chunks (never loading the full target
    log into memory at once) and yield one anomaly dict per flagged entry,
    in log order.

    An entry is flagged when the forest scores it as anomalous, or when one
    of its magnitude features exceeds `range_factor` (default: the model's)
    times the baseline bound. `anomaly_score` is the larger of the two
    scores, so it is above 0 exactly for flagged entries, and `reasons` says
    which check fired."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    factor = bundle.range_factor if range_factor is None else range_factor
    _check_range_factor(factor)
    _joblib, np, _IsolationForest, _StandardScaler = _require_sklearn()

    bounds = bundle.range_bounds or {}
    columns = [FEATURE_ORDER.index(name) for name in MAGNITUDE_FEATURES]
    limits = np.array([factor * bounds[name] if factor and bounds.get(name) is not None else math.inf
                       for name in MAGNITUDE_FEATURES])

    chunk: list[tuple[LogEntry, RequestFeatures]] = []

    def process(chunk_items: list[tuple[LogEntry, RequestFeatures]]) -> Iterator[dict]:
        if not chunk_items:
            return
        matrix = np.array([feat.to_vector() for _entry, feat in chunk_items], dtype=float)
        # IsolationForest.predict() == -1 is defined as decision_function() < 0;
        # testing the scores directly avoids scoring every row twice.
        forest_scores = -bundle.model.decision_function(bundle.scaler.transform(matrix))
        over = matrix[:, columns] / limits  # > 1: beyond the allowed range
        with np.errstate(divide="ignore"):  # log2(0) = -inf is fine: nowhere near the limit
            range_scores = _RANGE_SCORE_PER_DOUBLING * np.log2(over.max(axis=1))
        scores = np.maximum(forest_scores, range_scores)
        for index in np.flatnonzero(scores > 0):
            entry, feat = chunk_items[index]
            reasons = ["isolation forest"] if forest_scores[index] > 0 else []
            reasons += [
                f"{name}={getattr(feat, name)} is {getattr(feat, name) / bounds[name]:.1f}x "
                f"its baseline maximum ({bounds[name]:.4g})"
                for name, ratio in zip(MAGNITUDE_FEATURES, over[index], strict=True) if ratio > 1
            ]
            yield {
                "line_number": entry.line_number,
                "ip": entry.ip,
                "path": entry.path,
                "status": entry.status,
                # never display a flagged entry as 0.0 after rounding
                "anomaly_score": max(round(float(scores[index]), 6), 1e-06),
                "reasons": reasons,
                "features": feat.to_dict(),
                "raw_line": entry.raw_line,
            }

    for item in entries_and_features:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield from process(chunk)
            chunk = []

    yield from process(chunk)
