import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from hackdev_logsentinel import model as model_module  # noqa: E402
from hackdev_logsentinel.features import FEATURE_ORDER, extract_features  # noqa: E402
from hackdev_logsentinel.model import (  # noqa: E402
    BUNDLE_FORMAT_VERSION,
    DEFAULT_RANGE_FACTOR,
    EmptyBaselineError,
    ModelBundle,
    ModelError,
    fit_incremental,
    fit_model,
    load_bundle,
    save_bundle,
    scan_stream,
)
from hackdev_logsentinel.parsers import LogEntry, ParserStats, stream_log_file  # noqa: E402


def _write_baseline_log(path: Path, n: int = 200):
    lines = []
    for i in range(n):
        lines.append(
            f'10.0.0.{i % 250} - - [10/Oct/2023:13:{i % 60:02d}:00 -0700] '
            f'"GET /page{i % 10}.html HTTP/1.1" 200 {500 + i} "-" "Mozilla/5.0"\n'
        )
    path.write_text("".join(lines))


def _write_anomalous_log(path: Path):
    lines = [
        '10.0.0.1 - - [10/Oct/2023:14:00:00 -0700] "GET /page1.html HTTP/1.1" 200 500 "-" "Mozilla/5.0"\n',
        # A blatant SQLi-looking query string: long, high special-char ratio, high entropy.
        '10.0.0.99 - - [10/Oct/2023:14:00:01 -0700] '
        '"GET /login.php?id=1%27%20OR%20%271%27=%271%27--%20&x=%3Cscript%3Ealert(1)%3C%2Fscript%3E HTTP/1.1" '
        '200 90000 "-" "sqlmap/1.0"\n',
    ]
    path.write_text("".join(lines))


def test_fit_model_and_scan_flags_anomaly():
    with tempfile.TemporaryDirectory() as d:
        baseline_path = Path(d) / "baseline.log"
        target_path = Path(d) / "target.log"
        _write_baseline_log(baseline_path, n=300)
        _write_anomalous_log(target_path)

        stats = ParserStats()
        entries = stream_log_file(baseline_path, stats)
        features = (extract_features(e) for e in entries)
        bundle = fit_model(features, contamination=0.05)

        model_path = Path(d) / "model.joblib"
        save_bundle(bundle, model_path)
        loaded = load_bundle(model_path)

        target_stats = ParserStats()
        target_entries = list(stream_log_file(target_path, target_stats))
        pairs = ((e, extract_features(e)) for e in target_entries)
        anomalies = list(scan_stream(loaded, pairs, chunk_size=10))

        assert len(anomalies) >= 1
        # The huge, high-entropy SQLi-looking request should be the flagged one.
        flagged_lines = {a["line_number"] for a in anomalies}
        assert 2 in flagged_lines


def test_fit_model_raises_on_empty_baseline():
    with pytest.raises(ValueError):
        fit_model(iter([]), contamination=0.05)


def test_fit_incremental_combines_cached_and_new_data():
    with tempfile.TemporaryDirectory() as d:
        baseline_path = Path(d) / "baseline.log"
        _write_baseline_log(baseline_path, n=200)

        stats = ParserStats()
        features = (extract_features(e) for e in stream_log_file(baseline_path, stats))
        original_bundle = fit_model(features, contamination=0.05)
        original_cache_size = len(original_bundle.cached_sample)

        new_baseline_path = Path(d) / "baseline2.log"
        _write_baseline_log(new_baseline_path, n=50)
        new_stats = ParserStats()
        new_features = (extract_features(e) for e in stream_log_file(new_baseline_path, new_stats))

        updated_bundle = fit_incremental(original_bundle, new_features, contamination=0.05)
        assert updated_bundle.model is not None
        # The cache now stands for all 250 rows, and (being under the cap) holds all of them.
        assert original_cache_size == 200
        assert updated_bundle.n_samples_seen == 250
        assert len(updated_bundle.cached_sample) == 250

        # The updated model still flags the crafted SQLi/XSS request.
        target_path = Path(d) / "target.log"
        _write_anomalous_log(target_path)
        target_stats = ParserStats()
        target_entries = list(stream_log_file(target_path, target_stats))
        pairs = ((e, extract_features(e)) for e in target_entries)
        anomalies = list(scan_stream(updated_bundle, pairs))
        assert 2 in {a["line_number"] for a in anomalies}


def test_scan_handles_large_synthetic_log_streaming():
    """Correctness + reasonable performance on a large (50k-line) synthetic
    log, proving the streaming/chunked design actually scales."""
    with tempfile.TemporaryDirectory() as d:
        baseline_path = Path(d) / "baseline.log"
        _write_baseline_log(baseline_path, n=500)

        stats = ParserStats()
        features = (extract_features(e) for e in stream_log_file(baseline_path, stats))
        bundle = fit_model(features, contamination=0.02)

        large_path = Path(d) / "large.log"
        with large_path.open("w") as f:
            for i in range(50_000):
                f.write(
                    f'10.0.0.{i % 250} - - [10/Oct/2023:13:{i % 60:02d}:00 -0700] '
                    f'"GET /page{i % 10}.html HTTP/1.1" 200 {500 + (i % 100)} "-" "Mozilla/5.0"\n'
                )

        target_stats = ParserStats()
        target_entries = stream_log_file(large_path, target_stats)
        pairs = ((e, extract_features(e)) for e in target_entries)
        anomalies = list(scan_stream(bundle, pairs, chunk_size=2000))

        assert target_stats.total_lines == 50_000
        assert target_stats.unparsed_lines == 0
        assert isinstance(anomalies, list)  # completed without memory issues


# ---------------------------------------------------------------------------
# Helpers for the tests below
# ---------------------------------------------------------------------------


def _features(path: Path):
    return (extract_features(e) for e in stream_log_file(path, ParserStats()))


def _pairs(path: Path):
    return ((e, extract_features(e)) for e in stream_log_file(path, ParserStats()))


def _rows(n: int):
    """n distinct 6-feature rows whose first column is the row index."""
    return ([float(i)] * len(FEATURE_ORDER) for i in range(n))


@pytest.fixture(scope="module")
def baseline_bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("baseline") / "baseline.log"
    _write_baseline_log(path, n=300)
    return fit_model(_features(path), contamination=0.1)


# ---------------------------------------------------------------------------
# Bounded-memory sampling and proportional incremental merging
# ---------------------------------------------------------------------------


def test_uniform_sample_keeps_short_streams_whole_and_in_order():
    sample, seen = model_module._uniform_sample(_rows(123), 1000, np.random.default_rng(0))
    assert seen == 123
    assert sample[:, 0].tolist() == list(range(123))


def test_uniform_sample_is_bounded_ordered_and_unbiased():
    n, capacity, trials = 25_000, 400, 150
    inclusion = np.zeros(n)
    for trial in range(trials):
        sample, seen = model_module._uniform_sample(_rows(n), capacity, np.random.default_rng(trial))
        assert seen == n and len(sample) == capacity
        assert np.all(np.diff(sample[:, 0]) > 0)  # distinct rows, input order kept
        inclusion[sample[:, 0].astype(int)] += 1
    # Rows from every part of the stream are equally likely to be kept.
    per_decile = inclusion.reshape(10, -1).mean(axis=1) / (trials * capacity / n)
    assert np.all(np.abs(per_decile - 1) < 0.1), per_decile


def test_fit_model_caps_training_rows_but_counts_them_all(tmp_path):
    path = tmp_path / "baseline.log"
    _write_baseline_log(path, n=1500)
    bundle = fit_model(_features(path), contamination=0.05, cache_sample_size=100, max_fit_rows=500)
    assert bundle.n_samples_seen == 1500
    assert len(bundle.cached_sample) == 100
    assert bundle.model.max_samples_ == 256  # forest still trains normally on the sample


@pytest.mark.parametrize("n_old, n_new", [(1_000_000, 1_000_000), (1_000_000, 100_000), (300, 1_000_000), (200, 50)])
def test_merge_weights_history_and_new_data_by_population(n_old, n_new):
    history = np.zeros((5000, 6))
    new = np.ones((min(n_new, 100_000), 6))
    merged = model_module._merge_samples(history, n_old, new, n_new, 100_000, np.random.default_rng(0))
    assert merged[:, 0].mean() == pytest.approx(n_new / (n_old + n_new), abs=1e-3)


def test_update_preserves_contamination_unless_overridden(tmp_path, baseline_bundle):
    path = tmp_path / "more.log"
    _write_baseline_log(path, n=50)
    assert fit_incremental(baseline_bundle, _features(path)).contamination == 0.1
    assert fit_incremental(baseline_bundle, _features(path), contamination=0.02).contamination == 0.02


def test_invalid_contamination_rejected_before_reading_the_stream():
    consumed = []

    def stream():
        consumed.append(True)
        yield from ()

    for bad in (0, 0.6, -1, True, "auto"):
        with pytest.raises(ValueError, match="contamination"):
            fit_model(stream(), contamination=bad)
    assert not consumed


def test_empty_update_raises_empty_baseline_error(baseline_bundle):
    with pytest.raises(EmptyBaselineError):
        fit_incremental(baseline_bundle, iter([]))


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def test_scan_results_do_not_depend_on_chunk_size(tmp_path, baseline_bundle):
    target, anomalous = tmp_path / "target.log", tmp_path / "anomalous.log"
    _write_baseline_log(target, n=150)
    _write_anomalous_log(anomalous)
    with target.open("a") as fh:
        fh.write(anomalous.read_text())
        fh.write(f'10.0.0.7 - - [10/Oct/2023:14:00:02 -0700] "GET /{"x" * 400} HTTP/1.1" 200 500 "-" "M"\n')
    results = [list(scan_stream(baseline_bundle, _pairs(target), chunk_size=size)) for size in (1, 7, 2000)]
    assert results[0] == results[1] == results[2]
    assert results[0], "expected at least one anomaly"


def test_scan_flags_exactly_what_predict_flags(tmp_path, baseline_bundle):
    target = tmp_path / "target.log"
    _write_baseline_log(target, n=200)
    entries = list(stream_log_file(target, ParserStats()))
    matrix = np.array([extract_features(e).to_vector() for e in entries])
    predicted = baseline_bundle.model.predict(baseline_bundle.scaler.transform(matrix))
    expected = {e.line_number for e, p in zip(entries, predicted, strict=True) if p == -1}
    pure_forest = scan_stream(baseline_bundle, _pairs(target), chunk_size=64, range_factor=0)
    assert {a["line_number"] for a in pure_forest} == expected


def test_scan_boundary_matches_predict_semantics():
    # IsolationForest.predict() flags decision_function() < 0 strictly; a score of
    # exactly 0 is an inlier.
    class StubForest:
        def decision_function(self, matrix):
            return np.array([-0.1, 0.0, 0.1])[: len(matrix)]

    class Identity:
        def transform(self, matrix):
            return matrix

    bundle = ModelBundle(model=StubForest(), scaler=Identity(), feature_order=FEATURE_ORDER, cached_sample=None)
    entries = [LogEntry(i, "1.2.3.4", "", "GET", "/", "1.1", 200, 0, "", "", "") for i in (1, 2, 3)]
    flagged = list(scan_stream(bundle, ((e, extract_features(e)) for e in entries)))
    assert [(a["line_number"], a["anomaly_score"]) for a in flagged] == [(1, 0.1)]


def test_scan_rejects_non_positive_chunk_size(baseline_bundle):
    with pytest.raises(ValueError, match="chunk_size"):
        list(scan_stream(baseline_bundle, iter([]), chunk_size=0))


# ---------------------------------------------------------------------------
# Saving and loading model bundles
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_keeps_metadata(tmp_path, baseline_bundle):
    path = tmp_path / "nested" / "model.joblib"
    save_bundle(baseline_bundle, path)
    raw = joblib.load(path)
    assert raw["format_version"] == BUNDLE_FORMAT_VERSION
    loaded = load_bundle(path)
    assert loaded.n_samples_seen == 300
    assert loaded.contamination == 0.1
    assert loaded.feature_order == FEATURE_ORDER
    assert [p.name for p in path.parent.iterdir()] == ["model.joblib"]  # no temp files left behind


def test_failed_save_leaves_existing_model_intact(tmp_path, baseline_bundle, monkeypatch):
    path = tmp_path / "model.joblib"
    save_bundle(baseline_bundle, path)
    before = path.read_bytes()

    def exploding_dump(*_args, **_kwargs):
        raise KeyboardInterrupt  # e.g. Ctrl-C in the middle of `fit --update m -o m`

    monkeypatch.setattr(joblib, "dump", exploding_dump)
    with pytest.raises(KeyboardInterrupt):
        save_bundle(baseline_bundle, path)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["model.joblib"]


def test_load_bundle_from_before_metadata_was_saved(tmp_path, baseline_bundle):
    path = tmp_path / "v1.joblib"
    joblib.dump({"model": baseline_bundle.model, "scaler": baseline_bundle.scaler,
                 "feature_order": list(FEATURE_ORDER), "cached_sample": baseline_bundle.cached_sample}, path)
    loaded = load_bundle(path)
    assert loaded.n_samples_seen == len(baseline_bundle.cached_sample)


@pytest.mark.parametrize("corrupt", [
    lambda path, bundle: path.write_bytes(b"definitely not a pickle"),
    lambda path, bundle: joblib.dump(["a", "list"], path),
    lambda path, bundle: joblib.dump({"model": bundle.model}, path),
    lambda path, bundle: joblib.dump({"model": "not a forest", "scaler": bundle.scaler,
                                      "feature_order": FEATURE_ORDER, "cached_sample": bundle.cached_sample}, path),
    lambda path, bundle: joblib.dump({"model": bundle.model, "scaler": bundle.scaler,
                                      "feature_order": FEATURE_ORDER, "cached_sample": np.zeros((3, 2))}, path),
], ids=["not-a-pickle", "wrong-type", "missing-keys", "wrong-model", "bad-cache-shape"])
def test_load_bundle_rejects_invalid_files_with_model_error(tmp_path, baseline_bundle, corrupt):
    path = tmp_path / "bad.joblib"
    corrupt(path, baseline_bundle)
    with pytest.raises(ModelError):
        load_bundle(path)


def test_load_bundle_rejects_incompatible_features_and_newer_formats(tmp_path, baseline_bundle):
    base = {"model": baseline_bundle.model, "scaler": baseline_bundle.scaler,
            "feature_order": FEATURE_ORDER, "cached_sample": baseline_bundle.cached_sample}
    joblib.dump({**base, "feature_order": ("url_length", "status")}, tmp_path / "features.joblib")
    with pytest.raises(ModelError, match="Re-run `fit`"):
        load_bundle(tmp_path / "features.joblib")
    joblib.dump({**base, "format_version": BUNDLE_FORMAT_VERSION + 1}, tmp_path / "newer.joblib")
    with pytest.raises(ModelError, match="newer"):
        load_bundle(tmp_path / "newer.joblib")


# ---------------------------------------------------------------------------
# Range check: values far beyond anything in the baseline
# ---------------------------------------------------------------------------

SAMPLE_LOG = Path(__file__).resolve().parent.parent / "sample_access.log"


def _request(path="/", size=2000):
    return LogEntry(1, "6.6.6.6", "", "GET", path, "1.1", 200, size, "", "", "")


def _scan_one(bundle, entry, **kwargs):
    return list(scan_stream(bundle, [(entry, extract_features(entry))], **kwargs))


class _NeverAnomalousForest:
    contamination = 0.1

    def decision_function(self, matrix):
        return np.ones(len(matrix))


class _Identity:
    def transform(self, matrix):
        return matrix


def _range_only(bounds, factor=DEFAULT_RANGE_FACTOR):
    """A bundle whose forest never flags anything, to test the range check alone."""
    return ModelBundle(model=_NeverAnomalousForest(), scaler=_Identity(), feature_order=FEATURE_ORDER,
                       cached_sample=None, range_bounds=bounds, range_factor=factor)


@pytest.fixture(scope="module")
def demo_bundle():
    return fit_model(_features(SAMPLE_LOG), contamination=0.1)


def test_isolation_forest_alone_cannot_see_how_far_out_a_value_is(demo_bundle):
    # Why the range check exists: every split is drawn inside the training
    # range, so any url_length past the baseline's longest (68) scores the same.
    def forest_score(length):
        vector = extract_features(_request("/" + "a" * (length - 1))).to_vector()
        return -demo_bundle.model.decision_function(demo_bundle.scaler.transform([vector]))[0]

    assert forest_score(69) == forest_score(311) == forest_score(300_000)


def test_path_far_longer_than_anything_in_the_baseline_is_flagged(demo_bundle):
    (hit,) = _scan_one(demo_bundle, _request("/" + "A" * 310))
    assert hit["anomaly_score"] > 0
    assert hit["reasons"] == ["url_length=311 is 4.6x its baseline maximum (67.91)"]
    assert _scan_one(demo_bundle, _request("/" + "A" * 99)) == []  # 1.5x the baseline: within tolerance


def test_range_score_grows_with_distance(demo_bundle):
    scores = [_scan_one(demo_bundle, _request("/" + "a" * (n - 1)))[0]["anomaly_score"] for n in (200, 2_000, 20_000)]
    assert scores[0] < scores[1] < scores[2]


def test_range_limit_is_factor_times_bound():
    bundle = _range_only({"url_length": 100.0, "param_count": 1.0, "payload_length": 5000.0})
    assert _scan_one(bundle, _request("/" + "a" * 199)) == []  # 200 == 2 x 100
    (hit,) = _scan_one(bundle, _request("/" + "a" * 200))
    assert hit["reasons"] == ["url_length=201 is 2.0x its baseline maximum (100)"]
    assert 0 < hit["anomaly_score"] < 0.001
    assert _scan_one(bundle, _request("/?a=1&b=2")) == []  # 2 params == 2 x 1
    assert _scan_one(bundle, _request("/?a=1&b=2&c=3"))[0]["reasons"] == [
        "param_count=3 is 3.0x its baseline maximum (1)"]
    assert _scan_one(bundle, _request(size=10_001))[0]["reasons"] == [
        "payload_length=10001 is 2.0x its baseline maximum (5000)"]


def test_range_check_off_by_factor_zero_or_per_feature():
    bounds = {"url_length": 10.0, "param_count": 1.0, "payload_length": None}
    assert _scan_one(_range_only(bounds, factor=0), _request("/" + "a" * 5000)) == []
    bundle = _range_only(bounds)
    assert _scan_one(bundle, _request("/" + "a" * 5000))
    assert _scan_one(bundle, _request("/" + "a" * 5000), range_factor=0) == []  # per-scan override
    assert _scan_one(bundle, _request("/" + "a" * 40), range_factor=5) == []  # 41 < 5 x 10
    assert _scan_one(bundle, _request(size=10**12)) == []  # no size bound: the log never recorded sizes


def test_flagged_rows_never_show_a_zero_score():
    bundle = _range_only({"url_length": 1000.0, "param_count": 1.0, "payload_length": 1e12})
    (hit,) = _scan_one(bundle, _request(size=2 * 10**12 + 1))  # over the limit by a hair
    assert hit["anomaly_score"] == 1e-06


def test_range_bounds_learned_from_the_baseline():
    rng = np.random.default_rng(0)
    rows = np.zeros((10_000, len(FEATURE_ORDER)))
    rows[:, 0] = np.r_[rng.integers(20, 61, 9_995), [5_000] * 5]  # 5 attacks hidden in the baseline
    bounds = model_module._range_bounds(rows)
    assert bounds["url_length"] == 60.0  # the 99.9th percentile ignores the planted attacks
    assert bounds["param_count"] == 1.0  # always 0 (no query strings): floor of 1 keeps the check useful
    assert bounds["payload_length"] is None  # sizes never recorded: nothing to compare against
    rows[:, 4] = rng.integers(100, 5_000, 10_000)
    assert 4_900 < model_module._range_bounds(rows)["payload_length"] < 5_000


def test_update_recomputes_bounds_and_keeps_or_overrides_range_factor(tmp_path, baseline_bundle):
    longer = tmp_path / "longer.log"
    longer.write_text("".join(
        f'10.0.0.{i} - - [10/Oct/2023:13:00:00 -0700] '
        f'"GET /catalog/item-{i:04d}/details.html HTTP/1.1" 200 800 "-" "M"\n'
        for i in range(300)
    ))
    custom = replace(baseline_bundle, range_factor=3.0)
    updated = fit_incremental(custom, _features(longer))
    assert updated.range_factor == 3.0
    assert updated.range_bounds["url_length"] > baseline_bundle.range_bounds["url_length"]
    assert fit_incremental(custom, _features(longer), range_factor=0).range_factor == 0


def test_invalid_range_factor_rejected_before_reading_the_stream():
    for bad in (1, 0.5, -2, float("inf"), float("nan"), "2", True):
        with pytest.raises(ValueError, match="range_factor"):
            fit_model(iter([]), contamination=0.1, range_factor=bad)


# ---------------------------------------------------------------------------
# Compatibility of model files across releases
# ---------------------------------------------------------------------------


def _release_bundle(bundle, release):
    """A bundle dict exactly as LogSentinel `release` wrote it."""
    data = {"model": bundle.model, "scaler": bundle.scaler, "feature_order": list(FEATURE_ORDER),
            "cached_sample": bundle.cached_sample}
    if release == "2.1":
        data.update(format_version=2, logsentinel_version="2.1.0", n_samples_seen=bundle.n_samples_seen)
    return data


@pytest.mark.parametrize("release", ["2.0", "2.1"])
def test_models_from_older_releases_get_the_range_check(tmp_path, baseline_bundle, release):
    path = tmp_path / f"{release}.joblib"
    joblib.dump(_release_bundle(baseline_bundle, release), path)
    loaded = load_bundle(path)
    assert loaded.range_bounds == model_module._range_bounds(baseline_bundle.cached_sample)
    assert loaded.range_factor == DEFAULT_RANGE_FACTOR
    (hit,) = _scan_one(loaded, _request("/" + "a" * 1999))
    assert any(reason.startswith("url_length=2000 ") for reason in hit["reasons"])


def test_range_settings_round_trip(tmp_path, baseline_bundle):
    custom = replace(baseline_bundle, range_factor=np.float64(3.5),
                     range_bounds={"url_length": np.float64(500.0), "param_count": 3, "payload_length": None})
    save_bundle(custom, tmp_path / "m.joblib")
    loaded = load_bundle(tmp_path / "m.joblib")
    assert (loaded.range_bounds, loaded.range_factor) == (custom.range_bounds, 3.5)
    # Stored as plain Python floats: numpy scalar pickles don't load across numpy 1.x / 2.x.
    raw = joblib.load(tmp_path / "m.joblib")
    assert type(raw["range_factor"]) is float
    assert [type(raw["range_bounds"][name]) for name in ("url_length", "param_count")] == [float, float]


def test_new_models_stay_readable_by_older_releases(tmp_path, baseline_bundle):
    # 2.0 and 2.1 read only these keys, and 2.1 rejects format versions above 2.
    # The range settings are extra keys holding plain Python values, which both
    # ignore (they scan without the range check).
    save_bundle(baseline_bundle, tmp_path / "m.joblib")
    raw = joblib.load(tmp_path / "m.joblib")
    assert raw["format_version"] == 2
    assert {"model", "scaler", "feature_order", "cached_sample", "n_samples_seen"} <= set(raw)
    assert type(raw["range_factor"]) is float
    assert all(bound is None or type(bound) is float for bound in raw["range_bounds"].values())


def test_missing_bounds_are_completed_from_the_cached_sample(tmp_path, baseline_bundle):
    data = {**_release_bundle(baseline_bundle, "2.1"), "range_bounds": {"url_length": 999.0}}
    joblib.dump(data, tmp_path / "partial.joblib")
    loaded = load_bundle(tmp_path / "partial.joblib")
    derived = model_module._range_bounds(baseline_bundle.cached_sample)
    assert loaded.range_bounds == {**derived, "url_length": 999.0}


@pytest.mark.parametrize("field, value", [
    ("range_bounds", ["not", "a", "dict"]),
    ("range_bounds", {"url_length": -1.0}),
    ("range_bounds", {"url_length": float("nan")}),
    ("range_bounds", {"url_length": float("inf")}),
    ("range_bounds", {"url_length": "long"}),
    ("range_factor", 1.0),
    ("range_factor", "2"),
    ("range_factor", float("inf")),
])
def test_invalid_range_settings_in_a_model_file_are_rejected(tmp_path, baseline_bundle, field, value):
    joblib.dump({**_release_bundle(baseline_bundle, "2.1"), field: value}, tmp_path / "bad.joblib")
    with pytest.raises(ModelError):
        load_bundle(tmp_path / "bad.joblib")
