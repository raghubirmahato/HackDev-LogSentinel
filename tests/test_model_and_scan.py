import tempfile
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
    results =[list(scan_stream(baseline_bundle, _pairs(target), chunk_size=size)) for size in (1, 7, 2000)]
    assert results[0] == results[1] == results[2]
    assert results[0], "expected at least one anomaly"


def test_scan_flags_exactly_what_predict_flags(tmp_path, baseline_bundle):
    target = tmp_path / "target.log"
    _write_baseline_log(target, n=200)
    entries = list(stream_log_file(target, ParserStats()))
    matrix = np.array([extract_features(e).to_vector() for e in entries])
    predicted = baseline_bundle.model.predict(baseline_bundle.scaler.transform(matrix))
    expected = {e.line_number for e, p in zip(entries, predicted, strict=True) if p == -1}
    flagged = {a["line_number"] for a in scan_stream(baseline_bundle, _pairs(target), chunk_size=64)}
    assert flagged == expected


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
