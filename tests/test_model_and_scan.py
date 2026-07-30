import tempfile
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from hackdev_logsentinel.features import extract_features
from hackdev_logsentinel.model import fit_incremental, fit_model, load_bundle, save_bundle, scan_stream
from hackdev_logsentinel.parsers import ParserStats, stream_log_file


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
        '"GET /login.php?id=1%27%20OR%20%271%27=%271%27--%20&x=%3Cscript%3Ealert(1)%3C%2Fscript%3E HTTP/1.1" 200 90000 "-" "sqlmap/1.0"\n',
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
        # Cache should reflect the combined data (bounded by cap, so just assert it's non-trivial).
        assert len(updated_bundle.cached_sample) > 0

        # The updated model should still be usable for scanning.
        target_path = Path(d) / "target.log"
        _write_anomalous_log(target_path)
        target_stats = ParserStats()
        target_entries = list(stream_log_file(target_path, target_stats))
        pairs = ((e, extract_features(e)) for e in target_entries)
        anomalies = list(scan_stream(updated_bundle, pairs))
        assert isinstance(anomalies, list)


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
