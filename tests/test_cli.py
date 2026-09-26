"""End-to-end tests of the command-line interface (fit/scan orchestration,
output formats, exit codes, and error handling)."""

import gzip
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("sklearn")

import joblib  # noqa: E402

from hackdev_logsentinel import __version__, cli  # noqa: E402
from hackdev_logsentinel.cli import WEBHOOK_URL_ENV, main  # noqa: E402
from hackdev_logsentinel.features import FEATURE_ORDER  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_LOG = REPO_ROOT / "sample_access.log"
ATTACK_LINES = {24, 25, 26, 27}  # SQL injection x2, path traversal, XSS
SECRET_URL = "https://hooks.slack.com/services/T000/B000/SECRETTOKEN"


@pytest.fixture(scope="module")
def sample_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    assert main(["fit", str(SAMPLE_LOG), "-o", str(path), "--contamination", "0.1"]) == 0
    return path


def _scan_json(capsys, *argv) -> dict:
    assert main(["scan", *map(str, argv), "--format", "json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_scan_sample_log_flags_the_attacks(sample_model, capsys):
    result = _scan_json(capsys, SAMPLE_LOG, "--model", sample_model)
    assert result["stats"] == {"total_lines": 30, "parsed": 30, "unparsed": 0, "detected_format": "combined",
                               "anomalies_found": 3, "anomalies_reported": 3, "range_factor": 2.0}
    assert all(a["reasons"] == ["isolation forest"] for a in result["anomalies"])
    flagged = [a["line_number"] for a in result["anomalies"]]
    assert set(flagged) <= ATTACK_LINES
    scores = [a["anomaly_score"] for a in result["anomalies"]]
    assert scores == sorted(scores, reverse=True)
    assert all(set(a["features"]) == set(FEATURE_ORDER) for a in result["anomalies"])


def test_scan_text_output(sample_model, capsys):
    assert main(["scan", str(SAMPLE_LOG), "--model", str(sample_model)]) == 0
    out = capsys.readouterr().out
    assert "Total: 30  Parsed: 30  Unparsed: 0  Anomalies: 3" in out
    assert out.count("[line ") == 3


def test_scan_top_reports_highest_scores_but_counts_all(sample_model, capsys):
    full = _scan_json(capsys, SAMPLE_LOG, "--model", sample_model)
    top = _scan_json(capsys, SAMPLE_LOG, "--model", sample_model, "--top", "1")
    assert top["stats"]["anomalies_found"] == 3
    assert top["stats"]["anomalies_reported"] == 1
    assert top["anomalies"] == full["anomalies"][:1]


def test_scan_writes_output_file(sample_model, tmp_path, capsys):
    out_file = tmp_path / "results.json"
    printed = _scan_json(capsys, SAMPLE_LOG, "--model", sample_model, "-o", out_file)
    assert json.loads(out_file.read_text()) == printed


def test_scan_gzip_log_matches_plain(sample_model, tmp_path, capsys):
    gz_path = tmp_path / "access.log.1.gz"
    gz_path.write_bytes(gzip.compress(SAMPLE_LOG.read_bytes()))
    plain = _scan_json(capsys, SAMPLE_LOG, "--model", sample_model)
    assert _scan_json(capsys, gz_path, "--model", sample_model) == plain


def test_crafted_line_does_not_abort_the_scan(sample_model, tmp_path, capsys):
    target = tmp_path / "target.log"
    crafted = '6.6.6.6 - - [10/Oct/2023:14:09:00 -0700] "GET //[x HTTP/1.1" 404 9 "-" "x"\n'
    target.write_text(SAMPLE_LOG.read_text() + crafted)
    assert _scan_json(capsys, target, "--model", sample_model)["stats"]["parsed"] == 31


def test_scan_text_output_escapes_terminal_control_sequences(sample_model, tmp_path, capsys):
    # The SQL injection request from sample_access.log (flagged by the model),
    # with an OSC "set window title" escape sequence appended to its path.
    target = tmp_path / "evil.jsonl"
    path = "/products?id=1%20UNION%20SELECT%20username,password%20FROM%20users--\u001b]0;pwned\u0007"
    target.write_text(json.dumps({"ip": "203.0.113.9", "path": path, "status": 500, "size": 480}) + "\n")
    assert main(["scan", str(target), "--model", str(sample_model)]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out
    assert "\\x1b]0;pwned\\x07" in out


def test_fit_update_keeps_contamination_and_accumulates_rows(sample_model, tmp_path, capsys):
    updated = tmp_path / "updated.joblib"
    assert main(["fit", str(SAMPLE_LOG), "--update", str(sample_model), "-o", str(updated), "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "updated"
    assert summary["contamination"] == 0.1
    assert summary["total_baseline_rows"] == 60
    assert joblib.load(updated)["model"].contamination == 0.1


def test_fit_update_in_place(sample_model, tmp_path, capsys):
    model = tmp_path / "model.joblib"
    model.write_bytes(sample_model.read_bytes())
    assert main(["fit", str(SAMPLE_LOG), "--update", str(model), "-o", str(model), "--contamination", "0.2"]) == 0
    assert joblib.load(model)["model"].contamination == 0.2
    assert [p.name for p in tmp_path.iterdir()] == ["model.joblib"]


@pytest.mark.parametrize("argv, message", [
    (["fit", "missing.log"], "Baseline log file not found"),
    (["fit", str(SAMPLE_LOG), "--update", "missing.joblib"], "--update model file not found"),
    (["scan", str(SAMPLE_LOG), "--model", "missing.joblib"], "Model file not found"),
])
def test_missing_files_exit_1(argv, message, caplog):
    assert main(argv) == 1
    assert message in caplog.text


def test_scan_missing_target_exits_1(sample_model, caplog):
    assert main(["scan", "missing.log", "--model", str(sample_model)]) == 1
    assert "Target log file not found" in caplog.text


def test_corrupt_model_is_a_clean_error(tmp_path, caplog):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    assert main(["scan", str(SAMPLE_LOG), "--model", str(bad)]) == 1
    assert "Could not read model file" in caplog.text


def test_scan_where_nothing_parses_is_an_error_not_an_all_clear(sample_model, capsys, caplog):
    assert main(["scan", str(SAMPLE_LOG), "--model", str(sample_model), "--format-override", "json"]) == 1
    assert "None of the 30 lines" in caplog.text
    assert capsys.readouterr().out == ""


def test_fit_where_nothing_parses_is_an_error(tmp_path, caplog):
    assert main(["fit", str(SAMPLE_LOG), "-o", str(tmp_path / "m.joblib"), "--format-override", "w3c"]) == 1
    assert "None of the 30 lines" in caplog.text
    assert not (tmp_path / "m.joblib").exists()


def test_fit_empty_baseline_is_an_error(tmp_path, caplog):
    empty = tmp_path / "empty.log"
    empty.write_text("\n\n")
    assert main(["fit", str(empty), "-o", str(tmp_path / "m.joblib")]) == 1
    assert "contains no log lines" in caplog.text


def test_field_map_warning_for_non_json_logs(sample_model, capsys, caplog):
    _scan_json(capsys, SAMPLE_LOG, "--model", sample_model, "--field-map", '{"ip": "client"}')
    assert "--field-map only applies to JSON logs" in caplog.text


@pytest.mark.parametrize("extra", [
    ["--contamination", "0.9"],
    ["--contamination", "0"],
    ["--contamination", "abc"],
    ["--max-fit-rows", "0"],
    ["--range-factor", "1"],
    ["--range-factor", "-2"],
    ["--range-factor", "inf"],
    ["--range-factor", "abc"],
    ["--field-map", '["ip"]'],
    ["--field-map", '{"ipaddr": "x"}'],
    ["--field-map", '{"ip": ""}'],
    ["--field-map", "not json"],
])
def test_fit_usage_errors_fail_fast(extra, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["fit", str(SAMPLE_LOG), "-o", str(tmp_path / "m.joblib"), *extra])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("extra", [["--chunk-size", "0"], ["--top", "-1"], ["--range-factor", "0.5"],
                                   ["--webhook-url", "ftp://x/SECRETTOKEN"]])
def test_scan_usage_errors_fail_fast(extra, sample_model, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["scan", str(SAMPLE_LOG), "--model", str(sample_model), *extra])
    assert excinfo.value.code == 2
    assert "SECRETTOKEN" not in capsys.readouterr().err


def test_webhook_posted_with_summary(sample_model, capsys):
    with patch.object(cli, "post_webhook", return_value=True) as post:
        _scan_json(capsys, SAMPLE_LOG, "--model", sample_model, "--webhook-url", SECRET_URL)
    url, payload = post.call_args.args
    assert url == SECRET_URL
    assert "Anomalies: 3" in payload["text"]


def test_webhook_url_from_environment(sample_model, capsys, monkeypatch):
    monkeypatch.setenv(WEBHOOK_URL_ENV, SECRET_URL)
    with patch.object(cli, "post_webhook", return_value=True) as post:
        _scan_json(capsys, SAMPLE_LOG, "--model", sample_model)
    assert post.call_args.args[0] == SECRET_URL

    monkeypatch.setenv(WEBHOOK_URL_ENV, "")  # empty disables it
    with patch.object(cli, "post_webhook", return_value=True) as post:
        _scan_json(capsys, SAMPLE_LOG, "--model", sample_model)
    post.assert_not_called()


def test_failed_webhook_does_not_fail_the_scan(sample_model, capsys):
    with patch("hackdev_logsentinel.alerts.requests.post", side_effect=ConnectionError("down")):
        _scan_json(capsys, SAMPLE_LOG, "--model", sample_model, "--webhook-url", SECRET_URL)


@pytest.mark.parametrize("position", ["before", "after"])
def test_verbose_flag_before_or_after_subcommand(position, sample_model, capsys):
    scan = ["scan", str(SAMPLE_LOG), "--model", str(sample_model)]
    assert main(["-v", *scan] if position == "before" else [*scan, "-v"]) == 0


def test_broken_pipe_is_not_a_traceback(sample_model, monkeypatch):
    def closed_pipe(*_args, **_kwargs):
        raise BrokenPipeError

    monkeypatch.setattr(cli, "print", closed_pipe, raising=False)
    assert main(["scan", str(SAMPLE_LOG), "--model", str(sample_model)]) == 1


def test_real_broken_pipe_exits_quietly(sample_model, tmp_path):
    # Enough anomalies that the report overflows the pipe buffer, like `scan ... | head -1`.
    target = tmp_path / "noisy.log"
    target.write_text("".join(
        f'6.6.6.{i % 250} - - [10/Oct/2023:14:00:00 -0700] "GET /{"x" * (100 + i)}?id={i} HTTP/1.1" 500 9 "-" "x"\n'
        for i in range(3000)
    ))
    proc = subprocess.Popen([sys.executable, str(REPO_ROOT / "logsentinel.py"), "scan", str(target),
                             "--model", str(sample_model)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdout.readline()
    proc.stdout.close()
    stderr = proc.stderr.read().decode()
    proc.wait(timeout=120)
    assert "Traceback" not in stderr and "Exception ignored" not in stderr


@pytest.mark.parametrize("entrypoint", [["-m", "hackdev_logsentinel"], [str(REPO_ROOT / "logsentinel.py")]])
def test_entrypoints_report_version(entrypoint):
    result = subprocess.run([sys.executable, *entrypoint, "--version"], capture_output=True, text=True,
                            cwd=REPO_ROOT, check=True)
    assert result.stdout.strip() == f"logsentinel {__version__}"


def _long_path_log(tmp_path, length=311):
    target = tmp_path / "long.jsonl"
    path = "/" + "A" * (length - 1)
    target.write_text(json.dumps({"ip": "6.6.6.6", "path": path, "status": 200, "size": 2000}) + "\n")
    return target


def test_path_longer_than_anything_in_the_baseline_is_flagged(sample_model, tmp_path, capsys):
    # The demo baseline's longest URL is 68 characters; the forest alone scored this as normal.
    target = _long_path_log(tmp_path)
    result = _scan_json(capsys, target, "--model", sample_model)
    assert result["stats"]["anomalies_found"] == 1
    assert result["anomalies"][0]["reasons"] == ["url_length=311 is 4.6x its baseline maximum (67.91)"]

    assert main(["scan", str(target), "--model", str(sample_model)]) == 0
    assert "<- url_length=311 is 4.6x its baseline maximum (67.91)" in capsys.readouterr().out

    assert _scan_json(capsys, target, "--model", sample_model, "--range-factor", "0")["stats"]["anomalies_found"] == 0


def test_fit_reports_range_bounds(tmp_path, capsys):
    assert main(["fit", str(SAMPLE_LOG), "-o", str(tmp_path / "m.joblib"), "--format", "json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["range_factor"] == 2.0
    assert summary["range_bounds"] == {"url_length": pytest.approx(67.913), "param_count": 2.0,
                                       "payload_length": pytest.approx(9796.901)}


def test_range_factor_is_stored_and_inherited_by_updates(tmp_path, capsys):
    model = tmp_path / "m.joblib"
    assert main(["fit", str(SAMPLE_LOG), "-o", str(model), "--contamination", "0.1", "--range-factor", "5"]) == 0
    assert main(["fit", str(SAMPLE_LOG), "--update", str(model), "-o", str(model)]) == 0
    capsys.readouterr()  # discard the fit summaries
    assert joblib.load(model)["range_factor"] == 5.0
    target = _long_path_log(tmp_path, length=311)  # 4.6x the baseline: under 5x
    assert _scan_json(capsys, target, "--model", model)["stats"]["anomalies_found"] == 0
    assert _scan_json(capsys, target, "--model", model, "--range-factor", "4")["stats"]["anomalies_found"] == 1


def test_model_from_an_older_release_gets_the_range_check(sample_model, tmp_path, capsys):
    # A model file exactly as LogSentinel 2.0 wrote it: no metadata, no range bounds.
    current = joblib.load(sample_model)
    old = tmp_path / "v2.0.joblib"
    joblib.dump({key: current[key] for key in ("model", "scaler", "feature_order", "cached_sample")}, old)
    result = _scan_json(capsys, _long_path_log(tmp_path), "--model", old)
    assert result["anomalies"][0]["reasons"] == ["url_length=311 is 4.6x its baseline maximum (67.91)"]
    assert _scan_json(capsys, SAMPLE_LOG, "--model", old)["stats"]["anomalies_found"] == 3
