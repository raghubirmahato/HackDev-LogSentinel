# HackDev-LogSentinel

[![Test](https://github.com/raghubirrajmahato15/HackDev-LogSentinel/actions/workflows/test.yml/badge.svg)](https://github.com/raghubirrajmahato15/HackDev-LogSentinel/actions/workflows/test.yml)

ML-based web server log anomaly detector using Isolation Forest. Part of the **HackDev**
cybersecurity toolkit.

LogSentinel learns what "normal" traffic looks like from a baseline access log and then flags
anomalous requests in a target log — things like SQL injection probes, path traversal attempts, XSS
payloads, or unusually shaped requests that stand out from the baseline. It uses an unsupervised
`IsolationForest` model, so no labeled attack data is required.

## Features

- **Multiple log formats, auto-detected**: Apache/Nginx **combined** format, **JSON-lines** (one
  JSON object per line, with a configurable `--field-map`), and generic **W3C extended** log format
  (`#Fields:` header defines the columns). Malformed/unparsable lines are skipped and counted, never
  crash the run.
- **Streaming throughout**: both parsing and scanning process the log line-by-line / in bounded
  chunks (`--chunk-size`), never loading a large file fully into memory — verified against a
  50,000-line synthetic log in the test suite.
- **Incremental model updates** (`fit --update existing_model.joblib`): `IsolationForest` doesn't
  support true incremental fit, so a bounded random sample of the baseline's feature vectors is
  cached alongside the saved model; a later `--update` combines that cache with newly-provided
  baseline data and refits from the combined sample, instead of only ever fitting from scratch.
- **Webhook alerting** (`scan --webhook-url`): posts a Slack-compatible summary of anomalies found;
  a failed webhook is logged and never aborts the scan.
- Per-request feature extraction: URL length, query parameter count, special-character ratio,
  Shannon entropy of the query string, response payload size, HTTP method encoded as an integer.
- Text and JSON output, with a stats summary (total/parsed/unparsed lines, detected format,
  anomalies found).
- Ships with `sample_access.log`, a small example log so you can try it immediately.

## Installation

```bash
git clone https://github.com/raghubirrajmahato15/HackDev-LogSentinel.git
cd HackDev-LogSentinel
pip install -r requirements.txt
```

Requires Python 3.10+, `scikit-learn`, `numpy`, `joblib`. `requests` is only needed for
`--webhook-url` alerting.

## Usage

Train a baseline model (format auto-detected):

```bash
python logsentinel.py fit sample_access.log -o baseline_model.joblib --contamination 0.1
```

Scan a target log for anomalies:

```bash
python logsentinel.py scan sample_access.log --model baseline_model.joblib
```

Incrementally update an existing model with newer baseline data instead of refitting from scratch:

```bash
python logsentinel.py fit new_baseline_period.log --update baseline_model.joblib -o baseline_model.joblib
```

Scan a JSON-lines log with a custom field mapping, alert to a webhook, JSON output to a file:

```bash
python logsentinel.py scan app.jsonl --model baseline_model.joblib \
  --field-map '{"ip": "clientAddr", "path": "requestPath"}' \
  --webhook-url https://hooks.slack.com/services/... \
  --format json -o results.json
```

Force a format instead of auto-detecting, tune chunk size for a huge log:

```bash
python logsentinel.py scan access.w3c --model baseline_model.joblib --format-override w3c --chunk-size 5000
```

### Example JSON scan output (abridged)

```json
{
  "stats": {
    "total_lines": 30, "parsed": 30, "unparsed": 0,
    "detected_format": "combined", "anomalies_found": 4
  },
  "anomalies": [
    {
      "line_number": 25,
      "ip": "203.0.113.9",
      "path": "/products?id=1 UNION SELECT username,password FROM users--",
      "status": 500,
      "anomaly_score": 0.187432,
      "features": {
        "url_length": 60, "param_count": 1, "special_char_ratio": 0.216667,
        "entropy": 3.845, "payload_length": 480, "method_encoded": 0
      }
    }
  ]
}
```

## CLI flag reference

| Flag | Subcommand | Description |
|---|---|---|
| `baseline_log_file` | `fit` | Positional path to the baseline log file |
| `target_log_file` | `scan` | Positional path to the target log file |
| `--update EXISTING_MODEL` | `fit` | Incrementally update an existing model instead of fitting from scratch |
| `-o`, `--output` | `fit`, `scan` | `fit`: model save path (default `logsentinel_model.joblib`). `scan`: optional results file |
| `--model` | `scan` | Path to a model bundle saved by `fit` (required) |
| `--contamination` | `fit` | Expected proportion of anomalies in the baseline data (default `0.05`) |
| `--format {text,json}` | `fit`, `scan` | Output format (default `text`) |
| `--format-override {combined,json,w3c}` | `fit`, `scan` | Force a log format instead of auto-detecting |
| `--field-map JSON` | `fit`, `scan` | Field-name mapping for the JSON log format |
| `--chunk-size N` | `scan` | Rows scored per batch, bounds memory on huge logs (default `2000`) |
| `--webhook-url URL` | `scan` | POST a Slack-compatible summary here on completion |
| `-v`, `--verbose` | all | Enable verbose (DEBUG) logging |
| `--version` | — | Show version and exit |

## Project layout

```
logsentinel.py               Thin CLI entrypoint
hackdev_logsentinel/
  parsers.py                   Combined/JSON/W3C streaming parsers + format auto-detection
  features.py                    Numeric feature extraction
  model.py                        Fit, incremental update, streaming chunked scan
  alerts.py                        Webhook posting
  cli.py                            argparse wiring, orchestration
tests/                        pytest suite
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

Covers each parser against realistic fixture lines (including malformed-line handling), format
auto-detection per fixture file, a 50,000-line synthetic log proving the streaming design holds up
(consistent line/anomaly counts, no memory blowup), incremental fit producing a model that still
flags a known-anomalous crafted line, and webhook posting with a mocked HTTP endpoint (success and
failure cases — a failed webhook never fails the scan). ML-dependent tests use
`pytest.importorskip("sklearn")` to skip gracefully if scikit-learn isn't installed.

## Legal

LogSentinel is a defensive, blue-team tool. It is intended for analyzing web server access logs
that you own or are explicitly authorized to access and analyze. Do not use this tool against logs
or systems you do not have authorization for.
