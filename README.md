# HackDev-LogSentinel

ML-based web server log anomaly detector using Isolation Forest.

LogSentinel is a command-line tool that learns what "normal" traffic looks like from a
baseline Apache/Nginx access log (combined format) and then flags anomalous requests in
a target log file — things like SQL injection probes, path traversal attempts, XSS
payloads, or unusually shaped requests that stand out from the baseline. It uses an
unsupervised `IsolationForest` model, so no labeled attack data is required: you only
need a log file that represents typical, legitimate traffic.

## Features

- Regex parser for the standard Apache/Nginx **combined** log format, with graceful
  handling of malformed/unparsable lines (skipped and counted, with a warning summary)
- Per-request feature extraction: URL length, query parameter count, special-character
  ratio in path+query, Shannon entropy of the query string, response payload size, and
  HTTP method encoded as an integer
- `fit` subcommand: trains a `StandardScaler` + `IsolationForest` model on a baseline log
  and saves it to disk with `joblib`
- `scan` subcommand: scores a target log against a saved model and reports anomalous
  requests with their raw feature values and an anomaly score
- Text and JSON output modes, with a stats summary (total lines, parsed, anomalies found)
- Configurable contamination rate for tuning sensitivity
- Ships with `sample_access.log`, a ~30-line example log (mostly normal traffic, plus a
  couple of SQLi/XSS/path-traversal-looking requests) so you can try it immediately

## Installation

```bash
git clone https://github.com/raghubirrajmahato15/HackDev-LogSentinel.git
cd HackDev-LogSentinel
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

Train a baseline model from a known-good log file:

```bash
python logsentinel.py fit sample_access.log -o baseline_model.joblib --contamination 0.1
```

Scan a target log file for anomalies using that model:

```bash
python logsentinel.py scan sample_access.log --model baseline_model.joblib
```

Get JSON output and also write it to a file:

```bash
python logsentinel.py scan sample_access.log --model baseline_model.joblib --format json -o results.json
```

Verbose logging (useful for debugging parse issues):

```bash
python logsentinel.py -v fit sample_access.log -o baseline_model.joblib
```

### Example JSON scan output (abridged)

```json
{
  "stats": {
    "total_lines": 30,
    "parsed": 30,
    "unparsed": 0,
    "anomalies_found": 4
  },
  "anomalies": [
    {
      "line_number": 25,
      "ip": "203.0.113.9",
      "path": "/products?id=1 UNION SELECT username,password FROM users--",
      "status": 500,
      "anomaly_score": 0.187432,
      "features": {
        "url_length": 60,
        "param_count": 1,
        "special_char_ratio": 0.216667,
        "entropy": 3.845,
        "payload_length": 480,
        "method_encoded": 0
      }
    }
  ]
}
```

## CLI flag reference

| Flag | Subcommand | Description | Default |
|---|---|---|---|
| `baseline_log_file` | `fit` | Positional path to the baseline access log file | required |
| `target_log_file` | `scan` | Positional path to the target access log file | required |
| `-o`, `--output` | `fit`, `scan` | `fit`: path to save the model bundle. `scan`: optional path to also write results | `logsentinel_model.joblib` (fit) / none (scan) |
| `--model` | `scan` | Path to a model+scaler bundle saved by `fit` | required |
| `--contamination` | `fit` | Expected proportion of anomalies in the baseline data | `0.05` |
| `--format` | `fit`, `scan` | Output format: `text` or `json` | `text` |
| `-v`, `--verbose` | all | Enable verbose (DEBUG) logging | off |
| `-h`, `--help` | all | Show help and exit | — |

## Legal

LogSentinel is a defensive, blue-team tool. It is intended for analyzing web server
access logs that you own or are explicitly authorized to access and analyze (for
example, your own infrastructure, or systems you have written permission to assess).
Do not use this tool against logs or systems you do not have authorization for.
