# HackDev-LogSentinel

[![Test](https://github.com/raghubirmahato/HackDev-LogSentinel/actions/workflows/test.yml/badge.svg)](https://github.com/raghubirmahato/HackDev-LogSentinel/actions/workflows/test.yml)

ML-based web server log anomaly detector using Isolation Forest. Part of the **HackDev**
cybersecurity toolkit.

LogSentinel learns what "normal" traffic looks like from a baseline access log and then flags
anomalous requests in a target log — things like SQL injection probes, path traversal attempts, XSS
payloads, or unusually shaped requests that stand out from the baseline. It uses an unsupervised
`IsolationForest` model, so no labeled attack data is required.

## Features

- **Multiple log formats, auto-detected**: Apache/Nginx **combined** format (plus plain Common Log
  Format and nginx's default `main` format), **JSON-lines** (nginx `escape=json`, Caddy, Traefik and
  Elastic Common Schema field names recognized out of the box, or a custom `--field-map`), and
  generic **W3C extended** log format (`#Fields:` header defines the columns).
- **Gzip-rotated logs** (`access.log.2.gz`) are read transparently; a UTF-8 byte-order mark is ignored.
- **Evasion-resistant parsing**: log lines are attacker-influenced input, so parsing is deliberately
  lenient — raw spaces in the URL, escaped quotes in the User-Agent, non-standard methods and missing
  request lines are all still parsed and scored instead of silently dropped. Genuinely malformed
  lines are skipped and counted, never crash the run.
- **Bounded memory throughout**: parsing is line-by-line, scanning scores rows in chunks
  (`--chunk-size`), and fitting trains on a uniform random sample of at most `--max-fit-rows` rows
  (100,000 by default), so memory doesn't grow with the size of the log. Only the anomalies
  themselves are kept for the report; add `--top N` to bound those too.
- **Incremental model updates** (`fit --update existing_model.joblib`): a bounded uniform sample of
  every baseline row the model has seen is saved alongside it. An update merges that history with the
  new data *in proportion to how many rows each represents* and refits, so the model reflects the
  whole accumulated history instead of being dominated by the latest batch. The existing model's
  `--contamination` is kept unless you override it.
- **Webhook alerting** (`scan --webhook-url` or `$LOGSENTINEL_WEBHOOK_URL`): posts a Slack-compatible
  summary. Log-derived text is escaped so a crafted request can't ping `@channel` or plant links, the
  webhook URL is never written to logs, and a failed webhook never aborts the scan.
- **Range check for extreme values**: Isolation Forest can't tell *how far* beyond the baseline a
  value is — a 300,000-character URL scores exactly like the longest URL in the baseline. So every
  model also learns the baseline's maximum URL length, parameter count and response size, and flags
  requests beyond `--range-factor` times that (2× by default), with the reason in the report.
- Per-request feature extraction: URL length, query parameter count, special-character ratio,
  Shannon entropy of the query string, response payload size, HTTP method encoded as an integer.
- Text and JSON output with a stats summary; `--top N` keeps only the highest-scoring anomalies.
- Ships with `sample_access.log`, a small example log so you can try it immediately.

## Installation

```bash
git clone https://github.com/raghubirmahato/HackDev-LogSentinel.git
cd HackDev-LogSentinel
pip install ".[webhook]"     # installs the `logsentinel` command
```

Or run it straight from the checkout without installing:

```bash
pip install -r requirements.txt
python logsentinel.py --help          # or: python -m hackdev_logsentinel --help
```

Requires Python 3.10+, `scikit-learn`, `numpy`, `joblib`. `requests` is only needed for webhook
alerting (the `webhook` extra).

## Usage

Train a baseline model (format auto-detected):

```bash
logsentinel fit sample_access.log -o baseline_model.joblib --contamination 0.1
```

Scan a target log for anomalies:

```bash
logsentinel scan sample_access.log --model baseline_model.joblib
```

```
LogSentinel scan results (format: combined)

Total: 30  Parsed: 30  Unparsed: 0  Anomalies: 3

[line 25] score=0.0590 ip=203.0.113.9 status=500 path=/products?id=1%20UNION%20SELECT%20username,password%20FROM%20users--
[line 27] score=0.0489 ip=45.33.32.156 status=404 path=/wp-admin/admin-ajax.php?action=%3Cscript%3Ealert(1)%3C/script%3E
[line 24] score=0.0111 ip=203.0.113.9 status=500 path=/products?id=1'%20OR%20'1'='1
```

Requests far larger than anything in the baseline are flagged by the range check, with the reason
shown. Neither of these is flagged by the Isolation Forest alone:

```
[line 2] score=0.0408 ip=198.51.100.23 status=200 path=/index.html  <- payload_length=26000 is 2.7x its baseline maximum (9797)
[line 1] score=0.0254 ip=198.51.100.23 status=404 path=/images/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.png  <- url_length=162 is 2.4x its baseline maximum (67.91)
```

Incrementally update an existing model with newer baseline data instead of refitting from scratch
(rewriting the model in place is safe — it is saved atomically):

```bash
logsentinel fit new_baseline_period.log --update baseline_model.joblib -o baseline_model.joblib
```

Scan a JSON-lines log with a custom field mapping (dotted keys reach into nested objects), alert
to a webhook, and write JSON output to a file:

```bash
export LOGSENTINEL_WEBHOOK_URL=https://hooks.slack.com/services/...
logsentinel scan app.jsonl --model baseline_model.joblib \
  --field-map '{"ip": "clientAddr", "path": "request.target"}' \
  --format json -o results.json
```

Force a format instead of auto-detecting, tune chunk size and report only the worst offenders in a
huge, rotated log:

```bash
logsentinel scan access.w3c.gz --model baseline_model.joblib --format-override w3c --chunk-size 5000 --top 50
```

### Example JSON scan output (abridged)

```json
{
  "stats": {
    "total_lines": 30, "parsed": 30, "unparsed": 0,
    "detected_format": "combined", "anomalies_found": 3, "anomalies_reported": 3,
    "range_factor": 2.0
  },
  "anomalies": [
    {
      "line_number": 25,
      "ip": "203.0.113.9",
      "path": "/products?id=1%20UNION%20SELECT%20username,password%20FROM%20users--",
      "status": 500,
      "anomaly_score": 0.059023,
      "reasons": ["isolation forest"],
      "features": {
        "url_length": 68, "param_count": 1, "special_char_ratio": 0.161765,
        "entropy": 4.651975, "payload_length": 480, "method_encoded": 0
      },
      "raw_line": "203.0.113.9 - - [10/Oct/2023:14:04:05 -0700] \"GET /products?id=1%20UNION..."
    }
  ]
}
```

`anomaly_score` is the larger of two scores: the negated Isolation Forest decision function, and the
range check's score (0.1 per doubling beyond the allowed range). Anything above 0 is flagged, higher
means more anomalous, and `reasons` lists which check fired: `"isolation forest"`, or e.g.
`"url_length=311 is 4.6x its baseline maximum (67.91)"`. The "baseline maximum" is the 99.9th
percentile (at least 1), so a few attacks hidden in the baseline don't raise the bar; the response-size
check is skipped when the baseline log records no sizes.

## Supported log formats

| Format | Detection | Notes |
|---|---|---|
| `combined` | default | Apache/Nginx combined, Common Log Format (no referer/User-Agent), and formats with extra trailing fields such as nginx's default `main` (`"$http_x_forwarded_for"`). |
| `json` | first lines are JSON objects | Field names are resolved from `--field-map` first, then common aliases: flat names (`ip`, `path`, `status`, ...), nginx variables (`remote_addr`, `request_uri`, `body_bytes_sent`, `http_user_agent`, or a full `request` line), Caddy (`request.uri`, `request.headers.User-Agent`, ...), Traefik (`ClientHost`, `RequestPath`, ...) and ECS (`source.ip`, `url.original`, `http.response.status_code`, ...). Records without a request path (e.g. server startup messages in the same file) are counted as unparsed rather than scored as fake `GET /` requests. |
| `w3c` | a `#Fields:` directive | IIS and other W3C extended logs; field names are case-insensitive and the header may change mid-file. |

`--field-map` keys: `ip`, `timestamp`, `method`, `path`, `status`, `size`, `referer`, `user_agent`,
`request`. If a scan cannot parse *any* line of a non-empty log, it exits with an error instead of
reporting a false "0 anomalies".

## CLI flag reference

| Flag | Subcommand | Description |
|---|---|---|
| `baseline_log_file` | `fit` | Positional path to the baseline log file (plain or gzip) |
| `target_log_file` | `scan` | Positional path to the target log file (plain or gzip) |
| `--update EXISTING_MODEL` | `fit` | Incrementally update an existing model instead of fitting from scratch |
| `-o`, `--output` | `fit`, `scan` | `fit`: model save path (default `logsentinel_model.joblib`). `scan`: optional results file |
| `--model` | `scan` | Path to a model bundle saved by `fit` (required) |
| `--contamination` | `fit` | Expected proportion of anomalies in the baseline, in (0, 0.5] (default `0.05`; with `--update`, the existing model's value) |
| `--max-fit-rows N` | `fit` | Train on a uniform random sample of at most N baseline rows (default `100000`) |
| `--range-factor X` | `fit`, `scan` | Flag URL length, parameter count or response size beyond X times the baseline maximum. `fit`: stored in the model (default `2`; with `--update`, the existing model's value). `scan`: override for this scan. `0` turns the check off |
| `--format {text,json}` | `fit`, `scan` | Output format (default `text`) |
| `--format-override {combined,json,w3c}` | `fit`, `scan` | Force a log format instead of auto-detecting |
| `--field-map JSON` | `fit`, `scan` | Field-name mapping for the JSON log format (dotted keys for nested objects) |
| `--chunk-size N` | `scan` | Rows scored per batch, bounds memory on huge logs (default `2000`) |
| `--top N` | `scan` | Report only the N highest-scoring anomalies; all are still counted |
| `--webhook-url URL` | `scan` | POST a Slack-compatible summary here on completion (default: `$LOGSENTINEL_WEBHOOK_URL`) |
| `-v`, `--verbose` | all | Enable verbose (DEBUG) logging, including tracebacks for errors; accepted before or after the subcommand |
| `--version` | — | Show version and exit |

Exit status: `0` success, `1` error (missing/corrupt files, nothing parsable), `2` invalid
arguments, `130` interrupted.

## Model compatibility

Model files from every release work with every other release:

| Model file saved by | Loaded by 2.2 | Loaded by 2.0 / 2.1 |
|---|---|---|
| 2.0 or 2.1 | Yes, with the range check — its bounds are derived from the baseline sample stored in the model | Yes |
| 2.2 | Yes | Yes, without the range check (those releases don't have it) |

`fit --update` works across releases too; no model needs to be re-fitted. (An update made with 2.0
or 2.1 doesn't carry over a custom `--range-factor`; 2.2 then uses the default.) Models are pickles of
scikit-learn objects, so load them with the same scikit-learn version they were saved with
(scikit-learn warns otherwise).

## Security notes

- **Model files are pickles.** Loading a model runs code embedded in it, so only load models you
  created yourself or otherwise trust — treat a `.joblib` file like an executable.
- **Keep webhook URLs secret.** Slack/Discord/Teams webhook URLs are credentials. Prefer
  `LOGSENTINEL_WEBHOOK_URL` over `--webhook-url`, since command-line arguments are visible to other
  local users (`ps`) and end up in shell history. LogSentinel never logs more than the webhook's host.
- **Log content is untrusted.** Request paths and IPs are shown with control characters escaped
  (no ANSI escape injection into your terminal) and are escaped for Slack in alerts.

## Project layout

```
logsentinel.py            Thin CLI entrypoint (same as `python -m hackdev_logsentinel`)
hackdev_logsentinel/
  parsers.py              Combined/JSON/W3C streaming parsers, gzip support, format auto-detection
  features.py             Numeric feature extraction
  model.py                Bounded-memory fit, incremental update, safe save/load, chunked scan
  alerts.py               Webhook payloads and posting
  sanitize.py             Escaping of untrusted log content for display
  cli.py                  argparse wiring, orchestration
tests/                    pytest suite
```

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
ruff check .
```

The suite covers each parser against realistic fixture lines (including hostile and malformed
input: evasion attempts, crafted URLs, pathological JSON, truncated gzip files), format
auto-detection, a 50,000-line synthetic log for the streaming scan, statistical checks that the
bounded-memory sampler is unbiased and that incremental updates weight history correctly, the range
check (limits, scoring, and that the forest alone scores a 69- and a 300,000-character URL
identically), model save/load validation (corrupt, incompatible and interrupted saves, and model files
in every earlier release's format), the CLI end to end (exit codes,
output formats, usage errors), and webhook posting with a mocked endpoint — including that the
webhook secret never reaches the logs. ML-dependent tests use `pytest.importorskip("sklearn")` to
skip gracefully if scikit-learn isn't installed. CI runs the suite on Python 3.10–3.14.

## Legal

LogSentinel is a defensive, blue-team tool. It is intended for analyzing web server access logs
that you own or are explicitly authorized to access and analyze. Do not use this tool against logs
or systems you do not have authorization for.
