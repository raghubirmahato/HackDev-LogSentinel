#!/usr/bin/env python3
"""
HackDev-LogSentinel
====================

ML-based web server log anomaly detector using Isolation Forest. Parses
Apache/Nginx combined format, JSON-lines, and generic W3C extended log
format (auto-detected, plain or gzip-compressed), extracts numeric features
from each request, and flags anomalous requests relative to a trained
baseline.

Thin CLI entrypoint - the actual implementation lives in the
hackdev_logsentinel/ package (parsers.py, features.py, model.py, alerts.py,
sanitize.py, cli.py). Installing the package (`pip install .`) also provides
a `logsentinel` command.

This is a defensive (blue-team) tool intended for analyzing logs you own or
are authorized to access.
"""
import sys

from hackdev_logsentinel.cli import main

if __name__ == "__main__":
    sys.exit(main())
