"""Writes 48_report.md from experiment 48's log in outputs/ (2026-09-29)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

LOG = "48_wta_variants.log"

if __name__ == "__main__":
    reporting.log_report("48", "48_wta_variants.py", [LOG], {LOG: "wtavar48.log"})
