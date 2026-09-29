"""Writes 46_report.md from experiment 46's log in outputs/ (2026-09-29)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

LOG = "46_wta_scale.log"

if __name__ == "__main__":
    reporting.log_report("46", "46_wta_scale.py", [LOG], {LOG: "wtascale46.log"})
