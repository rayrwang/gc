"""Writes 47_report.md from experiment 47's log in outputs/ (2026-09-29)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

LOG = "47_wta_chaos.log"

if __name__ == "__main__":
    reporting.log_report("47", "47_wta_chaos.py", [LOG], {LOG: "wtachaos47.log"})
