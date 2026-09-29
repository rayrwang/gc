"""Writes 50_report.md from experiment 50's log in outputs/ (2026-09-29)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

LOG = "50_mnist_active_lock.log"

if __name__ == "__main__":
    reporting.log_report("50", "50_mnist_active_lock.py", [LOG], {LOG: "mnistlock50.log"})
