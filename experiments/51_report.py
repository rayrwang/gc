"""Writes 51_report.md from experiment 51's log in outputs/ (2026-09-29)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

LOG = "51_mnist_lock_2x2.log"

if __name__ == "__main__":
    reporting.log_report("51", "51_mnist_lock_2x2.py", [LOG], {LOG: "mnistlock51.log"})
