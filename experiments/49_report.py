"""Writes 49_report.md from experiment 49's two logs in outputs/ (2026-09-29)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

LOG = "49_carryover_lyapunov_check.log"
FIRST = "49_carryover_lyapunov_check_run1_actual_only.log"

if __name__ == "__main__":
    reporting.log_report(
        "49", "49_carryover_lyapunov_check.py", [LOG, FIRST],
        {LOG: "lyapcheck49.log", FIRST: "lyapcheck49_run1_actual_only.log"},
        ["- The second log is the script's first run, whose twin displaced the activity but not "
         "the accumulator; the docstring says why that was replaced. The results quote the first log."])
