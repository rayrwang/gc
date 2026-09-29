"""Writes 53_report.md from outputs/53_carryover_lock.json with 53's own report() (2026-09-29)."""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reporting  # noqa: E402

if __name__ == "__main__":
    m53 = importlib.import_module("53_carryover_lock")
    reporting.write("53", "53_carryover_lock.py",
                    ["53_carryover_lock.json", "53_carryover_lock.log", "53_carryover_lock.stdout"],
                    m53.report(),
                    {"53_carryover_lock.json": "carrylock53.json", "53_carryover_lock.log": "carrylock53.log",
                     "53_carryover_lock.stdout": "carrylock53.stdout"})
