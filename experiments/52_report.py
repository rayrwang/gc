"""Writes 52_report.md from outputs/52_wiring_lock.json with 52's own report() (2026-09-29)."""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reporting  # noqa: E402

if __name__ == "__main__":
    m52 = importlib.import_module("52_wiring_lock")
    reporting.write("52", "52_wiring_lock.py", ["52_wiring_lock.json", "52_wiring_lock.log", "52_wiring_lock.stdout"],
                    m52.report(),
                    {"52_wiring_lock.json": "wiringlock52.json", "52_wiring_lock.log": "wiringlock52.log",
                     "52_wiring_lock.stdout": "wiringlock52.stdout"})
