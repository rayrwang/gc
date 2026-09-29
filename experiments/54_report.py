"""Writes 54_report.md from outputs/54_inhibit_settle.json and its log (2026-09-29)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reporting  # noqa: E402

RAW = ["54_inhibit_settle.json", "54_inhibit_settle.log", "54_inhibit_settle.stdout"]


def body():
    d = json.load(open(os.path.join(reporting.OUT, RAW[0])))
    keys = ["final_change_mean_last50", "settled_step_below_1e-5", "active_fraction_final",
            "mean_abs_a_final", "inhib_mean_abs_last50"]
    out = [f"n_cols {d['n_cols']}, steps {d['steps']}", "",
           "summary per arm: " + ", ".join(keys), ""]
    for arm, s in d["summary"].items():
        out.append(f"  {arm:18}" + "".join(
            f"{'-' if s.get(k) is None else format(s[k], '.3g'):>12}" for k in keys))
    out += ["", "log (mean change per step, every 10 steps):",
            open(os.path.join(reporting.OUT, RAW[1])).read()]
    return "\n".join(out)


if __name__ == "__main__":
    reporting.write("54", "54_inhibit_settle.py", RAW, body(),
                    {"54_inhibit_settle.json": "inhibitsettle54.json", "54_inhibit_settle.log": "inhibitsettle54.log",
                     "54_inhibit_settle.stdout": "inhibitsettle54.stdout"})
