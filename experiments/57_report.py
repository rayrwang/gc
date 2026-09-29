"""Writes 57_report.md from outputs/57_keep_sweep_leak.jsonl (2026-09-29; split from
report_55to57.py, same tables)."""

import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import capture, f, jl, med, write  # noqa: E402

RAW = ["57_keep_sweep_leak.json", "57_keep_sweep_leak.jsonl", "57_keep_sweep_leak.log",
       "57_keep_sweep_leak.stdout", "57_keep_sweep_leak.stamp.json"]


def section57():
    rs = jl("57_keep_sweep_leak.jsonl")
    if not rs:
        return
    print("\n== 57:", len(rs), "cells", Counter(r["status"].split(":")[0] for r in rs))
    by = defaultdict(list)
    for r in rs:
        net, act, nm, keep, form, seed = r["tag"].split("|")
        by[(net, act, int(nm), float(keep), form)].append(r)
    KF = [(0.0, "code"), (0.25, "code"), (0.25, "leak"), (0.5, "code"), (0.5, "leak"),
          (0.75, "code"), (0.75, "leak"), (0.9, "code"), (0.9, "leak")]
    arms = ([(a, 1) for a in ("tanh", "sig4_0", "sig8_0", "step1", "tri0.3", "tri0.7", "raw", "relu", "tri1.0", "kwta4")]
            + [(a, 0) for a in ("tanh", "sig4_0", "sig8_0", "step1", "tri0.3", "tri0.7", "raw", "relu", "tri1.0", "kwta4",
                                "sign", "step0", "cstep", "wta")])
    for metric, spec in (("static", ".2f"), ("s_static", ".2f"), ("lock", ".2f"), ("rate", "+.2f"),
                         ("rate_fs", "+.2f"), ("s_level", ".2f"), ("sat", ".2f"), ("frozen", ".2f"),
                         ("c50", ".2f"), ("level", ".3g")):
        print(f"\n-- 57 {metric}: median over 4 seeds (D = all diverged); columns k0, then code/leak at 0.25, 0.5, 0.75, 0.9")
        for net in ("run39", "ex16"):
            print(f"  {net}")
            for act, nm in arms:
                cells = []
                for k, fm in KF:
                    rr = by.get((net, act, nm, k, fm), [])
                    ok = [r for r in rr if r["status"] == "ok"]
                    cells.append("    D" if rr and not ok else f(med([r.get(metric) for r in ok]), spec).rjust(5))
                print(f"    {act:7s} norm {nm}: {cells[0]} | " + " | ".join(
                    f"{cells[i]} {cells[i + 1]}" for i in range(1, 9, 2)))


if __name__ == "__main__":
    write("57", "57_keep_sweep_leak.py", RAW, capture(section57),
          {"57_keep_sweep_leak.json": "keepleak57.json", "57_keep_sweep_leak.log": "keepleak57.log",
           "57_keep_sweep_leak.stdout": "keepleak57.stdout", "57_keep_sweep_leak.stamp.json": "keepleak57.stamp.json"},
          ["- The one launch record is a test launch of 14:21 EDT that found all 1,728 jobs done and ran none; "
           "the run itself (07:23 to 08:07 EDT) predates the records."])
