"""
R-020 freeze computation: consumes the registered band-freeze run from the D3
store, computes the measured constants per the registered formulas, checks
E1-E5, and writes dire_frozen_constants.json. Run once per the entry; the
output file's hash goes into the verdict entry.

Usage: r020_freeze.py <run_id>
"""

import json
import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.d3 import D3Store

WINDOW = 50  # registered settle window (R-020 row 7)
FAMILIES = {"pc": "frozen:pc", "wake_sleep": "frozen:wake_sleep", "gc_ff": "frozen:gc_ff"}
CLASSES = {"cycle": "kp1", "constant": "kp0", "noise": "kz"}


def trailing_means(records, fixture, subject, label=None):
    out = []
    seeds = sorted({r["seed"] for r in records})
    for seed in seeds:
        xs = [r["score"] for r in records
              if r["fixture"] == fixture and r["subject"] == subject
              and r["seed"] == seed and (label is None or r["label"] == label)
              and r["score"] is not None]
        out.append(mean(xs[-WINDOW:]))
    return out


def main(run_id):
    store = D3Store(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "d3_store"))
    manifest, records = store.load_run(run_id)
    assert all(store.verify(run_id).values()), "integrity failure"
    seeds = manifest["config"]["seeds"]
    assert seeds == [101, 211, 307, 401, 503, 601, 701, 811], "wrong seeds for R-020"

    frozen = {"source_run": run_id, "seeds": seeds, "bars": {}, "kz_bands": {}}
    floor_cells = {}
    checks = {}

    for fam, twin in FAMILIES.items():
        # kp1 / kp0 bars: max(N-copy, own twin) trailing mean + 3 x pooled seed-SD
        for fixture in ("kp1", "kp0"):
            tw = trailing_means(records, fixture, twin)
            arms = [tw] if fixture == "kp0" else [tw, trailing_means(records, fixture, "null:copy_last")]
            pooled = [v for arm in arms for v in arm]
            bar = max(mean(a) for a in arms) + 3 * stdev(pooled)
            frozen["bars"][f"{fam}/{fixture}"] = round(bar, 4)
        # kz band: own twin's kz mean +/- 3 x seed-SD
        kz = trailing_means(records, "kz", twin)
        frozen["kz_bands"][fam] = [round(mean(kz) - 3 * stdev(kz), 4),
                                   round(mean(kz) + 3 * stdev(kz), 4)]
        # plateau floor cells: |offset| + 3 x SD per fixture class
        for cls, fixture in CLASSES.items():
            tm = trailing_means(records, fixture, twin)
            floor_cells[f"{fam}/{cls}"] = abs(mean(tm)) + 3 * stdev(tm)

    # R-021 amendment (RW, option A): per-family x fixture-class floors; the
    # global max conflated chance (noise) with structural alignment (constant/
    # cycle) and fired E5 on the first registered run
    frozen["plateau_floors"] = {k: round(v, 4) for k, v in floor_cells.items()}

    # E-checks (E5 recalibrated per R-021: noise cells are chance and stay
    # under 0.2; constant/cycle cells are structural alignment and must only
    # leave room for learners, under 0.9)
    for fam, twin in FAMILIES.items():
        lo, hi = frozen["kz_bands"][fam]
        m = mean(trailing_means(records, "kz", twin))
        checks[f"E1/{fam}"] = lo <= m <= hi
        checks[f"E3/{fam}"] = frozen["bars"][f"{fam}/kp1"] < 0.9
        checks[f"E4/{fam}"] = frozen["bars"][f"{fam}/kp1"] > mean(trailing_means(records, "kp1", twin))
        checks[f"E5/{fam}/noise"] = floor_cells[f"{fam}/noise"] <= 0.2
        checks[f"E5/{fam}/cycle"] = floor_cells[f"{fam}/cycle"] < 0.9
        checks[f"E5/{fam}/constant"] = floor_cells[f"{fam}/constant"] < 0.9
    sw_pre = [r["score"] for r in records if r["fixture"] == "sw"
              and r["subject"] == "null:copy_last" and r["label"] == "pre" and r["score"] is not None]
    plateau = mean(sw_pre[-WINDOW:])
    checks["E2"] = abs(plateau) < min(v for k, v in floor_cells.items() if k.endswith("/cycle"))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dire_frozen_constants.json")
    with open(out, "w") as f:
        json.dump(frozen, f, indent=1, sort_keys=True)
    print(json.dumps(frozen, indent=1, sort_keys=True))
    print("\nE-checks:")
    for k, v in sorted(checks.items()):
        print(f"  {k}: {'pass' if v else 'FAIL'}")
    print("\nverdict:", "PASS (constants frozen)" if all(checks.values())
          else "COMMISSIONING INCIDENT (see failing checks)")
    return all(checks.values())


if __name__ == "__main__":
    main(sys.argv[1])
