"""
Free-play trial of the R-025 row-8 responds statistic before RW rules on it:
no run has ever computed the proposed activity-movement reading, and a
declared statistic should be seen behaving before its bar is frozen.

Proposed construction under trial: on the ts fixture (cycle -> new cycle,
6 x 2000), at each of the 5 switches take every column's trailing activity
norm over the 500 steps before vs after the switch, average the absolute
change across columns, then average over switches. Read faithful against its
frozen twin on the identical per-seed CRN graphs: the row-8 bar proposal is
faithful > twin + 3 sigma(twin seed spread). This trial reports the measured
distributions so the ruling can accept, amend, or replace the construction.

All columns' activity is read from the host directly (exam records carry
predicted columns only, which would bias the reading). Summaries-only
(kind resp13); resume = rerun. Free-play seeds 0-9; registered seeds
untouched.
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rep = importlib.import_module("11_replicate_instar_relu")
sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("RESP_SMOKE"))
SEEDS = (0,) if SMOKE else tuple(range(10))
N_CYCLES, STEPS_PER, WINDOW = (3, 120, 40) if SMOKE else (6, 2000, 500)
WORKERS = 4 if SMOKE else 12

ARMS = [
    ("faithful", dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)),
    ("frozen", dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1,
                    frozen=True)),
]


def run_one(job):
    arm, knobs, seed = job
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS, **knobs)
    host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_resp13/w{os.getpid()}"),
                                    seed=seed, **flags))
    norms = {loc: [] for loc in host.agt.cols}  # per-step actual norm, all cols
    stream = de.ts_switch(N_CYCLES, STEPS_PER, seed=de.derive_seed("ts", "resp13", seed))
    for _label, a in stream:
        host.observe(a)
        for loc, c in host.agt.cols.items():
            norms[loc].append(float(c.nr_1.actual.norm()))
    switches = [k * STEPS_PER for k in range(1, N_CYCLES)]
    per_switch = []
    for t in switches:
        deltas = [abs(sum(v[t:t + WINDOW]) / WINDOW - sum(v[t - WINDOW:t]) / WINDOW)
                  for v in norms.values() if len(v) >= t + WINDOW]
        per_switch.append(sum(deltas) / len(deltas))
    responds = sum(per_switch) / len(per_switch)
    return {"arm": arm, "config": {**knobs, **flags}, "seed": seed,
            "config_hash": rep.arm_hash(arm, {**knobs, **flags},
                                        N_CYCLES * STEPS_PER)[:16],
            "responds": responds,
            "per_switch": [round(x, 5) for x in per_switch],
            "nan": any(x != x for v in norms.values() for x in v)}


def main():
    store = D3Store(sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "resp13"}
    jobs = [(arm, knobs, s) for arm, knobs in ARMS for s in SEEDS]
    todo = [j for j in jobs
            if (rep.arm_hash(j[0], dict(j[1]), N_CYCLES * STEPS_PER)[:16], j[2])
            not in done]
    print(f"resp13: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "resp13", "event_id":
                                 f"resp13:{s['config_hash']}:s{s['seed']}", **s})
            results.append(s)
            print(f"  resp13 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "resp13":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)

    print("\nresponds trial (proposed row-8 statistic, measured):", flush=True)
    stats = {}
    for arm, _ in ARMS:
        g = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
        vals = [r["responds"] for r in g]
        m = sum(vals) / len(vals)
        sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        stats[arm] = (m, sd)
        per = " ".join(f"{v:.4f}" for v in vals)
        print(f"{arm:<10} mean {m:.4f}  sd {sd:.4f}  per-seed: {per}")
    (fm, _), (zm, zs) = stats["faithful"], stats["frozen"]
    print(f"\nproposed bar preview: faithful mean {fm:.4f} vs twin + 3 sigma = "
          f"{zm + 3 * zs:.4f}  -> {'clears' if fm > zm + 3 * zs else 'DOES NOT clear'}",
          flush=True)
    print("per-seed pairing note: identical graphs per seed, so the paired "
          "differences are the sharper reading if the pooled bar is marginal.",
          flush=True)


if __name__ == "__main__":
    main()
