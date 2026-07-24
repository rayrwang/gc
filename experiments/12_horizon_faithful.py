"""
Horizon curve for the faithful recipe's world-imprint differential
(11_replicate_instar_relu found softhebb+triangle goes 9/10-positive at 6000
steps after reading zero at 2000-3000; 6000 was one doubling, not a measured
scale). Arms: faithful and its frozen twin on the identical per-seed CRN
graphs, horizons 1500/3000/6000/12000, 10 free-play seeds. The reading that
feeds the freeze page's length ruling: where faithful-minus-frozen stops
growing. Registered seeds stay untouched: 0-9 are free-play draws, reused
deliberately so the graphs match rep11's. Summaries-only (kind hz12);
resume = rerun.
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rep = importlib.import_module("11_replicate_instar_relu")
sweep = importlib.import_module("10_sweep_bare_x")
from experiments.d3 import D3Store

SMOKE = bool(os.environ.get("HZ_SMOKE"))
SEEDS = (0,) if SMOKE else tuple(range(10))
HORIZONS = ((120, 40),) if SMOKE else ((1500, 500), (3000, 500), (6000, 500),
                                       (12000, 500), (24000, 500))
# 24000 appended after the first pass: the gap was still growing at 12000
# (+0.098 -> +0.124), so the saturation criterion needed one more doubling;
# completed horizons resume from the ledger untouched
WORKERS = 4 if SMOKE else 12

ARMS = [
    ("faithful", dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)),
    ("frozen", dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1,
                    frozen=True)),
]


def main():
    store = D3Store(sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "hz12"}
    jobs = [(arm, knobs, s, steps, window)
            for arm, knobs in ARMS for s in SEEDS for steps, window in HORIZONS]
    todo = [j for j in jobs if (rep.arm_hash(j[0], j[1], j[3]), j[2]) not in done]
    print(f"hz12: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(rep.run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "hz12", "event_id":
                                 f"hz12:{s['config_hash']}:s{s['seed']}", **s})
            results.append(s)
            if n % 10 == 0 or n == len(todo):
                print(f"  hz12 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "hz12":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)

    print("\nhorizon curve (sign counts over seeds; reading, not ranking):", flush=True)
    print(f"{'arm':<10}{'steps':>7}{'pos':>5}{'neg':>5}{'nan':>5}{'mean diff':>11}"
          "  per-seed diffs")
    curve = {}
    for arm, _ in ARMS:
        for steps, _w in HORIZONS:
            g = sorted((r for r in rows if r["arm"] == arm and r["steps"] == steps),
                       key=lambda r: r["seed"])
            if not g:
                continue
            ds = [r["differential"] for r in g]
            ok = [d for d in ds if d is not None and d == d]
            md = sum(ok) / len(ok) if ok else float("nan")
            curve[(arm, steps)] = md
            per = " ".join(f"{d:+.3f}" if d is not None and d == d else "nan" for d in ds)
            print(f"{arm:<10}{steps:>7}{sum(1 for d in ok if d > 0):>5}"
                  f"{sum(1 for d in ok if d < 0):>5}{len(ds) - len(ok):>5}"
                  f"{md:>+11.3f}  {per}")
    print("\nfaithful minus frozen twin (the honest gap) per horizon:", flush=True)
    for steps, _w in HORIZONS:
        f, z = curve.get(("faithful", steps)), curve.get(("frozen", steps))
        if f is not None and z is not None:
            print(f"  {steps:>6}: {f - z:+.3f}")
    print("reading for the freeze page: the registered horizon wants the "
          "smallest step count after which this gap stops growing.", flush=True)


if __name__ == "__main__":
    main()
