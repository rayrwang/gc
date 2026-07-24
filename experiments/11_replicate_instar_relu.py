"""
Replication probe for the sweep's one draw-robust reading: the instar+relu
(ss=0.003) family showed a positive kp1-kz world-imprint differential on all
three sweep seeds (10_sweep_bare_x confirm tier), where the faithful
softhebb+triangle recipe reads zero. Question: does the sign survive being
looked at hard: 10 fresh graphs, doubled horizon, controls on the same
graphs.

Arms (all on the identical per-seed CRN graph, per the fixed-graph rule):
    instar_lr003/01/03  the family, three lr_e values
    dire_off            same substrate, E conns frozen at init: known answer
                        differential ~0 (a frozen random readout has no world
                        channel to gain)
    frozen              all learning off: same known answer
    faithful            softhebb+triangle+ss=1e-2: the sweep's recorded
                        zero-differential recipe, as the negative control

Horizons: 3000 (the sweep's confirm, for continuity) and 6000 (doubled).
Reading is threshold-shaped: per-arm sign counts over seeds, not means
alone. Summaries-only to the shared ledger (kind rep11); resume = rerun.
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import D3Store, canonical_bytes
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T
from src.agents import Dir

import hashlib

SMOKE = bool(os.environ.get("REP_SMOKE"))
SEEDS = (0,) if SMOKE else tuple(range(10))
HORIZONS = ((120, 40),) if SMOKE else ((3000, 500), (6000, 500))
WORKERS = 4 if SMOKE else 12

ARMS = [
    ("instar_lr003", dict(rule="instar", transport="relu", ss=3e-3, lr_e=0.03)),
    ("instar_lr01", dict(rule="instar", transport="relu", ss=3e-3, lr_e=0.1)),
    ("instar_lr03", dict(rule="instar", transport="relu", ss=3e-3, lr_e=0.3)),
    ("dire_off", dict(rule="instar", transport="relu", ss=3e-3, lr_e=0.1, dire=False)),
    ("frozen", dict(rule="instar", transport="relu", ss=3e-3, lr_e=0.1, frozen=True)),
    ("faithful", dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)),
]
if SMOKE:
    ARMS = ARMS[:2] + ARMS[3:4]


def arm_hash(arm, knobs, steps):
    return hashlib.sha256(canonical_bytes(
        {"arm": arm, "n_cols": sweep.N_COLS, "steps": steps, **knobs})).hexdigest()[:16]


def run_one(job):
    arm, knobs, seed, steps, window = job
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS, **knobs)
    host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_rep11/w{os.getpid()}"),
                                    seed=seed, **flags))
    records = []
    for fixture, fn in sweep.FIXTURES:
        de.run_stream_multi(host, fn(steps, seed=de.derive_seed(fixture, "rep11", seed)),
                            records.append, fixture=fixture, subject=f"rep:{arm}", seed=seed)
    nan = any(x != x for r in records for x in r["actual"])
    final_nan = any(bool(torch.isnan(c.nr_1.actual).any())
                    for c in host.agt.cols.values())
    tables = de.column_tables(records, window)
    margins = {}
    for fixture, _ in sweep.FIXTURES:
        cells = [(v["trail"], v["persist"]) for (f, _s, _c), v in tables.items()
                 if f == fixture and v["trail"] is not None and v["persist"] is not None]
        margins[fixture] = (sum(t - p for t, p in cells) / len(cells)) if cells else None
    diff = (margins["kp1"] - margins["kz"]
            if None not in margins.values() and not nan else None)
    return {"arm": arm, "config": {**knobs, **flags}, "steps": steps,
            "config_hash": arm_hash(arm, {**knobs, **flags}, steps), "seed": seed,
            "n_pred": len(host.agt.predicted), "nan": nan or final_nan,
            "margin_kp1": margins["kp1"], "margin_kz": margins["kz"],
            "differential": diff}


def main():
    store = D3Store(sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "rep11"}
    jobs = [(arm, knobs, s, steps, window)
            for arm, knobs in ARMS for s in SEEDS for steps, window in HORIZONS]
    todo = [j for j in jobs
            if (arm_hash(j[0], j[1], j[3]), j[2]) not in done]
    print(f"rep11: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "rep11", "event_id":
                                 f"rep11:{s['config_hash']}:s{s['seed']}", **s})
            results.append(s)
            if n % 10 == 0 or n == len(todo):
                print(f"  rep11 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "rep11":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)

    print("\nper-arm reading (sign counts over seeds; reading, not ranking):", flush=True)
    print(f"{'arm':<14}{'steps':>6}{'pos':>5}{'neg':>5}{'nan':>5}"
          f"{'mean kp1':>10}{'mean kz':>10}{'mean diff':>10}  per-seed diffs")
    for arm, _ in ARMS:
        for steps, _w in HORIZONS:
            g = sorted((r for r in rows if r["arm"] == arm and r["steps"] == steps),
                       key=lambda r: r["seed"])
            if not g:
                continue
            ds = [r["differential"] for r in g]
            ok = [d for d in ds if d is not None and d == d]
            pos = sum(1 for d in ok if d > 0)
            neg = sum(1 for d in ok if d < 0)
            bad = len(ds) - len(ok)
            m1 = sum(r["margin_kp1"] for r in g if r["margin_kp1"] is not None) / max(1, len(g))
            mz = sum(r["margin_kz"] for r in g if r["margin_kz"] is not None) / max(1, len(g))
            md = sum(ok) / len(ok) if ok else float("nan")
            per = " ".join(f"{d:+.3f}" if d is not None and d == d else "nan" for d in ds)
            print(f"{arm:<14}{steps:>6}{pos:>5}{neg:>5}{bad:>5}"
                  f"{m1:>+10.3f}{mz:>+10.3f}{md:>+10.3f}  {per}")
    print("\nknown answers: dire_off and frozen must sit at ~zero differential;"
          "\nfaithful is the recorded zero-differential recipe. any of them going"
          "\npositive with the family = machinery artifact, not world imprint.", flush=True)


if __name__ == "__main__":
    main()
