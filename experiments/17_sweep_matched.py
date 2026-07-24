"""
The 10 sweep redone under matched design (post-R-028-addendum, ledger #40):
same 216-config grid, same draw-luck rules, but every fixture runs on its
own fresh host from the identical per-seed init, so the kp1-kz differential
compares same-aged substrates and can only contain fixture content. The
old sweep's confirm differentials are artifact-contaminated and this
supersedes them.

Tiers: screen 400 (mechanical bars only), confirm 12000 matched: the
registered horizon, since the artifact's growth curve is gone and 12000 is
the only matched-validated reference point. The fanout-0 configs are the
known-zero null run through the real pipeline at the claim's horizon
(ledger #40 rule 3): they must read ~0 matched, and now actually test the
pipeline rather than the artifact's small-horizon shadow.

Memory-light via 14's run_light (trailing windows, no record retention).
Summaries-only, kinds sweep17-screen / sweep17-confirm; resume = rerun.
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
frz = importlib.import_module("14_registered_bare_freeze")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("SWEEP17_SMOKE"))
SEEDS = (0,) if SMOKE else (0, 1, 2)
SCREEN_STEPS, SCREEN_WINDOW = (60, 20) if SMOKE else (400, 100)
CONFIRM_STEPS, CONFIRM_WINDOW = (120, 40) if SMOKE else (12000, 500)
WORKERS = 4 if SMOKE else 12
STREAM_NS = "sweep17"


def run_one(job):
    tier, knobs, seed, steps, window = job
    out = {"tier": tier, "config": knobs, "seed": seed,
           "config_hash": sweep.config_hash(tier, knobs, steps)}
    margins = {}
    nan = False
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **knobs)
        host = BareHost(sweep.BareAgtXC(  # fresh host per fixture: matched
            cfg, _bare_path(f"dire_sweep17/w{os.getpid()}"), seed=seed))
        r = frz.run_light(host, fn(steps, seed=de.derive_seed(fixture, STREAM_NS, seed)),
                          window)
        margins[fixture] = r["margin"]
        nan = nan or r["nan"] or any(bool(torch.isnan(c.nr_1.actual).any())
                                     for c in host.agt.cols.values())
    out["margin_kp1"], out["margin_kz"] = margins["kp1"], margins["kz"]
    out["nan"] = nan
    out["differential"] = (margins["kp1"] - margins["kz"]
                           if None not in margins.values() and not nan else None)
    out["passed"] = (not nan) and None not in margins.values()
    return out


def stage(store, kind, jobs):
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == kind}
    todo = [j for j in jobs
            if (sweep.config_hash(j[0], j[1], j[3]), j[2]) not in done]
    print(f"{kind}: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": kind, "event_id":
                                 f"{kind}:{s['config_hash']}:s{s['seed']}", **s})
            out.append(s)
            if n % 25 == 0 or n == len(todo):
                print(f"  {kind} {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == kind:
            out.append(e)
    seen, uniq = set(), []
    for s in out:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    grid = sweep.grids()
    jobs = [(t, k, s, SCREEN_STEPS, SCREEN_WINDOW) for t, k in grid for s in SEEDS]
    screens = stage(store, "sweep17-screen", jobs)
    by_cfg = {}
    for s in screens:
        by_cfg.setdefault(s["config_hash"], []).append(s)
    passer_keys = {v[0]["config_hash"] for v in by_cfg.values()
                   if all(x["passed"] for x in v)}
    passers = [(t, k) for t, k in grid
               if sweep.config_hash(t, k, SCREEN_STEPS) in passer_keys]
    print(f"screen: {len(passers)}/{len(grid)} configs passed on all seeds", flush=True)

    jobs = [(t, k, s, CONFIRM_STEPS, CONFIRM_WINDOW) for t, k in passers for s in SEEDS]
    confirms = stage(store, "sweep17-confirm", jobs)
    by_cfg = {}
    for s in confirms:
        by_cfg.setdefault(s["config_hash"], []).append(s)

    print("\nmatched confirm (mean over seeds; reading, not ranking):", flush=True)
    print(f"{'tier':<5}{'config':<58}{'kp1':>8}{'kz':>8}{'diff':>8}{'signs':>7}")
    rows = []
    for g in by_cfg.values():
        vals = [x for x in g if x["differential"] is not None]
        if not vals:
            continue
        m1 = sum(x["margin_kp1"] for x in vals) / len(vals)
        mz = sum(x["margin_kz"] for x in vals) / len(vals)
        d = m1 - mz
        pos = sum(1 for x in vals if x["differential"] > 0)
        rows.append((vals[0]["tier"], vals[0]["config"], m1, mz, d,
                     f"{pos}/{len(vals)}"))
    for tier, knobs, m1, mz, d, signs in sorted(rows, key=lambda r: -r[4]):
        desc = ",".join(f"{k}={v}" for k, v in knobs.items())
        print(f"{tier:<5}{desc:<58}{m1:>+8.3f}{mz:>+8.3f}{d:>+8.3f}{signs:>7}")
    zeros = [(k, d) for t, k, _1, _z, d, _s in rows
             if t == "b" and k.get("input_fanout") == 0]
    bad = [(k, d) for k, d in zeros if abs(d) > 0.05]
    print(f"\nfanout-0 known zero at the claim's horizon: {len(zeros)} configs, "
          f"{'VIOLATED: ' + str(bad) if bad else 'all within |0.05|'}", flush=True)


if __name__ == "__main__":
    main()
