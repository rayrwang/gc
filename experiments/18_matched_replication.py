"""
Matched replication with controls for sweep17's candidates (the rep11
pattern with the #40 rules baked in). Candidates chosen on sweep17's seeds
0-2; this runs seeds 10-19, graphs no experiment has touched, so the
question is draw-generalization of matched-born readings, not design
validity.

Arms (controls share the candidate's exact graph per seed):
    raw01, raw003   softhebb/raw/beta1.0/ss1e-2 at lr_e 0.1 / 0.03: the
                    only family with positive absolute margins both
                    fixtures + 3/3 signs in sweep17
    raw_twin        same knobs frozen (one twin serves both lr_e arms)
    raw_direoff     same knobs, E frozen at init
    fan3, fan3_twin p_a=.3, p_e=.6, dist_exp=1, input_fanout=3 (+twin)
    fanb1, fanb1_twin  p_a=.15, p_e=.6, dist_exp=1, input_fanout=1 (+twin)

Reading: per family, differential minus its own twin's differential (the
honest gap), and the decomposition kp1-vs-twin-kp1 / kz-vs-twin-kz, so
"learned the cycle" separates from "collapsed on noise". Fresh host per
fixture (matched); streams namespace "rep18"; summaries-only (kind rep18);
resume = rerun.
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
rep = importlib.import_module("11_replicate_instar_relu")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("REP18_SMOKE"))
SEEDS = (10,) if SMOKE else tuple(range(10, 40))
# seeds 20-39 appended after the first pass: fan3 survived 8/10 with a
# marginal effect size (mean +0.069, sd ~0.16), so the sample extends
# before any retry event is designed around it; completed (arm, seed)
# cells resume from the ledger, so the dead families cost nothing extra
STEPS, WINDOW = (120, 40) if SMOKE else (12000, 500)
WORKERS = 4 if SMOKE else 12
NS = "rep18"

RAW = dict(rule="softhebb", transport="raw", beta=1.0, ss=1e-2, lr_e=0.1)
FAN3 = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3)
FANB1 = dict(p_a=0.15, p_e=0.6, dist_exp=1.0, input_fanout=1)

ARMS = [
    # raw family and fanb1 retired after the 10-19 pass (raw = linear-
    # oscillator regime, any frozen readout predicts it, twin unstable;
    # fanb1 gap +0.004): their 10-19 ledger entries still feed the report
    ("fan3", dict(FAN3)),
    ("fan3_twin", dict(FAN3, frozen=True)),
]
ALL_ARMS = [
    ("raw01", dict(RAW)),
    ("raw003", dict(RAW, lr_e=0.03)),
    ("raw_twin", dict(RAW, frozen=True)),
    ("raw_direoff", dict(RAW, dire=False)),
    ("fan3", dict(FAN3)),
    ("fan3_twin", dict(FAN3, frozen=True)),
    ("fanb1", dict(FANB1)),
    ("fanb1_twin", dict(FANB1, frozen=True)),
]
FAMILIES = [("raw01", "raw_twin"), ("raw003", "raw_twin"),
            ("fan3", "fan3_twin"), ("fanb1", "fanb1_twin")]


def run_one(job):
    arm, knobs, seed = job
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    margins, nan = {}, False
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **knobs)
        host = BareHost(sweep.BareAgtXC(  # fresh host per fixture: matched
            cfg, _bare_path(f"dire_rep18/w{os.getpid()}"), seed=seed, **flags))
        r = frz.run_light(host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                          WINDOW)
        margins[fixture] = r["margin"]
        nan = nan or r["nan"]
    return {"arm": arm, "config": {**knobs, **flags}, "seed": seed,
            "config_hash": rep.arm_hash(arm, {**knobs, **flags}, STEPS),
            "margin_kp1": margins["kp1"], "margin_kz": margins["kz"], "nan": nan,
            "differential": (margins["kp1"] - margins["kz"]
                             if None not in margins.values() and not nan else None)}


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "rep18"}
    jobs = [(arm, knobs, s) for arm, knobs in ARMS for s in SEEDS]
    todo = [j for j in jobs
            if (rep.arm_hash(j[0], dict(j[1]), STEPS), j[2]) not in done]
    print(f"rep18: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "rep18", "event_id":
                                 f"rep18:{s['config_hash']}:s{s['seed']}", **s})
            results.append(s)
            if n % 10 == 0 or n == len(todo):
                print(f"  rep18 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "rep18":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)

    def stats(arm):
        g = [r for r in rows if r["arm"] == arm]
        ds = [r["differential"] for r in g if r["differential"] is not None]
        m1 = [r["margin_kp1"] for r in g if r["margin_kp1"] is not None]
        mz = [r["margin_kz"] for r in g if r["margin_kz"] is not None]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        return {"n": len(g), "nan": len(g) - len(ds), "diff": mean(ds),
                "pos": sum(1 for d in ds if d > 0), "kp1": mean(m1), "kz": mean(mz),
                "per": " ".join(f"{d:+.3f}" for d in sorted(
                    (r["differential"] for r in g if r["differential"] is not None)))}

    print("\nper-arm (matched, fresh graphs):", flush=True)
    print(f"{'arm':<12}{'kp1':>8}{'kz':>8}{'diff':>8}{'signs':>7}{'nan':>5}  per-seed diffs")
    for arm, _ in ALL_ARMS:
        s = stats(arm)
        print(f"{arm:<12}{s['kp1']:>+8.3f}{s['kz']:>+8.3f}{s['diff']:>+8.3f}"
              f"{s['pos']:>4}/{s['n'] - s['nan']:<2}{s['nan']:>5}  {s['per']}")
    print("\nfamily readings vs own twin (honest gap + decomposition):", flush=True)
    for arm, twin in FAMILIES:
        a, t = stats(arm), stats(twin)
        print(f"{arm:<12} gap {a['diff'] - t['diff']:+.4f}   "
              f"kp1 vs twin {a['kp1'] - t['kp1']:+.4f}   "
              f"kz vs twin {a['kz'] - t['kz']:+.4f}"
              f"   ({'kp1-side' if abs(a['kp1'] - t['kp1']) > abs(a['kz'] - t['kz']) else 'kz-side'})")


if __name__ == "__main__":
    main()
