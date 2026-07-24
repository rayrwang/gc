"""
Channel-3 conflict-detector calibration (maturation step 1, per RW's
sequencing ruling and the provisional-acceptance policy: this is the
initial calibration event; later retunes are new events with new hashes).

Deployment shape: one detector instance per predicted column, calibrated
on that column's own residual stream after a declared burn-in. Streams are
regenerated deterministically (faithful + twin arms, seeds 20-22, kp1/kz/
ts at 12000, namespace "cal19"), per-column score series kept as plain
floats.

Known answers (planted by construction, fixed before tuning):
    ts   switches at 2000k: columns should OPEN within 400 steps of a
         switch and close before the next one; opens elsewhere = false
    kp1  after settling, quiet: post-2000 event rate ~ 0
    kz   noise is the world being noise: calibration should absorb it,
         low event rate
    twin frozen dynamics: near-zero events everywhere (instrument null)

Grid: k x dwell x cal_window (18 cells), burn-in fixed at 500 (declared).
Selection is threshold-shaped: the scorecard prints per cell; the chosen
cell is the provisional default (k=3, dwell=50, cal=500) if it sits in the
passing region, else the nearest passing cell: never the argmax of any
single statistic. Grid events go to a scratch store; the calibration
record (scorecard + chosen config hash) is one ledger entry (kind cal19)
in the real store.
"""

import importlib
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import ConflictOpenedDetector, D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("CAL_SMOKE"))
SEEDS = (20,) if SMOKE else (20, 21, 22)
STEPS, TS_CYCLES, TS_STEPS = (600, 3, 200) if SMOKE else (12000, 6, 2000)
BURN_IN = 100 if SMOKE else 500
SWITCH_WINDOW = 400
WORKERS = 3 if SMOKE else 6  # rep18 may still hold half the cores
NS = "cal19"
FAITHFUL = dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)
GRID = [dict(k=k, enter_dwell=d, exit_dwell=d, cal_window=c)
        for k in (2.0, 3.0, 4.0) for d in (25, 50, 100) for c in (300, 500)]
DEFAULT = dict(k=3.0, enter_dwell=50, exit_dwell=50, cal_window=500)


def gen_series(job):
    """regenerate one run, return per-column residual (score) series."""
    arm, seed, fixture = job
    knobs = dict(FAITHFUL)
    frozen = arm == "twin"
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **knobs)
    host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_cal19/w{os.getpid()}"),
                                    seed=seed, frozen=frozen))
    if fixture == "ts":
        stream = de.ts_switch(TS_CYCLES, TS_STEPS, seed=de.derive_seed("ts", NS, seed))
    elif fixture == "kp1":
        stream = de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", NS, seed))
    else:
        stream = de.kz_noise(STEPS, seed=de.derive_seed("kz", NS, seed))
    series = {loc: [] for loc in host.agt.predicted}
    prev_guess = None
    for item in stream:
        a = item[1] if isinstance(item, tuple) else item
        host.observe(a)
        acts = host.actuals()
        if prev_guess is not None:
            for loc, g in prev_guess.items():
                sc = de.score(g, acts[loc])
                series[loc].append(sc if sc is not None else 0.0)
        prev_guess = host.guesses()
    return {"arm": arm, "seed": seed, "fixture": fixture,
            "series": {f"{l[0]},{l[1]}": v for l, v in series.items()}}


def evaluate(run, cfg, scratch):
    """feed every column stream through a fresh detector; return counts."""
    n_switches = TS_CYCLES - 1
    switch_ts = [k * TS_STEPS for k in range(1, TS_CYCLES)]
    opens_at_switch = set()
    false_opens = opens = closes = 0
    fed = 0
    for col, xs in run["series"].items():
        det = ConflictOpenedDetector(scratch, cfg, "cal", col)
        for t, x in enumerate(xs[BURN_IN:], start=BURN_IN):
            det.feed(x)
        fed += max(0, len(xs) - BURN_IN)
        for e in det.events:
            t = e["t"] - 1 + BURN_IN  # feed index (1-based) -> stream step
            if e["kind"] == "conflict-opened":
                opens += 1
                hit = next((s for s in switch_ts if s <= t <= s + SWITCH_WINDOW), None)
                if run["fixture"] == "ts" and hit is not None:
                    opens_at_switch.add((run["seed"], col, hit))
                else:
                    false_opens += 1
            else:
                closes += 1
    return {"opens": opens, "closes": closes, "false_opens": false_opens,
            "col_switch_hits": len(opens_at_switch),
            "max_hits": len(run["series"]) * n_switches,
            "col_steps": fed}


def main():
    real = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    scratch = D3Store(tempfile.mkdtemp(prefix="cal19_"))
    jobs = [(arm, s, fx) for arm in ("faithful", "twin") for s in SEEDS
            for fx in ("ts", "kp1", "kz")]
    runs = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(gen_series, j) for j in jobs]
        for n, fut in enumerate(futs and as_completed(futs), 1):
            runs.append(fut.result())
            print(f"  series {n}/{len(jobs)}", flush=True)

    print(f"\nscorecard ({len(GRID)} cells; ts hit = cols opening within "
          f"{SWITCH_WINDOW} of a switch / all col-switch pairs; rates per 1k col-steps):",
          flush=True)
    print(f"{'k':>4}{'dwell':>7}{'cal':>5} | {'ts hit%':>8}{'ts false':>9}"
          f"{'kp1 rate':>9}{'kz rate':>9}{'twin rate':>10}")
    table = {}
    for cfg in GRID:
        agg = {}
        for run in runs:
            key = (run["arm"], run["fixture"])
            r = evaluate(run, cfg, scratch)
            a = agg.setdefault(key, {"opens": 0, "false": 0, "hits": 0,
                                     "max": 0, "steps": 0})
            a["opens"] += r["opens"]
            a["false"] += r["false_opens"]
            a["hits"] += r["col_switch_hits"]
            a["max"] += r["max_hits"]
            a["steps"] += r["col_steps"]
        rate = lambda key: (1000 * agg[key]["opens"] / agg[key]["steps"]
                            if key in agg and agg[key]["steps"] else 0.0)
        f_ts = agg.get(("faithful", "ts"), {})
        hit = 100 * f_ts.get("hits", 0) / max(1, f_ts.get("max", 1))
        ts_false = 1000 * f_ts.get("false", 0) / max(1, f_ts.get("steps", 1))
        twin_rate = sum(1000 * agg[k]["opens"] / agg[k]["steps"]
                        for k in agg if k[0] == "twin" and agg[k]["steps"]) / 3
        row = {"hit": hit, "ts_false": ts_false, "kp1": rate(("faithful", "kp1")),
               "kz": rate(("faithful", "kz")), "twin": twin_rate}
        table[tuple(sorted(cfg.items()))] = (cfg, row)
        print(f"{cfg['k']:>4}{cfg['enter_dwell']:>7}{cfg['cal_window']:>5} | "
              f"{hit:>7.1f}%{ts_false:>9.2f}{row['kp1']:>9.2f}{row['kz']:>9.2f}"
              f"{row['twin']:>10.2f}", flush=True)

    # threshold-shaped choice: passing region = plants seen, quiet elsewhere
    passing = [(cfg, r) for cfg, r in table.values()
               if r["hit"] >= 20 and r["ts_false"] < 1.0 and r["kp1"] < 0.5
               and r["kz"] < 0.5 and r["twin"] < 0.2]
    chosen = next((c for c, _ in passing if c == DEFAULT),
                  passing[0][0] if passing else None)
    print(f"\npassing region: {len(passing)}/{len(GRID)} cells "
          f"(bars: hit>=20%, ts false<1, kp1<0.5, kz<0.5, twin<0.2 per 1k)")
    verdict = chosen if chosen else "NONE PASS: constants need rework, no calibration issues"
    print(f"chosen: {verdict}")
    if chosen and not SMOKE:
        import hashlib
        from experiments.d3 import canonical_bytes
        h = hashlib.sha256(canonical_bytes(chosen)).hexdigest()[:12]
        real.append_ledger({"kind": "cal19", "event_id": f"cal19:{h}",
                            "chosen": chosen, "config_hash": h,
                            "burn_in": BURN_IN, "switch_window": SWITCH_WINDOW,
                            "scorecard": {str(k): v for k, (_c, v) in table.items()},
                            "streams": {"arms": ["faithful", "twin"],
                                        "seeds": list(SEEDS), "ns": NS}})
        print(f"calibration recorded: cal19 config_hash {h}")


if __name__ == "__main__":
    main()
