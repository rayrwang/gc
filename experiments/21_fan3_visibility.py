"""
Conflict-visibility reading on fan3, the one bare substrate with
demonstrated transport (rep18: +0.07 over twin at n=30). Question: when
the world switches, are the violations legible to the calibrated conflict
detector? cal19 answered "invisible" for the faithful recipe (no
transport, nothing to violate); cal20 answered 87% for the feedforward
host. fan3 is the case that decides whether the small transport effect
has a consumer, which scopes the registered event (RW ruling: the small
effect is registrable regardless; visibility decides whether detector
events ride along or legibility is declared out of scope).

Machinery: cal19's shape pointed at fan3: per-column residual streams on
ts (6x2000), one self-calibrating detector per column, full 18-cell grid
reported with the cal20-chosen cell (k=3, dwell 10, cal 200) as the
headline. Arms fan3 + fan3_twin, seeds 20-22 (cal19's seeds; graphs
differ by topology knobs as they must). Summaries printed; one ledger
entry (kind vis21) with the headline numbers.
"""

import hashlib
import importlib
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import ConflictOpenedDetector, D3Store, canonical_bytes
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("VIS_SMOKE"))
SEEDS = (20,) if SMOKE else (20, 21, 22)
TS_CYCLES, TS_STEPS = (3, 200) if SMOKE else (6, 2000)
BURN_IN = 100 if SMOKE else 500
SWITCH_WINDOW = 400
WORKERS = 3 if SMOKE else 6
NS = "vis21"
FAN3 = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)
GRID = [dict(k=k, enter_dwell=d, exit_dwell=d, cal_window=c)
        for k in (2.0, 3.0, 4.0) for d in (10, 25, 50) for c in (200, 300)]
CHOSEN = dict(k=3.0, enter_dwell=10, exit_dwell=10, cal_window=200)  # cal20


def gen_series(job):
    arm, seed = job
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **FAN3)
    host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_vis21/w{os.getpid()}"),
                                    seed=seed, frozen=(arm == "twin")))
    stream = de.ts_switch(TS_CYCLES, TS_STEPS, seed=de.derive_seed("ts", NS, seed))
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
    return {"arm": arm, "seed": seed,
            "series": {f"{l[0]},{l[1]}": v for l, v in series.items()}}


def evaluate(run, cfg, scratch):
    switch_ts = [k * TS_STEPS for k in range(1, TS_CYCLES)]
    hits = set()
    false = opens = fed = 0
    for col, xs in run["series"].items():
        det = ConflictOpenedDetector(scratch, cfg, "vis21", col)
        for x in xs[BURN_IN:]:
            det.feed(x)
        fed += max(0, len(xs) - BURN_IN)
        for e in det.events:
            if e["kind"] != "conflict-opened":
                continue
            opens += 1
            t = e["t"] - 1 + BURN_IN
            s = next((s for s in switch_ts if s <= t <= s + SWITCH_WINDOW), None)
            if s is not None:
                hits.add((run["seed"], col, s))
            else:
                false += 1
    return {"opens": opens, "hits": len(hits),
            "max_hits": len(run["series"]) * len(switch_ts),
            "false": false, "steps": fed}


def main():
    real = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    scratch = D3Store(tempfile.mkdtemp(prefix="vis21_"))
    jobs = [(arm, s) for arm in ("fan3", "twin") for s in SEEDS]
    runs = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(gen_series, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            runs.append(fut.result())
            print(f"  series {n}/{len(jobs)}", flush=True)

    print(f"\nfan3 conflict-visibility (hit = col-switch pairs opened within "
          f"{SWITCH_WINDOW}; rates per 1k col-steps):", flush=True)
    print(f"{'k':>4}{'dwell':>7}{'cal':>5} | {'fan3 hit%':>10}{'fan3 false':>11}"
          f"{'twin hit%':>10}{'twin rate':>10}")
    headline = None
    for cfg in GRID:
        agg = {}
        for run in runs:
            r = evaluate(run, cfg, scratch)
            a = agg.setdefault(run["arm"], {"opens": 0, "hits": 0, "max": 0,
                                            "false": 0, "steps": 0})
            for f in ("opens", "false", "steps"):
                a[f] += r[f]
            a["hits"] += r["hits"]
            a["max"] += r["max_hits"]
        f3, tw = agg["fan3"], agg["twin"]
        row = {"hit": 100 * f3["hits"] / max(1, f3["max"]),
               "false": 1000 * f3["false"] / max(1, f3["steps"]),
               "twin_hit": 100 * tw["hits"] / max(1, tw["max"]),
               "twin_rate": 1000 * tw["opens"] / max(1, tw["steps"])}
        if cfg == CHOSEN:
            headline = row
        print(f"{cfg['k']:>4}{cfg['enter_dwell']:>7}{cfg['cal_window']:>5} | "
              f"{row['hit']:>9.1f}%{row['false']:>11.2f}{row['twin_hit']:>9.1f}%"
              f"{row['twin_rate']:>10.2f}", flush=True)

    print(f"\nheadline (cal20-chosen cell): fan3 hit {headline['hit']:.1f}% "
          f"(faithful read ~0%, feedforward 87%), false {headline['false']:.2f}/1k, "
          f"twin {headline['twin_rate']:.2f}/1k", flush=True)
    if not SMOKE:
        real.append_ledger({"kind": "vis21", "event_id": "vis21:fan3",
                            "chosen_cell": CHOSEN, "headline": headline,
                            "seeds": list(SEEDS), "ns": NS,
                            "burn_in": BURN_IN, "switch_window": SWITCH_WINDOW})
        print("recorded: vis21")


if __name__ == "__main__":
    main()
