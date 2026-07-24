"""
Channel-3 conflict-detector calibration, part 2: against the commissioned
feedforward host (R-024), where world imprint provably exists, so the ts
switches are true known-positives (T_switch ~85: large, brief, real
conflicts). Part 1 (19) validated the nulls and discovered the bare
substrate offers no known-positive: an unimprinted substrate has nothing
for a world-switch to conflict with: replicating R-028 from the residual
side.

Single-stream form: the one-layer host has one input-level residual
stream per run. Known answers: ts switches must open within 200 steps
(bar: hit >= 80%: the feedforward conflicts are unmissable if the
construction works); kp1 post-settle quiet; kz quiet; frozen host = null.
Dwell grid sits below the ~85-step conflict duration. Chosen cell recorded
as the cal20 calibration event; the bare-rung deployment inherits the
construction with per-stream self-calibration, not these exact numbers.
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
from experiments.dire_hosts import HOSTS

SMOKE = bool(os.environ.get("CAL20_SMOKE"))
SEEDS = (0,) if SMOKE else (30, 31, 32)
TS_CYCLES, TS_STEPS = (3, 200) if SMOKE else (6, 2000)
FLAT_STEPS = 600 if SMOKE else 4000
BURN_IN = 100 if SMOKE else 300
SWITCH_WINDOW = 200
WORKERS = 3 if SMOKE else 6
NS = "cal20"
GRID = [dict(k=k, enter_dwell=d, exit_dwell=d, cal_window=c)
        for k in (2.0, 3.0, 4.0) for d in (10, 25, 50) for c in (200, 300)]
DEFAULT = dict(k=3.0, enter_dwell=25, exit_dwell=25, cal_window=300)


def gen_series(job):
    arm, seed, fixture = job
    host = HOSTS[arm](de.SYMBOL_DIM, de.derive_seed("subject", NS, seed))
    if fixture == "ts":
        stream = de.ts_switch(TS_CYCLES, TS_STEPS, seed=de.derive_seed("ts", NS, seed))
    elif fixture == "kp1":
        stream = de.kp1_cycle(FLAT_STEPS, seed=de.derive_seed("kp1", NS, seed))
    else:
        stream = de.kz_noise(FLAT_STEPS, seed=de.derive_seed("kz", NS, seed))
    scores = []
    de.run_stream(host, stream, lambda r: scores.append(
        r["score"] if r["score"] is not None else 0.0))
    return {"arm": arm, "seed": seed, "fixture": fixture, "scores": scores}


def evaluate(run, cfg, scratch):
    switch_ts = [k * TS_STEPS for k in range(1, TS_CYCLES)]
    det = ConflictOpenedDetector(scratch, cfg, "cal20", f"{run['arm']}/{run['seed']}")
    for x in run["scores"][BURN_IN:]:
        det.feed(x)
    opens = [e["t"] - 1 + BURN_IN for e in det.events if e["kind"] == "conflict-opened"]
    hits = {s for s in switch_ts if any(s <= t <= s + SWITCH_WINDOW for t in opens)}
    false = sum(1 for t in opens
                if not any(s <= t <= s + SWITCH_WINDOW for s in switch_ts))
    return {"opens": len(opens), "hits": len(hits), "max_hits": len(switch_ts),
            "false": false, "steps": max(0, len(run["scores"]) - BURN_IN)}


def main():
    real = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    scratch = D3Store(tempfile.mkdtemp(prefix="cal20_"))
    jobs = [(arm, s, fx) for arm in ("host:gc_ff", "control:frozen")
            for s in SEEDS for fx in ("ts", "kp1", "kz")]
    runs = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(gen_series, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            runs.append(fut.result())
            print(f"  series {n}/{len(jobs)}", flush=True)

    print(f"\nscorecard (hit = switches opened within {SWITCH_WINDOW}; "
          f"rates per 1k steps):", flush=True)
    print(f"{'k':>4}{'dwell':>7}{'cal':>5} | {'ts hit%':>8}{'ts false':>9}"
          f"{'kp1 rate':>9}{'kz rate':>9}{'frzn rate':>10}")
    table = []
    for cfg in GRID:
        agg = {}
        for run in runs:
            r = evaluate(run, cfg, scratch)
            key = (run["arm"], run["fixture"])
            a = agg.setdefault(key, {"opens": 0, "hits": 0, "max": 0,
                                     "false": 0, "steps": 0})
            for f in ("opens", "false", "steps"):
                a[f] += r[f]
            a["hits"] += r["hits"]
            a["max"] += r["max_hits"]
        ts = agg[("host:gc_ff", "ts")]
        hit = 100 * ts["hits"] / max(1, ts["max"])
        tsf = 1000 * ts["false"] / max(1, ts["steps"])
        rate = lambda k_: 1000 * agg[k_]["opens"] / max(1, agg[k_]["steps"])
        frzn = sum(rate(("control:frozen", fx)) for fx in ("ts", "kp1", "kz")) / 3
        row = {"hit": hit, "tsf": tsf, "kp1": rate(("host:gc_ff", "kp1")),
               "kz": rate(("host:gc_ff", "kz")), "frzn": frzn}
        table.append((cfg, row))
        print(f"{cfg['k']:>4}{cfg['enter_dwell']:>7}{cfg['cal_window']:>5} | "
              f"{hit:>7.1f}%{tsf:>9.2f}{row['kp1']:>9.2f}{row['kz']:>9.2f}"
              f"{frzn:>10.2f}", flush=True)

    passing = [(c, r) for c, r in table
               if r["hit"] >= 80 and r["tsf"] < 1.0 and r["kp1"] < 0.5
               and r["kz"] < 0.5 and r["frzn"] < 0.5]
    chosen = next((c for c, _ in passing if c == DEFAULT),
                  passing[0][0] if passing else None)
    print(f"\npassing region: {len(passing)}/{len(GRID)} "
          f"(bars: hit>=80%, ts false<1, kp1<0.5, kz<0.5, frozen<0.5 per 1k)")
    print(f"chosen: {chosen if chosen else 'NONE PASS'}")
    if chosen and not SMOKE:
        h = hashlib.sha256(canonical_bytes(chosen)).hexdigest()[:12]
        real.append_ledger({
            "kind": "cal20", "event_id": f"cal20:{h}", "chosen": chosen,
            "config_hash": h, "burn_in": BURN_IN, "switch_window": SWITCH_WINDOW,
            "passing": [c for c, _ in passing],
            "scorecard": [{"cfg": c, **r} for c, r in table],
            "note": ("construction validated on the commissioned feedforward "
                     "host; bare-rung deployment inherits the construction "
                     "with per-stream self-calibration. cal19 recorded the "
                     "bare substrate's null known-positive: an unimprinted "
                     "substrate gives a world-switch nothing to conflict "
                     "with (R-028 replicated from the residual side).")})
        print(f"calibration recorded: cal20 config_hash {h}")


if __name__ == "__main__":
    main()
