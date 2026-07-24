"""
Competition-knob sweep (RW go, 2026-07-22 ~22:4x BST): gate signedness x
beta on the fan3 base recipe, judged by the full post-R-032 reading set:

    degeneracy   majority-sign fraction + near-one-vs-rest, early vs late
                 windows (the antipodal-collapse statistic the debugger
                 observation invented)
    visibility   conflict-detector switch-hit rate at the cal20 cell
                 (fan3/signed/0.5 baseline: 3.3% free play, 1.0% registered)
    two-null differential: learner vs frozen twin (dynamics-null) and vs
                 dire_off (mechanism-null), matched fresh-host-per-fixture
    responds     rides free from the ts norm series
    boundedness  NaN anywhere = flagged

Grid: signed {True, False} x beta {0.25, 0.5, 1.0, 2.0} = 8 cells, fan3
topology throughout. The twin is knob-independent (frozen = no learning),
so one twin arm per seed serves all cells; dire_off learns Dir.A and is
per-cell. Seeds 50-52 (fresh free-play draws). Threshold-shaped reading:
the report flags cells with reduced late-window collapse, visibility above
the fan3 baseline, and any mechanism-null gap: never an argmax.

Summaries-only (kind comp25); resume = rerun; detached per standing rule.
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
frz = importlib.import_module("14_registered_bare_freeze")
rep = importlib.import_module("11_replicate_instar_relu")
from experiments import dire_exam as de
from experiments.d3 import ConflictOpenedDetector, D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("COMP_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
TS_CYCLES, TS_STEPS = (3, 80) if SMOKE else (6, 2000)
BURN_IN = 40 if SMOKE else 500
WORKERS = 4 if SMOKE else 12
NS = "comp25"
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)
CELLS = [dict(signed=s, beta=b) for s in (True, False)
         for b in (0.25, 0.5, 1.0, 2.0)]
DETECTOR_CELL = dict(k=3.0, enter_dwell=10, exit_dwell=10, cal_window=200)


def make_host(knobs, seed, frozen=False, dire=True):
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **knobs)
    return BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_comp25/w{os.getpid()}"),
                                    seed=seed, frozen=frozen, dire=dire))


def margins_job(arm, knobs, seed, frozen, dire):
    out = {}
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        host = make_host(knobs, seed, frozen=frozen, dire=dire)
        r = frz.run_light(host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                          WINDOW)
        out[fixture] = r["margin"]
        out[f"{fixture}_nan"] = r["nan"]
    out["diff"] = (out["kp1"] - out["kz"]
                   if None not in (out["kp1"], out["kz"]) else None)
    return out


def ts_job(knobs, seed):
    """one ts pass: degeneracy windows, visibility, responds."""
    host = make_host(knobs, seed)
    agt = host.agt
    stream = de.ts_switch(TS_CYCLES, TS_STEPS, seed=de.derive_seed("ts", NS, seed))
    series = {loc: [] for loc in agt.predicted}
    norm_series = {loc: [] for loc in agt.cols}
    total = TS_CYCLES * TS_STEPS
    windows = {"early": (BURN_IN, BURN_IN + 500 if not SMOKE else BURN_IN + 60),
               "late": (total - 1500, total - 1000) if not SMOKE else (total - 80, total - 20)}
    deg = {k: [0.0, 0, 0] for k in windows}
    prev_guess = None
    for i, item in enumerate(stream):
        a = item[1] if isinstance(item, tuple) else item
        host.observe(a)
        acts = {loc: c.nr_1.actual for loc, c in agt.cols.items()}
        if prev_guess is not None:
            for loc, g in prev_guess.items():
                sc = de.score(g, acts[loc])
                series[loc].append(sc if sc is not None else 0.0)
        for loc, v in acts.items():
            norm_series[loc].append(float(v.norm()))
        for wname, (lo, hi) in windows.items():
            if lo <= i < hi:
                for loc, x in acts.items():
                    if agt.is_i(loc):
                        continue
                    pos = int((x > 0).sum())
                    d = x.numel()
                    deg[wname][0] += max(pos, d - pos) / d
                    deg[wname][1] += (min(pos, d - pos) <= 2)
                    deg[wname][2] += 1
        prev_guess = host.guesses()
    scratch = D3Store(tempfile.mkdtemp(prefix="comp25_"))
    switch_ts = [k * TS_STEPS for k in range(1, TS_CYCLES)]
    hits, false = set(), 0
    for col, xs in series.items():
        det = ConflictOpenedDetector(scratch, DETECTOR_CELL, "comp25", str(col))
        for x in xs[BURN_IN:]:
            det.feed(x)
        for e in det.events:
            if e["kind"] != "conflict-opened":
                continue
            t = e["t"] - 1 + BURN_IN
            s = next((s for s in switch_ts if s <= t <= s + 400), None)
            if s is not None:
                hits.add((str(col), s))
            else:
                false += 1
    return {"deg": {k: (v[0] / max(1, v[2]), v[1] / max(1, v[2])) for k, v in deg.items()},
            "vis_hits": len(hits), "vis_max": len(series) * len(switch_ts),
            "vis_false": false,
            "responds": frz.responds_from_series(norm_series, WINDOW) if not SMOKE else 0.0}


def run_one(job):
    kind, cell, seed = job
    if kind == "twin":
        r = margins_job("twin", dict(BASE, signed=True, beta=0.5), seed,
                        frozen=True, dire=True)
        return {"kind_": "twin", "seed": seed, **r,
                "config_hash": rep.arm_hash("twin25", BASE, STEPS)}
    knobs = dict(BASE, **cell)
    tag = f"s{int(cell['signed'])}_b{cell['beta']}"
    if kind == "dire_off":
        r = margins_job("dire_off", knobs, seed, frozen=False, dire=False)
        return {"kind_": "dire_off", "cell": cell, "seed": seed, **r,
                "config_hash": rep.arm_hash(f"doff25_{tag}", knobs, STEPS)}
    m = margins_job("learner", knobs, seed, frozen=False, dire=True)
    t = ts_job(knobs, seed)
    return {"kind_": "learner", "cell": cell, "seed": seed, **m, **t,
            "config_hash": rep.arm_hash(f"lrn25_{tag}", knobs, STEPS)}


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "comp25"}
    jobs = [("twin", None, s) for s in SEEDS]
    jobs += [(k, c, s) for c in CELLS for k in ("learner", "dire_off") for s in SEEDS]
    todo = []
    for j in jobs:
        kind, cell, seed = j
        if kind == "twin":
            h = rep.arm_hash("twin25", BASE, STEPS)
        else:
            tag = f"s{int(cell['signed'])}_b{cell['beta']}"
            h = rep.arm_hash(("lrn25_" if kind == "learner" else "doff25_") + tag,
                             dict(BASE, **cell), STEPS)
        if (h, seed) not in done:
            todo.append(j)
    print(f"comp25: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "comp25", "event_id":
                                 f"comp25:{s['config_hash']}:s{s['seed']}", **{
                                     k: v for k, v in s.items() if k != "cell"},
                                 "cell": s.get("cell")})
            results.append(s)
            if n % 5 == 0 or n == len(todo):
                print(f"  comp25 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "comp25":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)

    tw = [r for r in rows if r["kind_"] == "twin"]
    twin_diff = sum(r["diff"] for r in tw if r["diff"] is not None) / max(1, len(tw))
    print(f"\nshared twin (dynamics-null) diff: {twin_diff:+.4f} over {len(tw)} seeds")
    print(f"{'signed':>7}{'beta':>6} | {'late maj%':>9}{'late 1vr%':>10}{'d-early':>9}"
          f"{'vis%':>6}{'false':>7}{'resp':>7} | {'diff':>7}{'vs twin':>8}{'vs doff':>8}{'nan':>4}")
    for cell in CELLS:
        L = [r for r in rows if r["kind_"] == "learner" and r.get("cell") == cell]
        D = [r for r in rows if r["kind_"] == "dire_off" and r.get("cell") == cell]
        if not L:
            continue
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        lm = mean([r["diff"] for r in L if r["diff"] is not None])
        dm = mean([r["diff"] for r in D if r["diff"] is not None])
        late = mean([r["deg"]["late"][0] for r in L])
        late1 = mean([r["deg"]["late"][1] for r in L])
        early = mean([r["deg"]["early"][0] for r in L])
        vis = 100 * sum(r["vis_hits"] for r in L) / max(1, sum(r["vis_max"] for r in L))
        fal = sum(r["vis_false"] for r in L)
        resp = mean([r["responds"] for r in L])
        nan = sum(1 for r in L + D if r.get("kp1_nan") or r.get("kz_nan"))
        print(f"{str(cell['signed']):>7}{cell['beta']:>6} | {late:>8.1%}{late1:>9.1%}"
              f"{late - early:>+9.1%}{vis:>5.1f}%{fal:>7}{resp:>7.3f} | "
              f"{lm:>+7.3f}{lm - twin_diff:>+8.3f}{lm - dm:>+8.3f}{nan:>4}", flush=True)
    print("\nreading guide: late maj%/1vr% = antipodal collapse (baseline signed/0.5"
          "\n~92%/78%); vis% = switch-hit (baseline 3.3%); vs twin = dynamics gap;"
          "\nvs doff = mechanism (E-specific) gap. flags, not ranks.", flush=True)


if __name__ == "__main__":
    main()
