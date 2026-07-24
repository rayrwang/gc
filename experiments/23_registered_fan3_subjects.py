"""
Registered subject event for the fan3 transport claim (R-031 registration,
R-032 verdict): fan3 against the constants R-029/R-030 froze (sha
6333ac07e5b0). Third instance of the subject-event pattern; single-clawed
by design: one gated claim, everything else recorded.

Arms:
    fan3        subject: kp1+kz, gated
    fan3_twin   control: drift gate vs R-029 aggregate hashes
    fan3_direoff control: drift gate + must-fail (full conjunction)
    fan3 ts     recorded arm, non-gating: per-column conflict-detector
                events at the cal20 cell (registered visibility), the
                responds value, and a provisional T_switch reading

Gates (constants frozen before this script was registered):
    gap:    mean(fan3 diff, floored scoring) - twin_diff_mean >= 0.0350
    signs:  fan3 per-seed diff > the SAME SEED's frozen twin diff
            (paired, the R-026 zero-point lesson) on >= 20 of 32
    boundedness: NaN in the subject = fail; in a control = incident
    must-fail: dire_off must NOT satisfy (gap AND signs) itself
    drift:  twin + dire_off aggregate traces bit-identical to R-029
    bite:   subject exclusion fraction and floored-vs-unfloored margin
            delta inside the audited envelope; outside = incident flag,
            never a verdict flip

Declared semantics unchanged: subject failure = registered negative,
control misbehavior = incident. SMOKE (SUBJ31_SMOKE=1): scratch store,
dummy seeds, tiny lengths, real constants file.
"""

import hashlib
import importlib
import json
import os
import sys
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
frz = importlib.import_module("14_registered_bare_freeze")
from experiments import dire_exam as de
from experiments.d3 import ConflictOpenedDetector, D3Store, Recorder, canonical_bytes
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("SUBJ31_SMOKE"))
HORIZON, WINDOW = (240, 60) if SMOKE else (12000, 500)
TS_CYCLES, TS_STEPS = (3, 80) if SMOKE else (6, 2000)
BURN_IN = 40 if SMOKE else 500
WORKERS = 4 if SMOKE else 12
STREAM_NS = "r029"
FAN3 = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)
DETECTOR_CELL = dict(k=3.0, enter_dwell=10, exit_dwell=10, cal_window=200)  # cal20
SETTLE = dict(fraction=0.1, atol=0.02)  # window/dwell passed at the call from WINDOW


def make_host(knobs, seed):
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **knobs)
    return BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_r031/w{os.getpid()}"),
                                    seed=seed, **flags))


def run_dual(host, stream, window, floor):
    """floored and unfloored per-column trails together, plus unfloored
    aggregates for the drift gate; the 15 pattern with the bite reading
    built in."""
    cols = host.agt.cols
    f_tr = {loc: deque(maxlen=window) for loc in host.agt.predicted}
    u_tr = {loc: deque(maxlen=window) for loc in host.agt.predicted}
    p_tr = {loc: deque(maxlen=window) for loc in cols}
    prev_actual = {loc: None for loc in cols}
    prev_guess = None
    agg = []
    n_scored = n_floored = 0
    for item in stream:
        a = item[1] if isinstance(item, tuple) else item
        host.observe(a)
        acts = {loc: c.nr_1.actual for loc, c in cols.items()}
        s_raw = []
        if prev_guess is not None:
            for loc, g in prev_guess.items():
                sc_u = de.score(g, acts[loc])
                sc_f = de.score(g, acts[loc], floor)
                if sc_u is not None:
                    u_tr[loc].append(sc_u)
                    s_raw.append(sc_u)
                    n_scored += 1
                if sc_f is not None:
                    f_tr[loc].append(sc_f)
                elif sc_u is not None:
                    n_floored += 1
        p_vals = []
        for loc, act in acts.items():
            pa = prev_actual[loc]
            if pa is not None:
                d = float(pa.norm() * act.norm())
                pv = float(pa @ act) / d if d > 0 else 0.0
                p_tr[loc].append(pv)
                p_vals.append(pv)
            prev_actual[loc] = act.clone()
        agg.append((round(sum(s_raw) / len(s_raw), 6) if s_raw else None,
                    round(sum(p_vals) / len(p_vals), 6) if p_vals else None))
        prev_guess = host.guesses()

    def margin(tr):
        ms = [sum(tr[loc]) / len(tr[loc]) - sum(p_tr[loc]) / len(p_tr[loc])
              for loc in host.agt.predicted if tr[loc] and p_tr[loc]]
        return sum(ms) / len(ms) if ms else None

    nan = any(x != x for pair in agg for x in pair if x is not None)
    return {"margin_f": margin(f_tr), "margin_u": margin(u_tr), "agg": agg,
            "nan": nan, "n_scored": n_scored, "n_floored": n_floored}


def run_arm_seed(job):
    arm, knobs, seed, floor = job
    out = {"arm": arm, "seed": seed, "fixtures": {}, "n_scored": 0, "n_floored": 0}
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        host = make_host(knobs, seed)
        r = run_dual(host, fn(HORIZON, seed=de.derive_seed(fixture, STREAM_NS, seed)),
                     WINDOW, floor)
        out["fixtures"][fixture] = {"margin_f": r["margin_f"], "margin_u": r["margin_u"],
                                    "nan": r["nan"], "agg": r["agg"],
                                    "agg_hash": frz.agg_hash(r["agg"])}
        out["n_scored"] += r["n_scored"]
        out["n_floored"] += r["n_floored"]
    for tag in ("f", "u"):
        m1, mz = (out["fixtures"]["kp1"][f"margin_{tag}"],
                  out["fixtures"]["kz"][f"margin_{tag}"])
        out[f"diff_{tag}"] = (m1 - mz) if None not in (m1, mz) else None
    return out


def run_ts_seed(job):
    """recorded arm: per-column detector events + responds + settle."""
    seed, = job
    host = make_host(dict(FAN3), seed)
    stream = de.ts_switch(TS_CYCLES, TS_STEPS, seed=de.derive_seed("ts", STREAM_NS, seed))
    cols = host.agt.cols
    series = {loc: [] for loc in host.agt.predicted}
    norm_series = {loc: [] for loc in cols}
    agg = []
    prev_guess = None
    for item in stream:
        a = item[1] if isinstance(item, tuple) else item
        host.observe(a)
        acts = {loc: c.nr_1.actual for loc, c in cols.items()}
        s_vals = []
        if prev_guess is not None:
            for loc, g in prev_guess.items():
                sc = de.score(g, acts[loc])
                series[loc].append(sc if sc is not None else 0.0)
                if sc is not None:
                    s_vals.append(sc)
        for loc, v in acts.items():
            norm_series[loc].append(float(v.norm()))
        agg.append(sum(s_vals) / len(s_vals) if s_vals else None)
        prev_guess = host.guesses()
    scratch = D3Store(tempfile.mkdtemp(prefix="r031_det_"))
    switch_ts = [k * TS_STEPS for k in range(1, TS_CYCLES)]
    hits, false, opens = set(), 0, 0
    for col, xs in series.items():
        det = ConflictOpenedDetector(scratch, DETECTOR_CELL, f"r031_s{seed}", str(col))
        for x in xs[BURN_IN:]:
            det.feed(x)
        for e in det.events:
            if e["kind"] != "conflict-opened":
                continue
            opens += 1
            t = e["t"] - 1 + BURN_IN
            s = next((s for s in switch_ts if s <= t <= s + 400), None)
            if s is not None:
                hits.add((str(col), s))
            else:
                false += 1
    responds = frz.responds_from_series(norm_series, WINDOW) if not SMOKE else 0.0
    xs = [s for s in agg if s is not None]
    settle = de.settled_at(xs, window=WINDOW, dwell=WINDOW,
                           plateau_floor=0.0, **SETTLE) if not SMOKE else None
    return {"seed": seed, "opens": opens, "hits": len(hits),
            "max_hits": len(series) * len(switch_ts), "false": false,
            "responds": responds, "settle": settle}


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fan3_frozen_constants.json"), "rb") as f:
        constants = json.loads(f.read())
    seeds = (902, 903) if SMOKE else tuple(constants["seeds"])
    floor = constants["activity_floor"]
    arms = [("fan3", dict(FAN3)), ("fan3_twin", dict(FAN3, frozen=True)),
            ("fan3_direoff", dict(FAN3, dire=False))]
    jobs = [(a, k, s, floor) for a, k in arms for s in seeds]
    ts_jobs = [(s,) for s in seeds]
    results, ts_results = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(run_arm_seed, j): "m" for j in jobs}
        futs.update({pool.submit(run_ts_seed, j): "t" for j in ts_jobs})
        done_n = 0
        for fut in as_completed(futs):
            (results if futs[fut] == "m" else ts_results).append(fut.result())
            done_n += 1
            if done_n % 10 == 0 or done_n == len(futs):
                print(f"  r031 {done_n}/{len(futs)}", flush=True)

    for r in results:
        rec = Recorder(store, {"run_id": f"r031_{r['arm']}_s{r['seed']}"},
                       {"arm": r["arm"], "seed": r["seed"], "horizon": HORIZON})
        for fixture, fr in r["fixtures"].items():
            for step, (s, p) in enumerate(fr["agg"]):
                rec.sink({"fixture": fixture, "subject": f"r031:{r['arm']}",
                          "seed": r["seed"], "step": step,
                          "agg_score": s, "agg_persist": p})
        rec.close()

    def rows(name):
        return sorted((r for r in results if r["arm"] == name), key=lambda r: r["seed"])

    fan, tw, doff = rows("fan3"), rows("fan3_twin"), rows("fan3_direoff")
    verdicts, incidents = {}, []

    drift_ok = True
    for rs, key in ((tw, "twin_agg_hashes"), (doff, "dire_off_agg_hashes")):
        for r in rs:
            for fx, fr in r["fixtures"].items():
                want = constants[key].get(f"{r['seed']}/{fx}")
                if want is not None and want != fr["agg_hash"]:
                    drift_ok = False
    verdicts["drift"] = drift_ok

    nan_arms = {r["arm"] for r in results
                if any(f["nan"] for f in r["fixtures"].values())}
    verdicts["boundedness_controls"] = not ({"fan3_twin", "fan3_direoff"} & nan_arms)

    twin_by_seed = dict(zip(constants["seeds"], constants["twin_diffs"]))
    fd = [(r["seed"], r["diff_f"]) for r in fan if r["diff_f"] is not None]
    f_mean = sum(d for _s, d in fd) / len(fd) if fd else float("nan")
    gap = f_mean - constants["twin_diff_mean"]
    paired = sum(1 for s, d in fd if d > twin_by_seed.get(s, 0.0))
    verdicts["fan3_gap_bar"] = gap >= constants["diff_bar_gap"] and "fan3" not in nan_arms
    verdicts["fan3_sign_bar"] = paired >= constants["sign_min"]

    dd = [(r["seed"], r["diff_f"]) for r in doff if r["diff_f"] is not None]
    d_mean = sum(d for _s, d in dd) / len(dd) if dd else float("nan")
    d_gap = d_mean - constants["twin_diff_mean"]
    d_paired = sum(1 for s, d in dd if d > twin_by_seed.get(s, 0.0))
    verdicts["dire_off_must_fail"] = not (d_gap >= constants["diff_bar_gap"]
                                          and d_paired >= constants["sign_min"])

    excl = sum(r["n_floored"] for r in fan) / max(1, sum(r["n_scored"] for r in fan))
    ud = [r["diff_u"] for r in fan if r["diff_u"] is not None]
    u_mean = sum(ud) / len(ud) if ud else float("nan")
    bite_delta = abs(f_mean - u_mean)
    caps = constants["bite_audit"]["caps"]
    if excl > caps["exclusion"] or bite_delta > caps["margin_delta"]:
        incidents.append(f"bite outside audited envelope: excl {excl:.3f}, "
                         f"delta {bite_delta:.4f}")

    t_hits = sum(t["hits"] for t in ts_results)
    t_max = sum(t["max_hits"] for t in ts_results)
    t_false = sum(t["false"] for t in ts_results)
    resp = [t["responds"] for t in ts_results]
    settles = [t["settle"] for t in ts_results if t["settle"] is not None]
    t_switch = sorted(settles)[len(settles) // 2] if settles else None

    print("\nVERDICTS:", verdicts, flush=True)
    print(f"fan3 diff mean {f_mean:+.4f} (gap {gap:+.4f} vs bar "
          f"{constants['diff_bar_gap']:.4f}); paired signs {paired}/{len(fd)} "
          f"(min {constants['sign_min']}); unfloored mean {u_mean:+.4f}")
    print(f"dire_off: gap {d_gap:+.4f}, paired {d_paired}/{len(dd)} -> "
          f"{'fails as registered' if verdicts['dire_off_must_fail'] else 'CLEARED = incident'}")
    print(f"bite: exclusion {excl:.3f}, delta {bite_delta:.4f} "
          f"(caps {caps['exclusion']}, {caps['margin_delta']})")
    print(f"recorded (non-gating): visibility {100 * t_hits / max(1, t_max):.1f}% "
          f"({t_hits}/{t_max} col-switch pairs, {t_false} false), responds mean "
          f"{sum(resp) / max(1, len(resp)):.4f}, T_switch median {t_switch}")
    if incidents:
        print("INCIDENTS:", incidents, flush=True)
    store.append_ledger({"kind": "r031-verdicts", "event_id": "r031:verdicts",
                         "verdicts": verdicts, "incidents": incidents,
                         "fan3_diffs": {s: round(d, 6) for s, d in fd},
                         "gap": round(gap, 6), "paired": paired,
                         "visibility": {"hits": t_hits, "max": t_max,
                                        "false": t_false},
                         "responds": [round(v, 6) for v in resp],
                         "t_switch": t_switch, "smoke": SMOKE})
    gates = all(verdicts.values())
    print(f"\nEVENT: {'ALL GATES GREEN' if gates else 'GATE FAILURE(S)'}")
    if not gates:
        sys.exit(1)


if __name__ == "__main__":
    main()
