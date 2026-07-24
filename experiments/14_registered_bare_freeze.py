"""
Registered band-freeze for the BareAgt rung (R-025 registration, R-026
outcome): controls only, fresh seeds, constants measured and frozen for the
subject event (15). Mirrors 06's role on the feedforward rung.

Arms: x frozen twin and x dire-off (faithful knobs, learning cut at the
declared point), on the eight registered seeds drawn at t0, never touched
by free play. Fixtures: kp1(12000), kz(12000), ts(6x2000), streams derived
under the "r025" namespace.

Frozen outputs (bare_frozen_constants.json + sha):
    twin differential mean/sigma  -> subject bar: gap >= 3 sigma + sign rule
    activity floor                -> 1st percentile of twin activity norms,
                                     pooled kp1+kz+ts, stride-4 sampling
    twin responds mean/sigma      -> subject bar: twin + 3 sigma
E-checks (tripwires, any failure = incident, no constants issue):
    E1 no NaN in any control run
    E2 twin differential sigma finite and in (0, 0.2)
    E3 activity floor > 0
    E4 twin responds sigma finite and > 0
    E5 drift: twin seed[0] kp1 rerun bit-identical (aggregate-trace hash)

Recording: per-step network aggregates (mean predicted-column score, mean
persistence) through the chunked Recorder per arm-seed; per-column detail
is regenerable deterministically from (config, seed, stream namespace),
declared in the registry entry. Memory-light by construction: the full
per-column record stream at this horizon does not fit sanely.

SMOKE mode (FREEZE_SMOKE=1) uses dummy seeds (900, 901), tiny lengths, and
a scratch store: it exists so the registered hash covers a script that has
executed end to end (R-023 lesson).
"""

import hashlib
import importlib
import json
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder, canonical_bytes
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("FREEZE_SMOKE"))
REGISTERED_SEEDS = (900, 901) if SMOKE else (3252, 4635, 5149, 5344, 6134, 7540, 7776, 9490)
HORIZON, WINDOW = (240, 60) if SMOKE else (12000, 500)
TS_CYCLES, TS_STEPS = (3, 80) if SMOKE else (6, 2000)
NORM_STRIDE = 4
FLOOR_PERCENTILE = 1.0
WORKERS = 4 if SMOKE else 12
STREAM_NS = "r025"

FAITHFUL_KNOBS = dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)
ARMS = [("twin", dict(FAITHFUL_KNOBS, frozen=True)),
        ("dire_off", dict(FAITHFUL_KNOBS, dire=False))]


def make_host(knobs, seed):
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS, **knobs)
    return BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_r025/w{os.getpid()}"),
                                    seed=seed, **flags))


def fixture_streams(seed):
    return [("kp1", de.kp1_cycle(HORIZON, seed=de.derive_seed("kp1", STREAM_NS, seed))),
            ("kz", de.kz_noise(HORIZON, seed=de.derive_seed("kz", STREAM_NS, seed))),
            ("ts", de.ts_switch(TS_CYCLES, TS_STEPS,
                                seed=de.derive_seed("ts", STREAM_NS, seed)))]


def run_light(host, stream, window, track_norms=False, track_norm_series=False):
    """memory-light drive: trailing per-column score/persistence windows,
    per-step aggregates, optional activity-norm sampling. tick alignment
    matches run_stream_multi (guess emitted at t scored against t+1)."""
    cols = host.agt.cols
    score_tr = {loc: deque(maxlen=window) for loc in host.agt.predicted}
    persist_tr = {loc: deque(maxlen=window) for loc in cols}
    prev_actual = {loc: None for loc in cols}
    prev_guess = None
    agg, norms = [], []
    norm_series = {loc: [] for loc in cols} if track_norm_series else None
    for step, item in enumerate(stream):
        a = item[1] if isinstance(item, tuple) else item
        host.observe(a)
        acts = {loc: c.nr_1.actual for loc, c in cols.items()}
        s_vals = []
        if prev_guess is not None:
            for loc, g in prev_guess.items():
                sc = de.score(g, acts[loc])
                if sc is not None:
                    score_tr[loc].append(sc)
                    s_vals.append(sc)
        p_vals = []
        for loc, act in acts.items():
            pa = prev_actual[loc]
            if pa is not None:
                d = float(pa.norm() * act.norm())
                pv = float(pa @ act) / d if d > 0 else 0.0
                persist_tr[loc].append(pv)
                p_vals.append(pv)
            prev_actual[loc] = act.clone()
        agg.append((round(sum(s_vals) / len(s_vals), 6) if s_vals else None,
                    round(sum(p_vals) / len(p_vals), 6) if p_vals else None))
        if track_norms and step % NORM_STRIDE == 0:
            norms.extend(float(v.norm()) for v in acts.values())
        if track_norm_series:
            for loc, v in acts.items():
                norm_series[loc].append(float(v.norm()))
        prev_guess = host.guesses()
    margins = [s - p for loc in host.agt.predicted
               if score_tr[loc] and persist_tr[loc]
               for s, p in [(sum(score_tr[loc]) / len(score_tr[loc]),
                             sum(persist_tr[loc]) / len(persist_tr[loc]))]]
    margin = sum(margins) / len(margins) if margins else None
    nan = any(x != x for pair in agg for x in pair if x is not None)
    return {"margin": margin, "agg": agg, "norms": norms,
            "norm_series": norm_series, "nan": nan}


def responds_from_series(norm_series, window):
    switches = [k * TS_STEPS for k in range(1, TS_CYCLES)]
    per = []
    for t in switches:
        deltas = [abs(sum(v[t:t + window]) / window - sum(v[t - window:t]) / window)
                  for v in norm_series.values() if len(v) >= t + window]
        per.append(sum(deltas) / len(deltas))
    return sum(per) / len(per)


def agg_hash(agg):
    return hashlib.sha256(canonical_bytes(agg)).hexdigest()[:16]


def run_arm_seed(job):
    arm, knobs, seed = job
    out = {"arm": arm, "seed": seed, "fixtures": {}}
    for fixture, stream in fixture_streams(seed):
        host = make_host(knobs, seed)
        r = run_light(host, stream, WINDOW,
                      track_norms=(arm == "twin"),
                      track_norm_series=(fixture == "ts"))
        out["fixtures"][fixture] = {
            "margin": r["margin"], "nan": r["nan"], "agg": r["agg"],
            "agg_hash": agg_hash(r["agg"]), "norms": r["norms"]}
        if fixture == "ts":
            out["responds"] = responds_from_series(r["norm_series"], WINDOW)
    out["differential"] = out["fixtures"]["kp1"]["margin"] - out["fixtures"]["kz"]["margin"]
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    jobs = [(arm, knobs, s) for arm, knobs in ARMS for s in REGISTERED_SEEDS]
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_arm_seed, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            print(f"  r025 {n}/{len(jobs)}", flush=True)

    # record aggregate traces, one recorder stamp per arm-seed
    for r in results:
        rec = Recorder(store, {"run_id": f"r025_{r['arm']}_s{r['seed']}"},
                       {"arm": r["arm"], "seed": r["seed"], "horizon": HORIZON,
                        "stream_ns": STREAM_NS})
        for fixture, fr in r["fixtures"].items():
            for step, (s, p) in enumerate(fr["agg"]):
                rec.sink({"fixture": fixture, "subject": f"r025:{r['arm']}",
                          "seed": r["seed"], "step": step,
                          "agg_score": s, "agg_persist": p})
        rec.close()

    twin = sorted((r for r in results if r["arm"] == "twin"), key=lambda r: r["seed"])
    doff = sorted((r for r in results if r["arm"] == "dire_off"), key=lambda r: r["seed"])
    td = [r["differential"] for r in twin]
    t_mean = sum(td) / len(td)
    t_sigma = (sum((v - t_mean) ** 2 for v in td) / max(1, len(td) - 1)) ** 0.5
    resp = [r["responds"] for r in twin]
    r_mean = sum(resp) / len(resp)
    r_sigma = (sum((v - r_mean) ** 2 for v in resp) / max(1, len(resp) - 1)) ** 0.5
    # floor over nonzero norms: structural zeros (unreachable columns) are
    # already excluded from scoring by the cosine zero-norm guard, and their
    # point mass would pin any low percentile to exactly 0 (smoke-caught)
    all_norms = [x for r in twin for f in r["fixtures"].values() for x in f["norms"]]
    pool_norms = sorted(x for x in all_norms if x > 0)
    zero_frac = 1 - len(pool_norms) / len(all_norms)
    floor = pool_norms[int(len(pool_norms) * FLOOR_PERCENTILE / 100)]

    # E5 drift: twin seed[0] kp1 rerun, bit-identical aggregates
    s0 = REGISTERED_SEEDS[0]
    fixture, stream = fixture_streams(s0)[0]
    rerun = run_light(make_host(dict(ARMS[0][1]), s0), stream, WINDOW)
    first = next(r for r in twin if r["seed"] == s0)
    e5 = agg_hash(rerun["agg"]) == first["fixtures"]["kp1"]["agg_hash"]

    any_nan = any(f["nan"] for r in results for f in r["fixtures"].values())
    checks = {"E1_no_nan": not any_nan,
              "E2_diff_sigma": 0 < t_sigma < 0.2,
              "E3_floor_positive": floor > 0,
              "E4_resp_sigma": r_sigma > 0,
              "E5_drift_bit_identical": e5}
    constants = {
        "seeds": list(REGISTERED_SEEDS), "horizon": HORIZON, "window": WINDOW,
        "ts": [TS_CYCLES, TS_STEPS], "stream_ns": STREAM_NS,
        "norm_stride": NORM_STRIDE, "floor_percentile": FLOOR_PERCENTILE,
        "twin_diff_mean": round(t_mean, 6), "twin_diff_sigma": round(t_sigma, 6),
        "diff_bar_gap": round(3 * t_sigma, 6), "sign_min": 7,
        "activity_floor": round(floor, 6),
        "zero_norm_fraction": round(zero_frac, 6),
        "resp_twin_mean": round(r_mean, 6), "resp_twin_sigma": round(r_sigma, 6),
        "resp_bar": round(r_mean + 3 * r_sigma, 6),
        "twin_diffs": [round(v, 6) for v in td],
        "dire_off_diffs": [round(r["differential"], 6) for r in doff],
        "twin_agg_hashes": {f"{r['seed']}/{fx}": fr["agg_hash"]
                            for r in twin for fx, fr in r["fixtures"].items()},
        "dire_off_agg_hashes": {f"{r['seed']}/{fx}": fr["agg_hash"]
                                for r in doff for fx, fr in r["fixtures"].items()},
    }
    b = canonical_bytes(constants)
    sha = hashlib.sha256(b).hexdigest()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bare_frozen_constants.json")
    if not SMOKE:
        with open(out_path, "wb") as f:
            f.write(b)
    store.append_ledger({"kind": "r025-freeze", "event_id": "r025:freeze",
                         "constants_sha": sha, "checks": checks, "smoke": SMOKE})
    print(f"\nE-checks: {checks}")
    print(f"twin diff {t_mean:+.4f} sigma {t_sigma:.4f} -> bar gap {3 * t_sigma:.4f}")
    print(f"activity floor (p{FLOOR_PERCENTILE}) {floor:.6f}  "
          f"responds twin {r_mean:.4f} sigma {r_sigma:.4f} -> bar {r_mean + 3 * r_sigma:.4f}")
    print(f"dire_off diffs: {[round(r['differential'], 4) for r in doff]}")
    print(f"constants sha {sha[:12]}  written: {not SMOKE}")
    if not all(checks.values()):
        print("E-CHECK FAILURE: incident, no constants issue")
        sys.exit(1)


if __name__ == "__main__":
    main()
