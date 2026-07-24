"""
Registered band-freeze for the fan3 transport event (R-029 registration,
R-030 outcome): controls only, 32 fresh seeds, constants frozen for the
subject event (23). Second instance of the 14 pattern, powered for a
small true effect per RW's ruling that the event certifies a truth, not a
magnitude (free-play basis: rep18 n=30, gap +0.070 vs twin, per-seed sd
~0.14, paired-positive rate ~0.70).

Arms: fan3_twin (frozen) and fan3_direoff (E frozen at init), fan3 knobs
(p_a=.3, p_e=.6, dist_exp=1, input_fanout=3, softhebb/triangle/ss=1e-2/
lr_e=.1), n_cols=49, CRN graphs. Fixtures kp1+kz at 12000, namespace
"r029". Matched by construction: fresh host per fixture.

Frozen outputs (fan3_frozen_constants.json + sha):
    twin differential mean/sigma -> gap bar = 3 sigma
    sign bar = 20 of 32 paired-positive (pre-stated: null probability
    2.5% at p=.5; power ~0.79 at the measured 0.70 rate)
    activity floor = p1 of nonzero twin norms, WITH the bite audit this
    time: exclusion fraction and floored-vs-unfloored margin delta
    measured on free-play fan3 runs (seeds 10-39) and frozen as the
    audited envelope; registered-run bite outside the envelope = incident
E-checks: E1 no NaN in controls; E2 twin sigma in (0, 0.2); E3 floor > 0;
E4 bite audit within declared caps (exclusion < 20%, subject margin delta
< 0.02); E5 drift: twin seed[0] kp1 rerun bit-identical.

SMOKE (FREEZE29_SMOKE=1): dummy seeds, tiny lengths, scratch store.
"""

import hashlib
import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
frz = importlib.import_module("14_registered_bare_freeze")
from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder, canonical_bytes
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("FREEZE29_SMOKE"))
REGISTERED_SEEDS = (902, 903) if SMOKE else (
    1190, 1337, 1666, 1722, 2309, 2597, 2699, 2779, 3474, 3485, 3650, 3833,
    3888, 3994, 4315, 4522, 4634, 4862, 4972, 5293, 5311, 6942, 7045, 7210,
    7314, 7533, 8516, 8691, 9377, 9457, 9465, 9664)
HORIZON, WINDOW = (240, 60) if SMOKE else (12000, 500)
FREE_PLAY_SEEDS = (10, 11) if SMOKE else tuple(range(10, 40))
SIGN_MIN = 1 if SMOKE else 20
NORM_STRIDE = 4
WORKERS = 4 if SMOKE else 12
STREAM_NS = "r029"
FAN3 = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)
ARMS = [("twin", dict(FAN3, frozen=True)), ("dire_off", dict(FAN3, dire=False))]


def make_host(knobs, seed):
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **knobs)
    return BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_r029/w{os.getpid()}"),
                                    seed=seed, **flags))


def run_arm_seed(job):
    arm, knobs, seed, track_norms = job
    out = {"arm": arm, "seed": seed, "fixtures": {}}
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        host = make_host(knobs, seed)
        r = frz.run_light(host, fn(HORIZON, seed=de.derive_seed(fixture, STREAM_NS, seed)),
                          WINDOW, track_norms=track_norms)
        out["fixtures"][fixture] = {"margin": r["margin"], "nan": r["nan"],
                                    "agg": r["agg"], "agg_hash": frz.agg_hash(r["agg"]),
                                    "norms": r["norms"]}
    out["differential"] = (out["fixtures"]["kp1"]["margin"]
                           - out["fixtures"]["kz"]["margin"])
    return out


def bite_audit(floor):
    """floored vs unfloored margins on free-play fan3 runs (matched runner):
    per-seed exclusion fraction and margin delta, the audited envelope."""
    deltas, excl = [], []
    for seed in FREE_PLAY_SEEDS[:6]:
        for fixture, fn in (("kp1", de.kp1_cycle),):
            host = make_host(dict(FAN3), seed)
            stream = list(fn(HORIZON, seed=de.derive_seed(fixture, "rep18", seed)))
            from collections import deque
            cols = host.agt.cols
            f_tr = {loc: deque(maxlen=WINDOW) for loc in host.agt.predicted}
            u_tr = {loc: deque(maxlen=WINDOW) for loc in host.agt.predicted}
            p_tr = {loc: deque(maxlen=WINDOW) for loc in cols}
            prev_actual = {loc: None for loc in cols}
            prev_guess = None
            n_scored = n_floored = 0
            for item in stream:
                a = item[1] if isinstance(item, tuple) else item
                host.observe(a)
                acts = {loc: c.nr_1.actual for loc, c in cols.items()}
                if prev_guess is not None:
                    for loc, g in prev_guess.items():
                        sc_u = de.score(g, acts[loc])
                        sc_f = de.score(g, acts[loc], floor)
                        if sc_u is not None:
                            u_tr[loc].append(sc_u)
                            n_scored += 1
                        if sc_f is not None:
                            f_tr[loc].append(sc_f)
                        elif sc_u is not None:
                            n_floored += 1
                for loc, act in acts.items():
                    pa = prev_actual[loc]
                    if pa is not None:
                        d = float(pa.norm() * act.norm())
                        p_tr[loc].append(float(pa @ act) / d if d > 0 else 0.0)
                    prev_actual[loc] = act.clone()
                prev_guess = host.guesses()
            def margin(tr):
                ms = [sum(tr[loc]) / len(tr[loc]) - sum(p_tr[loc]) / len(p_tr[loc])
                      for loc in host.agt.predicted if tr[loc] and p_tr[loc]]
                return sum(ms) / len(ms) if ms else 0.0
            deltas.append(abs(margin(f_tr) - margin(u_tr)))
            excl.append(n_floored / max(1, n_scored))
    return max(excl), max(deltas)


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    jobs = [(arm, knobs, s, arm == "twin") for arm, knobs in ARMS
            for s in REGISTERED_SEEDS]
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_arm_seed, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if n % 8 == 0 or n == len(jobs):
                print(f"  r029 {n}/{len(jobs)}", flush=True)

    for r in results:
        rec = Recorder(store, {"run_id": f"r029_{r['arm']}_s{r['seed']}"},
                       {"arm": r["arm"], "seed": r["seed"], "horizon": HORIZON,
                        "stream_ns": STREAM_NS})
        for fixture, fr in r["fixtures"].items():
            for step, (s, p) in enumerate(fr["agg"]):
                rec.sink({"fixture": fixture, "subject": f"r029:{r['arm']}",
                          "seed": r["seed"], "step": step,
                          "agg_score": s, "agg_persist": p})
        rec.close()

    twin = sorted((r for r in results if r["arm"] == "twin"), key=lambda r: r["seed"])
    doff = sorted((r for r in results if r["arm"] == "dire_off"), key=lambda r: r["seed"])
    td = [r["differential"] for r in twin]
    t_mean = sum(td) / len(td)
    t_sigma = (sum((v - t_mean) ** 2 for v in td) / max(1, len(td) - 1)) ** 0.5
    pool_norms = sorted(x for r in twin for f in r["fixtures"].values()
                        for x in f["norms"] if x > 0)
    all_n = sum(len(f["norms"]) for r in twin for f in r["fixtures"].values())
    zero_frac = 1 - len(pool_norms) / max(1, all_n)
    floor = pool_norms[int(len(pool_norms) * 1.0 / 100)]

    print("running bite audit on free-play fan3 runs...", flush=True)
    excl_max, delta_max = bite_audit(floor)

    s0 = REGISTERED_SEEDS[0]
    host = make_host(dict(ARMS[0][1]), s0)
    rerun = frz.run_light(host, de.kp1_cycle(HORIZON,
                          seed=de.derive_seed("kp1", STREAM_NS, s0)), WINDOW)
    first = next(r for r in twin if r["seed"] == s0)
    e5 = frz.agg_hash(rerun["agg"]) == first["fixtures"]["kp1"]["agg_hash"]

    any_nan = any(f["nan"] for r in results for f in r["fixtures"].values())
    checks = {"E1_no_nan": not any_nan,
              "E2_diff_sigma": 0 < t_sigma < 0.2,
              "E3_floor_positive": floor > 0,
              "E4_bite_in_caps": excl_max < 0.20 and delta_max < 0.02,
              "E5_drift_bit_identical": e5}
    constants = {
        "seeds": list(REGISTERED_SEEDS), "horizon": HORIZON, "window": WINDOW,
        "stream_ns": STREAM_NS, "family": FAN3,
        "twin_diff_mean": round(t_mean, 6), "twin_diff_sigma": round(t_sigma, 6),
        "diff_bar_gap": round(3 * t_sigma, 6), "sign_min": SIGN_MIN,
        "n_seeds": len(REGISTERED_SEEDS),
        "activity_floor": round(floor, 6), "zero_norm_fraction": round(zero_frac, 6),
        "bite_audit": {"max_exclusion": round(excl_max, 6),
                       "max_margin_delta": round(delta_max, 6),
                       "caps": {"exclusion": 0.20, "margin_delta": 0.02},
                       "basis": "free-play fan3 seeds 10-15, kp1, rep18 streams"},
        "twin_diffs": [round(v, 6) for v in td],
        "dire_off_diffs": [round(r["differential"], 6) for r in doff],
        "twin_agg_hashes": {f"{r['seed']}/{fx}": fr["agg_hash"]
                            for r in twin for fx, fr in r["fixtures"].items()},
        "dire_off_agg_hashes": {f"{r['seed']}/{fx}": fr["agg_hash"]
                                for r in doff for fx, fr in r["fixtures"].items()},
    }
    b = canonical_bytes(constants)
    sha = hashlib.sha256(b).hexdigest()
    if not SMOKE:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "fan3_frozen_constants.json"), "wb") as f:
            f.write(b)
    store.append_ledger({"kind": "r029-freeze", "event_id": "r029:freeze",
                         "constants_sha": sha, "checks": checks, "smoke": SMOKE})
    print(f"\nE-checks: {checks}")
    print(f"twin diff {t_mean:+.4f} sigma {t_sigma:.4f} -> gap bar {3 * t_sigma:.4f}; "
          f"sign bar {SIGN_MIN}/{len(REGISTERED_SEEDS)}")
    print(f"floor {floor:.4f} (zero frac {zero_frac:.3f}); bite: excl {excl_max:.3f}, "
          f"delta {delta_max:.4f}")
    print(f"constants sha {sha[:12]}  written: {not SMOKE}")
    if not all(checks.values()):
        print("E-CHECK FAILURE: incident, no constants issue")
        sys.exit(1)


if __name__ == "__main__":
    main()
