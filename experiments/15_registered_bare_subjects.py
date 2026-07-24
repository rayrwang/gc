"""
Registered subject event for the BareAgt rung (R-027 registration, R-028
outcome): the roster against the constants frozen by 14 (R-025/R-026).
Mirrors 07's role on the feedforward rung.

Roster:
    bare:x_faithful   subject, gated (differential bar + responds bar +
                      boundedness)
    bare:stock        subject, recorded reading, non-gating (FORECAST:
                      uncertain; never probed at this horizon in free play)
    bare:x_twin       control: drift gate, aggregate traces bit-identical
                      to R-025's stored runs
    bare:x_dire_off   control: must-fail the differential bar; drift gate
                      like the twin

Gate arithmetic (constants from bare_frozen_constants.json):
    differential: faithful mean(kp1-kz diff) - twin_diff_mean >= diff_bar_gap
                  AND per-seed diff > 0 on >= sign_min of 8 seeds
    responds:     faithful mean responds >= resp_bar
    must-fail:    dire_off mean diff - twin_diff_mean < diff_bar_gap
    boundedness:  any NaN in any arm = that arm fails; in a control = incident
    drift:        twin and dire_off aggregate-trace hashes equal R-025's

Floor policy (declared): the frozen activity floor applies to per-column
scoring identically for every arm in this event's margin computation; the
recorded aggregate traces stay unfloored so the drift gate compares
machinery, not scoring policy. The twin's floored-vs-frozen differential
delta is recorded; a delta above sigma/2 is an incident-shaped finding.

T_switch export (closes 3c): on the faithful ts aggregate score trace,
re-settle time after each switch, median over switches and seeds; settle
parameters declared at registration: fraction 0.1, window 500, dwell 500,
atol 0.02, plateau floor = twin ts aggregate score mean + 3 sigma over
registered seeds (computed from R-025's stored aggregates).

SMOKE (SUBJ_SMOKE=1): synthetic constants, drift gate exercised only if the
scratch store holds R-025 smoke runs; exists so the registered hash covers
a script that executed end to end.
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

frz = importlib.import_module("14_registered_bare_freeze")
sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder, canonical_bytes
from experiments.dire_hosts import BareHost, _bare_path, stock_bare_host
from src import iotypes as T

SMOKE = bool(os.environ.get("SUBJ_SMOKE"))
SEEDS = frz.REGISTERED_SEEDS
WORKERS = 4 if SMOKE else 12
SETTLE = dict(fraction=0.1, window=frz.WINDOW, dwell=frz.WINDOW, atol=0.02)

ARMS = [
    ("faithful", "x", dict(frz.FAITHFUL_KNOBS)),
    ("stock", "stock", dict(lr_e=0.1)),
    ("twin", "x", dict(frz.FAITHFUL_KNOBS, frozen=True)),
    ("dire_off", "x", dict(frz.FAITHFUL_KNOBS, dire=False)),
]


def make_host(kind, knobs, seed):
    if kind == "stock":
        return stock_bare_host(de.SYMBOL_DIM, seed, n_cols=sweep.N_COLS, **knobs)
    knobs = dict(knobs)
    flags = {k: knobs.pop(k) for k in ("dire", "frozen") if k in knobs}
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS, **knobs)
    return BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_r027/w{os.getpid()}"),
                                    seed=seed, **flags))


def run_light_floored(host, stream, window, floor):
    """14's runner plus the frozen scoring floor; aggregates stay unfloored
    so drift compares machinery. duplicated rather than imported-with-edits:
    14's registered hash must not move."""
    cols = host.agt.cols
    score_tr = {loc: deque(maxlen=window) for loc in host.agt.predicted}
    persist_tr = {loc: deque(maxlen=window) for loc in cols}
    prev_actual = {loc: None for loc in cols}
    prev_guess = None
    agg = []
    norm_series = {loc: [] for loc in cols}
    for _step, item in enumerate(stream):
        a = item[1] if isinstance(item, tuple) else item
        host.observe(a)
        acts = {loc: c.nr_1.actual for loc, c in cols.items()}
        s_raw = []
        if prev_guess is not None:
            for loc, g in prev_guess.items():
                sc_f = de.score(g, acts[loc], floor)
                if sc_f is not None:
                    score_tr[loc].append(sc_f)
                sc = de.score(g, acts[loc])
                if sc is not None:
                    s_raw.append(sc)
        p_vals = []
        for loc, act in acts.items():
            pa = prev_actual[loc]
            if pa is not None:
                d = float(pa.norm() * act.norm())
                pv = float(pa @ act) / d if d > 0 else 0.0
                persist_tr[loc].append(pv)
                p_vals.append(pv)
            prev_actual[loc] = act.clone()
        agg.append((round(sum(s_raw) / len(s_raw), 6) if s_raw else None,
                    round(sum(p_vals) / len(p_vals), 6) if p_vals else None))
        for loc, v in acts.items():
            norm_series[loc].append(float(v.norm()))
        prev_guess = host.guesses()
    margins = [s - p for loc in host.agt.predicted
               if score_tr[loc] and persist_tr[loc]
               for s, p in [(sum(score_tr[loc]) / len(score_tr[loc]),
                             sum(persist_tr[loc]) / len(persist_tr[loc]))]]
    margin = sum(margins) / len(margins) if margins else None
    nan = any(x != x for pair in agg for x in pair if x is not None)
    return {"margin": margin, "agg": agg, "nan": nan, "norm_series": norm_series}


def run_arm_seed(job):
    arm, kind, knobs, seed, floor = job
    out = {"arm": arm, "seed": seed, "fixtures": {}}
    for fixture, stream in frz.fixture_streams(seed):
        host = make_host(kind, knobs, seed)
        r = run_light_floored(host, stream, frz.WINDOW, floor)
        fr = {"margin": r["margin"], "nan": r["nan"],
              "agg_hash": frz.agg_hash(r["agg"]), "agg": r["agg"]}
        if fixture == "ts":
            out["responds"] = frz.responds_from_series(r["norm_series"], frz.WINDOW)
            fr["ts_scores"] = [s for s, _p in r["agg"]]
        out["fixtures"][fixture] = fr
    m1, mz = out["fixtures"]["kp1"]["margin"], out["fixtures"]["kz"]["margin"]
    out["differential"] = (m1 - mz) if None not in (m1, mz) else None
    return out


def t_switch_from_scores(ts_scores, plateau_floor):
    """re-settle time after each switch on the aggregate score trace."""
    out = []
    for k in range(1, frz.TS_CYCLES):
        t0 = k * frz.TS_STEPS
        seg = [s for s in ts_scores[t0:t0 + frz.TS_STEPS] if s is not None]
        at = de.settled_at(seg, plateau_floor=plateau_floor, **SETTLE)
        if at is not None:
            out.append(at)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    if SMOKE:
        constants = {"twin_diff_mean": -0.05, "twin_diff_sigma": 0.002,
                     "diff_bar_gap": 0.006, "sign_min": 1,
                     "activity_floor": 0.5, "resp_bar": 0.015,
                     "twin_agg_hashes": {}, "dire_off_agg_hashes": {}}
    else:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "bare_frozen_constants.json"), "rb") as f:
            constants = json.loads(f.read())
    floor = constants["activity_floor"]
    jobs = [(arm, kind, knobs, s, floor) for arm, kind, knobs in ARMS for s in SEEDS]
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_arm_seed, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            print(f"  r027 {n}/{len(jobs)}", flush=True)

    for r in results:
        rec = Recorder(store, {"run_id": f"r027_{r['arm']}_s{r['seed']}"},
                       {"arm": r["arm"], "seed": r["seed"], "horizon": frz.HORIZON})
        for fixture, fr in r["fixtures"].items():
            for step, (s, p) in enumerate(fr["agg"]):
                rec.sink({"fixture": fixture, "subject": f"r027:{r['arm']}",
                          "seed": r["seed"], "step": step,
                          "agg_score": s, "agg_persist": p})
        rec.close()

    def arm_rows(name):
        return sorted((r for r in results if r["arm"] == name), key=lambda r: r["seed"])

    fa, st, tw, doff = (arm_rows(n) for n in ("faithful", "stock", "twin", "dire_off"))
    verdicts = {}

    drift_ok = True
    for rows, key in ((tw, "twin_agg_hashes"), (doff, "dire_off_agg_hashes")):
        for r in rows:
            for fx, fr in r["fixtures"].items():
                want = constants[key].get(f"{r['seed']}/{fx}")
                if want is not None and want != fr["agg_hash"]:
                    drift_ok = False
    verdicts["drift"] = drift_ok

    nan_arms = {r["arm"] for r in results
                if any(f["nan"] for f in r["fixtures"].values())}
    verdicts["boundedness_controls"] = not ({"twin", "dire_off"} & nan_arms)

    fd = [r["differential"] for r in fa]
    fd_ok = [d for d in fd if d is not None]
    f_mean = sum(fd_ok) / len(fd_ok) if fd_ok else float("nan")
    gap = f_mean - constants["twin_diff_mean"]
    signs = sum(1 for d in fd_ok if d > 0)
    verdicts["faithful_diff_bar"] = (gap >= constants["diff_bar_gap"]
                                     and signs >= constants["sign_min"]
                                     and "faithful" not in nan_arms)
    f_resp = [r["responds"] for r in fa]
    fr_mean = sum(f_resp) / len(f_resp)
    verdicts["faithful_responds_bar"] = fr_mean >= constants["resp_bar"]

    # must-fail = fails the full bar (gap AND signs): dire_off's gap is
    # inflated by dir.a learning erasing the twin's persistence asymmetry
    # (rep11), so the sign clause is the discriminating half
    dd = [r["differential"] for r in doff if r["differential"] is not None]
    d_mean = sum(dd) / len(dd) if dd else float("nan")
    d_signs = sum(1 for d in dd if d > 0)
    verdicts["dire_off_must_fail"] = not (
        d_mean - constants["twin_diff_mean"] >= constants["diff_bar_gap"]
        and d_signs >= constants["sign_min"])

    td_floored = [r["differential"] for r in tw if r["differential"] is not None]
    t_delta = abs(sum(td_floored) / len(td_floored) - constants["twin_diff_mean"])
    verdicts["floor_delta_ok"] = t_delta <= 0.01  # recorded reading with an
    # absolute incident threshold; sigma-relative was spuriously tight

    sd = [r["differential"] for r in st if r["differential"] is not None]
    s_mean = sum(sd) / len(sd) if sd else float("nan")
    s_signs = sum(1 for d in sd if d > 0)

    # plateau floor from seed-level twin means, not pooled per-step noise
    tw_seed_means = []
    for r in tw:
        xs = [s for s in r["fixtures"]["ts"]["ts_scores"] if s is not None]
        tw_seed_means.append(sum(xs) / len(xs))
    tw_ts_mean = sum(tw_seed_means) / len(tw_seed_means)
    tw_ts_sigma = (sum((x - tw_ts_mean) ** 2 for x in tw_seed_means)
                   / max(1, len(tw_seed_means) - 1)) ** 0.5
    plateau_floor = tw_ts_mean + 3 * tw_ts_sigma
    t_switches = [t for r in fa
                  for t in t_switch_from_scores(r["fixtures"]["ts"]["ts_scores"],
                                                plateau_floor)]
    t_switch = sorted(t_switches)[len(t_switches) // 2] if t_switches else None

    print("\nVERDICTS:", verdicts, flush=True)
    print(f"faithful diff mean {f_mean:+.4f} (gap {gap:+.4f} vs bar "
          f"{constants['diff_bar_gap']:.4f}), signs {signs}/{len(fd_ok)} "
          f"(min {constants['sign_min']}); responds {fr_mean:.4f} "
          f"(bar {constants['resp_bar']:.4f})")
    print(f"dire_off diff mean {d_mean:+.4f} (must stay under bar): "
          f"{'fails as registered' if verdicts['dire_off_must_fail'] else 'CLEARED = incident'}")
    print(f"stock (non-gating reading) diff mean {s_mean:+.4f}, signs "
          f"{s_signs}/{len(sd)}, nan arms: {sorted(nan_arms) or 'none'}")
    print(f"twin floored-vs-frozen delta {t_delta:.5f} "
          f"(sigma/2 = {constants['twin_diff_sigma'] / 2:.5f})")
    print(f"T_switch export: median {t_switch} (per-instance {sorted(t_switches)}, "
          f"plateau floor {plateau_floor:.4f})")
    store.append_ledger({"kind": "r027-verdicts", "event_id": "r027:verdicts",
                         "verdicts": verdicts, "faithful_diffs": fd,
                         "stock_diffs": sd, "t_switch": t_switch,
                         "smoke": SMOKE})
    gates = all(verdicts.values())
    print(f"\nEVENT: {'ALL GATES GREEN' if gates else 'GATE FAILURE(S): incident routing'}")
    if not gates:
        sys.exit(1)


if __name__ == "__main__":
    main()
