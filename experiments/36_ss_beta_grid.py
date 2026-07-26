"""
Is step size a real lever, or just a clock? (RW, 2026-07-26.)
Free play, seeds 50-52, fan3, Dir.E FROZEN, rule=softhebb throughout.

WHERE THIS COMES FROM. Run 34 swept the gate temperature at the incumbent
ss=1e-2 and found the axis bimodal with nothing in between: every arm from
beta 0.5 to 2 learns, collapses, and runs the weights away (|W| 11.3 -> 1728
at the incumbent b0.5, peaking 2.1e7 at b1.5), while every arm at beta 3+ is
uncollapsed only because it is inert (|W| equal to frozen to 3 s.f.). Run 35
then decomposed the growth exactly:

    d|W|^2 = 2*ss*[ <W, outer(x,gate)> - sum_j gu[j]*|w[:,j]|^2 ]
              ^ hebb                      ^ decay

and found (a) the two terms cancel to three significant figures in the inert
arms, which is Oja's fixed point working as designed, (b) the prediction is
accurate within 3x wherever the system is stable and undershoots by up to
9800x wherever it runs away, always undershooting, and (c) the runaway arms
ACCELERATE (half2/half1 up to 8.8x) while the stable arms are flat. So the
first-order term is a correct instantaneous rate and the process compounds.

THE OBVIOUS NEXT LEVER IS ss, AND THE OBVIOUS NULL IS THAT IT IS NOT A LEVER.
Both terms above are linear in ss, so to first order the per-step log-growth
rate is linear in ss and log(|W|_final/|W|_0) ~ ss * c * N. Under that story
**ss and step count trade off exactly**: 12000 steps at ss=1e-3 is 1200 steps
at ss=1e-2, and lowering ss buys nothing except a slower clock. That is the
null, it is the most likely outcome, and this run is built to test it rather
than to sweep and squint.

THE TEST. Two things make the null falsifiable:
  1. **The normalised rate `meas_rate / ss`.** If growth is a pure time
     rescaling this is CONSTANT in ss at fixed beta. If it rises with ss, the
     compounding is superlinear and ss is doing something a clock cannot.
  2. **Explicit time-rescaled control arms.** For each low-ss cell at 12000
     steps there is a matched ss=1e-2 arm run for the step count that gives
     the same ss*N product. If the null holds these pairs land on the same
     |W| and the same collapse statistics.

ARMS. ss x beta grid at 12000 steps, plus the rescaling controls, plus one
frozen twin (which does not depend on ss). ss=1e-2 is the incumbent; 3e-3 is
the only other value the 216-config sweeps ever used (10_sweep_bare_x:145),
and neither was ever measured against |W| because |W| did not exist as a
measurement until run 34.

METRICS, reusing 31/34/35 rather than reimplementing:
  maj / minority / n_patterns / act_norm   collapse, threshold-free (31)
  mean_gu / frac_neg                        gate*u diagnostics (35). Net gate
                                            mass is deliberately absent: it is
                                            a function of beta and d only, so
                                            it does not vary along the ss axis,
                                            and run 34 already has it per beta.
  w0 / w1 / meas_rate                       the growth actually observed
  hebb_rate / decay_rate / pred_rate        the exact decomposition (35)
  rate_per_ss = meas_rate / ss              THE PRIMARY READ
  differential                              kp1 - kz margin, Dir.E frozen

READING, fixed before the run:
  rate_per_ss flat in ss at every beta, AND the rescaled control pairs match
        -> ss IS A CLOCK. It is not an independent lever, the substrate has
           no stable learning regime at any (ss, beta), and this axis is
           closed. Report it and stop sweeping the rule's own knobs.
  some cell has maj near the 58% chance floor, |W| clearly moved off frozen,
  and a stable (non-accelerating) trajectory
        -> A REAL CELL EXISTS. That is the configuration to build on, and it
           is the first one found in this whole arc.
  rate_per_ss rises with ss but no good cell
        -> superlinear compounding, ss helps sublinearly. Report the exponent,
           do not smooth it, and treat the loop gain as the target.
  control pairs diverge from their matched low-ss arms
        -> ss and N are NOT interchangeable and the first-order account is
           incomplete in a way run 35 did not see. Report the direction.

Summaries only (kind ssb36).
"""

import importlib
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
frz = importlib.import_module("14_registered_bare_freeze")
rep = importlib.import_module("11_replicate_instar_relu")
probe = importlib.import_module("27_hebb_e_probe")
arule = importlib.import_module("31_a_rule_degeneracy")
beta34 = importlib.import_module("34_beta_sweep")
gu35 = importlib.import_module("35_decay_term_probe")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("SSB_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
FULL = 240 if SMOKE else 12000
WORKERS = 2 if SMOKE else 12
NS = "comp25"
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            transport="triangle", rule="softhebb", lr_e=0.1, signed=True)

SS_VALS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
BETA_VALS = (0.5, 1.0, 2.0, 4.0)
REF_SS = 1e-2                       # the incumbent, and the rescaling anchor


def _arms():
    """(name, ss, beta, steps, a_frozen)."""
    out = [("frozen", REF_SS, 0.5, FULL, True)]
    for b in BETA_VALS:
        for ss in SS_VALS:
            out.append((f"b{b:g}_ss{ss:g}", ss, b, FULL, False))
    # time-rescaled controls: same ss*N product as each low-ss arm, run at the
    # incumbent ss. If ss is only a clock these must match their partners.
    for b in BETA_VALS:
        for ss in SS_VALS:
            if ss >= REF_SS:
                continue
            n = max(int(round(FULL * ss / REF_SS)), 60)
            out.append((f"b{b:g}_rescale{ss:g}", REF_SS, b, n, False))
    return tuple(out)


ARMS = _arms()


def arm_cfg(ss, beta):
    return dict(BASE, ss=ss, beta=beta)


def cfg_hash(name, ss, beta, steps, a_frozen):
    return rep.arm_hash(f"ssb36:{name}",
                        dict(arm_cfg(ss, beta), a_frozen=a_frozen), steps)


def run_one(job):
    seed, name, ss, beta, steps, a_frozen = job
    cfg_d = arm_cfg(ss, beta)
    window = max(min(500, steps // 4), 10)
    out = {"seed": seed, "arm": name, "ss": ss, "beta": beta, "steps": steps}
    gu35.install_recorder()          # patches fc.lrn_oja_gated, process-local
    try:
        for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
            gu35.reset_stats()
            cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                                  n_cols=sweep.N_COLS, **cfg_d)
            host = BareHost(sweep.BareAgtXC(
                cfg, _bare_path(f"ssb36/w{os.getpid()}"), seed=seed,
                dire=False, a_frozen=a_frozen))
            agt = host.agt
            w0 = beta34.w_norm(agt)
            r = frz.run_light(host,
                              fn(steps, seed=de.derive_seed(fixture, NS, seed)),
                              window)
            out[fixture] = r["margin"]
            out[f"{fixture}_nan"] = r["nan"]
            if fixture != "kp1":
                continue
            w1 = beta34.w_norm(agt)
            S = gu35.STATS
            out["w0"], out["w1"] = w0, w1
            out["maj"], out["minority"], out["n_patterns"] = \
                arule.degeneracy(agt)
            norms = [float(c.nr_1.actual.norm()) for l, c in agt.cols.items()
                     if not agt.is_i(l)]
            out["act_norm"] = sum(norms) / max(len(norms), 1)
            ne = max(S["n_elem"], 1)
            # net gate mass is not recoverable from gu35's recorder (it logs
            # gate*u, not gate sums) and it is a function of beta and d only,
            # so it does not vary along the ss axis. Run 34 has it per beta.
            # mean_gu IS available and is the diagnostic that matters here.
            out["mean_gu"] = S["sum_gu"] / ne
            out["frac_neg"] = S["n_neg"] / ne
            nr = max(S["n_rel"], 1)
            out["hebb_rate"] = S["sum_rel_h"] / nr / 2
            out["decay_rate"] = S["sum_rel_d"] / nr / 2
            out["pred_rate"] = out["hebb_rate"] + out["decay_rate"]
            out["meas_rate"] = (math.log(w1 / w0) / steps
                                if w0 > 0 and w1 > 0 else float("nan"))
            # THE PRIMARY READ: constant in ss iff ss is a pure time rescaling
            out["rate_per_ss"] = (out["meas_rate"] / ss if ss else float("nan"))
            tr = S["traj"]
            cps = S["n_calls"] / max(steps, 1)
            mid = len(tr) // 2
            out["half1"] = (gu35.log_slope(tr[:mid]) * cps if mid >= 3
                            else float("nan"))
            out["half2"] = (gu35.log_slope(tr[mid:]) * cps
                            if len(tr) - mid >= 3 else float("nan"))
    finally:
        fc.lrn_oja_gated = gu35._ORIG_LRN
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out.get("kp1"), out.get("kz"))
                           else None)
    out["config_hash"] = cfg_hash(name, ss, beta, steps, a_frozen)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "ssb36"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, n, ss, b, st, f) for (n, ss, b, st, f) in ARMS for s in SEEDS
            if (cfg_hash(n, ss, b, st, f), s) not in done]
    print(f"ssb36: {len(ARMS)*len(SEEDS)} cells, {len(todo)} to run "
          f"(Dir.E FROZEN, rule=softhebb, ss x beta at {FULL} steps + "
          f"time-rescaled controls at ss={REF_SS:g})", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "ssb36", "event_id":
                                 f"ssb36:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['arm']:>16}: "
                  f"maj {s['maj']:.1%} |a| {s['act_norm']:.2f} "
                  f"|W| {s['w0']:.1f}->{s['w1']:.4g} "
                  f"rate/ss {s['rate_per_ss']:+.4f}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "ssb36"]

    def agg(name, ss, beta, steps, a_frozen):
        g = [e for e in led
             if e.get("config_hash") == cfg_hash(name, ss, beta, steps, a_frozen)]
        if not g:
            return None
        f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
        r = {k: f(k) for k in ("maj", "n_patterns", "act_norm", "w0", "w1", "mean_gu",
                               "hebb_rate", "decay_rate", "pred_rate",
                               "meas_rate", "rate_per_ss", "half1", "half2",
                               "differential")}
        r["arm"], r["ss"], r["beta"], r["steps"] = name, ss, beta, steps
        return r

    rows = {a[0]: agg(*a) for a in ARMS}
    fz = rows.get("frozen")
    wref = fz["w1"] if fz else None
    print(f"\nss x beta, Dir.E frozen, rule=softhebb, seeds {list(SEEDS)}, "
          f"{FULL} steps")
    if fz:
        print(f"frozen twin: maj {fz['maj']:.1%} |a| {fz['act_norm']:.2f} "
              f"|W| {fz['w1']:.2f}")
    print("rate_per_ss is meas_rate/ss. CONSTANT down an ss column means ss "
          "is a pure time rescaling and buys nothing.\n")

    print(f"{'beta':>5} {'ss':>8} {'maj':>7} {'patt':>6} {'|a|':>6} "
          f"{'|W|':>11} {'xfrozen':>9} {'meas':>10} {'rate/ss':>9} "
          f"{'hebb':>10} {'decay':>10} {'h2/h1':>7}")
    for b in BETA_VALS:
        for ss in SS_VALS:
            r = rows.get(f"b{b:g}_ss{ss:g}")
            if not r:
                continue
            xf = r["w1"] / wref if wref else float("nan")
            h = (r["half2"] / r["half1"]
                 if r["half1"] and r["half1"] == r["half1"] and r["half1"] != 0
                 else float("nan"))
            print(f"{b:5g} {ss:8.0e} {r['maj']:7.1%} {r['n_patterns']:6.2f} "
                  f"{r['act_norm']:6.2f} {r['w1']:11.4g} {xf:9.3g} "
                  f"{r['meas_rate']:+10.2e} {r['rate_per_ss']:+9.4f} "
                  f"{r['hebb_rate']:+10.2e} {r['decay_rate']:+10.2e} "
                  f"{h:7.2f}")
        print()

    print("time-rescaled controls: same ss*N product, run at the incumbent ss.")
    print(f"{'beta':>5} {'orig ss':>9} {'orig |W|':>10} {'ctrl steps':>11} "
          f"{'ctrl |W|':>10} {'ratio':>8} {'orig maj':>9} {'ctrl maj':>9}")
    pair_gaps = []
    for b in BETA_VALS:
        for ss in SS_VALS:
            if ss >= REF_SS:
                continue
            o = rows.get(f"b{b:g}_ss{ss:g}")
            c = rows.get(f"b{b:g}_rescale{ss:g}")
            if not o or not c:
                continue
            ratio = c["w1"] / o["w1"] if o["w1"] else float("nan")
            pair_gaps.append(abs(math.log(ratio)) if ratio > 0 else float("nan"))
            print(f"{b:5g} {ss:9.0e} {o['w1']:10.4g} {c['steps']:11d} "
                  f"{c['w1']:10.4g} {ratio:8.3g} {o['maj']:9.1%} "
                  f"{c['maj']:9.1%}")
    if SMOKE:
        print(f"\nSMOKE at {FULL} steps: reading DISABLED. The runaway is "
              "cumulative and 240 steps is 2% of the real horizon. Smoke "
              "shows the code paths execute, nothing else.")
        return

    print("\nreading:")
    # 1. is rate_per_ss constant down each ss column?
    spreads = []
    for b in BETA_VALS:
        vals = [rows[f"b{b:g}_ss{ss:g}"]["rate_per_ss"] for ss in SS_VALS
                if rows.get(f"b{b:g}_ss{ss:g}")]
        vals = [v for v in vals if v == v]
        if len(vals) >= 3 and min(map(abs, vals)) > 0:
            spread = max(map(abs, vals)) / min(map(abs, vals))
            spreads.append((b, spread))
            print(f"  beta {b:g}: rate_per_ss spans {min(vals):+.4f} to "
                  f"{max(vals):+.4f}, ratio {spread:.2f}x")
    clock = spreads and all(s < 2.0 for _, s in spreads)
    # 2. any cell both alive and uncollapsed?
    good = []
    if fz:
        for b in BETA_VALS:
            for ss in SS_VALS:
                r = rows.get(f"b{b:g}_ss{ss:g}")
                if not r:
                    continue
                moved = wref and 1.5 * wref < r["w1"] < 20 * wref
                if r["maj"] < 0.80 and moved and r["act_norm"] > 0.5 * fz["act_norm"]:
                    good.append(r)
    pg = [g for g in pair_gaps if g == g]
    pairs_match = pg and max(pg) < math.log(2.0)
    print(f"  time-rescaled pairs match within 2x: {bool(pairs_match)}"
          + (f" (worst {math.exp(max(pg)):.2f}x)" if pg else " (no pairs)"))
    if good:
        print("  LIVE CELLS (maj < 80%, |W| between 1.5x and 20x frozen, "
              "activity alive): "
              + ", ".join(f"{g['arm']} (maj {g['maj']:.1%}, "
                          f"|W| {g['w1']:.3g})" for g in good))
        print("  -> the first configuration in this arc that both learns and "
              "stays readable. Confirm on fresh seeds before building on it.")
    elif clock and pairs_match:
        print("  NO live cell, rate_per_ss flat, controls match.")
        print("  -> ss IS A CLOCK. Not an independent lever. The rule has no "
              "stable learning regime at any (ss, beta) on this substrate, "
              "and sweeping its own knobs further is closed. The next move "
              "has to change the rule or the architecture, not its numbers.")
    else:
        print("  NO live cell, and the null did not hold cleanly either.")
        print("  -> report the shape as measured. Superlinear compounding or "
              "control divergence both mean the first-order account from run "
              "35 is incomplete; say which, and do not smooth it.")
    print("\nh2/h1 is the ratio of second-half to first-half log-growth slope. "
          "Above 1 means the runaway accelerates, which is the feedback "
          "signature run 35 found; near 1 means constant-rate growth.")


if __name__ == "__main__":
    main()
