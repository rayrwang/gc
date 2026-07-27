"""
The collapse happens before the weights move. Did they move, or did I measure
the wrong thing about them? (RW, 2026-07-26.)
Free play, seeds 50-52, fan3, Dir.E FROZEN, rule=softhebb.

WHERE THIS COMES FROM. Run 36 swept ss x beta and found the collapse arrives
FIRST and the weight runaway arrives later:

    beta 0.5   ss=3e-4   maj 74.7%   patterns 1.00   |W| 1.01x frozen
               ss=1e-3   maj 83.2%   patterns 0.91   |W| 1.18x
               ss=3e-3   maj 93.6%   patterns 0.71   |W| 2.07x
               ss=1e-2   maj 95.3%   patterns 0.70   |W|  153x
               ss=3e-2   maj 94.5%   patterns 0.62   |W| 4.9e13x
    (frozen twin: maj 57.0%, |W| 11.32)

Majority-sign is already at 74.7% against a 57% chance floor while the weight
norm has moved ONE PERCENT. By the time the collapse is essentially complete
the norm has merely doubled, and then it climbs thirteen more orders of
magnitude while the collapse barely changes.

THE OBVIOUS SUSPICION, AND IT IS ABOUT THE INSTRUMENT, NOT THE SUBSTRATE.
|W| is a Frobenius norm, and **a norm is blind to direction**. A one percent
change in magnitude is entirely compatible with a large rotation if the
change is concentrated. So "the weights did not move" may simply be "the
weights moved and I measured the one property that cannot see it". That is
the same mistake as never measuring |W| at all, one level down, and it is
ledger #22 for the third time in two days: point the new instrument at the
old verdict.

WHY THIS RUN AND NOT THE NORMALISATION PORT. The BN/centring port was aimed
at the activity's common mode, a diagnosis that runs 35 and 36 have now
superseded twice. Worse, it would corrupt the primary statistic: `maj` is
max(pos,neg)/d, a measure of sign asymmetry, and CENTRING THE ACTIVITY FORCES
pos ~ neg BY CONSTRUCTION. Porting BN now would drive maj toward 50% whether
or not anything real improved, i.e. it would fix the thermometer. Direction
has to be measured before any normalisation is added.

WHAT IS NEW HERE.
  cos_init      mean over Dir.A conns of cos(W_0, W_t), flattened. How far
                each weight matrix has ROTATED from its initialisation,
                independent of how its norm changed.
  mean_abs_cos  mean |cosine| between the COLUMNS of each Dir.A matrix, i.e.
                how similar the receiving units' incoming weight vectors have
                become. This is the statistic runs 32 and 33 used on CIFAR,
                where it ran 0.080 -> 0.987 under collapse and was the tell
                that channels had merged onto one prototype. It has never
                once been measured on the bare rung.
  a TIME TRACE  maj, |W|, cos_init and mean_abs_cos sampled through training,
                so the ordering is MEASURED rather than inferred from an ss
                sweep. Run 36 inferred it; this measures it.
  crossing times t_maj75 / t_cos99 / t_cos95 / t_w110: the first step at
                which each quantity passes its threshold. These four numbers
                are the answer on their own.

Dir.E is frozen and the kp1/kz margin machinery is deliberately NOT used:
this run asks nothing about prediction, so the driver is a bare observe loop
and the differential is not computed.

READING, fixed before the run:
  cos_init falls clearly while |W| stays flat
        -> the weights ROTATE at near-constant norm. The collapse is a
           DIRECTIONAL phenomenon, |W| was the wrong statistic, and no
           normalisation scheme was ever going to touch it.
  cos_init stays near 1 AND mean_abs_cos stays near its init value, while
  maj still climbs
        -> the weights genuinely did not change and the collapse is a
           property of the RECURRENT DYNAMICS at near-initial weights. That
           is stranger and more interesting, and needs a different run: the
           frozen twin sits at 57%, so a tiny nudge would be tipping the
           system into a basin that the initial weights already contain.
  in the trace, t_maj75 < t_cos95
        -> collapse leads, weights follow.
  t_cos95 < t_maj75
        -> weights lead, collapse follows.
  mean_abs_cos climbing toward 1 like CIFAR's 0.080 -> 0.987
        -> the same channel-merging failure as CIFAR, on a substrate where it
           has never been looked for.

Reference: for random unit vectors in R^32 the expected |cos| is about
sqrt(2/(pi*32)) ~ 0.14, so the frozen arm's mean_abs_cos should land near
there. Anything much above that is structure, not initialisation.

Summaries plus a 25-point trace per cell (kind wdir37).
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
rep = importlib.import_module("11_replicate_instar_relu")
probe = importlib.import_module("27_hebb_e_probe")
arule = importlib.import_module("31_a_rule_degeneracy")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T
from src.agents import Dir

SMOKE = bool(os.environ.get("WDIR_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS = 240 if SMOKE else 12000
TRACE_PTS = 25
WORKERS = 2 if SMOKE else 12
NS = "comp25"
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            transport="triangle", rule="softhebb", lr_e=0.1, signed=True)

# the window run 36 says matters: collapse turns on, norm does not move
SS_VALS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
BETA_VALS = (0.5, 2.0)


def _arms():
    out = [("frozen", 1e-2, 0.5, True)]
    for b in BETA_VALS:
        for ss in SS_VALS:
            out.append((f"b{b:g}_ss{ss:g}", ss, b, False))
    return tuple(out)


ARMS = _arms()


def arm_cfg(ss, beta):
    return dict(BASE, ss=ss, beta=beta)


def cfg_hash(name, ss, beta, a_frozen):
    return rep.arm_hash(f"wdir37:{name}",
                        dict(arm_cfg(ss, beta), a_frozen=a_frozen), STEPS)


def snapshot(agt):
    """the Dir.A weights at t=0, keyed by (sender, target)."""
    return {(loc, tloc): w.detach().clone()
            for loc, col in agt.cols.items()
            for (tloc, d), w in col.conns.items() if d == Dir.A}


def wstats(agt, W0):
    """DIRECTION and magnitude of the Dir.A weights.

    cos_init     cos(W_0, W_t) per conn, flattened: pure rotation from init,
                 insensitive to how the norm changed.
    mean_abs_cos mean |cos| between COLUMNS of W, per conn: how similar the
                 receiving units' incoming vectors have become. CIFAR's tell.
    w_norm       Frobenius, kept so the two can be compared directly.
    """
    ci, cc, wn = [], [], []
    for loc, col in agt.cols.items():
        for (tloc, d), w in col.conns.items():
            if d != Dir.A:
                continue
            w0 = W0.get((loc, tloc))
            if w0 is None:
                continue
            a, b = w.flatten().float(), w0.flatten().float()
            den = float(a.norm() * b.norm())
            if den > 0:
                ci.append(float(a @ b) / den)
            u = torch.nn.functional.normalize(w.float(), dim=0)
            c = u.T @ u
            off = c - torch.diag(torch.diag(c))
            k = len(c)
            if k > 1:
                cc.append(float(off.abs().sum()) / (k * (k - 1)))
            wn.append(float(w.norm()))
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return m(ci), m(cc), m(wn)


def run_one(job):
    seed, name, ss, beta, a_frozen = job
    cfg_d = arm_cfg(ss, beta)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                          n_cols=sweep.N_COLS, **cfg_d)
    host = BareHost(sweep.BareAgtXC(
        cfg, _bare_path(f"wdir37/w{os.getpid()}"), seed=seed,
        dire=False, a_frozen=a_frozen))
    agt = host.agt
    W0 = snapshot(agt)
    ci0, cc0, w0 = wstats(agt, W0)

    every = max(STEPS // TRACE_PTS, 1)
    trace = []
    cross = {"t_maj75": None, "t_cos99": None, "t_cos95": None, "t_w110": None}
    # bare observe loop: this run asks nothing about prediction, so no margin
    # machinery, no kz fixture, no run_light.
    for step, a in enumerate(de.kp1_cycle(STEPS,
                                          seed=de.derive_seed("kp1", NS, seed))):
        host.observe(a)
        if step % every and step != STEPS - 1:
            continue
        maj, _, npat = arule.degeneracy(agt)
        ci, cc, wn = wstats(agt, W0)
        trace.append([step, round(maj, 5), round(wn, 4),
                      round(ci, 6), round(cc, 6)])
        if cross["t_maj75"] is None and maj > 0.75:
            cross["t_maj75"] = step
        if cross["t_cos99"] is None and ci < 0.99:
            cross["t_cos99"] = step
        if cross["t_cos95"] is None and ci < 0.95:
            cross["t_cos95"] = step
        if cross["t_w110"] is None and wn > 1.10 * w0:
            cross["t_w110"] = step

    maj, minority, npat = arule.degeneracy(agt)
    ci, cc, wn = wstats(agt, W0)
    norms = [float(c.nr_1.actual.norm()) for l, c in agt.cols.items()
             if not agt.is_i(l)]
    return {"seed": seed, "arm": name, "ss": ss, "beta": beta,
            "maj": maj, "minority": minority, "n_patterns": npat,
            "act_norm": sum(norms) / max(len(norms), 1),
            "w0": w0, "w_norm": wn, "cos_init": ci, "mean_abs_cos": cc,
            "cos_init0": ci0, "mean_abs_cos0": cc0,
            "trace": trace, **cross,
            "config_hash": cfg_hash(name, ss, beta, a_frozen)}


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "wdir37"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, n, ss, b, f) for (n, ss, b, f) in ARMS for s in SEEDS
            if (cfg_hash(n, ss, b, f), s) not in done]
    print(f"wdir37: {len(ARMS)*len(SEEDS)} cells, {len(todo)} to run "
          f"(Dir.E FROZEN, rule=softhebb, {STEPS} steps, "
          f"{TRACE_PTS}-point trace)", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "wdir37", "event_id":
                                 f"wdir37:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['arm']:>12}: "
                  f"maj {s['maj']:.1%} cos_init {s['cos_init']:+.4f} "
                  f"|cos| {s['mean_abs_cos']:.3f} "
                  f"|W| {s['w0']:.1f}->{s['w_norm']:.4g}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "wdir37"]

    def agg(name, ss, beta, a_frozen):
        g = [e for e in led
             if e.get("config_hash") == cfg_hash(name, ss, beta, a_frozen)]
        if not g:
            return None
        f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
        r = {k: f(k) for k in ("maj", "n_patterns", "act_norm", "w0", "w_norm",
                               "cos_init", "mean_abs_cos", "cos_init0",
                               "mean_abs_cos0")}
        for k in ("t_maj75", "t_cos99", "t_cos95", "t_w110"):
            v = [e.get(k) for e in g if e.get(k) is not None]
            r[k] = (sum(v) / len(v)) if v else None
        r["arm"], r["ss"], r["beta"] = name, ss, beta
        r["trace"] = g[0].get("trace")
        return r

    rows = {a[0]: agg(*a) for a in ARMS}
    fz = rows.get("frozen")
    print(f"\ndirection vs magnitude, Dir.E frozen, seeds {list(SEEDS)}, "
          f"{STEPS} steps")
    if fz:
        print(f"frozen twin: maj {fz['maj']:.1%} |W| {fz['w_norm']:.2f} "
              f"cos_init {fz['cos_init']:+.4f} |cos| {fz['mean_abs_cos']:.3f}")
    print("reference: random unit vectors in R^32 give E|cos| ~ 0.14, so the "
          "frozen |cos| should land near there.")
    print("cos_init = cos(W_0, W_t): 1.000 means the weights never turned.\n")

    hdr = (f"{'beta':>5} {'ss':>8} {'maj':>7} {'|W|/W0':>8} {'cos_init':>9} "
           f"{'|cos|':>7} {'t_maj75':>8} {'t_cos99':>8} {'t_cos95':>8} "
           f"{'t_w110':>7}")
    for b in BETA_VALS:
        print(hdr if b == BETA_VALS[0] else "")
        for ss in SS_VALS:
            r = rows.get(f"b{b:g}_ss{ss:g}")
            if not r:
                continue
            g = lambda k: ("-" if r[k] is None else f"{r[k]:.0f}")
            print(f"{b:5g} {ss:8.0e} {r['maj']:7.1%} "
                  f"{r['w_norm']/r['w0']:8.3g} {r['cos_init']:+9.4f} "
                  f"{r['mean_abs_cos']:7.3f} {g('t_maj75'):>8} "
                  f"{g('t_cos99'):>8} {g('t_cos95'):>8} {g('t_w110'):>7}")

    if SMOKE:
        print(f"\nSMOKE at {STEPS} steps: reading DISABLED. Collapse and "
              "runaway are both cumulative; 240 steps is 2% of the horizon.")
        return

    live = [r for r in rows.values() if r and r["arm"] != "frozen"]
    print("\nreading:")
    rot = [r for r in live
           if r["cos_init"] < 0.95 and r["w_norm"] < 1.2 * r["w0"]]
    still = [r for r in live
             if r["cos_init"] > 0.99 and r["maj"] > 0.70]
    lead_collapse = [r["arm"] for r in live
                     if r["t_maj75"] is not None and r["t_cos95"] is not None
                     and r["t_maj75"] < r["t_cos95"]]
    lead_weights = [r["arm"] for r in live
                    if r["t_maj75"] is not None and r["t_cos95"] is not None
                    and r["t_cos95"] < r["t_maj75"]]
    print(f"  ROTATED at near-constant norm (cos_init < 0.95, |W| < 1.2x): "
          f"{[r['arm'] for r in rot] or 'none'}")
    print(f"  COLLAPSED without turning (cos_init > 0.99, maj > 70%): "
          f"{[r['arm'] for r in still] or 'none'}")
    print(f"  collapse leads (t_maj75 < t_cos95): {lead_collapse or 'none'}")
    print(f"  weights lead (t_cos95 < t_maj75): {lead_weights or 'none'}")
    if fz:
        merged = [r["arm"] for r in live
                  if r["mean_abs_cos"] > 3 * fz["mean_abs_cos"]]
        print(f"  channels merging (|cos| > 3x frozen): {merged or 'none'}")
    if rot:
        print("  -> DIRECTIONAL. The weights turn at near-constant norm, |W| "
              "was blind to it, and no normalisation scheme addresses it. The "
              "next question is what they turn TOWARD.")
    elif still:
        print("  -> NOT THE WEIGHTS. The collapse happens with the weights "
              "essentially unturned, so it is a property of the recurrent "
              "dynamics at near-initial weights. The frozen twin sits at 57%, "
              "so learning is tipping the system into a basin the "
              "INITIALISATION already contains. That is a different run.")
    else:
        print("  -> mixed or neither. Report the table as measured; do not "
              "pick a story the crossing times do not support.")
    print("\ntrace of the incumbent (b0.5_ss1e-2), step / maj / |W| / cos_init "
          "/ |cos|:")
    tr = (rows.get("b0.5_ss0.01") or {}).get("trace")
    for row in (tr or [])[:TRACE_PTS]:
        print("   ", row)


if __name__ == "__main__":
    main()
