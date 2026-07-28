"""
Does ANY gate variant avoid the rank-1 collapse? (RW, 2026-07-26.)
Free play, seeds 50-52, fan3, Dir.E FROZEN, rule=softhebb, beta 0.5 throughout.

WHY. Run 37 identified the mechanism: the Dir.A weight matrices ROTATE to
rank 1. cos(W_0, W_t) falls 1.000 -> 0.069 and mean |cos| between the columns
of each matrix runs 0.142 -> 0.966, reproducing the CIFAR channel-merging
signature (0.080 -> 0.987 in runs 32/33) on a substrate where nobody had
looked. A rank-1 matrix maps every input onto one output direction, which is
exactly why every column ends up sharing a sign pattern. So the antipodal
collapse and the weight rotation are one event seen through two instruments.

The anti-Hebbian repulsion in the signed gate's loser term exists precisely to
stop this, and on CIFAR it does: |cos| holds at 0.080 for 50000 steps. Here it
fails completely. So the question is now specific: **is there a gate shape that
makes the repulsion work on this substrate**, or does the whole family collapse?

WHAT HAS AND HAS NOT BEEN TRIED, and this run exists because of the gaps:
  argmax     the incumbent. Collapses (run 37, |cos| 0.966).
  centred    resp - resp.mean(). gate29 measured maj 66% against argmax's 95%
             and it is the ONLY arm that ever looked both active and
             uncollapsed. But that was with the old statistics: **nobody has
             ever run it with cos_init or mean_abs_cos**, which are the
             instruments that turned out to matter.
  shift      g - f*sum(g)/d. Built in run 33 SPECIFICALLY to separate the two
             things centring conflates: it zeroes the row sum while preserving
             every pairwise difference EXACTLY, whereas centring also inverts
             the loser ordering (argmax gives loser i the gate -p_i, so a
             stronger loser is repelled harder; centring gives it p_i - 1/d,
             so a stronger loser is ATTRACTED). **The shift has never been run
             on the recurrent rung**, only on CIFAR, where net mass is +0.305
             and it degraded monotonically. Here net mass is -0.845, the
             opposite sign, so the CIFAR result does not transfer.
  topk       STRUCK in gate29, but for the STATISTIC, not for failing: the old
             one-vs-rest measure was min(pos,neg) <= 2, which hard-codes a
             one-winner gate, so a k-winner gate reads 0% collapse by
             construction while being totally collapsed. n_patterns and
             mean_abs_cos are k-agnostic, so it is measurable now.
  unsigned   signed=False, plain softmax, net mass exactly +1. The far end of
             the axis, already a supported flag.

Beta is pinned at 0.5 so the gate SHAPE is the only variable. Run 34 already
swept temperature and found no beta works.

METRICS, the run-37 set plus the gate's own quantity:
  cos_init      cos(W_0, W_t) per Dir.A conn. Pure rotation from init.
  mean_abs_cos  mean |cos| between COLUMNS of each conn. Frozen sits at 0.142,
                which matches E|cos| ~ sqrt(2/(pi*32)) for random unit vectors
                in R^32, so anything well above that is structure. This is the
                rank-1 detector and the primary read.
  net_mass      mean gate row sum, since every variant changes it and run 33
                found it load-bearing on CIFAR.
  maj / n_patterns / act_norm / w_norm   the threshold-free collapse set (31).
  trace + crossing times, so ordering is measured rather than inferred.

READING, fixed before the run:
  some arm keeps |cos| near the frozen 0.142 AND moves |W| off frozen AND
  keeps maj near the 58% chance floor
        -> THE FIRST WORKING RULE in this whole arc. Confirm on fresh seeds
           before building anything on it.
  every arm drifts toward |cos| ~ 0.97
        -> the collapse is a property of GATED-OJA ITSELF, not of the gate
           shape, and the gate family is exhausted. The next move has to
           change the update rule, not its gating.
  shift arms collapse despite preserving pairwise differences exactly
        -> loser ORDERING is definitively not the mechanism, on either
           substrate. Run 33's full run already suggested this on CIFAR
           (shift100 preserves ordering and was the WORST arm there); this
           would settle it.
  an arm is uncollapsed but |W| ~ frozen
        -> INERT, not solved. Same trap as oja and basic in arule31. The
           reading below flags it rather than counting it as a candidate.

Summaries plus a 25-point trace per cell (kind gate38).
"""

import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
rep = importlib.import_module("11_replicate_instar_relu")
probe = importlib.import_module("27_hebb_e_probe")
arule = importlib.import_module("31_a_rule_degeneracy")
wdir = importlib.import_module("37_weight_direction")
from concurrent.futures import ProcessPoolExecutor, as_completed
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("GATE38_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS = 240 if SMOKE else 12000
TRACE_PTS = 25
WORKERS = 2 if SMOKE else 12
NS = "comp25"
BETA = 0.5
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            transport="triangle", rule="softhebb", lr_e=0.1, ss=1e-2,
            beta=BETA, signed=True)

# (name, kind, param, a_frozen)
ARMS = (("frozen", "argmax", 0.0, True),
        ("argmax", "argmax", 0.0, False),
        ("centred", "centred", 0.0, False),
        ("shift25", "shift", 0.25, False),
        ("shift50", "shift", 0.50, False),
        ("shift75", "shift", 0.75, False),
        ("shift100", "shift", 1.00, False),
        ("topk2", "topk", 2, False),
        ("topk4", "topk", 4, False),
        ("unsigned", "unsigned", 0.0, False))

_ORIG = fc.softmax_wta
GSUM = []


def install(kind, param):
    """patch fc.softmax_wta per worker. gate29's pattern, extended."""
    base = _ORIG

    def rec(g):
        GSUM.append(float(g.sum()))
        return g

    if kind == "argmax":
        f = lambda u, beta=1.0, signed=False: rec(base(u, beta, signed))
    elif kind == "unsigned":
        f = lambda u, beta=1.0, signed=False: rec(base(u, beta, False))
    elif kind == "centred":
        def f(u, beta=1.0, signed=False):
            resp = torch.softmax(beta * u, dim=-1)
            return rec(resp - resp.mean() if signed else resp)
    elif kind == "shift":
        def f(u, beta=1.0, signed=False):
            g = base(u, beta, signed)
            if signed:
                # subtract a CONSTANT: every pairwise difference survives
                # exactly, only the row sum changes (scales by 1 - param)
                g = g - param * g.sum() / g.numel()
            return rec(g)
    elif kind == "topk":
        def f(u, beta=1.0, signed=False):
            resp = torch.softmax(beta * u, dim=-1)
            if not signed:
                return rec(resp)
            g = -resp
            idx = torch.topk(u, int(param)).indices
            g[idx] *= -1          # k winners positive, the rest repelled
            return rec(g)
    else:
        raise ValueError(kind)
    fc.softmax_wta = f


def cfg_hash(name, kind, param, a_frozen):
    return rep.arm_hash(f"gate38:{name}",
                        dict(BASE, gate=kind, gparam=param, a_frozen=a_frozen),
                        STEPS)


def run_one(job):
    seed, name, kind, param, a_frozen = job
    install(kind, param)
    GSUM.clear()
    try:
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **BASE)
        host = BareHost(sweep.BareAgtXC(
            cfg, _bare_path(f"gate38/w{os.getpid()}"), seed=seed,
            dire=False, a_frozen=a_frozen))
        agt = host.agt
        W0 = wdir.snapshot(agt)
        ci0, cc0, w0 = wdir.wstats(agt, W0)

        every = max(STEPS // TRACE_PTS, 1)
        trace, cross = [], {"t_maj75": None, "t_cos95": None, "t_cos90": None}
        for step, a in enumerate(de.kp1_cycle(
                STEPS, seed=de.derive_seed("kp1", NS, seed))):
            host.observe(a)
            if step % every and step != STEPS - 1:
                continue
            maj, _, npat = arule.degeneracy(agt)
            ci, cc, wn = wdir.wstats(agt, W0)
            trace.append([step, round(maj, 5), round(wn, 4),
                          round(ci, 6), round(cc, 6)])
            if cross["t_maj75"] is None and maj > 0.75:
                cross["t_maj75"] = step
            if cross["t_cos95"] is None and ci < 0.95:
                cross["t_cos95"] = step
            if cross["t_cos90"] is None and cc > 0.90:
                cross["t_cos90"] = step

        maj, minority, npat = arule.degeneracy(agt)
        ci, cc, wn = wdir.wstats(agt, W0)
        norms = [float(c.nr_1.actual.norm()) for l, c in agt.cols.items()
                 if not agt.is_i(l)]
        return {"seed": seed, "arm": name, "gate": kind, "gparam": param,
                "maj": maj, "minority": minority, "n_patterns": npat,
                "act_norm": sum(norms) / max(len(norms), 1),
                "w0": w0, "w_norm": wn, "cos_init": ci, "mean_abs_cos": cc,
                "mean_abs_cos0": cc0,
                "net_mass": (sum(GSUM) / len(GSUM)) if GSUM else float("nan"),
                "trace": trace, **cross,
                "config_hash": cfg_hash(name, kind, param, a_frozen)}
    finally:
        fc.softmax_wta = _ORIG


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "gate38"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, n, k, p, f) for (n, k, p, f) in ARMS for s in SEEDS
            if (cfg_hash(n, k, p, f), s) not in done]
    print(f"gate38: {len(ARMS)*len(SEEDS)} cells, {len(todo)} to run "
          f"(Dir.E FROZEN, rule=softhebb, beta {BETA} pinned, {STEPS} steps)",
          flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "gate38", "event_id":
                                 f"gate38:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['arm']:>9}: "
                  f"maj {s['maj']:.1%} |cos| {s['mean_abs_cos']:.3f} "
                  f"cos_init {s['cos_init']:+.4f} net {s['net_mass']:+.3f} "
                  f"|W| {s['w0']:.1f}->{s['w_norm']:.4g}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "gate38"]

    def agg(name, kind, param, a_frozen):
        g = [e for e in led
             if e.get("config_hash") == cfg_hash(name, kind, param, a_frozen)]
        if not g:
            return None
        f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
        r = {k: f(k) for k in ("maj", "n_patterns", "act_norm", "w0", "w_norm",
                               "cos_init", "mean_abs_cos", "net_mass")}
        for k in ("t_maj75", "t_cos95", "t_cos90"):
            v = [e.get(k) for e in g if e.get(k) is not None]
            r[k] = (sum(v) / len(v)) if v else None
        r["arm"] = name
        return r

    rows = {a[0]: agg(*a) for a in ARMS}
    fz = rows.get("frozen")
    print(f"\ngate family x rank-1 collapse, beta {BETA}, Dir.E frozen, "
          f"seeds {list(SEEDS)}, {STEPS} steps")
    if fz:
        print(f"frozen twin: maj {fz['maj']:.1%} |W| {fz['w_norm']:.2f} "
              f"|cos| {fz['mean_abs_cos']:.3f} (random-vector baseline ~0.14)")
    print("|cos| is the rank-1 detector: near 0.14 means columns still "
          "distinct, near 1.0 means the matrix has collapsed to one "
          "direction.\n")
    print(f"{'arm':>9} {'net':>8} {'maj':>7} {'patt':>6} {'|a|':>6} "
          f"{'|W|/W0':>9} {'cos_init':>9} {'|cos|':>7} {'t_maj75':>8} "
          f"{'t_cos90':>8}  status")
    for (name, kind, param, a_frozen) in ARMS:
        r = rows.get(name)
        if not r:
            continue
        g = lambda k: ("-" if r[k] is None else f"{r[k]:.0f}")
        moved = fz and r["w_norm"] > 1.3 * fz["w_norm"]
        merged = fz and r["mean_abs_cos"] > 3 * fz["mean_abs_cos"]
        st = ("INERT" if name != "frozen" and not moved else
              "RANK-1" if merged else "ok")
        print(f"{name:>9} {r['net_mass']:+8.3f} {r['maj']:7.1%} "
              f"{r['n_patterns']:6.2f} {r['act_norm']:6.2f} "
              f"{r['w_norm']/r['w0']:9.4g} {r['cos_init']:+9.4f} "
              f"{r['mean_abs_cos']:7.3f} {g('t_maj75'):>8} {g('t_cos90'):>8}"
              f"  {st}")

    if SMOKE:
        print(f"\nSMOKE at {STEPS} steps: reading DISABLED. Rank-1 collapse is "
              "cumulative; 240 steps is 2% of the horizon.")
        return

    live = [r for r in rows.values() if r and r["arm"] != "frozen"]
    print("\nreading:")
    if fz:
        working = [r for r in live
                   if r["mean_abs_cos"] < 2 * fz["mean_abs_cos"]
                   and r["w_norm"] > 1.3 * fz["w_norm"]
                   and r["maj"] < 0.75
                   and r["act_norm"] > 0.5 * fz["act_norm"]]
        inert = [r["arm"] for r in live if r["w_norm"] <= 1.3 * fz["w_norm"]]
        rank1 = [r["arm"] for r in live
                 if r["mean_abs_cos"] > 3 * fz["mean_abs_cos"]]
        print(f"  INERT (|W| never moved, uncollapsed for the wrong reason): "
              f"{inert or 'none'}")
        print(f"  RANK-1 (|cos| > 3x frozen): {rank1 or 'none'}")
        shifts = [r for r in live if r["arm"].startswith("shift")]
        if shifts:
            sr = [r["arm"] for r in shifts
                  if r["mean_abs_cos"] > 3 * fz["mean_abs_cos"]]
            print(f"  shift arms that collapsed anyway: {sr or 'none'} "
                  "(these preserve every pairwise gate difference EXACTLY, so "
                  "if they collapse, loser ordering is not the mechanism)")
        if working:
            print("  ** CANDIDATE RULES: "
                  + ", ".join(f"{r['arm']} (|cos| {r['mean_abs_cos']:.3f}, "
                              f"maj {r['maj']:.1%}, |W| x"
                              f"{r['w_norm']/r['w0']:.2f})" for r in working))
            print("  -> the first arm in this arc that both learns and stays "
                  "readable. Confirm on fresh seeds before building on it.")
        else:
            print("  -> NO candidate. Every gate shape either collapses to "
                  "rank 1 or does nothing. The collapse is a property of "
                  "GATED-OJA ITSELF, not of the gate, and the gate family is "
                  "exhausted. The next move changes the update rule.")
    print("\nnet mass is the mean gate row sum. Run 33 found it load-bearing "
          "on CIFAR (+0.305 there); it is -0.845 for argmax here, the opposite "
          "sign, which is why the CIFAR shift result does not transfer.")


if __name__ == "__main__":
    main()
