"""
Does the sign of gate*u explain the |W| runaway? (RW, 2026-07-26.)
Free play, seeds 50-52, fan3, Dir.E FROZEN. Same arms as beta34.

WHAT RUN 34 FOUND, AND WHY IT NEEDS A MECHANISM BEFORE A FIX. The beta sweep
came back bimodal with nothing in between: every signed-gate arm from beta
0.5 to 2 learns and collapses, and every arm at beta 3+ is uncollapsed only
because it is inert (|W| equal to frozen to three significant figures). The
surprise was the middle column:

      arm          net      |W|        vs frozen
   frozen         +nan     11.32          1x
     b0.5       -0.845      1728        153x     <- THE INCUMBENT PRESET
       b1       -0.610   3.31e+05      29202x
    b1.25       -0.455   2.72e+06     240590x
     b1.5       -0.294   2.13e+07    1877413x   <- peak, then it falls back
    b1.75       -0.107   1.16e+07    1025409x
       b2       +0.061   2.43e+05      21427x
       b3       +0.998     11.24       0.99x
 unsigned       +1.000     30.66       2.71x

b0.5 is not a new arm. It is the faithful preset that comp25, gate29,
arule31 and every fan3 registered run has used. Run 34 was simply the first
experiment ever to measure a Dir.A weight norm on this substrate (grep:
w_norm appears only in 32, 33 and 34, and 27's e_norm_growth is the Dir.E
side). So the runaway is presumably months old and nothing was looking.

THE ARITHMETIC THAT MAKES THIS TESTABLE. From funcs.py:241,

    lrn_oja_gated(x, w, gate, u, ss) = w + ss*(outer(x, gate) - w*(gate*u))

`gate*u` is elementwise over d_y and `w*(gate*u)` broadcasts over the last
axis, so column j of w evolves as

    w[:,j]  <-  w[:,j] * (1 - ss*gate[j]*u[j])  +  ss*x*gate[j]

The second term is a DECAY term only when gate[j]*u[j] > 0. When it is
negative the multiplier exceeds 1 and that column grows geometrically. With
a signed gate every loser has gate[j] < 0, and roughly half of u is negative
on this substrate (arule31 measured 69.6% negative postsynaptic activity),
so about half the loser columns should be in geometric growth every step. At
ss=0.01 over 12000 steps a persistent product of -0.05 gives a factor of 400.

That is a hypothesis with an arithmetic mechanism, not a finding. This run
tests it directly rather than throwing a known-good normaliser (BN) at an
undiagnosed phenomenon, which would produce an outcome with no mechanism.
Run 33's lesson: |W| is what turned "centring is bad" into "net mass sustains
the weights against decay".

THE MEAN-FIELD VERSION OF THAT STORY IS WRONG, AND THE SMOKE SAID SO BEFORE
THE RUN. mean_gu came back POSITIVE in every arm (+0.010 to +0.128), which
under the mean-field reading predicts |W| SHRINKING, against a measured 153x
growth at b0.5. Two things were missing. The additive Hebbian term was
ignored, and the decay term is weighted by COLUMN NORMS, not uniform. So
this run measures the exact first-order decomposition instead:

    d|w|^2 = 2*ss*[ <w, outer(x,gate)> - sum_j gu[j]*|w[:,j]|^2 ]

with <w, outer(x,gate)> = x @ w @ gate, so no outer product is formed. An
unweighted mean of gu can be positive while the weighted sum is negative,
if the columns sitting in the negative regime are the large ones. That is
exactly the heterogeneity the mean-field version washed out.

MEASURED, by patching fc.lrn_oja_gated in-process (it is not torch.compiled;
src/funcs.py untouched). Each conn is updated once per step, so a per-call
mean is a per-step rate:
  hebb_rate      ss * <w, outer(x,gate)> / |w|^2, halved for d log|w|.
                 The Hebbian term's contribution to the growth rate.
  decay_rate     -ss * sum_j gu[j]*|w[:,j]|^2 / |w|^2, halved. The gated-Oja
                 term. Negative means it shrinks the weights.
  pred_rate      hebb_rate + decay_rate, the predicted per-step log-growth.
  w1_pred        w0 * exp(pred_rate * STEPS), against the measured w1.
  mean_gu        UNWEIGHTED mean of gate*u, kept as a diagnostic so the
                 weighted/unweighted divergence is visible.
  frac_neg       fraction of elements with gate*u < 0.
  half1 / half2  per-step log-slopes of the sampled |W| trajectory over the
                 first and second halves. Equal slopes mean a constant
                 exponential rate; half2 >> half1 means the balance between
                 hebb and decay SHIFTS during training, which the smoke
                 already hints at (b0.5's 240-step rate extrapolates to
                 |W| 11.9 at 12000 steps, against a measured 1728).

READING, fixed before the run:
  w1_pred within 3x of w1 on all but at most one arm
                        -> CONFIRMED. Whichever term dominates is what BN
                           has to change, and that becomes a prediction the
                           BN run can fail.
  signs and ordering track but magnitudes do not
                        -> PARTIAL. Higher-order effects or cross-step
                           correlation carry the rest; half1 vs half2 says
                           which.
  neither sign nor ordering tracks
                        -> REFUTED. Next suspects: recurrence (w feeds
                           activity feeds gate feeds w), and CIFAR's
                           ~1024-position averaging against this
                           single-sample update.

Summaries only (kind gu35).
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
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("GU_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 2 if SMOKE else 12
NS = "comp25"
BASE = beta34.BASE          # identical substrate config to run 34
ARMS = beta34.ARMS          # identical arms, so |W| is directly comparable
SAMPLE_EVERY = 200 if SMOKE else 20000   # trajectory samples, in rule calls

_ORIG_LRN = fc.lrn_oja_gated
STATS = {}


def reset_stats():
    STATS.clear()
    STATS.update(n_calls=0, n_elem=0, sum_gu=0.0, n_neg=0, sum_neg_gu=0.0,
                 sum_rel_h=0.0, sum_rel_d=0.0, n_rel=0, traj=[])


def install_recorder():
    """record the gate*u statistics that decide growth vs decay, per column.

    fc.lrn_oja_gated is a plain function (not torch.compile'd) and the call
    site in BareAgtX.step resolves it through the module, so replacing the
    attribute takes effect. Everything runs on CPU here, so float() costs no
    device sync.
    """
    base = _ORIG_LRN

    def lrn(x, w, gate, u, ss=1e-2):
        gu = gate * u
        STATS["n_calls"] += 1
        STATS["n_elem"] += gu.numel()
        STATS["sum_gu"] += float(gu.sum())
        neg = gu[gu < 0]
        STATS["n_neg"] += neg.numel()
        if neg.numel():
            STATS["sum_neg_gu"] += float(neg.sum())
        # EXACT first-order decomposition of the change in |w|^2:
        #   d|w|^2 = 2*ss*[ <w, outer(x,gate)> - sum_j gu[j]*|w[:,j]|^2 ]
        # and <w, outer(x,gate)> = x @ w @ gate, so no outer product is
        # formed. Note the decay term is WEIGHTED BY COLUMN NORMS: an
        # unweighted mean of gu can be positive while this is negative, if
        # the columns sitting in the negative regime are the large ones.
        colsq = (w * w).sum(0)
        wsq = float(colsq.sum())
        if wsq > 0:
            h = 2 * ss * float((x @ w) @ gate) / wsq
            d = -2 * ss * float((gu * colsq).sum()) / wsq
            STATS["sum_rel_h"] += h
            STATS["sum_rel_d"] += d
            STATS["n_rel"] += 1
        if STATS["n_calls"] % SAMPLE_EVERY == 0:
            STATS["traj"].append((STATS["n_calls"], float(w.norm())))
        return base(x, w, gate, u, ss=ss)

    fc.lrn_oja_gated = lrn


def log_slope(pts):
    """least-squares slope of log(norm) against call index. nan if unusable."""
    pts = [(i, n) for i, n in pts if n > 0]
    if len(pts) < 3:
        return float("nan")
    xs = [i for i, _ in pts]
    ys = [math.log(n) for _, n in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def cfg_hash(name, beta, signed, a_frozen):
    return rep.arm_hash(f"gu35:{name}",
                        dict(beta34.arm_cfg(beta, signed), a_frozen=a_frozen),
                        STEPS)


def run_one(job):
    seed, name, beta, signed, a_frozen = job
    cfg_d = beta34.arm_cfg(beta, signed)
    ss = cfg_d["ss"]
    out = {"seed": seed, "arm": name, "beta": beta, "signed": signed}
    install_recorder()
    try:
        for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
            reset_stats()
            cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                                  n_cols=sweep.N_COLS, **cfg_d)
            host = BareHost(sweep.BareAgtXC(
                cfg, _bare_path(f"gu35/w{os.getpid()}"), seed=seed,
                dire=False, a_frozen=a_frozen))
            agt = host.agt
            w0 = beta34.w_norm(agt)          # BEFORE any learning
            r = frz.run_light(host,
                              fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                              WINDOW)
            out[fixture] = r["margin"]
            out[f"{fixture}_nan"] = r["nan"]
            if fixture != "kp1":
                continue
            w1 = beta34.w_norm(agt)
            out["w0"], out["w1"] = w0, w1
            out["maj"], out["minority"], out["n_patterns"] = \
                arule.degeneracy(agt)
            ne = max(STATS["n_elem"], 1)
            out["mean_gu"] = STATS["sum_gu"] / ne
            out["frac_neg"] = STATS["n_neg"] / ne
            out["mean_neg_gu"] = (STATS["sum_neg_gu"] / STATS["n_neg"]
                                  if STATS["n_neg"] else float("nan"))
            out["meas_rate"] = (math.log(w1 / w0) / STEPS
                                if w0 > 0 and w1 > 0 else float("nan"))
            # each conn is updated once per step, so a per-call mean IS the
            # per-step rate. d log|w| = d|w|^2 / (2 |w|^2), hence the halving.
            nr = max(STATS["n_rel"], 1)
            out["hebb_rate"] = STATS["sum_rel_h"] / nr / 2
            out["decay_rate"] = STATS["sum_rel_d"] / nr / 2
            out["pred_rate"] = out["hebb_rate"] + out["decay_rate"]
            out["w1_pred"] = w0 * math.exp(out["pred_rate"] * STEPS)
            # trajectory slopes convert to per-step rates via calls-per-step
            cps = STATS["n_calls"] / max(STEPS, 1)
            tr = STATS["traj"]
            mid = len(tr) // 2
            out["calls_per_step"] = cps
            out["half1"] = log_slope(tr[:mid]) * cps if mid >= 3 else float("nan")
            out["half2"] = log_slope(tr[mid:]) * cps if len(tr) - mid >= 3 \
                else float("nan")
    finally:
        fc.lrn_oja_gated = _ORIG_LRN
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out.get("kp1"), out.get("kz"))
                           else None)
    out["config_hash"] = cfg_hash(name, beta, signed, a_frozen)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "gu35"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, n, b, g, f) for (n, b, g, f) in ARMS for s in SEEDS
            if (cfg_hash(n, b, g, f), s) not in done]
    print(f"gu35: {len(ARMS)*len(SEEDS)} cells, {len(todo)} to run "
          f"(Dir.E FROZEN, rule=softhebb, arms identical to beta34)",
          flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "gu35", "event_id":
                                 f"gu35:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['arm']}: "
                  f"hebb {s['hebb_rate']:+.2e} decay {s['decay_rate']:+.2e} "
                  f"pred {s['pred_rate']:+.2e} meas {s['meas_rate']:+.2e} "
                  f"|W| {s['w0']:.1f}->{s['w1']:.4g}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "gu35"]
    print(f"\ngate*u vs weight growth, Dir.E frozen, seeds {list(SEEDS)}, "
          f"{STEPS} steps")
    print("exact first-order decomposition: d|W|^2 = 2*ss*[<W, outer(x,gate)> "
          "- sum_j gu[j]*|w[:,j]|^2]. hebb and decay are those two terms as "
          "per-step log-growth rates; pred is their sum.\n")
    print(f"{'arm':>9} {'mean_gu':>9} {'hebb':>10} {'decay':>10} "
          f"{'pred':>10} {'meas':>10} {'|W| pred':>11} {'|W| meas':>11} "
          f"{'half1':>10} {'half2':>10}")
    rows = []
    for (name, beta, signed, a_frozen) in ARMS:
        g = [e for e in led
             if e.get("config_hash") == cfg_hash(name, beta, signed, a_frozen)]
        if not g:
            continue
        f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
        row = {k: f(k) for k in ("mean_gu", "frac_neg", "mean_neg_gu",
                                 "w0", "w1", "w1_pred", "hebb_rate",
                                 "decay_rate", "pred_rate", "meas_rate",
                                 "half1", "half2")}
        row["arm"] = name
        rows.append(row)
        print(f"{name:>9} {row['mean_gu']:+9.4f} {row['hebb_rate']:+10.2e} "
              f"{row['decay_rate']:+10.2e} {row['pred_rate']:+10.2e} "
              f"{row['meas_rate']:+10.2e} {row['w1_pred']:11.4g} "
              f"{row['w1']:11.4g} {row['half1']:+10.2e} {row['half2']:+10.2e}")

    live = [r for r in rows if r["arm"] != "frozen"
            and r["meas_rate"] == r["meas_rate"]
            and r["pred_rate"] == r["pred_rate"]]
    if SMOKE:
        print(f"\nSMOKE at {STEPS} steps: the reading below is DISABLED. The "
              "runaway is cumulative (|W| moves 11.3 -> 1728 at 12000 steps "
              "and barely at all at 240), so growth rates measured here are "
              "noise around zero. Smoke shows the code paths execute.")
        return
    print("\nreading:")
    if len(live) >= 3:
        close = [r["arm"] for r in live
                 if r["w1_pred"] > 0 and r["w1"] > 0
                 and abs(math.log(r["w1_pred"] / r["w1"])) < math.log(3)]
        sign_ok = all((r["pred_rate"] > 0) == (r["meas_rate"] > 0)
                      for r in live)
        order_ok = all(
            (a["pred_rate"] <= b["pred_rate"])
            == (a["meas_rate"] <= b["meas_rate"])
            for i, a in enumerate(live) for b in live[i + 1:])
        hebb_wins = [r["arm"] for r in live
                     if abs(r["hebb_rate"]) > abs(r["decay_rate"])]
        print(f"  arms whose predicted |W| is within 3x of measured: "
              f"{close or 'none'} (of {len(live)})")
        print(f"  sign of pred matches sign of meas: {sign_ok}")
        print(f"  ordering of meas matches ordering of pred: {order_ok}")
        print(f"  arms where the HEBBIAN term dominates: {hebb_wins or 'none'}")
        print(f"  arms where the DECAY term dominates: "
              f"{[r['arm'] for r in live if r['arm'] not in hebb_wins] or 'none'}")
        if len(close) >= max(1, len(live) - 1):
            print("  -> CONFIRMED. The first-order decomposition accounts for "
                  "the growth. Whichever term dominates is what BN has to "
                  "change, and that is a prediction the BN run can fail.")
        elif sign_ok and order_ok:
            print("  -> PARTIAL. The decomposition gets the direction and the "
                  "ordering but not the magnitude, so higher-order or "
                  "cross-step correlation is carrying the rest.")
        else:
            print("  -> REFUTED. First-order local analysis does not explain "
                  "it. Next suspects: recurrence (w feeds activity feeds "
                  "gate), and CIFAR's ~1024-position averaging against this "
                  "single-sample update.")
    else:
        print("  too few readable arms to judge; report the table as-is.")
    print("\nmean_gu is the UNWEIGHTED mean of gate*u, kept as a diagnostic. "
          "decay_rate is the same quantity WEIGHTED BY COLUMN NORMS, which is "
          "what actually enters d|W|^2. They can differ in sign when the "
          "columns sitting in the negative regime are the large ones.")
    print("half1 vs half2 are per-step log-slopes of the sampled |W| "
          "trajectory. Equal slopes mean growth at a constant exponential "
          "rate, which is what the first-order story predicts.")


if __name__ == "__main__":
    main()
