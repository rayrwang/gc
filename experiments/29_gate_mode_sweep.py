"""
Gate winner-count (Dir.A) x E-rule (Dir.E) cross (RW, 2026-07-25).

Two marginal sweeps preceded this and neither can see what this is for.
  27_hebb_e_probe swept Dir.E's rule with the Dir.A gate pinned at argmax.
  The gate half of this file swept Dir.A's gate with Dir.E pinned at delta.
RW's question: cross them.

THE GATE HALF. COMP25 swept the three named competition knobs (sign,
temperature, power) and found the axis alone does not fix the substrate. All
three were swept with a FOURTH thing pinned: how many units get positive
learning credit. softmax_wta always exempts exactly one:

    resp = softmax(beta*u); gate = -resp; gate[argmax] *= -1

so its NET mass is 2*p_win - 1, negative whenever the winner holds under half
the softmax mass. Measured at beta 0.5: -0.786. Every column takes a net
negative update on nearly every step, one rotating exemption. That is the
shape of the collapse COMP25 measured (94% majority-sign / 89% near-one-vs-
rest at beta 0.5, easing to 87%/50% at beta 2.0 where p_win is larger).
    argmax    control, exactly one positive, net mass -0.786
    centered  gate = resp - resp.mean(); mean(softmax) == 1/d exactly, so
              "beat uniform to get credit". Net mass +0.000 by construction.
    topk4     four winners. Separates "more than one winner" from "zero net
              mass", which centered confounds.

THE CROSS, and why it can kill the standing result. Round 1 found oja the
only error-free E rule that stayed bounded, beating frozen 3/3 paired, while
hebb diverged at x139. Both were measured on a substrate at 95% majority-sign,
i.e. collapsed. Two live possibilities:
  (a) hebb diverges BECAUSE the substrate is collapsed (activity pinned to one
      axis => large correlated post, and hebb has no bound but the NLMS
      denominator). On a centered-gate substrate it may stay finite.
  (b) OJA'S WIN IS AN ARTIFACT OF THE BROKEN SUBSTRATE. If what oja's -w*y*y
      decay was protecting against is the collapse itself, then fixing the
      gate should shrink or erase its advantage, and oja is not a better E
      rule, only a better rule for a collapsed one.
(b) is the one to want, because it is how the standing finding dies cheaply
rather than expensively later.

Pre-registered readings, in order:
  PRIMARY   does the gate fix the collapse? centered/topk4 cut near-one-vs-
            rest vs argmax => argmax implicated. No change => look elsewhere.
  CROSS     does the E-rule ranking change with the gate? oja's margin over
            delta shrinking under centered supports (b).
  BOUNDED   does hebb stop diverging under centered? supports (a).
beta fixed at 0.5: the default every registered fan3 run used, where the
collapse is worst and where round 1's oja finding was measured, so the
comparison is matched. beta 2.0 is a follow-up if an interaction shows.

Patches fc.softmax_wta inside each worker rather than editing dire_hosts.py,
so it cannot disturb runs in flight. If a mode wins it becomes a real
BareXCfg knob and this probe is superseded. Summaries only (kind gate29).
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
frz = importlib.import_module("14_registered_bare_freeze")
rep = importlib.import_module("11_replicate_instar_relu")
probe = importlib.import_module("27_hebb_e_probe")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("GATE_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 2 if SMOKE else 18  # each worker ~= 1 core (measured); 6 left 18 cores idle
NS = "comp25"
MODES = ("argmax", "centered", "topk4")
E_RULES = ("delta", "oja", "hebb")
BETA = 0.5
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1, beta=BETA)

_ORIG = fc.softmax_wta
GATE_SUMS = []


def make_gate(mode):
    def gate_fn(u, beta=1.0, signed=False):
        if mode == "argmax" or not signed:
            g = _ORIG(u, beta=beta, signed=signed)
        else:
            resp = torch.softmax(beta * u, dim=-1)
            if mode == "centered":
                g = resp - resp.mean()
            elif mode == "topk4":
                g = -resp
                g[torch.topk(u, min(4, u.numel())).indices] *= -1
            else:
                raise ValueError(mode)
        GATE_SUMS.append(float(g.sum()))
        return g
    return gate_fn


def cfg_hash(mode, e_rule):
    return rep.arm_hash(f"gate29:{mode}:{e_rule}", BASE, STEPS)


def degeneracy(agt):
    """comp25's pair: majority-sign fraction, near-one-vs-rest (min <= 2)."""
    maj = ovr = n = 0
    for loc, c in agt.cols.items():
        if agt.is_i(loc):
            continue
        x = c.nr_1.actual
        pos, d = int((x > 0).sum()), x.numel()
        maj += max(pos, d - pos) / d
        ovr += (min(pos, d - pos) <= 2)
        n += 1
    return maj / max(n, 1), ovr / max(n, 1)


def run_one(job):
    seed, mode, e_rule = job
    fc.softmax_wta = make_gate(mode)  # per-worker; shared code untouched
    GATE_SUMS.clear()
    out = {"seed": seed, "mode": mode, "e_rule": e_rule, "beta": BETA}
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **BASE)
        host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"gate29/w{os.getpid()}"),
                                        seed=seed, dire_rule=e_rule))
        agt = host.agt
        r = frz.run_light(host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                          WINDOW)
        out[fixture] = r["margin"]
        out[f"{fixture}_nan"] = r["nan"]
        if fixture == "kp1":
            out["maj"], out["ovr"] = degeneracy(agt)
            out["gate_sum"] = sum(GATE_SUMS) / len(GATE_SUMS) if GATE_SUMS else None
            _, out["e_norm_max"] = probe.e_norm_stats(agt)
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out["kp1"], out["kz"]) else None)
    out["config_hash"] = cfg_hash(mode, e_rule)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "gate29"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    jobs = [(s, m, r) for m in MODES for r in E_RULES for s in SEEDS
            if (cfg_hash(m, r), s) not in done]
    print(f"gate29: {len(MODES)*len(E_RULES)*len(SEEDS)} cells "
          f"({len(MODES)} gates x {len(E_RULES)} E-rules x {len(SEEDS)} seeds), "
          f"{len(jobs)} to run, beta={BETA}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "gate29", "event_id":
                                 f"gate29:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(jobs)} s{s['seed']} {s['mode']}/{s['e_rule']}: "
                  f"maj {s['maj']:.0%} ovr {s['ovr']:.0%} "
                  f"Emax {s['e_norm_max']:.0f} diff {s['differential']}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "gate29"]
    tab = {}
    for m in MODES:
        for r in E_RULES:
            g = [e for e in led if e.get("config_hash") == cfg_hash(m, r)]
            if not g:
                continue
            f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
            emx = probe.finite([e.get("e_norm_max") for e in g])
            tab[(m, r)] = {"maj": f("maj"), "ovr": f("ovr"), "diff": f("differential"),
                           "gsum": f("gate_sum"),
                           "emax": max(emx) if emx else float("nan"),
                           "div": (not emx) or len(emx) < len(g) or max(emx) > 1e3,
                           "diffs": probe.finite([e.get("differential") for e in g])}

    print(f"\ngate mode (Dir.A) x E-rule (Dir.E), beta {BETA}, seeds {list(SEEDS)}")
    print("comp25 argmax reference at beta 0.5: 94% maj / 89% one-v-rest\n")
    print(f"{'gate':>9} {'E-rule':>7} {'maj':>6} {'one-v-rest':>11} {'net gate':>9} "
          f"{'E-max':>10} {'diff':>9}  status")
    for m in MODES:
        for r in E_RULES:
            c = tab.get((m, r))
            if not c:
                continue
            print(f"{m:>9} {r:>7} {c['maj']:6.0%} {c['ovr']:11.0%} "
                  f"{c['gsum']:+9.3f} {c['emax']:10.1f} {c['diff']:+9.4f}  "
                  f"{'DIVERGED' if c['div'] else 'ok'}")

    print("\nPRIMARY, does the gate fix the collapse (read down the delta column):")
    base = tab.get(("argmax", "delta"))
    for m in MODES:
        c = tab.get((m, "delta"))
        if not c or not base:
            continue
        d = c["ovr"] - base["ovr"]
        print(f"  {m:>9}: one-v-rest {c['ovr']:.0%} ({d:+.0%} vs argmax)  "
              + ("ARGMAX IMPLICATED" if d < -0.10 else
                 "no material change" if abs(d) <= 0.10 else "worse"))

    print("\nCROSS, does the E-rule ranking survive fixing the gate:")
    for m in MODES:
        o, dl = tab.get((m, "oja")), tab.get((m, "delta"))
        if not o or not dl:
            continue
        adv = o["diff"] - dl["diff"]
        print(f"  {m:>9}: oja - delta = {adv:+.4f}"
              + ("   (round-1 argmax value was +0.177)" if m == "argmax" else ""))
    a = tab.get(("argmax", "oja"))
    c = tab.get(("centered", "oja"))
    d0 = tab.get(("argmax", "delta"))
    d1 = tab.get(("centered", "delta"))
    if a and c and d0 and d1:
        s0, s1 = a["diff"] - d0["diff"], c["diff"] - d1["diff"]
        print(f"\n  oja's advantage over delta: {s0:+.4f} under argmax -> "
              f"{s1:+.4f} under centered")
        print("  " + ("SHRINKS: oja's win was substantially an artifact of the "
                      "collapsed substrate (possibility b)" if s1 < 0.5 * s0 else
                      "HOLDS: oja is a better E rule independent of the collapse"))

    print("\nBOUNDED, does hebb stop diverging when the gate is fixed:")
    for m in MODES:
        h = tab.get((m, "hebb"))
        if h:
            print(f"  {m:>9}: E-max {h['emax']:.1f}  "
                  f"{'still DIVERGED' if h['div'] else 'BOUNDED (possibility a)'}")


if __name__ == "__main__":
    main()
