"""
Is the antipodal collapse a SUBSTRATE problem or a SOFTHEBB problem?
(RW, 2026-07-25.) Free play, seeds 50-52, fan3, Dir.E FROZEN throughout.

Why this was never asked. The 216-config sweeps varied the Dir.A rule across
{softhebb, oja, instar, basic} over 650+ runs, but their ledger rows carry
alive/passed/silent/honest/nan/n_pred/margins and NO degeneracy statistic:
the collapse measure did not exist yet, it was invented by the
24_watch_fan3 debugger observation months later. Everything after that
observation (comp25, aoff26, the fan3 registered runs, hebb27/ojab2/gate29)
pinned rule="softhebb", the faithful preset. So the collapse has only ever
been measured on ONE A-rule.

And it matters, because only softhebb uses the gate:
    softhebb -> fc.softmax_wta(y, beta, signed) then lrn_oja_gated
    oja      -> fc.lrn_oja        (no gate)
    instar   -> fc.lrn_instar     (no gate)
    basic    -> fc.lrn_basic      (no gate)
The argmax gate is what gate29 implicated as the cause (net mass -0.845,
one-vs-rest 93% -> 0% when centred). Three of the four A-rules cannot have
a gate-induced collapse because they have no gate. So "the substrate
collapses" may be "the faithful preset collapses", and gate29's centred fix
may be a fix to softhebb rather than to the substrate.

Dir.E is frozen in every cell so nothing on the E side confounds the read.

STATISTICS, and note the old one is deliberately not used as a gate.
gate29 found that `min(pos,neg) <= 2` hard-codes a one-winner gate: a
4-winner gate collapses to a minority of exactly 4, fails the test, and the
metric reports 0% collapse on a totally collapsed network. Replacements here
are threshold-free:
    maj          majority-sign fraction, max(pos,neg)/d  (comp25 refs:
                 twin ~58% = chance for d=32, collapsed learner ~94%)
    minority     min(pos,neg)/d as a CONTINUOUS value, no threshold
    n_patterns   distinct sign patterns across columns / n columns. Total
                 collapse => every column shares one pattern => ~0.
                 This is the k-agnostic measure; it cannot be fooled by
                 changing how many units are in the minority.
    maj_sd       spread of maj ACROSS SEEDS, reported at aggregation. This
                 was the tell that exposed topk4: identical to 4 s.f. across
                 seeds means the state is determined by the rule, not the
                 data. Near-zero spread is itself a collapse signature.
    act_norm     mean activation norm, so a dead network is not read as a
                 healthy uncollapsed one.
Summaries only (kind arule31).
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
from src import iotypes as T

SMOKE = bool(os.environ.get("ARULE_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 2 if SMOKE else 12
NS = "comp25"
A_RULES = ("softhebb", "oja", "instar", "basic")
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            transport="triangle", ss=1e-2, lr_e=0.1, beta=0.5)


def cfg_hash(rule):
    return rep.arm_hash(f"arule31:{rule}", dict(BASE, rule=rule), STEPS)


def degeneracy(agt):
    """threshold-free collapse statistics."""
    majs, mins, pats = [], [], set()
    for loc, c in agt.cols.items():
        if agt.is_i(loc):
            continue
        x = c.nr_1.actual
        d = x.numel()
        pos = int((x > 0).sum())
        majs.append(max(pos, d - pos) / d)
        mins.append(min(pos, d - pos) / d)
        pats.add(tuple((x > 0).tolist()))
    n = max(len(majs), 1)
    return (sum(majs) / n, sum(mins) / n, len(pats) / n)


def run_one(job):
    seed, rule = job
    out = {"seed": seed, "a_rule": rule}
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **dict(BASE, rule=rule))
        # Dir.E frozen everywhere: this is a pure Dir.A read
        host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"arule31/w{os.getpid()}"),
                                        seed=seed, dire=False))
        agt = host.agt
        r = frz.run_light(host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                          WINDOW)
        out[fixture] = r["margin"]
        out[f"{fixture}_nan"] = r["nan"]
        if fixture == "kp1":
            out["maj"], out["minority"], out["n_patterns"] = degeneracy(agt)
            norms = [float(c.nr_1.actual.norm()) for l, c in agt.cols.items()
                     if not agt.is_i(l)]
            out["act_norm"] = sum(norms) / max(len(norms), 1)
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out["kp1"], out["kz"]) else None)
    out["config_hash"] = cfg_hash(rule)
    return out


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "arule31"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, r) for r in A_RULES for s in SEEDS
            if (cfg_hash(r), s) not in done]
    print(f"arule31: {len(A_RULES)*len(SEEDS)} cells, {len(todo)} to run "
          f"(Dir.E FROZEN in all)", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "arule31", "event_id":
                                 f"arule31:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['a_rule']}: "
                  f"maj {s['maj']:.1%} minority {s['minority']:.1%} "
                  f"patterns {s['n_patterns']:.2f} |a| {s['act_norm']:.2f}",
                  flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "arule31"]
    print("\nDir.A rule x collapse, Dir.E frozen, seeds "
          f"{list(SEEDS)}, {STEPS} steps")
    print("refs: twin (no learning) ~58% maj = chance for d=32; "
          "collapsed softhebb learner ~94%; gate29 centred ~66%\n")
    print(f"{'A-rule':>9} {'gate?':>6} {'maj':>7} {'maj_sd':>8} {'minority':>9} "
          f"{'patterns':>9} {'|a|':>7} {'diff':>9}")
    for r in A_RULES:
        g = sorted([e for e in led if e.get("config_hash") == cfg_hash(r)],
                   key=lambda x: x["seed"])
        if not g:
            continue
        majs = probe.finite([e.get("maj") for e in g])
        f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
        print(f"{r:>9} {'yes' if r=='softhebb' else 'no':>6} "
              f"{f('maj'):7.1%} {sd(majs):8.4f} {f('minority'):9.1%} "
              f"{f('n_patterns'):9.2f} {f('act_norm'):7.2f} "
              f"{f('differential'):+9.4f}")

    print("\nreading:")
    print("  maj near 58% and patterns near 1.0  -> not collapsed")
    print("  maj near 94% and patterns near 0    -> collapsed")
    print("  maj_sd near 0 across seeds          -> state set by the rule, "
          "not the data (this is what exposed topk4)")
    print("  |a| near 0                          -> dead, not healthy: a "
          "silent net is trivially 'uncollapsed'")
    print("\nIF only softhebb collapses: the two days of collapse work were "
          "about the faithful preset, not the substrate, and gate29's centred "
          "gate is a softhebb fix.")
    print("IF all four collapse: it is substrate-level, the gate is one route "
          "in, and there must be a second mechanism since three of these have "
          "no gate at all.")


if __name__ == "__main__":
    main()
