"""
Dir.E rule-form x step-size sweep at the bare rung (RW, 2026-07-25).
Free play, seeds 50-52, comp25 streams, so these cells are directly
comparable to the 2x2 that 26_a_off_probe.py completed.

Motivation. AOFF26 showed delta converges to a near-perfect, exactly
fixture-blind predictor: the failure is AT the least-squares optimum, not
short of it. COMP25 generalised that to "E-learning actively obscures".
Both are properties of a least-squares objective. An error-free rule has no
such objective, so no reason to land on that fixed point. This keeps the
target and drops the least-squares criterion: a third axis alongside
AOFF26's change-the-TARGET and change-the-DYNAMICS.

Round 1 (lr_e=0.1 only) found: rig validated (delta -0.1201 and frozen
-0.0137 reproduce comp25's learner/dire_off cells to 4dp); oja the only
error-free rule that stayed bounded, beating both references 3/3 paired;
hebb, instar and softhebb all diverged. So round 1 could not separate
"this rule does not help" from "this rule diverges at this step size".
That is what the lr_e axis is for.

Reading, decided before the run:
    an arm BEATS FROZEN            -> rule form matters, not just amount
    an arm lands frozen..delta     -> amount-of-E-learning axis, so the
                                      answer is a different target instead
    an arm at or below delta       -> no recovery
Vitals: E-norm divergence is the ONLY discriminating vital. Majority-sign is
A-side antipodal collapse (95.3% in every arm INCLUDING frozen, where E never
updates), so it is reported and never gated on: round 1 gated at 0.93 and
made every arm unreadable, which was a harness bug, not a finding.
A diverged arm is a finding about the rule at that step size, not a
differential. Do not add normalisation to rescue one; that is a new rule.
Summaries only (kind hebb27).
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
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("HEBB_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 2 if SMOKE else 6
NS = "comp25"  # same streams as comp25/aoff26 for cell-to-cell comparability
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2)

# error-free rules differ in what bounds them:
#   hebb      nothing but the NLMS denominator
#   oja       the -w*y*y decay term: negative regardless of sign(y), so it
#             survives this substrate's 69.6%-negative postsynaptic activity
#   instar    prototype (x-w)y: Grossberg's rule ASSUMES y >= 0. With signed
#             soft-WTA it drives w AWAY from x, force ~ (x-w) => exponential
#             divergence. A type error, not a step-size problem (measured:
#             effective step 0.007 median, nowhere near the overshoot bound).
#             Kept in the sweep so the divergence is on the record at every lr.
#   softhebb  oja gated by soft-WTA on the target
# (bcm dropped: lrn_adaptive needs Activs for its sliding threshold, which the
# E channel does not maintain. basic dropped: dw ~ xy, identical to hebb.)
RULES = ("delta", "hebb", "oja", "instar", "softhebb")
LR_ES = (0.1,) if SMOKE else (0.001, 0.01, 0.1)
FROZEN_LR = 0.1  # E never updates, so one cell suffices as the reference


def fan3(lr_e):
    return dict(BASE, lr_e=lr_e)


def cfg_hash(rule, lr_e):
    return rep.arm_hash(f"hebb27:{rule}", fan3(lr_e), STEPS)


def jobs():
    out = [(s, "frozen", FROZEN_LR) for s in SEEDS]
    out += [(s, r, lr) for lr in LR_ES for r in RULES for s in SEEDS]
    return out


def e_norm_stats(agt):
    """Total and max Dir.E conn norm: the divergence vital."""
    from src.agents import Dir
    tot = mx = 0.0
    for _, col in agt.cols.items():
        for (loc, direction), w in col.conns.items():
            if direction is Dir.E:
                n = float(torch.linalg.norm(w))
                tot += n
                mx = max(mx, n)
    return tot, mx


def run_one(job):
    seed, rule, lr_e = job
    out = {"seed": seed, "rule": rule, "lr_e": lr_e}
    deg_sum = deg_n = 0
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **fan3(lr_e))
        kw = dict(dire=False) if rule == "frozen" else dict(dire_rule=rule)
        host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_hebb27/w{os.getpid()}"),
                                        seed=seed, **kw))
        agt = host.agt
        e0, _ = e_norm_stats(agt)
        r = frz.run_light(host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                          WINDOW)
        out[fixture] = r["margin"]
        out[f"{fixture}_nan"] = r["nan"]
        if fixture == "kp1":
            e1, e1max = e_norm_stats(agt)
            out["e_norm_growth"] = e1 / max(e0, 1e-9)
            out["e_norm_max"] = e1max
            for loc, c in agt.cols.items():
                if agt.is_i(loc):
                    continue
                x = c.nr_1.actual
                pos = int((x > 0).sum())
                deg_sum += max(pos, x.numel() - pos) / x.numel()
                deg_n += 1
    out["deg_final"] = deg_sum / max(1, deg_n)
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out["kp1"], out["kz"]) else None)
    out["config_hash"] = cfg_hash(rule, lr_e)
    return out


def finite(xs):
    return [x for x in xs
            if x is not None and x == x and abs(x) != float("inf")]


def cell(led, rule, lr_e):
    """Ledger rows for exactly this (rule, lr_e, STEPS). Filtering on
    config_hash is what keeps smoke rows out: round 1 grouped by rule alone
    and silently mixed 240-step rows into the 12000-step aggregates."""
    h = cfg_hash(rule, lr_e)
    g = sorted([e for e in led if e.get("config_hash") == h],
               key=lambda r: r["seed"])
    ds = finite([e.get("differential") for e in g])
    gro = finite([e.get("e_norm_growth") for e in g])
    emx = finite([e.get("e_norm_max") for e in g])
    return {
        "n": len(g), "rows": g, "diffs": ds,
        "mean": sum(ds) / len(ds) if ds else float("nan"),
        "kp1": finite([e.get("kp1") for e in g]),
        "kz": finite([e.get("kz") for e in g]),
        "growth": max(gro) if gro else float("nan"),
        "maj": sum(e["deg_final"] for e in g) / len(g) if g else float("nan"),
        "diverged": bool(g) and (len(gro) < len(g) or len(emx) < len(g)
                                 or max(gro) > 10 or max(emx) > 1e3),
    }


def avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "hebb27"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [j for j in jobs() if (cfg_hash(j[1], j[2]), j[0]) not in done]
    print(f"hebb27: {len(jobs())} cells, {len(todo)} to run, "
          f"{len(jobs()) - len(todo)} reused from ledger", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "hebb27", "event_id":
                                 f"hebb27:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)}  seed {s['seed']} {s['rule']} "
                  f"lr_e={s['lr_e']}: diff {s['differential']}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "hebb27"]
    fro = cell(led, "frozen", FROZEN_LR)
    print(f"\nE-rule x step-size sweep (fan3, seeds {list(SEEDS)}, "
          f"comp25 streams, {STEPS} steps)")
    print(f"reference: frozen {fro['mean']:+.4f} (E held at init, A learning); "
          f"comp25 dire_off signed/0.5 = -0.0137")
    print("majority-sign is A-side collapse, same in every arm incl. frozen: "
          "reported, never gated\n")
    print(f"{'rule':>9} {'lr_e':>7} {'diff':>9} {'kp1':>9} {'kz':>9} "
          f"{'maj%':>6} {'E-grow':>12} {'vs frozen':>10}  status")
    winners = []
    for rule in RULES:
        for lr in LR_ES:
            c = cell(led, rule, lr)
            if not c["n"]:
                continue
            vs = c["mean"] - fro["mean"]
            print(f"{rule:>9} {lr:>7} {c['mean']:+9.4f} {avg(c['kp1']):+9.4f} "
                  f"{avg(c['kz']):+9.4f} {c['maj']:6.1%} {c['growth']:12.2f} "
                  f"{vs:+10.4f}  {'DIVERGED' if c['diverged'] else 'ok'}")
            print(f"{'':>17} per-seed "
                  + " ".join(f"{d:+.4f}" for d in c["diffs"]))
            if not c["diverged"] and len(c["diffs"]) == len(fro["diffs"]):
                pair = [d - f for d, f in zip(c["diffs"], fro["diffs"])]
                if all(p > 0 for p in pair):
                    winners.append((vs, rule, lr, pair))

    print("\nstable AND beats frozen on every seed (paired):")
    if not winners:
        print("  none")
    for vs, rule, lr, pair in sorted(winners, reverse=True):
        print(f"  {rule} @ lr_e={lr}: mean {vs:+.4f} over frozen, per-seed "
              + " ".join(f"{p:+.4f}" for p in pair)
              + f"  (n={len(pair)}, sign-test p~{0.5 ** len(pair):.3f}, "
                "suggestive only)")
    print("\nDIVERGED means unreadable at that step size, NOT 'does not help'. "
          "Separating those two is the whole point of the lr_e axis.")


if __name__ == "__main__":
    main()
