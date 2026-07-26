"""
Confirmatory follow-up on oja (RW, 2026-07-25): fresh seeds, beta 2.0.

Round 1 (27_hebb_e_probe, exploratory) found oja the only error-free E rule
that stayed bounded, beating both references 3/3 paired at beta 0.5, lr_e 0.1.
n=3 gives p~0.125, so that is a hypothesis, not a result. This is the
confirmatory pass and it is pre-specified end to end.

Three deliberate choices, all made before any number here exists:

1. FRESH SEEDS (60-67, eight of them). The hypothesis was generated on seeds
   50-52, so confirming on 50-52 would be circular. Nothing in this file
   touches those seeds.

2. BETA 2.0, not the 0.5 default. comp25 measured both cells:
       beta 0.5:  dire_off -0.0137, learner -0.1201  (E-damage -0.106)
       beta 2.0:  dire_off +0.2392, learner +0.0163  (E-damage -0.223)
   Two reasons. There is twice as much destroyed signal available to
   recover, so a real effect should be roughly twice the size. And the bar
   becomes meaningful: beating a strongly positive reference (+0.24) is a
   claim, beating a slightly-negative one (-0.014) is barely distinguishable
   from the frozen twin at -0.048.

3. lr_e IS PICKED ON STABILITY ALONE, never on differential. The rule,
   fixed here: take the LARGEST lr_e in the round-1 sweep at which oja was
   non-divergent on every seed (E-norm growth <= 10 and max <= 1e3). If the
   sweep has not run, this file refuses to proceed rather than guess.
   Picking the lr_e that maximised the differential would be forking-paths;
   picking the largest stable one is a rule that cannot see the outcome.

Reading, fixed now: paired per-seed oja - frozen across the eight seeds.
    8/8 or 7/8 positive  -> holds up; worth writing as a registered arm
    5-6/8                -> unresolved at n=8, not a result
    <= 4/8               -> round 1 was seed luck, drop it
Two-sided sign test p is printed. Vitals unchanged: E-norm divergence is the
only gate, majority-sign is A-side collapse and is reported only.
Summaries only (kind ojab2).
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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

SMOKE = bool(os.environ.get("OJAB2_SMOKE"))
SEEDS = (60, 61) if SMOKE else (60, 61, 62, 63, 64, 65, 66, 67)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 2 if SMOKE else 18  # each worker ~= 1 core (measured); 6 left 18 cores idle
NS = "ojab2"  # fresh stream namespace: these seeds have no comp25 history
BETA = 2.0
ARMS = ("oja", "delta", "frozen")
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, beta=BETA)


def pick_lr_e():
    """Largest lr_e at which round-1 oja was stable on every seed. Stability
    only: this function never looks at a differential."""
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "hebb27"]
    stable = []
    for lr in sorted(probe.LR_ES, reverse=True):
        c = probe.cell(led, "oja", lr)
        if c["n"] and not c["diverged"]:
            stable.append(lr)
    if not stable:
        raise SystemExit(
            "refusing to run: no stable oja cell in the round-1 sweep ledger. "
            "Run 27_hebb_e_probe.py first. Guessing an lr_e here would make "
            "the pre-specification worthless.")
    return stable[0]


def cfg_for(lr_e):
    return dict(BASE, lr_e=lr_e)


def run_one(job):
    seed, arm, lr_e = job
    out = {"seed": seed, "arm": arm, "lr_e": lr_e, "beta": BETA}
    deg_sum = deg_n = 0
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **cfg_for(lr_e))
        kw = dict(dire=False) if arm == "frozen" else dict(dire_rule=arm)
        host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"ojab2/w{os.getpid()}"),
                                        seed=seed, **kw))
        agt = host.agt
        e0, _ = probe.e_norm_stats(agt)
        r = frz.run_light(host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                          WINDOW)
        out[fixture] = r["margin"]
        out[f"{fixture}_nan"] = r["nan"]
        if fixture == "kp1":
            e1, e1max = probe.e_norm_stats(agt)
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
    out["config_hash"] = rep.arm_hash(f"ojab2:{arm}", cfg_for(lr_e), STEPS)
    return out


def sign_p(k, n):
    """two-sided sign test, exact."""
    from math import comb
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def main():
    lr_e = pick_lr_e()
    print(f"ojab2: lr_e={lr_e} picked by the stability rule (largest "
          f"non-divergent oja cell in round 1); beta={BETA}; seeds "
          f"{list(SEEDS)} (fresh, no overlap with the 50-52 that generated "
          f"the hypothesis)", flush=True)
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "ojab2"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, a, lr_e) for a in ARMS for s in SEEDS
            if (rep.arm_hash(f"ojab2:{a}", cfg_for(lr_e), STEPS), s) not in done]
    print(f"  {len(SEEDS) * len(ARMS)} cells, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "ojab2", "event_id":
                                 f"ojab2:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} seed {s['seed']} {s['arm']}: "
                  f"diff {s['differential']}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "ojab2"]
    by = {}
    for a in ARMS:
        h = rep.arm_hash(f"ojab2:{a}", cfg_for(lr_e), STEPS)
        rows = sorted([e for e in led if e.get("config_hash") == h],
                      key=lambda r: r["seed"])
        by[a] = {r["seed"]: r for r in rows}

    print(f"\noja confirmatory, beta {BETA}, lr_e {lr_e}, {len(SEEDS)} fresh seeds")
    print(f"{'arm':>8} {'diff':>9} {'E-grow':>9} {'maj%':>7}  status")
    for a in ARMS:
        rows = list(by[a].values())
        if not rows:
            continue
        ds = probe.finite([r.get("differential") for r in rows])
        gro = probe.finite([r.get("e_norm_growth") for r in rows])
        emx = probe.finite([r.get("e_norm_max") for r in rows])
        div = len(gro) < len(rows) or (gro and max(gro) > 10) or (emx and max(emx) > 1e3)
        print(f"{a:>8} {probe.avg(ds):+9.4f} {(max(gro) if gro else float('nan')):9.2f} "
              f"{probe.avg([r['deg_final'] for r in rows]):7.1%}  "
              f"{'DIVERGED' if div else 'ok'}")

    print(f"\npaired per-seed, oja minus each reference:")
    for ref in ("frozen", "delta"):
        pairs = [(s, by["oja"][s]["differential"] - by[ref][s]["differential"])
                 for s in SEEDS
                 if s in by["oja"] and s in by[ref]
                 and by["oja"][s]["differential"] is not None
                 and by[ref][s]["differential"] is not None]
        if not pairs:
            continue
        k = sum(1 for _, d in pairs if d > 0)
        n = len(pairs)
        print(f"  vs {ref}: {k}/{n} positive, p={sign_p(k, n):.4f}  "
              + " ".join(f"s{s}:{d:+.3f}" for s, d in pairs))
        if ref == "frozen":
            verdict = ("HOLDS UP: worth writing as a registered arm" if k >= n - 1
                       else "UNRESOLVED at this n" if k >= n * 0.6
                       else "DROP: round 1 was seed luck")
            print(f"  pre-registered reading vs frozen: {verdict}")


if __name__ == "__main__":
    main()
