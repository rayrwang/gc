"""
Oja at its better step sizes, fresh seeds (RW, 2026-07-25). EXPLORATORY.

Status, stated first because it is the whole point of keeping this separate
from 28_oja_beta2_confirm:

  28 is CONFIRMATORY. Its lr_e was fixed by a rule written before any of its
     numbers existed (largest non-divergent oja cell in round 1), which chose
     0.1. That rule could not see a differential.
  30 (this file) is EXPLORATORY. lr_e 0.001 and 0.01 are here BECAUSE round 1
     showed oja looks better there (+0.283 and +0.219 vs +0.056 at 0.1).
     That is selection on the outcome. It does not make the numbers wrong, it
     makes them hypothesis-generating rather than hypothesis-testing, and no
     result from this file may be reported as confirmation of anything.

What it buys anyway. Round 1's cross-step-size pattern (oja beats delta by
+0.17 to +0.22 at every lr_e) rested on three seeds, and those means were
dominated by seed 52, which threw a large positive differential in nearly
every arm. Re-running the same grid on eight fresh seeds tests whether the
pattern is a property of the rules or of that one graph draw. The pattern
holding across three step sizes on eight seeds is a much better reason to
believe it than any single cell.

Design, matched to 28 so the two compose into one grid:
  seeds 60-67 (same fresh set as 28; no overlap with 50-52)
  beta 2.0, ns "ojab2" streams, 12000 steps
  arms: oja, delta at lr_e in {0.001, 0.01}
  frozen is NOT re-run: E never updates, so it is lr_e-independent, and 28's
  frozen cells on these same seeds are the reference. If 28 has not finished,
  this file refuses rather than inventing a reference.

Reading: per-seed paired oja - delta and oja - frozen at each lr_e, with the
exact sign test. What to look for is CONSISTENCY across lr_e, not the best
cell. A single strong cell at a step size chosen after the fact is worth
very little; the same sign at every step size on fresh seeds is worth more.
Summaries only (kind ojalr30).
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
conf = importlib.import_module("28_oja_beta2_confirm")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("OJALR_SMOKE"))
SEEDS = conf.SEEDS if not SMOKE else (60, 61)
STEPS, WINDOW = (conf.STEPS, conf.WINDOW) if not SMOKE else (240, 60)
WORKERS = 2 if SMOKE else 18  # each worker ~= 1 core (measured); 6 left 18 cores idle
NS = conf.NS          # same streams as 28
BETA = conf.BETA      # 2.0
RULES = ("oja", "delta")
LR_ES = (0.001, 0.01)
CONFIRM_LR = 0.1      # 28's cell; referenced, never re-run here


def cfg_for(lr_e):
    return dict(conf.BASE, lr_e=lr_e)


def cfg_hash(rule, lr_e):
    return rep.arm_hash(f"ojalr30:{rule}", cfg_for(lr_e), STEPS)


def frozen_ref(store):
    """28's frozen cells on the same seeds. E never updates so this is
    lr_e-independent; refuse rather than fabricate a reference."""
    h = rep.arm_hash("ojab2:frozen", cfg_for(CONFIRM_LR), STEPS)
    rows = {e["seed"]: e for e in store.ledger()
            if e.get("kind") == "ojab2" and e.get("config_hash") == h}
    if len(rows) < len(SEEDS):
        raise SystemExit(
            f"refusing to run: need 28's frozen cells on all {len(SEEDS)} seeds "
            f"as the reference, found {len(rows)}. Let 28 finish first.")
    return rows


def run_one(job):
    seed, rule, lr_e = job
    out = {"seed": seed, "rule": rule, "lr_e": lr_e, "beta": BETA}
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **cfg_for(lr_e))
        host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"ojalr30/w{os.getpid()}"),
                                        seed=seed, dire_rule=rule))
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
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out["kp1"], out["kz"]) else None)
    out["config_hash"] = cfg_hash(rule, lr_e)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    fro = frozen_ref(store)
    print(f"ojalr30 EXPLORATORY: lr_e {LR_ES} chosen AFTER seeing round 1; "
          f"28's lr_e={CONFIRM_LR} cell remains the only confirmatory one.")
    print(f"  seeds {list(SEEDS)}, beta {BETA}, frozen reference reused from 28",
          flush=True)
    led = [e for e in store.ledger() if e.get("kind") == "ojalr30"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, r, lr) for lr in LR_ES for r in RULES for s in SEEDS
            if (cfg_hash(r, lr), s) not in done]
    print(f"  {len(RULES)*len(LR_ES)*len(SEEDS)} cells, {len(todo)} to run",
          flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "ojalr30", "event_id":
                                 f"ojalr30:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['rule']} lr={s['lr_e']}: "
                  f"diff {s['differential']}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "ojalr30"]
    ojab2 = [e for e in store.ledger() if e.get("kind") == "ojab2"]

    def cells(kind_led, tag, rule, lr):
        h = (rep.arm_hash(f"{tag}:{rule}", cfg_for(lr), STEPS))
        return {e["seed"]: e for e in kind_led if e.get("config_hash") == h}

    print(f"\noja vs delta on {len(SEEDS)} FRESH seeds, beta {BETA}, by step size")
    print(f"{'lr_e':>7} {'status':>13} {'oja':>9} {'delta':>9} {'frozen':>9} "
          f"{'oja-delta':>10} {'sign':>7} {'p':>7}")
    for lr in sorted(set(LR_ES) | {CONFIRM_LR}):
        if lr == CONFIRM_LR:
            o, d = cells(ojab2, "ojab2", "oja", lr), cells(ojab2, "ojab2", "delta", lr)
            status = "CONFIRMATORY"
        else:
            o, d = cells(led, "ojalr30", "oja", lr), cells(led, "ojalr30", "delta", lr)
            status = "exploratory"
        common = [s for s in SEEDS if s in o and s in d and s in fro
                  and o[s].get("differential") is not None
                  and d[s].get("differential") is not None]
        if not common:
            print(f"{lr:>7} {status:>13}   (no data yet)")
            continue
        od = [o[s]["differential"] - d[s]["differential"] for s in common]
        k = sum(1 for x in od if x > 0)
        print(f"{lr:>7} {status:>13} "
              f"{probe.avg([o[s]['differential'] for s in common]):+9.4f} "
              f"{probe.avg([d[s]['differential'] for s in common]):+9.4f} "
              f"{probe.avg([fro[s]['differential'] for s in common]):+9.4f} "
              f"{probe.avg(od):+10.4f} {k:>3}/{len(od):<3} "
              f"{conf.sign_p(k, len(od)):7.4f}")

    print("\nWhat to read: consistency of the oja-delta sign ACROSS step sizes, "
          "not the size of the best cell.")
    print("The 0.001 and 0.01 rows were selected after seeing round 1 and cannot "
          "confirm anything on their own; only the 0.1 row was pre-specified.")


if __name__ == "__main__":
    main()
