"""
The factorial's fourth cell (RW question, 2026-07-23): Dir.A frozen, Dir.E
learning: the E-rule against a stationary, uncollapsed, passively
world-coupled target. Decomposes comp25's anti-correlation finding: if
this arm shows a healthy differential, Dir.A is the saboteur (collapse +
moving target) and the E-rule is fine; if E still converges fixture-blind
here, the delta rule itself prefers the endogenous component.

Arms on comp25's exact graphs (seeds 50-52, fan3 topology, matched
fresh-host-per-fixture, namespace comp25 streams for comparability):
    a_off      A frozen, E learning  (the new cell)
    twin       both frozen           (re-used from comp25 ledger if present)
Existing cells for the 2x2 read: learner + dire_off from comp25 (signed,
beta 0.5). Also records the degeneracy statistic (should match twin: no
A-learning, no collapse) and E-prediction quality (kp1/kz margins: how
well E predicts the frozen dynamics at all). Summaries-only (kind aoff26).
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

SMOKE = bool(os.environ.get("AOFF_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 3 if SMOKE else 6
NS = "comp25"  # same streams as comp25 for cell-to-cell comparability
FAN3 = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)


def run_one(job):
    seed, = job
    out = {"seed": seed}
    deg_sum = deg_n = 0
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **FAN3)
        host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_aoff26/w{os.getpid()}"),
                                        seed=seed, a_frozen=True))
        agt = host.agt
        stream = fn(STEPS, seed=de.derive_seed(fixture, NS, seed))
        r = frz.run_light(host, stream, WINDOW)
        out[fixture] = r["margin"]
        out[f"{fixture}_nan"] = r["nan"]
        if fixture == "kp1":  # late-window degeneracy on the same pass's end state
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
    out["config_hash"] = rep.arm_hash("aoff26", FAN3, STEPS)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "aoff26"}
    todo = [(s,) for s in SEEDS
            if (rep.arm_hash("aoff26", FAN3, STEPS), s) not in done]
    print(f"aoff26: {len(SEEDS)} jobs, {len(SEEDS) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "aoff26", "event_id":
                                 f"aoff26:{s['config_hash']}:s{s['seed']}", **s})
            results.append(s)
            print(f"  aoff26 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "aoff26":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)
    rows.sort(key=lambda r: r["seed"])

    # pull comp25's signed/0.5 cells for the 2x2
    comp = [e for e in store.ledger() if e.get("kind") == "comp25"
            and e.get("cell") in ({"signed": True, "beta": 0.5}, None)]
    def cell(kind_):
        g = [e for e in comp if e.get("kind_") == kind_
             and (kind_ == "twin" or e.get("cell") == {"signed": True, "beta": 0.5})]
        ds = [e["diff"] for e in g if e.get("diff") is not None]
        return sum(ds) / len(ds) if ds else float("nan")

    ds = [r["differential"] for r in rows if r["differential"] is not None]
    m = sum(ds) / len(ds) if ds else float("nan")
    print("\nthe 2x2 (fan3 topology, seeds 50-52, matched):", flush=True)
    print(f"  A on,  E on   (learner):  diff {cell('learner'):+.4f}   [comp25]")
    print(f"  A on,  E off  (dire_off): diff {cell('dire_off'):+.4f}   [comp25]")
    print(f"  A off, E off  (twin):     diff {cell('twin'):+.4f}   [comp25]")
    print(f"  A off, E on   (a_off):    diff {m:+.4f}   [this probe]")
    print(f"\na_off detail: per-seed diffs "
          + " ".join(f"{d:+.4f}" for d in ds)
          + f"; kp1 margin {sum(r['kp1'] for r in rows)/len(rows):+.4f}, "
          f"kz margin {sum(r['kz'] for r in rows)/len(rows):+.4f}; "
          f"final majority-sign {sum(r['deg_final'] for r in rows)/len(rows):.1%} "
          f"(twin ~58%, collapsed learner ~94%)", flush=True)


if __name__ == "__main__":
    main()
