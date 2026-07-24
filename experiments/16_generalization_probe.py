"""
Post-R-028 disambiguation probe (free play): why did the faithful
differential read +0.083 (9/10 positive) on development graphs in free
play and -0.019 (1/8) on fresh registered graphs?

The 2x2: graphs {development 0-9, registered 3252..9490} x stream
namespace {rep11 (free play's), r025 (the registered event's)}. Two cells
exist: (0-9, rep11) = hz12's +0.083 9/10; (registered, r025) = R-028's
weak read. This probe runs the two missing cells at 12000 steps,
unfloored per-column margins (free play's own method), so graph-family
and symbol-draw effects separate. If (0-9, r025) stays strongly positive,
the development graphs are special (overfit-to-draws); if it collapses,
the symbol streams carry the luck; if both missing cells are weak, free
play's whole reading rode a graphs-x-streams interaction.

Post-event use of registered seeds is diagnostic only: the pre-event ban
existed for severity, the verdict is in, and any future registered event
draws fresh seeds regardless. Summaries-only (kind gen16); resume = rerun.
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rep = importlib.import_module("11_replicate_instar_relu")
sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("GEN_SMOKE"))
STEPS, WINDOW = (120, 40) if SMOKE else (12000, 500)
WORKERS = 4 if SMOKE else 12
DEV_SEEDS = (0,) if SMOKE else tuple(range(10))
REG_SEEDS = (900,) if SMOKE else (3252, 4635, 5149, 5344, 6134, 7540, 7776, 9490)
KNOBS = dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1)

CELLS = [("dev_r025ns", DEV_SEEDS, "r025"),
         ("reg_rep11ns", REG_SEEDS, "rep11")]


def run_one(job):
    cell, seed, ns = job
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **KNOBS)
    host = BareHost(sweep.BareAgtXC(cfg, _bare_path(f"dire_gen16/w{os.getpid()}"),
                                    seed=seed))
    records = []
    for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
        de.run_stream_multi(host, fn(STEPS, seed=de.derive_seed(fixture, ns, seed)),
                            records.append, fixture=fixture, subject=cell, seed=seed)
    tables = de.column_tables(records, WINDOW)
    margins = {}
    for fixture in ("kp1", "kz"):
        cells_ = [(v["trail"], v["persist"]) for (f, _s, _c), v in tables.items()
                  if f == fixture and v["trail"] is not None and v["persist"] is not None]
        margins[fixture] = (sum(t - p for t, p in cells_) / len(cells_)) if cells_ else None
    diff = (margins["kp1"] - margins["kz"]
            if None not in margins.values() else None)
    return {"cell": cell, "seed": seed, "ns": ns, "differential": diff,
            "config_hash": rep.arm_hash(cell, {"ns": ns, **KNOBS}, STEPS)}


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    done = {(e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == "gen16"}
    jobs = [(cell, s, ns) for cell, seeds, ns in CELLS for s in seeds]
    todo = [j for j in jobs
            if (rep.arm_hash(j[0], {"ns": j[2], **KNOBS}, STEPS), j[1]) not in done]
    print(f"gen16: {len(jobs)} jobs, {len(jobs) - len(todo)} in ledger, "
          f"{len(todo)} to run", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "gen16", "event_id":
                                 f"gen16:{s['config_hash']}:s{s['seed']}", **s})
            results.append(s)
            if n % 6 == 0 or n == len(todo):
                print(f"  gen16 {n}/{len(todo)}", flush=True)
    for e in store.ledger():
        if e.get("kind") == "gen16":
            results.append(e)
    seen, rows = set(), []
    for s in results:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            rows.append(s)
    print("\n2x2 completion (existing cells: dev x rep11ns = +0.083 9/10 [hz12];"
          "\nregistered x r025ns = -0.012 unfloored 5/8 [R-028 diagnostics]):", flush=True)
    for cell, _seeds, _ns in CELLS:
        g = sorted((r for r in rows if r["cell"] == cell), key=lambda r: r["seed"])
        ds = [r["differential"] for r in g if r["differential"] is not None]
        m = sum(ds) / len(ds) if ds else float("nan")
        pos = sum(1 for d in ds if d > 0)
        per = " ".join(f"{d:+.3f}" for d in ds)
        print(f"{cell:<14} mean {m:+.4f}  {pos}/{len(ds)} positive  per-seed: {per}")


if __name__ == "__main__":
    main()
