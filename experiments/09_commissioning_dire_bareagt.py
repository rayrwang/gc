"""
Dir.E on the real recurrent substrate: BareAgt takes the exam (machinery in
dire_exam.py, agents in dire_hosts.py, recorder in d3.py).

Era: FREE PLAY, single configuration, no sweeps (second act per the plan's
pending list). Read the report against the pre-run prediction slate in the
plan fork: distance profiles, E/A coverage, thin-but-real margins, revisit
vs the 0.854 memoryless ceiling, boundedness/quiescence, scale-blindness
watch. Predictability here is a commissioning virtue, not the project's
good: see the plan's scope note.

Subjects (five): bare:stock (unmodified BareAgt + wrapped delta), bare:x
(faithful shadow sharing stock's seed: the bitwise pair, a deliberate
exception to seed namespacing, declared via drive_fixtures' seed_of),
bare:x_frozen, bare:x_dire_off, deep:ff (machinery control).

Per (subject, seed) the manifest declares the drawn topology's luck: input
out-degree and targets, input-reachable set (BFS over A-conns), predicted
set (E-conn targets), E-conn count. Deaf draws are loud.

Modes:
    09_commissioning_dire_bareagt.py             full run
    09_commissioning_dire_bareagt.py judge <id>  fresh-process ledger replay
"""

import contextlib
import io
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder
from experiments.dire_hosts import DeepFFHost, stock_bare_host, x_bare_host
from experiments.provenance import stamp
from src.agents import Dir

SEEDS = [0, 1, 2]        # free play
N_COLS = 49              # 7x7: die-out guard (RW: 16 risks quiescence)
N_STEPS = 2000          # convergence-length rerun (first contact showed
SW_CYCLE, SW_NOISE = 2000, 1000  # scores still rising at 400: undertrained
TS_CYCLES, TS_STEPS = 6, 1000    # vs broken is the question this settles)
DEEP_DIMS = (64, 64)
PROVISIONAL_SETTLE = {"fraction": 0.1, "window": 50, "dwell": 100,
                      "atol": 0.02, "plateau_floor": 0.1}  # unregistered

SUBJECTS = {
    "bare:stock": lambda dim, seed: stock_bare_host(dim, seed, n_cols=N_COLS),
    "bare:x": lambda dim, seed: x_bare_host(dim, seed, n_cols=N_COLS),
    "bare:x_frozen": lambda dim, seed: x_bare_host(dim, seed, n_cols=N_COLS, frozen=True),
    "bare:x_dire_off": lambda dim, seed: x_bare_host(dim, seed, n_cols=N_COLS, dire=False),
    "deep:ff": lambda dim, seed: DeepFFHost(dim, seed, dims=DEEP_DIMS),
}

PAIR = ("bare:stock", "bare:x")  # must share draws to be a bitwise shadow


def seed_of(name, seed):
    if name in PAIR:
        return de.derive_seed("subject", "bare-pair", seed)
    return de.derive_seed("subject", name, seed)


STORE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "d3_store")


def probe_topology(name, seed):
    """construct one instance and declare its drawn luck for the manifest."""
    with contextlib.redirect_stdout(io.StringIO()):
        host = SUBJECTS[name](de.SYMBOL_DIM, seed_of(name, seed))
    agt = host.agt
    in_loc = agt.I_cols[0].loc
    a_edges = defaultdict(list)
    e_in, a_in = defaultdict(set), defaultdict(set)
    for col in agt.cols.values():
        for (tloc, d) in col.conns:
            if d == Dir.A:
                a_edges[col.loc].append(tloc)
                a_in[tloc].add(col.loc)
            else:
                e_in[tloc].add(col.loc)
    reach, frontier = {in_loc}, [in_loc]
    while frontier:
        frontier = [t for f in frontier for t in a_edges[f] if t not in reach]
        reach.update(frontier)
    coverage = {}
    for tloc in agt.predicted:
        cov = (len(a_in[tloc] & e_in[tloc]) / len(a_in[tloc])) if a_in[tloc] else None
        coverage[f"{tloc[0]},{tloc[1]}"] = None if cov is None else round(cov, 3)
    return {
        "input_out_degree": len(a_edges[in_loc]),
        "input_targets": sorted(map(str, a_edges[in_loc])),
        "reachable": len(reach) - 1,
        "predicted": sorted(map(str, agt.predicted)),
        "e_conns": len(agt.e_conns) if hasattr(agt, "e_conns") else len(agt.predicted),
        "coverage": coverage,
        "input_loc": str(in_loc),
        "distance": {f"{loc[0]},{loc[1]}": round(math.dist(in_loc, loc), 3)
                     for loc in agt.cols},
    }


def silence_probe(name, seed, steps=200):
    """fresh instance through a short cycle stream: count silent and nan cols."""
    with contextlib.redirect_stdout(io.StringIO()):
        host = SUBJECTS[name](de.SYMBOL_DIM, seed_of(name, seed))
    peak = defaultdict(float)
    nan_cols = set()
    for a in de.kp1_cycle(steps, seed=seed):
        host.observe(a)
        for loc, c in host.agt.cols.items():
            v = c.nr_1.actual
            if torch.isnan(v).any():
                nan_cols.add(loc)
            else:
                peak[loc] = max(peak[loc], float(v.abs().max()))
    silent = [loc for loc, p in peak.items() if p < 1e-6 and loc not in nan_cols]
    return {"silent": len(silent), "nan": len(nan_cols), "cols": len(host.agt.cols)}


def pair_equality(records):
    """bit-level check of the stock/shadow pair from the recorded streams."""
    seqs = defaultdict(list)
    for r in sorted(records, key=lambda r: (r["fixture"], r["seed"], r["step"], r.get("column", ""))):
        if r["subject"] in PAIR and "column" in r:
            seqs[(r["subject"], r["fixture"], r["seed"])].append(
                (r["step"], r["column"], r["guess_hash"], r["score"]))
    lines, ok = [], True
    for fixture in sorted({k[1] for k in seqs}):
        for seed in sorted({k[2] for k in seqs}):
            a = seqs.get((PAIR[0], fixture, seed), [])
            b = seqs.get((PAIR[1], fixture, seed), [])
            if a == b:
                continue
            ok = False
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                         min(len(a), len(b)))
            lines.append(f"  DIVERGES {fixture} s{seed} at index {first}")
    if ok:
        lines.append("  identical across all fixtures and seeds")
    return ok, lines


def print_report(run_id, manifest, report, integrity, records):
    cfg = manifest["config"]
    print(f"run {run_id}  era {cfg['era']}  "
          f"({sum(1 for v in integrity.values() if v)}/{len(integrity)} objects verified)")
    print(f"free play, seeds {cfg['seeds']}, n_cols {cfg['n_cols']}: settles nothing\n")

    print("topology luck per (subject, seed):")
    for key, t in cfg["topology"].items():
        flag = "  DEAF" if t["input_out_degree"] == 0 else ""
        print(f"  {key:<22} out-deg {t['input_out_degree']:>2}  reach {t['reachable']:>2}  "
              f"predicted {len(t['predicted']):>2}  e_conns {t['e_conns']:>3}{flag}")
    print("\nstability (fresh 200-step cycle probe):")
    for key, sres in cfg["silence"].items():
        print(f"  {key:<22} silent {sres['silent']:>2}/{sres['cols']}  nan {sres['nan']}")
    nan_scores = defaultdict(int)
    for r in records:
        if r["score"] is not None and r["score"] != r["score"]:
            nan_scores[r["subject"]] += 1
    print("  nan scores in recorded runs: " +
          (str(dict(nan_scores)) if nan_scores else "none"))

    print("\nnetwork aggregate (mean over predicted cols), trailing mean per seed:")
    for fixture in ("kp0", "kp1", "kz", "revisit"):
        for name, per_seed in sorted(report["fixtures"].get(fixture, {}).items()):
            tm = [f"{r['trailing_mean']:+.3f}" if r["trailing_mean"] is not None else "  n/a"
                  for r in per_seed]
            print(f"{fixture:>8} {name:<18} {tm}")
        print()

    print("per-column, kp1 and kz (mean across seeds): distance, coverage, "
          "predict, persist, margin:")
    def _norm_keys(d):
        return {k.replace("(", "").replace(")", "").replace(", ", ","): v
                for k, v in d.items()}
    topo0 = {}
    for k, v in cfg["topology"].items():
        if k.endswith("/s0"):
            v = dict(v, distance=_norm_keys(v["distance"]),
                     coverage=_norm_keys(v["coverage"]))
            topo0[k.split("/s")[0]] = v
    for fixture in ("kp1", "kz"):
        rows = [(subj, col, v) for (fx, subj, col), v in report["columns"].items()
                if fx == fixture and subj in PAIR[:1] + ("bare:x_frozen",)]
        for subj, col, v in sorted(rows, key=lambda r: topo0.get(r[0], {}).get("distance", {}).get(r[1], 9)):
            t = topo0.get(subj, {})
            d = t.get("distance", {}).get(col)
            cov = t.get("coverage", {}).get(col)
            tr, pe = v["trail"], v["persist"]
            m = None if tr is None or pe is None else tr - pe
            fmt = lambda x: "  n/a" if x is None else f"{x:+.3f}"
            print(f"  {fixture} {subj:<12} {col:<8} d={d if d is not None else '?':<5} "
                  f"cov={cov if cov is not None else ' -':<5} "
                  f"predict {fmt(tr)}  persist {fmt(pe)}  margin {fmt(m)}")
        print()

    print("sandwich (network aggregate):")
    for name, per_seed in sorted(report["readings"]["sw"].items()):
        r = per_seed[0]
        pre = sum(x["pre_trailing"] for x in per_seed) / len(per_seed)
        post = sum(x["post_immediate"] for x in per_seed) / len(per_seed)
        print(f"  {name:<18} pre {pre:+.3f}  post0 {post:+.3f}  "
              f"settle {r['settle_pre']}/{r['settle_post']} (s0)")

    print("\nstock-vs-shadow bit equality:")
    _, lines = pair_equality(records)
    print("\n".join(lines))

    print("\npromise verdicts (refusals expected: no bareagt bands exist):")
    for k, v in report["verdicts"].items():
        print(f"  {k}: {v}")


def main():
    store = D3Store(STORE_ROOT)
    if len(sys.argv) > 2 and sys.argv[1] == "judge":
        run_id = sys.argv[2]
        manifest, records = store.load_run(run_id)
        report = de.judge(records, settle=manifest["config"]["provisional_settle"])
        print_report(run_id, manifest, report, store.verify(run_id), records)
        return

    topology = {f"{n}/s{s}": probe_topology(n, s) for n in SUBJECTS for s in SEEDS}
    silence = {f"{n}/s{s}": silence_probe(n, s)
               for n in SUBJECTS if n.startswith("bare") for s in SEEDS}

    s = stamp(seed=SEEDS[0], extra_files=[os.path.abspath(__file__)])
    fixtures = de.standard_fixtures(N_STEPS, SW_CYCLE, SW_NOISE, TS_CYCLES, TS_STEPS) \
        + [("revisit", lambda seed: de.revisit_cycle(N_STEPS, seed=seed))]
    config = {"era": "free-play", "seeds": SEEDS, "n_cols": N_COLS,
              "deep_dims": list(DEEP_DIMS), "n_steps": N_STEPS,
              "sw": [SW_CYCLE, SW_NOISE], "ts": [TS_CYCLES, TS_STEPS],
              "provisional_settle": PROVISIONAL_SETTLE,
              "subjects": sorted(SUBJECTS), "pair": list(PAIR),
              "fixtures": [f for f, _ in fixtures],
              "topology": topology, "silence": silence}
    rec = Recorder(store, s, config)
    with contextlib.redirect_stdout(io.StringIO()):
        de.drive_fixtures(SUBJECTS, rec.sink, SEEDS, fixtures, seed_of=seed_of)
    run_id, mhash = rec.close()
    print(f"recorded {run_id}  manifest {mhash[:12]}")
    print(f"replay with: 09_commissioning_dire_bareagt.py judge {run_id}\n")
    manifest, records = store.load_run(run_id)
    report = de.judge(records, settle=PROVISIONAL_SETTLE)
    print_report(run_id, manifest, report, store.verify(run_id), records)


if __name__ == "__main__":
    main()
