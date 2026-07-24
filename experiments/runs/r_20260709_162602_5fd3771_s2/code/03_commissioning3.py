"""03: Commissioning attempt #3 — R-009 (R-007 design + escalation deltas).

Deltas from 02: (1) SGD 784-256-256-10, read LAST hidden (rep-level forgetting
grows with depth); (2) probe_every=800 so stale evals exist at exactly
fit+1600; (3) GATING construct = short-lag mean-centered drift-S (probe fitted
at each phase boundary, evaluated one phase later, per-fit-point centered);
at-gate drift, raw short-lag drift, content all recorded non-gating;
(4) RandomProj widened to 256. Gates unchanged from R-007.

FINDINGS (2026-07-08, 64 runs r_130501..140235, verdict FAIL -> R-010):
Gate (b) 12/12. Gate (a): PERFECT DEGENERACY — D_ord = D_seed to 4 decimals,
S = 1.00, MC-p = 1.0 on all nine drift constructs. Stale probes = constant
functions ([1.0, 0x9] profiles); deep-arm PR ~5/256 (neural collapse), the
5-dim subspace rotates wholesale per phase => frozen linear maps mean-shift-
dominated. Content acc 0.80: arm learns fine; the linear-probe ruler is the
broken part (3 attempts, 3 different constructs, 1 root cause). kNN's
neighborhood signal survived all three -> construct review (geometry/RSA).
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.determinism import gen
from experiments.harness import SGDArm, RandomProjArm, run_curriculum
from experiments.provenance import load_stamped
from experiments.tasks import class_group_tasks, probe_sets, washout_task

SEEDS = list(range(8))
GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
ORDERS = {"forward": (0, 1, 2, 3, 4), "reverse": (4, 3, 2, 1, 0),
          "perm3": (4, 1, 0, 3, 2), "perm4": (3, 4, 2, 1, 0)}   # frozen in R-007
GATED = ("sgd", "randproj")
RUNS_DIR = "experiments/runs"
GATE_STEP = 8000
FIT_STEPS = (1600, 3200, 4800, 6400)
PROBE_NAMES = ("ridge", "knn", "logistic")
N_NULL = 10_000


def alias(arm, oname, seed):
    return f"{RUNS_DIR}/comm3_{arm}_{oname}_s{seed}.jsonl"


def make_schedule(oname, seed):
    tasks = class_group_tasks(seed, GROUPS)
    sched = [(tasks[i], 1600) for i in ORDERS[oname]]
    sched.append((washout_task(seed, n=2000), 2000))
    return sched


def factory_for(arm):
    if arm == "sgd":
        return lambda s: SGDArm(784, 10, hidden=(256, 256), lr=1e-2, seed=s)
    return lambda s: RandomProjArm(784, hidden=256, seed=s)


def run_all():
    ptr, pte = probe_sets(seed=0, n_train=400, n_test=400)
    for arm, oname, seed in itertools.product(GATED, ORDERS, SEEDS):
        ap = alias(arm, oname, seed)
        if os.path.islink(ap):
            print(f"skip {ap}", flush=True)
            continue
        t0 = time.time()
        run_curriculum(factory_for(arm), make_schedule(oname, seed), seed=seed,
                       out_path=ap, probe_every=800,
                       extra_files=[os.path.abspath(__file__)],
                       probe_train=ptr, probe_test=pte, probe_fit_steps=FIT_STEPS)
        print(f"done {ap} in {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------- analysis
SHORTLAG = tuple((fs, fs + 1600) for fs in FIT_STEPS)   # eval one phase after fit

def _stale_vec(rows, fit, ev_at, probe):
    row = next((r for r in rows if r.get("step") == ev_at and "stale_eval" in r), None)
    if row is None:
        return None
    e = row["stale_eval"].get(str(fit)) or row["stale_eval"].get(fit)
    if e is None:
        return None
    return torch.tensor([v if v is not None else 0.0 for v in e[probe]])

def profile_of(rows, construct, probe, step=GATE_STEP):
    if construct in ("shortlag", "shortlag_raw"):    # GATING (centered) + raw
        vec = []
        for fit, ev_at in SHORTLAG:
            pc = _stale_vec(rows, fit, ev_at, probe)
            if pc is None:
                return None
            vec.append(pc - pc.mean() if construct == "shortlag" else pc)
        return torch.cat(vec)
    row = next((r for r in rows if r.get("step") == step and "per_class" in r), None)
    if row is None:
        return None
    if construct == "content":
        vec = row["per_class"][probe]
    else:                                            # at-gate drift (recorded)
        se = row.get("stale_eval", {})
        vec = []
        for fs in FIT_STEPS:
            e = se.get(str(fs)) or se.get(fs)
            if e is None:
                return None
            vec += e[probe]
    return torch.tensor([v if v is not None else 0.0 for v in vec])


def load_profiles(arm, construct, probe, step=GATE_STEP):
    """{order: {seed: vec}} or None if incomplete."""
    out = {o: {} for o in ORDERS}
    for oname, seed in itertools.product(ORDERS, SEEDS):
        ap = alias(arm, oname, seed)
        if not os.path.exists(ap):
            return None
        _, rows = load_stamped(ap)
        v = profile_of(rows, construct, probe, step)
        if v is None:
            return None
        out[oname][seed] = v
    return out


def S_of(P):
    """P: {order: {seed: vec}} -> (D_order, D_seed, S). K-order generalization."""
    onames = list(P)
    d_order = []
    for s in SEEDS:
        for a, b in itertools.combinations(onames, 2):
            d_order.append((P[a][s] - P[b][s]).norm())
    d_seed = []
    for o in onames:
        for a, b in itertools.combinations(SEEDS, 2):
            d_seed.append((P[o][a] - P[o][b]).norm())
    do = float(torch.stack(d_order).mean())
    ds = float(torch.stack(d_seed).mean())
    return do, ds, do / (ds + 1e-12)


def mc_null(P, n=N_NULL):
    """Within-seed relabelings of order assignments, gen-seeded (R-007)."""
    g = gen(0, "c1_null")
    onames = list(P)
    null = []
    for _ in range(n):
        P2 = {o: {} for o in onames}
        for s in SEEDS:
            perm = torch.randperm(len(onames), generator=g).tolist()
            for i, o in enumerate(onames):
                P2[o][s] = P[onames[perm[i]]][s]
        null.append(S_of(P2)[2])
    return null


def analyze():
    lines = ["R-009 COMMISSIONING #3 ANALYSIS", "=" * 50]
    gate_a, gate_b = {}, {}
    for arm in GATED:
        lines.append(f"\n--- {arm} @ step {GATE_STEP} ---")
        for construct in ("shortlag", "shortlag_raw", "drift", "content"):
            for p in PROBE_NAMES:
                P = load_profiles(arm, construct, p)
                if P is None:
                    print(f"{arm}/{construct}/{p}: incomplete"); return None
                do, ds, S = S_of(P)
                tag = f"{construct}-{p}"
                if arm == "sgd" and construct == "shortlag":
                    null = mc_null(P)
                    n_ge = sum(1 for v in null if v >= S - 1e-12)
                    pval = (1 + n_ge) / (1 + len(null))
                    gate_a[p] = pval <= 0.001
                    lines.append(f"  {tag:18s} D_ord={do:.4f} D_seed={ds:.4f} "
                                 f"S={S:6.2f}  MC-p={pval:.5f} ({n_ge}/{len(null)})")
                else:
                    lines.append(f"  {tag:18s} D_ord={do:.4f} D_seed={ds:.4f} S={S:6.2f}")
                if arm == "randproj":
                    gate_b[(construct, p)] = do < 1e-9
    ga = sum(gate_a.values()) >= 2
    gb = all(gate_b.values())
    verdict = "PASS" if (ga and gb) else "FAIL"
    lines.append(f"\nGATE (a) SGD short-lag centered drift-S p<=0.001 on >=2/3 probes: "
                 f"{sum(gate_a.values())}/3 -> {'OK' if ga else 'FAIL'}")
    lines.append(f"GATE (b) RandProj D_order=0, all constructs x probes: "
                 f"{sum(gate_b.values())}/12 -> {'OK' if gb else 'FAIL'}")
    lines.append(f"\nCOMMISSIONING #3 VERDICT: {verdict}")



    run_ids = sorted({load_stamped(alias(a, o, s))[0]["run_id"]
                      for a in GATED for o in ORDERS for s in SEEDS})
    lines.append(f"\nEvidence: {len(run_ids)} stamped runs, {run_ids[0]} .. {run_ids[-1]}")
    lines.append("Cost cards unchanged from R-006 (same arms); tuning-eval count: 0, 0.")
    text = "\n".join(lines)
    print(text)
    with open(f"{RUNS_DIR}/commissioning3_verdict.txt", "w") as f:
        f.write(text + "\n")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run_all()
    analyze()
