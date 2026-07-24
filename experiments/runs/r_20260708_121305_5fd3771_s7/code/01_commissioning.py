"""01: Commissioning run — R-004 as amended by R-005. REGISTERED; gates locked.

Question: does C1's statistic separate a known-history-dependent arm (SGD) from
a known-history-independent arm (RandomProj)? Instrument test, no thesis contact.

Design (R-004/R-005): 8 seeds x 2 orders x 2 gated arms (+MNISTAgt 2 seeds,
non-gating, via --mnistagt). AB = A01234(4k) -> B56789(4k) -> washout(2k);
BA = task order reversed. Fixed probe sets (test split, 400/400, seed-0),
3 probes every 1000 steps. GATE reads step 8000 (end of training) ONLY.

Primary: per probe p, P_p = 10-dim per-class accuracy; S_p = D_order/D_seed
(seed-matched pairs / within-order pairs). Gates (R-005): (a) SGD's S_p beats
all 2^8 paired sign-flip relabelings (exact p=1/256) on >=2/3 probes;
(b) RandomProj's D_order = 0 within float tolerance on all 3 (pipeline check).
Secondary (non-gating): S_p(t) curve at every probed depth. Fixed 3.0/1.5
thresholds = stated expectations only.

FINDINGS (2026-07-08, runs r_20260708_0130..0144, verdict FAIL -> R-006):
Gate (b) perfect: RandProj D_order = 0.0 exactly, 3/3 probes — pipeline clean.
Gate (a) failed: SGD S = 1.12/1.85/0.94, perm-p 2/256, 2/256, 115/256 — the
retrained probe REPAIRS forgetting at read-time (forgetting lives in the head;
128-d MNIST features survive task B; random-proj floor 0.76 proves features
barely gate MNIST decoding). S(t) 1.3-1.7 with one task seen -> ~1.0 after
both: endpoint reps near path-independent even when behavior isn't. Washout
is a no-op for supervised arms (y=None => no learning). Redesign = R-007.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.harness import SGDArm, RandomProjArm, mnist_hebb_arm, run_curriculum
from experiments.provenance import load_stamped
from experiments.tasks import digit_split_tasks, probe_sets, washout_task

SEEDS = list(range(8))                       # R-005 design bump
ORDERS = ("AB", "BA")
GATED = ("sgd", "randproj")
RUNS_DIR = "experiments/runs"
GATE_STEP = 8000                             # end of training, pre-washout
PROBE_NAMES = ("ridge", "knn", "logistic")


def alias(arm, order, seed):
    return f"{RUNS_DIR}/comm_{arm}_{order}_s{seed}.jsonl"


def make_schedule(order, seed):
    A, B = digit_split_tasks(seed)
    W = washout_task(seed, n=2000)
    ab = [(A, 4000), (B, 4000), (W, 2000)]
    return ab if order == "AB" else [(ab[1][0], 4000), (ab[0][0], 4000), (W, 2000)]


def factory_for(arm, seed):
    if arm == "sgd":
        return lambda s: SGDArm(784, 10, hidden=128, lr=1e-2, seed=s)
    if arm == "randproj":
        return lambda s: RandomProjArm(784, hidden=128, seed=s)
    if arm == "mnistagt":
        from src import iotypes as T
        return lambda s: mnist_hebb_arm([T.I_Vector(784)], [T.O_Vector(10)], seed=s)
    raise ValueError(arm)


def run_all(arms, seeds):
    ptr, pte = probe_sets(seed=0, n_train=400, n_test=400)   # FIXED for all runs
    for arm, order, seed in itertools.product(arms, ORDERS, seeds):
        ap = alias(arm, order, seed)
        if os.path.islink(ap):                                # resume: run exists
            print(f"skip {ap} (done)", flush=True)
            continue
        t0 = time.time()
        run_curriculum(factory_for(arm, seed), make_schedule(order, seed),
                       seed=seed, out_path=ap, probe_every=1000,
                       extra_files=[os.path.abspath(__file__)],
                       probe_train=ptr, probe_test=pte)
        print(f"done {ap} in {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------- analysis
def profiles(arm, step):
    """{probe: {order: {seed: 10-dim tensor}}} at a given step; None if missing."""
    out = {p: {o: {} for o in ORDERS} for p in PROBE_NAMES}
    for order, seed in itertools.product(ORDERS, SEEDS):
        ap = alias(arm, order, seed)
        if not os.path.exists(ap):
            return None
        _, rows = load_stamped(ap)
        row = next((r for r in rows if r.get("step") == step and "per_class" in r), None)
        if row is None:
            return None
        for p in PROBE_NAMES:
            out[p][order][seed] = torch.tensor([v if v is not None else 0.0
                                                for v in row["per_class"][p]])
    return out


def S_of(PAB, PBA, seeds):
    d_order = torch.stack([(PAB[s] - PBA[s]).norm() for s in seeds]).mean()
    within = []
    for grp in (PAB, PBA):
        for a, b in itertools.combinations(seeds, 2):
            within.append((grp[a] - grp[b]).norm())
    d_seed = torch.stack(within).mean()
    return float(d_order), float(d_seed), float(d_order / (d_seed + 1e-12))


def signflip_null(PAB, PBA, seeds):
    """All 2^n paired relabelings (swap AB/BA within a seed); returns S list."""
    null = []
    for mask in range(2 ** len(seeds)):
        A2, B2 = {}, {}
        for i, s in enumerate(seeds):
            if mask >> i & 1:
                A2[s], B2[s] = PBA[s], PAB[s]
            else:
                A2[s], B2[s] = PAB[s], PBA[s]
        null.append(S_of(A2, B2, seeds)[2])
    return null


def analyze():
    lines = ["R-004/R-005 COMMISSIONING ANALYSIS", "=" * 50]
    gate_a_probes, gate_b_ok = {}, {}
    for arm in GATED:
        prof = profiles(arm, GATE_STEP)
        if prof is None:
            print(f"{arm}: runs incomplete"); return None
        lines.append(f"\n--- {arm} @ step {GATE_STEP} (end of training) ---")
        for p in PROBE_NAMES:
            PAB, PBA = prof[p]["AB"], prof[p]["BA"]
            do, ds, S = S_of(PAB, PBA, SEEDS)
            null = signflip_null(PAB, PBA, SEEDS)
            n_ge = sum(1 for v in null if v >= S - 1e-12)
            pval = n_ge / len(null)
            lines.append(f"  {p:9s} D_order={do:.4f} D_seed={ds:.4f} "
                         f"S={S:6.2f}  perm-p={pval:.4f} ({n_ge}/{len(null)})")
            if arm == "sgd":
                gate_a_probes[p] = (pval <= 1.0 / 256 + 1e-9)
            if arm == "randproj":
                gate_b_ok[p] = (do < 1e-9)
                if do >= 1e-9:
                    lines.append(f"    !! nonzero order-effect on the known-zero arm "
                                 f"= PIPELINE BUG (charter S13 incident)")
    ga = sum(gate_a_probes.values()) >= 2
    gb = all(gate_b_ok.values())
    verdict = "PASS" if (ga and gb) else "FAIL"
    lines.append(f"\nGATE (a) sensitivity  (SGD p=1/256 on >=2/3 probes): "
                 f"{sum(gate_a_probes.values())}/3 -> {'OK' if ga else 'FAIL'}")
    lines.append(f"GATE (b) false-positive (RandProj D_order=0 on 3/3):  "
                 f"{sum(gate_b_ok.values())}/3 -> {'OK' if gb else 'FAIL'}")
    lines.append(f"\nCOMMISSIONING VERDICT: {verdict}")

    lines.append("\nSECONDARY (non-gating): S_p(t) across the run (SGD):")
    hdr = "  step  " + "".join(f"{p:>10s}" for p in PROBE_NAMES)
    lines.append(hdr)
    for t in range(1000, 10001, 1000):
        prof = profiles("sgd", t)
        if prof is None:
            continue
        row = f"  {t:5d} "
        for p in PROBE_NAMES:
            row += f"{S_of(prof[p]['AB'], prof[p]['BA'], SEEDS)[2]:10.2f}"
        lines.append(row + ("   <- GATE" if t == GATE_STEP else
                            "   (washout)" if t > GATE_STEP else ""))

    lines.append("\nCost cards (S6, analytic v0, MAC*2): sgd ~6.1e5 FLOPs/step "
                 "(fwd+bwd 784x128+128x10); randproj ~2.0e5 (fwd only). "
                 "Tuning-evaluation count: sgd 0, randproj 0.")
    run_ids = sorted({load_stamped(alias(a, o, s))[0]["run_id"]
                      for a in GATED for o in ORDERS for s in SEEDS})
    lines.append(f"\nEvidence: {len(run_ids)} stamped runs, "
                 f"{run_ids[0]} .. {run_ids[-1]}")
    text = "\n".join(lines)
    print(text)
    with open(f"{RUNS_DIR}/commissioning_verdict.txt", "w") as f:
        f.write(text + "\n")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--mnistagt", action="store_true", help="run the non-gating third arm (slow)")
    args = ap.parse_args()
    if args.mnistagt:
        run_all(["mnistagt"], seeds=[0, 1])
    elif args.analyze_only:
        analyze()
    else:
        run_all(GATED, SEEDS)
        analyze()
