"""
Is it the NET MASS or the DIFFERENTIAL? (RW, 2026-07-25.)

32_cifar_gate_arms found the centred gate catastrophic on CIFAR: -14/-21/-18
vs frozen, mean_abs_cos 0.080 -> 0.766 (channels merged onto one prototype),
|W| 52.6 -> 9.8. Meanwhile gate29 found argmax's net-negative mass causes the
antipodal collapse on the recurrent substrate. Both measured correctly; the
inference "so removing it is an improvement" was wrong.

But `resp - resp.mean()` confounds two changes. The argmax gate
    winner +p_win, losers -p_i,  sum = 2*p_win - 1  (measured -0.845)
decomposes into a DIFFERENTIAL (winner up, losers down, relative to each
other: this is the anti-Hebbian repulsion that makes channels tile) and a
COMMON MODE (every unit pushed down by the same amount: this is what feeds
back and compounds in recurrence). Centring zeroes the common mode AND
redistributes who gets positive credit, and at low beta the softmax is nearly
flat so the whole gate nearly vanishes. |W| 9.8 says it collapsed the gate,
not rebalanced it.

THE CLEAN SEPARATION: subtract a CONSTANT from every entry.
    g' = g - f * sum(g) / d
Every pairwise difference g_i - g_j is untouched, so the differential is
exactly preserved. The sum scales as (1 - f) * sum(g). So:
    f = 0    original argmax gate, net mass -0.845
    f = 1    net mass exactly 0, differential IDENTICAL to original
    f = 0.25/0.5/0.75  the curve between
If performance is flat across f and only `centered` falls over, the net mass
is irrelevant and centring's damage is entirely from destroying the
differential. If performance degrades with f, the net mass is load-bearing on
image data too, and gate29's "fix" trades a recurrent pathology for a
feedforward one.

READING fixed before the run:
  flat in f, centered bad      -> differential is the function; net mass is
                                  free to remove; gate29's fix should be done
                                  by SHIFTING, not centring
  degrades with f              -> net mass is load-bearing here; the collapse
                                  is a cost the repulsion must impose
  non-monotone                 -> report it, do not smooth it
Arms: frozen, original(f=0), shift25, shift50, shift75, shift100, centered.
Patches fc.softmax_wta_batched in-process only. src/agents.py untouched.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib

import torch

arms32 = importlib.import_module("32_cifar_gate_arms")
from src import funcs as fc
from src.agents import CIFARAgt
from src.envs import CIFARDataset

SEED = arms32.SEED
WARMUP = arms32.WARMUP
N_TRAIN_EVAL, N_TEST_EVAL = arms32.N_TRAIN_EVAL, arms32.N_TEST_EVAL
_ORIG_GATE = fc.softmax_wta_batched

ARMS = ("frozen", "original", "shift25", "shift50", "shift75", "shift100",
        "centered")
GATE_SUMS = []


def install(arm):
    fc.softmax_wta_batched = _ORIG_GATE
    if arm in ("frozen", "original"):
        base = _ORIG_GATE

        def g0(u, beta=1.0, signed=False):
            g = base(u, beta, signed)
            GATE_SUMS.append(float(g.sum(-1).mean()))
            return g
        fc.softmax_wta_batched = g0
        return
    if arm == "centered":
        def gc(u, beta=1.0, signed=False):
            resp = torch.softmax(beta * u, dim=-1)
            g = resp - resp.mean(-1, keepdim=True) if signed else resp
            GATE_SUMS.append(float(g.sum(-1).mean()))
            return g
        fc.softmax_wta_batched = gc
        return
    f = int(arm.replace("shift", "")) / 100.0
    base = _ORIG_GATE

    def gs(u, beta=1.0, signed=False):
        g = base(u, beta, signed)
        if signed:
            # subtract a constant per row: every pairwise difference survives,
            # the row sum scales by (1 - f)
            g = g - f * g.sum(-1, keepdim=True) / g.shape[-1]
        GATE_SUMS.append(float(g.sum(-1).mean()))
        return g
    fc.softmax_wta_batched = gs


def run_arm(arm, imgs, labels, timgs, tlabels, eval_tr, n_steps):
    install(arm)
    GATE_SUMS.clear()
    torch.manual_seed(SEED)
    agt = CIFARAgt()
    gen = torch.Generator(device=imgs.device).manual_seed(SEED)
    t0 = time.perf_counter()
    for _ in range(WARMUP):
        agt.step([imgs[torch.randint(len(imgs), (1,), generator=gen).item()]],
                 use_lrn=False, training=True)
    if arm != "frozen":
        for i in range(n_steps):
            agt.step([imgs[torch.randint(len(imgs), (1,), generator=gen).item()]])
            if (i + 1) % 25000 == 0:
                print(f"    {arm} {i+1}/{n_steps} {time.perf_counter()-t0:.0f}s",
                      flush=True)
    rtr = arms32.reps(agt, imgs[eval_tr])
    rte = arms32.reps(agt, timgs[:N_TEST_EVAL])
    mu, sd = rtr.mean(0), rtr.std(0) + 1e-6
    p = arms32.probes((rtr - mu) / sd, labels[eval_tr],
                      (rte - mu) / sd, tlabels[:N_TEST_EVAL])
    w = agt.W[0]
    wn = torch.nn.functional.normalize(w, dim=0)
    cos = wn.T @ wn
    off = cos - torch.diag(torch.diag(cos))
    p["mean_abs_cos"] = float(off.abs().sum() / (len(cos) * (len(cos) - 1)))
    p["w_norm"] = float(w.norm())
    p["net_mass"] = (sum(GATE_SUMS) / len(GATE_SUMS)) if GATE_SUMS else float("nan")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(dev)
    print(f"device {dev}, steps {a.steps}", flush=True)

    tr, te = CIFARDataset(train=True), CIFARDataset(train=False)
    imgs = (torch.tensor(tr.cifar.data).permute(0, 3, 1, 2).float() / 255).to(dev)
    labels = torch.tensor(tr.cifar.targets, device=dev)
    timgs = (torch.tensor(te.cifar.data).permute(0, 3, 1, 2).float() / 255).to(dev)
    tlabels = torch.tensor(te.cifar.targets, device=dev)
    eval_tr = torch.randperm(
        len(imgs), generator=torch.Generator(device=dev).manual_seed(0))[:N_TRAIN_EVAL]

    res = {}
    for arm in a.arms.split(","):
        print(f"\n=== {arm} ===", flush=True)
        res[arm] = run_arm(arm, imgs, labels, timgs, tlabels, eval_tr, a.steps)
        r = res[arm]
        print(f"  {arm}: kNN {r['kNN']:.2f} ridge {r['ridge']:.2f} "
              f"logit {r['logistic']:.2f} netmass {r['net_mass']:+.3f} "
              f"|cos| {r['mean_abs_cos']:.3f} |W| {r['w_norm']:.1f}", flush=True)

    f = res.get("frozen")
    print(f"\n{'arm':>10} {'netmass':>9} {'kNN':>7} {'ridge':>7} {'logit':>7} "
          f"{'dkNN':>7} {'dridge':>7} {'dlogit':>7} {'|cos|':>7} {'|W|':>7}")
    for arm in a.arms.split(","):
        r = res.get(arm)
        if not r:
            continue
        d = ("" if not f else
             f"{r['kNN']-f['kNN']:+7.2f} {r['ridge']-f['ridge']:+7.2f} "
             f"{r['logistic']-f['logistic']:+7.2f}")
        print(f"{arm:>10} {r['net_mass']:+9.3f} {r['kNN']:7.2f} {r['ridge']:7.2f} "
              f"{r['logistic']:7.2f} {d} {r['mean_abs_cos']:7.3f} {r['w_norm']:7.1f}")
    print("\nshift arms preserve every pairwise gate difference exactly and only "
          "change the row sum. So a flat curve across shift25..shift100 means the "
          "net mass is free to remove and centring's damage was from destroying "
          "the differential, not from zeroing the sum.")


if __name__ == "__main__":
    main()
