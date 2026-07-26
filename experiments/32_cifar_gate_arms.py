"""
Does the gate fix survive contact with a real task? (RW, 2026-07-25.)

The toy-substrate arc found: (a) the antipodal collapse is caused by the
argmax gate's net-negative mass, not by the substrate (gate29, one-vs-rest
93% -> 0% under a mean-centred gate); (b) it belongs to SOFTHEBB
specifically, since only softhebb calls the gate at all and oja/basic sit at
chance while instar dies (arule31: maj 95.3 / 57.8 / 57.8 / 100-but-dead).
All of that is on kp1/kz fixtures with no external ground truth.

CIFAR-10 has ground truth. If the centred gate is a genuine improvement it
should not hurt here, and if the collapse was costing representation quality
it should help. The example's own docstring says the opposite is expected:
"gc's own rules (BCM, plain instar/oja) all fail here; the signed gate is
required." So this is a real test with a stated prior against the change.

ARMS (identical init, seed, data order, warmup; only the update differs):
  frozen    no learning at all: the control every gain is measured against
  original  softhebb as shipped: signed soft-WTA gate (argmax exempts one
            channel per location) feeding the gated-Oja update
  centered  same update, gate replaced by resp - resp.mean() (net mass 0)
  oja       plain Oja, NO gate:    dw = x.T@y - w*(y*y).sum(0)
  instar    plain instar, NO gate: dw = x.T@y - w*y.sum(0)
The two ungated rules use y = triangle(u), the activation that actually
propagates, matching how BareAgtX's A-side dispatch feeds lrn_oja/lrn_instar.

READING, fixed before the run. Report absolute kNN/ridge/logistic and the
delta vs frozen, which is the example's own convention ("read the learning
gains, absolute numbers are confounded by a frozen head-start"). Then:
  centered >= original    -> the gate fix generalises off the toy fixtures
  centered <  original    -> the argmax gate is doing useful work on real
                             data that it was not doing on kp1/kz, and the
                             collapse is a cost worth paying here
  oja/instar collapse     -> reproduces the example's stated prior and the
                             arule31 finding (instar is a type error on
                             signed activations) on a second substrate
Implementation: patches fc.softmax_wta_batched and fc.lrn_oja_gated_batched
inside this process only. src/agents.py is untouched.
"""

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ex = importlib.import_module("examples.09_cifar10") if False else None
from src import funcs as fc
from src.agents import CIFARAgt
from src.envs import CIFARDataset

SEED = 0
WARMUP = 1000
N_TRAIN_EVAL, N_TEST_EVAL = 3000, 1000
ARMS = ("frozen", "original", "centered", "oja", "instar")

_ORIG_GATE = fc.softmax_wta_batched
_ORIG_LRN = fc.lrn_oja_gated_batched


def install(arm):
    """Patch the two functions CIFARAgt calls. Returns nothing; process-local."""
    fc.softmax_wta_batched = _ORIG_GATE
    fc.lrn_oja_gated_batched = _ORIG_LRN
    if arm in ("frozen", "original"):
        return
    if arm == "centered":
        def gate(u, beta=1.0, signed=False):
            resp = torch.softmax(beta * u, dim=-1)
            return resp - resp.mean(-1, keepdim=True) if signed else resp
        fc.softmax_wta_batched = gate
        return
    # ungated rules: ignore the gate entirely, use y = triangle(u)
    power = 0.7

    def lrn(x, w, g, u, ss=1e-2):
        y = fc.triangle_batched(u, power)
        xy = x.T @ y
        if arm == "oja":
            return w + ss * (xy - w * (y * y).sum(0)) / max(len(x), 1)
        return w + ss * (xy - w * y.sum(0)) / max(len(x), 1)   # instar
    fc.lrn_oja_gated_batched = lrn


def probes(rtr, ytr, rte, yte):
    import torch.nn.functional as F
    out = {}
    # representation health first: a degenerate rep makes every probe
    # meaningless, and ridge dies outright on a singular Gram (seen on oja).
    out["live_dims"] = float((rtr.std(0) > 1e-4).float().mean())
    out["rep_std"] = float(rtr.std())
    sim = F.normalize(rte, dim=1) @ F.normalize(rtr, dim=1).T
    out["kNN"] = (torch.mode(ytr[sim.topk(20, dim=1).indices], dim=1).values
                  == yte).float().mean().item() * 100
    tgt = F.one_hot(ytr, 10).double()
    a = torch.cat([rtr, torch.ones(len(rtr), 1)], 1).double()
    b = torch.cat([rte, torch.ones(len(rte), 1)], 1).double()
    gram = a.T @ a
    gram.diagonal().add_(80.0)
    try:
        wts = torch.linalg.solve(gram, a.T @ tgt)
        out["ridge"] = ((b @ wts).argmax(1) == yte).float().mean().item() * 100
    except Exception as e:  # singular => the rep collapsed; that IS the finding
        print(f"    ridge failed ({type(e).__name__}): rep is degenerate, "
              f"live_dims={out['live_dims']:.3f} rep_std={out['rep_std']:.2e}",
              flush=True)
        out["ridge"] = float("nan")
    torch.manual_seed(SEED)
    clf = torch.nn.Linear(rtr.shape[1], 10).float()
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(clf(rtr), ytr).backward()
        opt.step()
    out["logistic"] = (clf(rte).argmax(1) == yte).float().mean().item() * 100
    return out


def reps(agt, imgs):
    return torch.stack([(agt.step([i], use_lrn=False, training=False),
                         agt.get_representations().float())[1] for i in imgs])


def run_arm(arm, imgs, labels, timgs, tlabels, eval_tr, n_steps):
    install(arm)
    torch.manual_seed(SEED)
    agt = CIFARAgt()
    gen = torch.Generator(device=imgs.device).manual_seed(SEED)
    t0 = time.perf_counter()
    for i in range(WARMUP):  # BN warmup, no learning, identical in every arm
        agt.step([imgs[torch.randint(len(imgs), (1,), generator=gen).item()]],
                 use_lrn=False, training=True)
    if arm != "frozen":
        for i in range(n_steps):
            agt.step([imgs[torch.randint(len(imgs), (1,), generator=gen).item()]])
            if (i + 1) % 10000 == 0:
                print(f"    {arm} {i+1}/{n_steps}  {time.perf_counter()-t0:.0f}s",
                      flush=True)
    rtr, rte = reps(agt, imgs[eval_tr]), reps(agt, timgs[:N_TEST_EVAL])
    mu, sd = rtr.mean(0), rtr.std(0) + 1e-6
    p = probes((rtr - mu) / sd, labels[eval_tr], (rte - mu) / sd, tlabels[:N_TEST_EVAL])
    p["secs"] = time.perf_counter() - t0
    # weight health: are the channels still distinct, or collapsed?
    w = agt.W[0]
    wn = torch.nn.functional.normalize(w, dim=0)
    cos = (wn.T @ wn)
    off = cos - torch.diag(torch.diag(cos))
    p["mean_abs_cos"] = float(off.abs().sum() / (len(cos) * (len(cos) - 1)))
    p["w_norm"] = float(w.norm())
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
    print(f"device {dev}, steps {a.steps}, arms {a.arms}", flush=True)

    tr, te = CIFARDataset(train=True), CIFARDataset(train=False)
    imgs = (torch.tensor(tr.cifar.data).permute(0, 3, 1, 2).float() / 255).to(dev)
    labels = torch.tensor(tr.cifar.targets, device=dev)
    timgs = (torch.tensor(te.cifar.data).permute(0, 3, 1, 2).float() / 255).to(dev)
    tlabels = torch.tensor(te.cifar.targets, device=dev)
    eval_tr = torch.randperm(len(imgs),
                             generator=torch.Generator(device=dev).manual_seed(0))[:N_TRAIN_EVAL]

    res = {}
    for arm in a.arms.split(","):
        print(f"\n=== {arm} ===", flush=True)
        res[arm] = run_arm(arm, imgs, labels, timgs, tlabels, eval_tr, a.steps)
        print(f"  {arm}: " + "  ".join(f"{k} {v:.2f}" for k, v in res[arm].items()),
              flush=True)

    print(f"\n{'arm':>9} {'kNN':>7} {'ridge':>7} {'logit':>7} "
          f"{'dkNN':>7} {'dridge':>7} {'dlogit':>7} {'|cos|':>7} {'|W|':>8} {'live':>7}")
    f = res.get("frozen")
    for arm in a.arms.split(","):
        r = res.get(arm)
        if not r:
            continue
        d = ("" if not f else
             f"{r['kNN']-f['kNN']:+7.2f} {r['ridge']-f['ridge']:+7.2f} "
             f"{r['logistic']-f['logistic']:+7.2f}")
        print(f"{arm:>9} {r['kNN']:7.2f} {r['ridge']:7.2f} {r['logistic']:7.2f} "
              f"{d} {r['mean_abs_cos']:7.3f} {r['w_norm']:8.1f} "
              f"{r['live_dims']:7.3f}")
    print("\ndeltas are vs the frozen control (same init, same warmup, no learning).")
    print("mean_abs_cos: mean |cosine| between layer-1 channel weights. High = "
          "channels collapsed onto one prototype; the anti-Hebbian repulsion "
          "exists to keep this low.")


if __name__ == "__main__":
    main()
