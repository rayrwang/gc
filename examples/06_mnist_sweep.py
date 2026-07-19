
"""
Controlled sweep harness: does a local Hebbian rule learn representations that beat
a frozen random net of the same init? This is the fast research engine behind the
quantified gaps in 04/05: a batched, vectorized reimplementation of the agent's
learning, decoupled from the Col/Agt framework so whole design axes can be swept in
seconds.

One controlled run: init a dense net, clone it into a learning copy and a frozen copy
(same init), train the learning copy with the chosen Hebbian rule, then probe both
with ridge on standardized final-layer features. The learn-vs-frozen gap (same init,
multi-seed) is the signal; a linear probe on the input is the floor.

Sweepable per cfg: rule (basic/instar/oja/bcm), competition (none/triangle/softmax/
layernorm/topk), weight-norm, hidden dims (-> depth), init scale, input norm
(raw/bin/std/zca), activation (relu/tanh/id/spike). Competition shapes the local
update (ylearn) while the uncompeted activation is propagated and read as the rep.

Scope/caveats: MNIST and fully-connected only (the conv line lives in 09/10); ridge
probe only; batched (B=500), a fast approximation of the agent's per-sample online
stepping, so learning-rate effects can differ (selftest checks the batched instar/oja
math matches fc per-sample at B=1). Headline finding (reproduced by __main__): without
competition the correlational rules (basic/instar/oja) collapse to chance while BCM
self-stabilizes via its sliding threshold; with softmax + weight-norm all four tie
(~84%): the rule is not the differentiator, the architecture is.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import statistics

import torch
import torchvision

import src.funcs as fc

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
torch.set_default_dtype(torch.float32)


# ---------- data (load once, vectorized) ----------
def load(split):
    ds = torchvision.datasets.MNIST("data", train=(split == "train"), download=True)
    X = ds.data.reshape(-1, 784).float().to(DEV) / 255.0
    Y = torch.zeros(len(ds), 10, device=DEV)
    Y[torch.arange(len(ds)), ds.targets.to(DEV)] = 1.0
    return X, Y

Xtr, Ytr = load("train")
Xte, Yte = load("test")


def normalize_input(X, kind, stats=None):
    if kind == "bin":
        return torch.where(X > 0.0, 1.0, 0.0), None
    if kind == "raw":
        return X, None
    if kind == "std":
        if stats is None:
            stats = (X.mean(0), X.std(0) + 1e-3)
        return (X - stats[0]) / stats[1], stats
    if kind == "zca":
        if stats is None:
            mu = X.mean(0)
            Xc = X - mu
            cov = (Xc.T @ Xc) / len(Xc)
            U, S, _ = torch.linalg.svd(cov)
            W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-2)) @ U.T
            stats = (mu, W)
        return (X - stats[0]) @ stats[1], stats
    raise ValueError(kind)


# ---------- activation / competition ----------
def act(a, kind):
    if kind == "relu":
        return torch.relu(a)
    if kind == "tanh":
        return torch.tanh(a)
    if kind == "id":
        return a
    if kind == "spike":
        return torch.where(a < 1.0, 0.0, 1.0)
    raise ValueError(kind)


def compete(z, kind):
    if kind == "none":
        return z
    if kind == "triangle":  # Coates-Ng: only above-average neurons fire
        return torch.relu(z - z.mean(1, keepdim=True))
    if kind == "softmax":
        return torch.softmax(z, dim=1)
    if kind == "layernorm":  # across neurons per sample (subtractive + divisive)
        return (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-6)
    if kind == "topk":  # keep top-10% per sample
        k = max(1, z.shape[1] // 10)
        thr = z.topk(k, dim=1).values[:, -1:]
        return torch.where(z >= thr, z, 0.0)
    raise ValueError(kind)


# ---------- net ----------
def init_net(dims, in_dim, scale, gen):
    W = []
    fan = in_dim
    for d in dims:
        W.append(scale * torch.randn(fan, d, generator=gen, device=DEV) / fan**0.5)
        fan = d
    return W


def wnorm_(W):  # unit L2 norm per output neuron (column)
    for w in W:
        w /= (w.norm(dim=0, keepdim=True) + 1e-8)


def forward(W, h, cfg):
    # per layer: (presyn, dense, ylearn). dense = activation (propagated + read as
    # the rep); ylearn = competed signal used only for the local update.
    layers = []
    for w in W:
        dense = act(h @ w, cfg["act"])
        y = compete(dense, cfg["comp"])
        layers.append((h, dense, y))
        h = dense
    return layers


def hebb_update(W, layers, cfg, theta):
    ss = cfg["ss"]
    for li, (x, _dense, y) in enumerate(layers):
        B = x.shape[0]
        XY = x.T @ y / B
        rule = cfg["rule"]
        if rule == "basic":
            dw = ss * XY
        elif rule == "instar":
            dw = ss * (XY - W[li] * y.mean(0, keepdim=True))
        elif rule == "oja":
            dw = ss * (XY - W[li] * (y * y).mean(0, keepdim=True))
        elif rule == "bcm":
            th = theta[li]
            th.mul_(0.8).add_(0.2 * (y * y).mean(0))     # sliding threshold EMA
            dw = ss * (x.T @ (y * (y - th)) / B)
        else:
            raise ValueError(rule)
        W[li] += dw
        if cfg["wnorm"]:
            W[li] /= (W[li].norm(dim=0, keepdim=True) + 1e-8)


# ---------- probe (ridge on standardized final reps) ----------
def reps_of(W, X, cfg, istats):
    Xn, _ = normalize_input(X, cfg["norm"], istats)
    return forward(W, Xn, cfg)[-1][1]  # dense activation of last layer


def _solve(Rtr, Ytr_, Rte, Yte_):
    Rtr = torch.nan_to_num(Rtr)
    Rte = torch.nan_to_num(Rte)
    n = Rtr.shape[0]
    Rtr = torch.cat([Rtr, torch.ones(Rtr.shape[0], 1)], 1).double()
    Rte = torch.cat([Rte, torch.ones(Rte.shape[0], 1)], 1).double()
    lam = 0.01 * n  # ridge scaled to standardized features (R^T R diag ~ n)
    A = Rtr.T @ Rtr + lam * torch.eye(Rtr.shape[1], dtype=torch.float64)
    V = torch.linalg.solve(A, Rtr.T @ Ytr_.double())
    return ((Rte @ V).argmax(1) == Yte_.argmax(1)).float().mean().item() * 100


def ridge_acc(W, cfg, istats, ntr=10000, nte=2000):
    Rtr = reps_of(W, Xtr[:ntr], cfg, istats)
    Rte = reps_of(W, Xte[:nte], cfg, istats)
    mu, sd = Rtr.mean(0), Rtr.std(0) + 1e-6
    return _solve((Rtr - mu) / sd, Ytr[:ntr], (Rte - mu) / sd, Yte[:nte])


def linear_baseline(cfg, istats, ntr=10000, nte=2000):
    Xn, _ = normalize_input(Xtr[:ntr], cfg["norm"], istats)
    Xnte, _ = normalize_input(Xte[:nte], cfg["norm"], istats)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    return _solve((Xn - mu) / sd, Ytr[:ntr], (Xnte - mu) / sd, Yte[:nte])


# ---------- one controlled run: learn vs frozen, same init ----------
def run(cfg, seed, n_learn=30000, B=500):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    _, istats = normalize_input(Xtr, cfg["norm"])  # fit norm on full train
    in_dim = normalize_input(Xtr[:2], cfg["norm"], istats)[0].shape[1]

    W0 = init_net(cfg["dims"], in_dim, cfg["scale"], gen)
    if cfg["wnorm"]:
        wnorm_(W0)
    W_frozen = [w.clone() for w in W0]
    W_learn = [w.clone() for w in W0]

    theta = [torch.ones(d) for d in cfg["dims"]]
    perm = torch.randperm(len(Xtr), generator=gen, device=DEV)[:n_learn]
    for i in range(0, n_learn, B):
        Xn, _ = normalize_input(Xtr[perm[i:i + B]], cfg["norm"], istats)
        hebb_update(W_learn, forward(W_learn, Xn, cfg), cfg, theta)

    return {"learn": ridge_acc(W_learn, cfg, istats),
            "frozen": ridge_acc(W_frozen, cfg, istats)}


def sweep(cfgs, seeds=(0, 1, 2)):
    """Run each cfg-override over seeds; print learn / frozen / gap / linear floor."""
    base = {"rule": "instar", "comp": "none", "wnorm": False, "dims": [256],
            "scale": 3.0, "norm": "raw", "act": "relu", "ss": 0.01}
    print(f"{'rule':7} {'comp':9} {'wn':5} {'dims':8} {'sc':4} {'norm':5} {'act':5} "
          f"| {'learn':>12} {'frozen':>12} {'gap':>8} {'lin':>6}")
    for over in cfgs:
        cfg = {**base, **over}
        L = [run(cfg, s) for s in seeds]
        learn = [r["learn"] for r in L]
        frozen = [r["frozen"] for r in L]
        lin = linear_baseline(cfg, normalize_input(Xtr, cfg["norm"])[1])
        lm, fm = statistics.mean(learn), statistics.mean(frozen)
        ls = statistics.pstdev(learn) if len(learn) > 1 else 0
        fs = statistics.pstdev(frozen) if len(frozen) > 1 else 0
        print(f"{cfg['rule']:7} {cfg['comp']:9} {str(cfg['wnorm']):5} "
              f"{str(cfg['dims']):8} {cfg['scale']:<4} {cfg['norm']:5} {cfg['act']:5} "
              f"| {lm:5.1f}±{ls:3.1f}   {fm:5.1f}±{fs:3.1f}   {lm - fm:+6.1f}  {lin:5.1f}")


def selftest():
    """The batched rules must match fc's per-sample rules at batch size 1."""
    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.rand(64, generator=g, device=DEV)
    y = torch.rand(32, generator=g, device=DEV)
    w = torch.randn(64, 32, generator=g, device=DEV) * 0.1
    X, Y = x[None], y[None]  # batch of 1
    mine = w + 0.02 * (X.T @ Y - w * Y.mean(0, keepdim=True))
    assert torch.allclose(mine, fc.lrn_instar(x, w, y, ss=0.02), atol=1e-5), "instar mismatch"
    mine = w + 0.02 * (X.T @ Y - w * (Y * Y).mean(0, keepdim=True))
    assert torch.allclose(mine, fc.lrn_oja(x, w, y, ss=0.02), atol=1e-5), "oja mismatch"
    print("selftest OK (batched instar/oja match fc)\n")


if __name__ == "__main__":
    selftest()
    print("=== no competition: basic/instar/oja collapse, BCM self-stabilizes ===")
    sweep([{"rule": r, "comp": "none", "wnorm": False} for r in ("basic", "instar", "oja", "bcm")])
    print("\n=== + softmax competition + weight-norm: the correlational rules recover ===")
    sweep([{"rule": r, "comp": "softmax", "wnorm": True} for r in ("basic", "instar", "oja", "bcm")])
