"""
CIFAR-10: a local, ONLINE Hebbian conv stack -- no backprop, no objective, one sample at a
time. Three conv layers trained ONLY by a local rule, read as a pooled feature map and
probed (kNN/ridge/logistic) vs a SAME-INIT frozen control; the gap is what learning adds.

RULE -- oja-signed, which IS SoftHebb's update (Moraitis et al., arXiv:2209.11883): a
prototype rule gated by a SIGNED soft-WTA. Per location, softmax over channels; the winner
moves its weight TOWARD the input (Hebbian), losers move AWAY (anti-Hebbian). Loser
repulsion makes channels tile instead of collapsing onto one prototype.

WHY THIS ISN'T JUST A WORSE SoftHebb: it's the same rule, on purpose. gc's own rules (BCM,
instar, plain oja) were tested and FAIL on CIFAR -- below the random control -- so the
SoftHebb-class signed rule is simply the local primitive that works. What differs: (1) gc
runs it fully ONLINE / single-sample (SoftHebb batches); (2) in gc this is a SUBSTRATE for
an objective-free association architecture (the Dir.E critic), not a standalone classifier.
We adopt the best known local rule and verify it survives gc's online constraint; the
divergence from SoftHebb is what gets built ON this, not the CIFAR number.

TWO NON-OBVIOUS LEVERS (a naive conv-Hebb stack fails without both):
 - Triangle activation relu(u - mean_c(u))^0.7 -- a GRADED forward code. A hard-WTA/softmax
   activation near-one-hots the signal and collapses over depth (3 layers -> ~13% kNN).
 - SOFT weight-norm -- let ||w|| drift via the rule's decay; do NOT hard-project each step.
   Hard projection makes learning DESTRUCTIVE (drops below the frozen baseline).
ZCA whitening of layer-1 patches (label-free, both arms) lifts the baseline; kept, though
an offline shortcut a local decorrelation should eventually replace.

RESULTS (CIFAR-10, widetop config; gain vs same-init frozen). This script runs 20k steps
= gc's kNN PEAK, which is a TRANSIENT early-stopping point, NOT a stable optimum. The rep
sits on a kNN<->logistic tradeoff and gets WORSE with more training:
    steps      kNN          ridge        logistic
    20k     50.7 (+7.8)   48.4 (-1.8)   52.5 (-1.9)   <- kNN peak (this script's operating pt)
    50k     47.4 (+4.5)   47.9 (-2.3)   57.0 (+2.6)
    100k    43.8 (+0.9)   50.3 (+0.1)   58.0 (+3.6)
    200k    36.2 (-6.7)   48.9 (-1.3)   57.6 (+3.2)   <- kNN now BELOW frozen
Prolonged training COLLAPSES kNN (the prototype over-specializes, no homeostatic regularizer)
while logistic creeps to a ~58 plateau and ridge never moves. There is no step count where
gc is good on all three at once.

TRAINING-MATCHED comparison (both ~1 epoch / 50k images):
    gc @ 50k                       47.4   47.9   57.0
    SoftHebb online   (batch 1)    51.5   63.7   62.2   <- holds all three high
    SoftHebb batched  (batch 10)   54.6   65.7   62.7
    SoftHebb native readout (50k-sample + dropout + 50-epoch linear): 79.9
Matched on training, gc trails SoftHebb on ALL three probes. (A transient gc peak does reach
kNN ~51 at 20k, but that is early-stopping luck, not parity.)

WHY gc only TRADES kNN<->logistic while SoftHebb holds all three: leading suspect is
BatchNorm-DURING-TRAINING (gc uses per-sample instance-norm). BN standardizes each channel
across the batch every step -- a homeostatic regularizer that keeps channels balanced/used,
which would both prevent the prototype collapse (the kNN crash) AND hold linear separability.
Untested in gc; an online running-stats version would keep it in-spirit. This, not training
length or per-layer tuning, is the open gap to SoftHebb.

float32 (the Oja outer products overflow float16 over many patches).
  tensorboard --logdir runs/cifar
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import statistics

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.envs import CIFARDataset

SEEDS = (0, 1)
N_STEPS = 20000                                       # online single-sample steps
N_TRAIN_EVAL, N_TEST_EVAL = 3000, 1000                # rep subset for the probes
K_NN = 20
LAYERS = [(3, 192, 5), (192, 384, 3), (384, 1536, 3)]  # (in, out, kernel); stride1+pad, pool2 between
BASE_LR = 0.1                                          # fixed; calibrated for this width
POWER = 0.7                                            # Triangle exponent
SIGNED_T = 1.0                                         # soft-WTA gate temperature
INIT_SCALE = 3.0


class ConvHebb:
    """3-layer conv trained ONLY by the local oja-signed rule (no backprop)."""

    def __init__(self, confs, learns=True, whiten=None):
        self.confs, self.learns, self.whiten = confs, learns, whiten
        self.W = [INIT_SCALE * torch.randn(ic * k * k, oc) / (ic * k * k) ** 0.5 for ic, oc, k in confs]
        self.rep = None

    def _signed_gate(self, u):
        """SoftHebb gate: winner Hebbian (+), losers anti-Hebbian (-), magnitude = responsibility."""
        gate = -torch.softmax(u * SIGNED_T, dim=1)
        gate[torch.arange(u.shape[0]), u.argmax(1)] *= -1
        return gate

    def step(self, img, use_lrn=True):
        fmap = img
        for li, (ic, oc, k) in enumerate(self.confs):
            if li > 0:  # per-sample instance-norm on deeper layers
                fmap = (fmap - fmap.mean((1, 2), keepdim=True)) / (fmap.std((1, 2), keepdim=True) + 1e-5)
            H, Wd = fmap.shape[1], fmap.shape[2]
            x = F.unfold(fmap.unsqueeze(0), k, stride=1, padding=(k - 1) // 2)[0].T  # (H*W, ic*k*k)
            if li == 0:                                     # ZCA-whiten layer-1 patches
                x = (x - self.whiten["mu"]) @ self.whiten["W"]
            u = x @ self.W[li]
            y = F.relu(u - u.mean(1, keepdim=True)) ** POWER  # Triangle activation (graded)
            if use_lrn and self.learns:
                g = self._signed_gate(u)
                dW = (x.T @ g - self.W[li] * (g * u).sum(0, keepdim=True)) / x.shape[0]  # oja-signed
                self.W[li] = self.W[li] + BASE_LR * dW       # soft norm: no hard projection
            fmap = F.avg_pool2d(y.T.reshape(oc, H, Wd).unsqueeze(0), 2)[0]
        self.rep = F.adaptive_avg_pool2d(fmap.unsqueeze(0), 2).reshape(-1).clone()

    def get_representations(self):
        return self.rep


# Frozen classifiers (each: train feats/labels, test feats/labels -> accuracy %)
def knn(rtr, ytr, rte, yte, k=K_NN):
    sim = F.normalize(rte, dim=1) @ F.normalize(rtr, dim=1).T
    return (torch.mode(ytr[sim.topk(k, dim=1).indices], dim=1).values == yte).float().mean().item() * 100


def ridge(rtr, ytr, rte, yte, lam=80.0):
    targets = F.one_hot(ytr, 10).double()
    a = torch.cat([rtr, torch.ones(len(rtr), 1)], 1).double()
    b = torch.cat([rte, torch.ones(len(rte), 1)], 1).double()
    weights = torch.linalg.solve(a.T @ a + lam * torch.eye(a.shape[1], dtype=torch.float64), a.T @ targets)
    return ((b @ weights).argmax(1) == yte).float().mean().item() * 100


def logistic(rtr, ytr, rte, yte, steps=400, lr=0.05):
    clf = torch.nn.Linear(rtr.shape[1], 10).float()
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(clf(rtr), ytr).backward()
        opt.step()
    return (clf(rte).argmax(1) == yte).float().mean().item() * 100


PROBES = {"kNN": knn, "ridge": ridge, "logistic": logistic}


def standardize(rtr, rte):
    mu, sd = rtr.mean(0), rtr.std(0) + 1e-6
    return (rtr - mu) / sd, (rte - mu) / sd


def representations(agt, imgs):
    """Read the pooled conv feature map for each image (learning off)."""
    out = []
    for img in imgs:
        agt.step(img, use_lrn=False)
        out.append(agt.get_representations().float())
    return torch.stack(out)


def fit_whitening(train_imgs):
    """ZCA on layer-1 patches: label-free input preprocessing, fit once, shared by both arms."""
    ic, oc, k = LAYERS[0]
    p = F.unfold(train_imgs[:2000], k, stride=1, padding=(k - 1) // 2).permute(0, 2, 1).reshape(-1, ic * k * k)
    p = p[torch.randperm(len(p))[:100_000]]
    mu = p.mean(0)
    xc = (p - mu).double()
    ev, evec = torch.linalg.eigh((xc.T @ xc) / len(xc))
    return {"mu": mu, "W": (evec @ torch.diag((ev + 0.1).rsqrt()) @ evec.T).to(mu.dtype)}


def evaluate(agt, etr, ytr, ete, yte):
    rtr, rte = standardize(representations(agt, etr), representations(agt, ete))
    return {n: p(rtr, ytr, rte, yte) for n, p in PROBES.items()}


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    train_ds, test_ds = CIFARDataset(train=True), CIFARDataset(train=False)
    train_imgs = (torch.tensor(train_ds.cifar.data).permute(0, 3, 1, 2).float() / 255).to(device)
    train_labels = torch.tensor(train_ds.cifar.targets, device=device)
    test_imgs = (torch.tensor(test_ds.cifar.data).permute(0, 3, 1, 2).float() / 255).to(device)
    test_labels = torch.tensor(test_ds.cifar.targets, device=device)
    n_train = len(train_imgs)
    whiten = fit_whitening(train_imgs)

    eval_tr = torch.randperm(n_train, generator=torch.Generator(device=device).manual_seed(0))[:N_TRAIN_EVAL]
    ytr, yte = train_labels[eval_tr], test_labels[:N_TEST_EVAL]
    etr, ete = train_imgs[eval_tr], test_imgs[:N_TEST_EVAL]

    writer = SummaryWriter("runs/cifar")
    print("tensorboard --logdir runs/cifar\n")
    results = {"learn": [], "control": []}
    for seed in SEEDS:
        order = torch.randint(0, n_train, (N_STEPS,), generator=torch.Generator(device=device).manual_seed(seed), device=device)
        torch.manual_seed(seed)
        learn_agt = ConvHebb(LAYERS, learns=True, whiten=whiten)
        torch.manual_seed(seed)                                  # SAME init as the control
        control_agt = ConvHebb(LAYERS, learns=False, whiten=whiten)
        for i in order:
            learn_agt.step(train_imgs[i], use_lrn=True)
        acc = {cond: evaluate(agt, etr, ytr, ete, yte)
               for cond, agt in (("learn", learn_agt), ("control", control_agt))}
        results["learn"].append(acc["learn"])
        results["control"].append(acc["control"])
        print(f"seed {seed}: " + "  ".join(
            f"{p} learn {acc['learn'][p]:.1f} / control {acc['control'][p]:.1f}" for p in PROBES))

    print(f"\n{'probe':9} {'learned':>8} {'frozen':>8} {'gain':>6}")
    for p in PROBES:
        learned = statistics.mean(r[p] for r in results["learn"])
        frozen = statistics.mean(r[p] for r in results["control"])
        print(f"{p:9} {learned:8.1f} {frozen:8.1f} {learned - frozen:+6.1f}")
        writer.add_scalars(p, {"learn": learned, "control": frozen}, N_STEPS)
    writer.flush()
