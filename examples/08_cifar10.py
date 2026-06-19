"""
CIFAR-10: a local, ONLINE Hebbian conv stack -- no backprop, no objective, no batching, one
sample at a time. Reproduces the SoftHebb recipe (Moraitis et al., arXiv:2209.11883) under
gc's online/single-sample constraint, with NO offline oracles. Three conv layers trained
ONLY by a local rule, read as a pooled feature map and probed (kNN/ridge/logistic) vs a
SAME-INIT frozen control; the gap is what learning adds.

RULE -- oja-signed, which IS SoftHebb's update: a prototype rule gated by a SIGNED soft-WTA.
Per location, softmax over channels; the winner moves its weight TOWARD the input (Hebbian),
losers move AWAY (anti-Hebbian). Loser repulsion makes channels tile instead of collapsing
onto one prototype. gc's own rules (BCM, plain instar/oja) were tested and FAIL on CIFAR --
below the random control -- so the SoftHebb-class signed rule is the local primitive that
works. gc's divergence from SoftHebb is what gets built ON this (the Dir.E critic), not here.

FOUR INGREDIENTS, each load-bearing (drop any and it breaks):
 - Triangle activation relu(u-mean_c(u))^0.7 -- a GRADED forward code. A hard-WTA/softmax
   activation near-one-hots the signal and collapses over depth (3 layers -> ~13% kNN).
 - SOFT weight-norm -- let ||w|| drift via the rule's decay; do NOT hard-project each step.
   Hard projection makes learning DESTRUCTIVE (drops below the frozen baseline).
 - Online BatchNorm (current-image stats at train, running at eval) at every layer -- the
   homeostatic regularizer. Without it, kNN is a TRANSIENT peak that collapses with training
   (52->36); with it the rep is STABLE under indefinite (continual) training.
 - NO whitening -- BN does all the normalization. Whitening (even an online ZCA) actually
   SUPPRESSES the kNN learning (it pre-captures the structure the prototype would learn).

RESULTS (CIFAR-10, kNN/ridge/logistic %, matched on training: 50k=1 epoch, 200k=4 epochs):
    config                       kNN   ridge  logistic
    gc no-learning (frozen)     41.7   52.0    59.2
    gc           50k            48.2   53.5    59.8
    gc          200k            47.3   58.8    63.3
    SoftHebb b1   50k           50.9   63.4    61.8
    SoftHebb b1  200k           47.7   59.6    58.9
    SoftHebb b10  50k           53.9   65.9    62.9
    SoftHebb b10 200k           50.2   61.7    58.8
    SoftHebb native readout      --     --     79.9
    (50k-sample + dropout + 50-epoch linear)

KEY: SoftHebb is tuned for 1 epoch and DEGRADES with more training (b1 50.9->47.7 kNN); gc is
STABLE -- no collapse, ridge/logistic even rise. So at 1 epoch SoftHebb leads, but at 4 epochs
(the continual regime gc targets) gc TIES SoftHebb-online on kNN/ridge and BEATS it on logistic
(63.3 vs 58.9). gc's edge is stability under indefinite training; SoftHebb's is the early-stop
peak plus its per-layer schedule (temperature/power/LR), deliberately not ported here.

Other findings, in brief: gc's own rules (BCM, plain instar/oja) all FAIL here -- the signed
anti-Hebbian gate is required. The four ingredients above are each necessary -- hard-WTA
activation collapses with depth; hard weight-norm makes learning destructive; without online
BN the kNN gain is a transient peak that COLLAPSES (52->36) under continued training; and
whitening (even an online ZCA matching the offline oracle) SUPPRESSES the kNN gain, so none is
used. Per-neuron adaptive-LR was tested and HURTS kNN. SoftHebb's larger linear-probe gaps are
mostly its BatchNorm conditioning + channel count, not representation quality.

float32 (the Oja outer products overflow float16 over many patches).
  tensorboard --logdir runs/cifar
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents import CIFARAgt          # the recipe lives in the agent; this script is the harness
from src.envs import CIFARDataset

SEED = 0
N_STEPS = 50000                                       # online single-sample steps (~1 epoch)
EVAL_INTERVAL = 10000                                 # report learn-vs-frozen progress
WARMUP = 1000                                         # populate BN stats before learning
N_TRAIN_EVAL, N_TEST_EVAL = 3000, 1000
K_NN = 20


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
    """Read the pooled conv feature map for each image (learning off, BN in eval mode)."""
    out = []
    for img in imgs:
        agt.step([img], use_lrn=False, training=False)
        out.append(agt.get_representations().float())
    return torch.stack(out)


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
    gen = torch.Generator(device=device).manual_seed(SEED)
    eval_tr = torch.randperm(n_train, generator=torch.Generator(device=device).manual_seed(0))[:N_TRAIN_EVAL]
    ytr, yte = train_labels[eval_tr], test_labels[:N_TEST_EVAL]
    etr, ete = train_imgs[eval_tr], test_imgs[:N_TEST_EVAL]
    warm = torch.randint(0, n_train, (WARMUP,), generator=gen, device=device)
    order = torch.randint(0, n_train, (N_STEPS,), generator=gen, device=device)

    writer = SummaryWriter("runs/cifar")
    print("tensorboard --logdir runs/cifar\n")
    torch.manual_seed(SEED)
    learn_agt = CIFARAgt()
    torch.manual_seed(SEED)                                   # SAME init as the control
    control_agt = CIFARAgt()                                  # never gets use_lrn=True -> stays frozen
    for agt in (learn_agt, control_agt):                      # warm up BN stats before learning
        for i in warm:
            agt.step([train_imgs[i]], use_lrn=False, training=True)
    control = evaluate(control_agt, etr, ytr, ete, yte)       # frozen baseline (constant)
    print("frozen baseline:  " + "   ".join(f"{p} {control[p]:.1f}" for p in PROBES), flush=True)

    for step, i in enumerate(order, 1):
        learn_agt.step([train_imgs[i]], use_lrn=True, training=True)
        if step % EVAL_INTERVAL == 0:
            acc = evaluate(learn_agt, etr, ytr, ete, yte)
            print(f"step {step:>6}:  " + "   ".join(
                f"{p} {acc[p]:5.1f} ({acc[p] - control[p]:+.1f})" for p in PROBES), flush=True)
            for p in PROBES:
                writer.add_scalars(p, {"learn": acc[p], "control": control[p]}, step)
    writer.flush()
