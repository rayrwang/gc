
"""
Evaluate the CIFAR-10 conv-BCM agent's learned representations with FROZEN probes
(kNN, ridge, logistic), offline -- the CIFAR analogue of 05_mnist_all_probes.

A single Hebbian conv layer (CIFARAgt) trained by BCM, read as a spatially
avg-pooled feature map. Same controlled protocol as 05: a learning agent and a
SAME-INIT frozen control, periodically frozen, features standardised, each probe
fit. kNN / ridge / logistic on the raw flattened pixels is the image baseline.

Findings (controlled same-init A/B, single seed, 5000 online steps):

1. Random conv is a strong baseline. A frozen, untrained conv + ReLU + avg-pool
   gets ridge ~47% vs ~29% on raw pixels -- +18 from an UNtrained conv. CIFAR
   gives a high ceiling for a representation, but the random init already claims
   most of it (the "random features are hard to beat" result, more pronounced
   here than on MNIST where random ReLU sat near the linear floor).

2. Plain conv-BCM (no competition, no whitening) DEGRADES the rep, monotonically:
   every probe ends BELOW its frozen control (kNN -1.2, ridge -1.3, logistic -4.6
   at step 5000) and the learn curve declines steadily while the control is flat.
   Without competition or whitening BCM chases the dominant DC / low-frequency
   patch directions and collapses the diverse random filters toward redundant
   ones -> lower effective rank -> below random. This CONFIRMS the prediction in
   04's finding 1 ("expect competition to become necessary on harder / natural
   data"): on natural images the caveat is binding, and learning HURTS.

3. ZCA whitening of the patches flips the sign. CIFARAgt.fit_whitening centers
   and decorrelates patches, killing the DC dominance, fit once and applied
   EQUALLY to learn and control (label-free input preprocessing). Two effects:
   (a) whitening alone lifts the random baseline ~+3 every probe (an untrained
   conv on decorrelated input is just better); (b) the learning gap flips from
   collapse to a small robust positive -- kNN +0.9, ridge +0.7, logistic +0.9 at
   step 5000, stable (no decline), consistent across all three probes incl.
   parameter-free kNN (the same ~+1 signature as MNIST). So learning on CIFAR is
   no longer harmful, now weakly helpful.

   Is whitening cheating? No in the way that matters: it is label-free (input
   statistics only) and applied equally to both arms, so it raises learn and
   control together and the learn-vs-control gap stays clean and attributable to
   BCM; and it is biologically grounded (retinal / LGN center-surround ~ whitening,
   and the BCM / ICA / sparse-coding natural-image literature all whiten). But it
   IS a non-local, offline, batch-fit linear transform -- a shortcut the local-
   learning program should eventually replace with an ONLINE LOCAL decorrelation,
   which is the same machinery as lateral-inhibition competition. Treat the ZCA as
   a fixed oracle proving "decorrelated input lets BCM help instead of hurt";
   making it local is future work (and overlaps with competition, TBD). Note most
   of the absolute accuracy is the preprocessing, not the learning (~+3 vs ~+1).

float32 (not the float16 the MNIST agent uses): conv-BCM accumulates over many
patches and overflows float16 more easily, and CIFAR is harder, so start clean.

  tensorboard --logdir runs/cifar
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import random

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# isort: off
from src.agents import CIFARAgt
from src.envs import CIFARDataset

SEED = 0
N_STEPS = 5000          # online learning steps
EVAL_INTERVAL = 500
N_TRAIN_EVAL = 4000     # rep subset for fitting the probes
N_TEST_EVAL = 1500
K_NN = 20


# Frozen classifiers (each: train feats/labels, test feats/labels -> accuracy %)
def knn(rtr, ytr, rte, yte, k=K_NN):
    sim = F.normalize(rte, dim=1) @ F.normalize(rtr, dim=1).T
    pred = torch.mode(ytr[sim.topk(k, dim=1).indices], dim=1).values
    return (pred == yte).float().mean().item() * 100


def ridge(rtr, ytr, rte, yte, lam=80.0):
    targets = F.one_hot(ytr, 10).double()
    a = torch.cat([rtr, torch.ones(len(rtr), 1)], 1).double()
    b = torch.cat([rte, torch.ones(len(rte), 1)], 1).double()
    eye = lam * torch.eye(a.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(a.T @ a + eye, a.T @ targets)
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
    reps = []
    for img in imgs:
        agt.step([img], use_lrn=False, disable_print=True)
        reps.append(agt.get_representations().float())
    return torch.stack(reps)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Data as bulk tensors (N, 3, 32, 32)
    train_ds, test_ds = CIFARDataset(train=True), CIFARDataset(train=False)
    train_imgs = (torch.tensor(train_ds.cifar.data).permute(0, 3, 1, 2).float() / 255).to(device)
    train_labels = torch.tensor(train_ds.cifar.targets, device=device)
    test_imgs = (torch.tensor(test_ds.cifar.data).permute(0, 3, 1, 2).float() / 255).to(device)
    test_labels = torch.tensor(test_ds.cifar.targets, device=device)
    eval_tr = torch.randperm(len(train_imgs))[:N_TRAIN_EVAL]
    eval_te = torch.arange(N_TEST_EVAL)
    ytr, yte = train_labels[eval_tr], test_labels[eval_te]

    # Learning agent and a frozen control sharing the SAME random init
    torch.manual_seed(SEED)
    learn_agt = CIFARAgt()
    torch.manual_seed(SEED)
    control_agt = CIFARAgt()
    agents = {"learn": (learn_agt, True), "control": (control_agt, False)}

    # ZCA-whiten the patches: input preprocessing, fit once and SHARED by both
    # agents (identical for learn and control, so the only difference stays use_lrn)
    learn_agt.fit_whitening(train_imgs[:2000])
    control_agt.whiten_mu, control_agt.whiten_W = learn_agt.whiten_mu, learn_agt.whiten_W

    # Probes on the raw flattened pixels, once, as the image baseline (the floor)
    base_tr, base_te = standardize(
        train_imgs[eval_tr].reshape(N_TRAIN_EVAL, -1),
        test_imgs[eval_te].reshape(N_TEST_EVAL, -1))
    baseline = {name: probe(base_tr, ytr, base_te, yte) for name, probe in PROBES.items()}

    writer = SummaryWriter("runs/cifar")
    print("\nImage baseline:  " + "   ".join(f"{n} {baseline[n]:.1f}%" for n in PROBES))
    print("tensorboard --logdir runs/cifar\n")

    for step in range(N_STEPS + 1):
        i = random.randrange(len(train_imgs))
        for agt, lrn in agents.values():
            agt.step([train_imgs[i]], use_lrn=lrn, disable_print=True)

        if step % EVAL_INTERVAL == 0:
            feats = {
                cond: standardize(representations(agt, train_imgs[eval_tr]),
                                  representations(agt, test_imgs[eval_te]))
                for cond, (agt, _) in agents.items()}
            print(f"Step {step}")
            for pname, probe in PROBES.items():
                accs = {cond: probe(feats[cond][0], ytr, feats[cond][1], yte) for cond in agents}
                accs["image"] = baseline[pname]
                print(f"  {pname:9} learn {accs['learn']:5.1f}%   control {accs['control']:5.1f}%   "
                      f"(image {accs['image']:.1f}%)   gap {accs['learn'] - accs['control']:+.1f}")
                writer.add_scalars(pname, accs, step)
            writer.flush()
