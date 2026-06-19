"""
CIFAR-100 substrate probe: the CIFAR-10 example (08_cifar10.py) on the harder 100-class task
(1% chance). Same recipe (src.agents.CIFARAgt -- oja-signed + Triangle + soft-norm + online BN,
no whitening, 24576-dim rep), same online single-sample protocol, same probes (kNN/ridge/
logistic) vs a SAME-INIT frozen control. See 08 for the recipe, the four load-bearing
ingredients, and the SoftHebb (Moraitis et al., arXiv:2209.11883) lineage. Kept deliberately
parallel to 08 -- sync changes across both.

The CIFAR-100 story mirrors CIFAR-10. Read LEARNING GAINS (Δ vs each arm's OWN frozen):
    config                  1 epoch (50k)         4 epochs (200k)
    gc                      +4.1 / +2.0 / +1.6    +2.4 / +7.3 / +3.7
    SoftHebb b1 (online)    +3.6 / +2.9 / +9.0    +2.2 / -1.7 / +5.9

At 1 epoch SoftHebb learns more (especially logistic, +9.0 vs +1.6) -- gc is NOT competitive at a
single pass. The one thing gc does that SoftHebb (as run) doesn't is hold up under CONTINUED
training: gc's gains are stable-to-RISING (ridge 25.7->31.0) while SoftHebb's DECAY -- its ridge
falls BELOW its own frozen by 4 epochs. By 200k gc edges SoftHebb on ridge and ~ties on
kNN/logistic. Same honest result as 08: "gc's gains don't collapse under continual training,"
NOT "gc beats SoftHebb." Same caveat: SoftHebb's per-layer temperature/power/LR schedule was not
ported, so its 4-epoch decay may be un-annealed overtraining, not a property of induction; needs
a scheduled baseline + more seeds. Single config, ~1 seed.

ABSOLUTE NUMBERS (kNN/ridge/logistic %, chance 1%; gains above are vs each frozen row):
    gc:       frozen 15.1/23.7/20.1   50k 19.2/25.7/21.7   200k 17.5/31.0/23.8
    SoftHebb: frozen 15.8/25.3/18.1   b1 50k 19.4/28.2/27.1   b1 200k 18.0/23.6/24.0
Frozen rows differ slightly (same architecture; forward-path/seed, see 08). The recipe lives in
src.agents.CIFARAgt; this script is the harness. float32 (Oja products overflow fp16).
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before torch)

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents import CIFARAgt
from src.envs import CIFAR100Dataset

SEED = 0
N_STEPS = 50000                                       # online single-sample steps (~1 epoch)
EVAL_INTERVAL = 10000                                 # report learn-vs-frozen progress
WARMUP = 1000                                         # populate BN stats before learning
N_TRAIN_EVAL, N_TEST_EVAL = 3000, 1000
N_CLASSES = 100
K_NN = 20


# Frozen classifiers (each: train feats/labels, test feats/labels -> accuracy %)
def knn(rtr, ytr, rte, yte, k=K_NN):
    sim = F.normalize(rte, dim=1) @ F.normalize(rtr, dim=1).T
    return (torch.mode(ytr[sim.topk(k, dim=1).indices], dim=1).values == yte).float().mean().item() * 100


def ridge(rtr, ytr, rte, yte, lam=80.0):
    targets = F.one_hot(ytr, N_CLASSES).double()
    a = torch.cat([rtr, torch.ones(len(rtr), 1)], 1).double()
    b = torch.cat([rte, torch.ones(len(rte), 1)], 1).double()
    gram = a.T @ a
    gram.diagonal().add_(lam)             # ridge penalty in-place (avoids a 24576^2 eye -> OOM at this rep dim)
    weights = torch.linalg.solve(gram, a.T @ targets)
    return ((b @ weights).argmax(1) == yte).float().mean().item() * 100


def logistic(rtr, ytr, rte, yte, steps=400, lr=0.05):
    torch.manual_seed(SEED)                       # deterministic classifier init (the only RNG in eval)
    clf = torch.nn.Linear(rtr.shape[1], N_CLASSES).float()
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
    torch.use_deterministic_algorithms(True)              # fully reproducible run-to-run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    train_ds, test_ds = CIFAR100Dataset(train=True), CIFAR100Dataset(train=False)
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

    writer = SummaryWriter("runs/cifar100")
    print("tensorboard --logdir runs/cifar100\n")
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
