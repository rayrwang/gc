"""
CIFAR-10 substrate probe: a local, online Hebbian conv stack (no backprop, no objective, one
sample at a time), three conv layers trained only by a local rule, probed (kNN/ridge/logistic)
vs a same-init frozen control. The gap is what learning adds.

TL;DR: the learned representation is stable under continued training (its gains don't
collapse); the earlier headline that it "beats SoftHebb" was wrong (see Finding). This file
measures representation quality from local unsupervised learning, nothing more.

Finding: read the learning gains (Δ vs each arm's own frozen); absolute numbers are confounded by
a ~3-4pt frozen head-start (forward-path/seed; see bottom). Δ kNN / ridge / logistic:
    config                    1 epoch (50k)           4 epochs (200k)
    gc                         +6.9 / +3.5  / +1.9    +5.2 / +7.1 / +4.2
    SoftHebb b1 (online)       +7.7 / +9.6  / +8.4    +4.5 / +5.8 / +5.5
    SoftHebb b10 (batched)    +10.7 / +12.1 / +9.5    +7.0 / +7.9 / +5.4

At 1 epoch SoftHebb learns far more; gc is not competitive at a single pass. The one thing gc
does that SoftHebb (as run) doesn't is hold up under continued training: gc's gains are stable-
to-rising while SoftHebb's peak-then-decay. By 4 epochs gc edges online SoftHebb on kNN/ridge,
still trails it on logistic, batched SoftHebb leads throughout. The result: "gc's gains don't
collapse under continual training," not "gc beats SoftHebb."

Caveat, not yet a fair continual test: SoftHebb's per-layer temperature/power/LR schedule
(tuned for 1 epoch) was not ported, so its 4-epoch decay may be un-annealed overtraining rather
than a real method difference. Needs a scheduled/annealed SoftHebb baseline + more seeds before
"gc stays stable where SoftHebb decays" is earned. Single config, ~1 seed.

----- reference below (how it works; skip unless you care) -----

Rule: oja-signed = SoftHebb's update (Moraitis et al., arXiv:2209.11883): per-location softmax
over channels, winner moves toward the input, losers move away (anti-Hebbian repulsion makes
channels tile instead of collapsing onto one prototype). gc's own rules (BCM, plain instar/oja)
all fail here; the signed gate is required.

Four ingredients, each necessary: (1) Triangle activation relu(u-mean_c)^0.7, a graded code;
hard-WTA collapses over depth. (2) Soft weight-norm: hard projection makes learning
destructive. (3) Online BatchNorm: the homeostatic regularizer; without it kNN is a transient
peak that collapses (52->36). (4) No whitening: BN handles it; even online ZCA suppresses the
kNN gain. (Per-neuron adaptive-LR also tested; hurts kNN.)

Architecture (identical to SoftHebb, ~5.65M params): three conv 96->384->1536, kernels 5/3/3,
stride-1+pad; spatial 32->16->8->4 via MaxPool(4,s2) after layers 1-2, AvgPool(2) after 3;
affine-free BN before each conv; final 4x4 map read whole -> 24576-dim rep (= SoftHebb's
flatten, dim-matched). The conv stack (weights, topology) is identical to SoftHebb; gc differs
by not porting SoftHebb's per-layer schedule: the temperature/power shaping each layer's
activation (forward path: SoftHebb 0.7/1.4/1.0, gc a fixed 0.7, so even the frozen baselines
differ ~3-4pt) and the LR (training: gc fixed), plus minor padding/init/BN-momentum/seed.
Recipe in src.agents.CIFARAgt; this script is the harness. float32 (Oja products overflow fp16).

Absolute numbers (kNN/ridge/logistic %; gains above are vs each frozen row):
    gc:       frozen 44.3/57.1/57.7        50k 51.2/60.6/59.6        200k 49.5/64.2/61.9
    SoftHebb: frozen 43.2/53.8/53.4    b1  50k 50.9/63.4/61.8    b1  200k 47.7/59.6/58.9
                                       b10 50k 53.9/65.9/62.9    b10 200k 50.2/61.7/58.8

    SoftHebb native readout 79.9 logistic (50k-sample + dropout + 50-epoch linear, a much
    harder probe, not comparable). Frozen rows differ ~3-4pt from those forward-path differences
    (above), not from architecture.
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before torch)

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents import CIFARAgt
from src.envs import CIFARDataset

SEED = 0
N_STEPS = 50000        # online single-sample steps (~1 epoch)
EVAL_INTERVAL = 10000  # report learn-vs-frozen progress
WARMUP = 1000          # populate BN stats before learning
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
    gram = a.T @ a
    gram.diagonal().add_(lam)  # ridge penalty in-place (avoids a 24576^2 eye -> OOM at this rep dim)
    weights = torch.linalg.solve(gram, a.T @ targets)
    return ((b @ weights).argmax(1) == yte).float().mean().item() * 100


def logistic(rtr, ytr, rte, yte, steps=400, lr=0.05):
    torch.manual_seed(SEED)  # deterministic classifier init (the only RNG in eval)
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
    torch.use_deterministic_algorithms(True)  # fully reproducible run-to-run
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
    torch.manual_seed(SEED)               # same init as the control
    control_agt = CIFARAgt()              # never gets use_lrn=True -> stays frozen
    for agt in (learn_agt, control_agt):  # warm up BN stats before learning
        for i in warm:
            agt.step([train_imgs[i]], use_lrn=False, training=True)
    control = evaluate(control_agt, etr, ytr, ete, yte)  # frozen baseline (constant)
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
