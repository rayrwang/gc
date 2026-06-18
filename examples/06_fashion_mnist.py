
"""
Fashion-MNIST version of 05: same MNISTAgt (width-128 post-ReLU hidden, raw input,
BCM), same controlled protocol. Drop-in -- 28x28 grayscale, 10 classes -- so only the
dataset changes. A harder MNIST (~5-10 pts lower) that, unlike MNIST, fills the frame.

Result: at the committed online rate it behaves like MNIST -- BCM weakly beats the
frozen control (logistic +2.4, kNN +0.8) but stays below the linear-on-pixels floor.
Whitening does NOT help here (raw/std/zca 3-seed sweep, _sweep.py): per-feature
z-score is fatal (dead pixels -> inf), and ZCA only "rescued" Fashion when an over-hot
ss=0.01 made raw BCM collapse -- a learning-rate artifact, not a real need. So the
whitening rescue is CIFAR/conv-specific (CIFAR degrades even at the gentle ss);
grayscale FC datasets don't need it.

  tensorboard --logdir runs/fashion_mnist
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import random

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# isort: off
from src.agents import MNISTCfg, MNISTAgt
from src.envs import MNISTEnv, MNISTEnvCfg, FashionMNISTDataset

SEED = 0
N_STEPS = 5000          # online learning steps
EVAL_INTERVAL = 500
N_TRAIN_EVAL = 4000     # rep subset for fitting the probes
N_TEST_EVAL = 1500
K_NN = 20


def get_representations(agt: MNISTAgt) -> torch.Tensor:
    return agt.cols[1, 0].nr_1[0].clone()  # the single post-ReLU hidden layer


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
    """Read the post-ReLU hidden layer for each image (learning off)."""
    reps = []
    for img in imgs:
        agt.step([img], use_lrn=False, disable_print=True)
        reps.append(get_representations(agt).float())
    return torch.stack(reps)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Data as bulk tensors
    train_ds, test_ds = FashionMNISTDataset(train=True), FashionMNISTDataset(train=False)
    train_imgs = (train_ds.mnist.data.reshape(-1, 784).half() / 255).to(device)
    train_labels = train_ds.mnist.targets.to(device)
    test_imgs = (test_ds.mnist.data.reshape(-1, 784).half() / 255).to(device)
    test_labels = test_ds.mnist.targets.to(device)
    eval_tr = torch.randperm(len(train_imgs))[:N_TRAIN_EVAL]
    eval_te = torch.arange(N_TEST_EVAL)
    ytr, yte = train_labels[eval_tr], test_labels[eval_te]

    # Learning agent and a frozen control sharing the SAME random init
    ispec, ospec = MNISTEnv.get_specs(MNISTEnvCfg("passive"))
    torch.manual_seed(SEED)
    learn_agt = MNISTAgt(MNISTCfg(ispec, ospec), f"{project_root_path}/saves/fashion_learn")
    torch.manual_seed(SEED)
    control_agt = MNISTAgt(MNISTCfg(ispec, ospec), f"{project_root_path}/saves/fashion_control")
    agents = {"learn": (learn_agt, True), "control": (control_agt, False)}

    # Probes on the raw image, once, as the image baseline (the floor)
    base_tr, base_te = standardize(
        train_imgs[eval_tr].float(), test_imgs[eval_te].float())
    baseline = {name: probe(base_tr, ytr, base_te, yte) for name, probe in PROBES.items()}

    writer = SummaryWriter("runs/fashion_mnist")
    print("\nImage baseline:  " + "   ".join(f"{n} {baseline[n]:.1f}%" for n in PROBES))
    print("tensorboard --logdir runs/fashion_mnist\n")

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
