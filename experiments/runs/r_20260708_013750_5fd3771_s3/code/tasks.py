"""Task streams for the harness — the shared data layer under every instrument.

C1 = permuted schedules over these tasks + an analysis; A1 = A->B->A cycles over
the same tasks. One data module, many instruments.

MNIST digit-splits: task A = digits 0-4, task B = digits 5-9 (the canonical
split-MNIST continual pair). Washout = noise images (no learnable structure).
Probe sets come from the MNIST TEST split (never streamed to arms as training),
fixed per seed, labels one-hot over all 10 classes (§7d fixed global recipe).

Data regime (§8): streams are built single-pass by default — a stream is a
permutation of distinct samples, each seen at most once per run. If a schedule
asks for more steps than the stream holds, that run is in the repeated regime
and MUST declare it at registration.
"""
import torch

from experiments.determinism import gen
from experiments.harness import Task


def _root():
    import sys, os
    r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if r not in sys.path:
        sys.path.insert(0, r)
    return r


def _load_mnist(train=True):
    _root()
    from src.envs import MNISTDataset
    ds = MNISTDataset(train=train)
    return ds.mnist.data.reshape(len(ds.mnist), -1).float() / 255.0, ds.mnist.targets


def _load_fashion(train=True):
    _root()
    from src.envs import FashionMNISTDataset
    ds = FashionMNISTDataset(train=train)
    return ds.mnist.data.reshape(len(ds.mnist), -1).float() / 255.0, ds.mnist.targets


def _load_cifar10(train=True):
    _root()
    import torchvision
    ds = torchvision.datasets.CIFAR10("data", train=train, download=False)
    X = torch.tensor(ds.data).reshape(len(ds.data), -1).float() / 255.0   # (N, 3072)
    return X, torch.tensor(ds.targets)


DATASETS = {  # name -> (loader, in_dim, n_classes); drop new ones in here
    "mnist":   (_load_mnist,   784,  10),
    "fashion": (_load_fashion, 784,  10),
    "cifar10": (_load_cifar10, 3072, 10),
}


def _onehot(y, n=10):
    out = torch.zeros(n)
    out[int(y)] = 1
    return out


def split_tasks(seed, dataset="mnist", classes_a=(0, 1, 2, 3, 4), classes_b=(5, 6, 7, 8, 9)):
    """Returns (taskA, taskB): shuffled single-pass streams of (x, onehot)."""
    loader, in_dim, n_classes = DATASETS[dataset]
    X, Y = loader(train=True)
    tasks = []
    for name, classes in (("A" + "".join(map(str, classes_a)), classes_a),
                          ("B" + "".join(map(str, classes_b)), classes_b)):
        mask = torch.isin(Y, torch.tensor(classes))
        Xc, Yc = X[mask], Y[mask]
        perm = torch.randperm(len(Xc), generator=gen(seed, f"stream_{name}"))
        samples = [(Xc[i], _onehot(Yc[i], n_classes)) for i in perm.tolist()]
        tasks.append(Task(f"{dataset}-{name}", samples))
    return tasks


def digit_split_tasks(seed, **kw):
    return split_tasks(seed, dataset="mnist", **kw)


def washout_task(seed, n=10000, in_dim=784):
    """Structureless noise stream (uniform [0,1] images, dummy labels): the
    post-training tail for persistence tests. Nothing here is learnable."""
    g = gen(seed, "washout")
    # y=None: a NEUTRAL tail. (zeros(10) would argmax to class 0 and actively
    # TEACH supervised arms that noise is class 0 — caught 2026-07-07 pre-run.)
    samples = [(torch.rand(in_dim, generator=g), None) for _ in range(n)]
    return Task("washout", samples)


def probe_sets(seed, dataset="mnist", n_train=500, n_test=500, classes=tuple(range(10))):
    """Fixed probe train/eval sets from the TEST split (never streamed to arms).
    Same sets for every arm and every checkpoint of a run (§7d)."""
    loader, in_dim, n_classes = DATASETS[dataset]
    X, Y = loader(train=False)
    mask = torch.isin(Y, torch.tensor(classes))
    Xc, Yc = X[mask], Y[mask]
    perm = torch.randperm(len(Xc), generator=gen(seed, "probe_sets"))
    idx = perm[: n_train + n_test].tolist()
    pts = [(Xc[i], _onehot(Yc[i], n_classes)) for i in idx]
    return pts[:n_train], pts[n_train:]


def selftest():
    a, b = digit_split_tasks(seed=0)
    assert len(a.samples) > 25000 and len(b.samples) > 25000
    xs, ys = a.samples[0]
    assert xs.shape == (784,) and ys.shape == (10,) and 0 <= xs.max() <= 1
    labels_a = {int(y.argmax()) for _, y in a.samples[:200]}
    labels_b = {int(y.argmax()) for _, y in b.samples[:200]}
    assert labels_a <= {0, 1, 2, 3, 4} and labels_b <= {5, 6, 7, 8, 9}
    w = washout_task(seed=0, n=100)
    assert all(y is None for _, y in w.samples)
    ptr, pte = probe_sets(seed=0, n_train=50, n_test=50)
    tr_ids = {id(x) for x, _ in ptr}
    assert not any(id(x) in tr_ids for x, _ in pte)
    a2, _ = digit_split_tasks(seed=0)
    assert torch.equal(a.samples[0][0], a2.samples[0][0])   # deterministic streams
    fa, fb = split_tasks(seed=0, dataset="fashion")
    assert fa.samples[0][0].shape == (784,) and fa.name.startswith("fashion-")
    print(f"selftest passed: A={len(a.samples)} B={len(b.samples)} samples, "
          f"probe {len(ptr)}/{len(pte)}, washout structureless, streams deterministic.")


if __name__ == "__main__":
    selftest()
