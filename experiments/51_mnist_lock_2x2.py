"""
Example 16's lock as a 2x2 on its own frozen network: activation (tri0.3 or
wta) by input (active: the output's argmax picks the digit, a fresh image of
it each step; passive: a random image of any digit each step). RW,
2026-09-26: "wait so did you say theat 16 vs 45 lock vs keep changing isn't
probalby due to the diffnet acitvaiotns (triangle vs wta)?", then "run it".

Premises: as experiment 50 (src BareAgt defaults, 200 bulk columns on the GPU,
keep 0, RMS norm on, 500 steps, last 200 read), with Dir.A learning off in
every cell (lrn_oja_gated made the identity). wta replaces the triangle on
what each bulk column sends (fc.atv_triangle patched to a one-hot at the
column's argmax times the weights); the input column still sends raw. The
tri0.3 / active cell repeats 50's learning-off arm.

Usage:
    venv/bin/python experiments/51_mnist_lock_2x2.py            # the 4 cells in parallel
    venv/bin/python experiments/51_mnist_lock_2x2.py wta passive
"""

import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the MNIST loader reads ./data

STEPS, WINDOW, SEED, N_COLS = 500, 200, 0, 200
LOG = os.path.join(ROOT, "experiments", "mnistlock51.log")


def arm(act, mode):
    import torch
    from src import funcs as fc
    from src import iotypes as T
    from src.agents import BareAgt, BareCfg
    from src.envs import MNISTDataset

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    fc.lrn_oja_gated = lambda x, w, gate, u, ss=1e-2: w
    if act == "wta":
        def _wta_out(x, w, power=0.3):
            o = torch.zeros_like(x)
            o[int(torch.argmax(x))] = 1.0
            return o @ w
        fc.atv_triangle = _wta_out
    random.seed(SEED)
    torch.manual_seed(SEED)

    mnist = MNISTDataset(train=True)
    targets = mnist.mnist.targets.cpu()
    by_digit = {d: (targets == d).nonzero().flatten().tolist() for d in range(10)}
    image = mnist[random.randrange(len(mnist))][0]

    cfg = BareCfg(N_COLS, [T.I_Vector(28 * 28)], [T.O_Vector(10)])
    agt = BareAgt(cfg, os.path.join(ROOT, "saves", "mnistlock51", f"{act}_{mode}"))
    bulk = [c for loc, c in agt.cols.items() if not agt.is_io(loc)]
    states, digits = [], []
    t0 = time.time()
    for step in range(STEPS):
        out, = agt.step([image], disable_print=True)
        if mode == "passive":
            k = random.randrange(len(mnist))
            image = mnist[k][0]
            digits.append(int(targets[k]))
        elif not torch.allclose(out, torch.zeros(10, device=out.device)):
            d = int(torch.argmax(out))
            image = mnist[random.choice(by_digit[d])][0]
            digits.append(d)
        else:
            digits.append(-1)
        if step >= STEPS - WINDOW:
            states.append(torch.stack([c.nr_1.actual.float().cpu() for c in bulk]))
    rate = STEPS / (time.time() - t0)
    X = torch.stack(states)                                   # steps x cols x 32

    def corr(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)

    c1 = float(corr(X[:-1], X[1:]).mean(0).median())
    c10 = float(corr(X[:-10], X[10:]).mean(0).median())
    sign_fixed = float((torch.sign(X) == torch.sign(X[:1])).all(0).float().mean())
    dswitch = sum(a != b for a, b in zip(digits[1:], digits[:-1])) / (len(digits) - 1)
    return (f"{act:6} {mode:7}: step-to-step pattern correlation {c1:.3f}, "
            f"10 steps apart {c10:.3f}; units whose sign never changed over the last {WINDOW} steps "
            f"{sign_fixed:.3f}; mean |state| {float(X.abs().mean()):.3f}; chosen digit changes on "
            f"{dswitch:.2f} of steps (last 20 digits: {digits[-20:]}); {rate:.1f} steps/s")


def main():
    if len(sys.argv) == 3:
        print(arm(sys.argv[1], sys.argv[2]))
        return
    cells = [(a, m) for a in ("tri0.3", "wta") for m in ("active", "passive")]
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as ex:
        lines = list(ex.map(arm, *zip(*cells)))
    with open(LOG, "a") as fh:
        for line in lines:
            print(line)
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
