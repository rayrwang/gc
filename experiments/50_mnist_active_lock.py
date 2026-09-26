"""
Why does example 16's network lock in place while run 39's wta cell keeps
drifting? Learning, or the activation? (RW, 2026-09-26: "though even with it
slowed down it sgradually changes wheile example 16 is lcoked inplace, that
might be due to the leanring ig or mabey somewhat due to the activation".)

Premises: example 16 itself, in one process.
  host        src BareAgt with its defaults (stock wiring 0.3/d^2 and 0.6/d^2
              from io columns, one conn per pair, weight scale 2.0), 200 bulk
              columns on the GPU, one 784-dim input column, one 10-dim output
              column
  activity    tri0.3 on what each column sends, keep 0 (the accumulator is
              zeroed each step), RMS norm on
  input       MNIST "active" as in src/envs.py: each step the output's argmax
              picks the digit and a fresh random training image of it is shown
              (an all-zero output keeps the image); example 16 does this through
              an env process and queues, here inline
  varied      Dir.A learning on (stock: lrn_oja_gated, softmax_wta beta 0.5
              signed, ss 1e-2) against off (lrn_oja_gated made the identity);
              Dir.E never learns in BareAgt; same seed, so both arms start
              from the same network and the same first image
  measures    per bulk column, the correlation of its 32 values between steps
              t and t+1 and t and t+10 (median over columns, over the last 200
              steps); the fraction of units whose sign never changes over the
              last 200 steps; mean |state|; how often the chosen digit changes;
              steps per second

Usage:
    venv/bin/python experiments/50_mnist_active_lock.py          # both arms in parallel
    venv/bin/python experiments/50_mnist_active_lock.py on       # one arm
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
LOG = os.path.join(ROOT, "experiments", "mnistlock50.log")


def arm(learning):
    import torch
    from src import funcs as fc
    from src import iotypes as T
    from src.agents import BareAgt, BareCfg
    from src.envs import MNISTDataset

    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    if not learning:
        fc.lrn_oja_gated = lambda x, w, gate, u, ss=1e-2: w
    random.seed(SEED)
    torch.manual_seed(SEED)

    mnist = MNISTDataset(train=True)
    targets = mnist.mnist.targets.cpu()
    by_digit = {d: (targets == d).nonzero().flatten().tolist() for d in range(10)}
    image = mnist[random.randrange(len(mnist))][0]

    cfg = BareCfg(N_COLS, [T.I_Vector(28 * 28)], [T.O_Vector(10)])
    agt = BareAgt(cfg, os.path.join(ROOT, "saves", "mnistlock50", "on" if learning else "off"))
    bulk = [c for loc, c in agt.cols.items() if not agt.is_io(loc)]
    states, digits = [], []
    t0 = time.time()
    for step in range(STEPS):
        out, = agt.step([image], disable_print=True)
        if not torch.allclose(out, torch.zeros(10, device=out.device)):
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
    return (f"learning {'on ' if learning else 'off'}: step-to-step pattern correlation {c1:.3f}, "
            f"10 steps apart {c10:.3f}; units whose sign never changed over the last {WINDOW} steps "
            f"{sign_fixed:.3f}; mean |state| {float(X.abs().mean()):.3f}; chosen digit changes on "
            f"{dswitch:.2f} of steps (last 20 digits: {digits[-20:]}); {rate:.1f} steps/s")


def main():
    if len(sys.argv) == 2:
        print(arm(sys.argv[1] == "on"))
        return
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as ex:
        lines = list(ex.map(arm, (False, True)))
    with open(LOG, "a") as fh:
        for line in lines:
            print(line)
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
