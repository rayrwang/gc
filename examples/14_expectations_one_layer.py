"""
Expectations (Dir.E) on a one-hidden-layer feedforward substrate: a
one-step-ahead prediction exam with its own controls. One layer on purpose:
the guess is read from the first hidden layer's activity, and stacking more
layers leaves that path untouched (tested: depths 1-3 score identically), so
for this exam extra depth is decoration. Depth starts mattering when
prediction becomes per-layer, which is the recurrent substrate's shape, not
this one.

The agent runs Dir.A forward (SoftHebb: signed soft-WTA gated Oja, Triangle
activation, online norm) and Dir.E backward: a map from hidden activity at
t-1 to input at t, trained by the delta rule (Hebbian in form, error-valued:
pre x residual, normalized). At each step the guess is published before the
next input is visible, then scored by cosine against what actually arrived.

Three situations with known answers, three controls that must not pass:
    cycle     four orthogonal symbols repeating: structure exists, find it
    noise     fresh random vector each step: nothing to find, claim nothing
    sandwich  cycle -> noise -> same cycle: does noise exposure destroy
              what was learned (retention), and is relearning slower than
              learning was (burnout) or faster (savings)?

    copy-last null    predicts the previous input: scores 0 on the cycle by
                      construction (no consecutive repeats)
    frozen twin       same init, learning off: whatever untrained dynamics
                      score
    dir.e off         same substrate, forward learning on, backward map
                      frozen: shows the prediction lives in Dir.E, not Dir.A

Two reference mechanisms sit alongside for calibration: canonical temporal
predictive coding (settling inference + error-hebbian learning, both hinges
of the family on) and a temporal Helmholtz machine trained by wake-sleep
(one-pass recognition, no settling; recognition learns on its own dreams).

Results (seeds 0/1/2, mean trailing score; retention = post-noise cycle
score immediately on return, re-settle vs first-settle in steps):

    subject      cycle    noise    retention  resettle
    dir.e on     +1.00    -0.00    +0.78       64 vs 82 (faster: savings)
    pc           +1.00    -0.00    +0.25      262 vs 57 (5x slower: burnout)
    wake-sleep   +1.00    +0.01    +0.94       50 vs 53
    dir.e off    -0.02    +0.02    n/a        n/a
    frozen twin  +0.01    +0.02    n/a        n/a
    copy-last    -0.00    -0.00    n/a        n/a

The backward direction carries all of it: with Dir.E frozen the same
substrate predicts nothing. The delta rule reaches 1.00 where pure Hebb
plateaus near 0.8 (crosstalk: delta is Hebb plus unlearning of the already-
predicted part). Prediction alone does not separate the three learners: the
sandwich does. Canonical PC keeps settling-and-learning on the noise, drags
its maps, returns to its cycle at a quarter strength and relearns it five
times slower than it first did (burnout); wake-sleep's normalized steps
barely move; dir.e-on loses a fifth of its score but relearns faster than
it learned (savings). Noise-robustness, not prediction, is where the
mechanisms differ.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

import src.funcs as fc

DIM = 32
HIDDEN = 64
N_STEPS = 400          # cycle / noise length
SW = (400, 200, 400)   # sandwich: cycle, noise, cycle
SEEDS = (0, 1, 2)
WINDOW = 50            # trailing window for scores and settling


def derive_seed(*parts):
    # every role gets its own stream so a subject can never share a fixture's rng
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "big")


def symbols(n, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, DIM, generator=g).T)
    return q.T[:n].contiguous()


def cycle(steps, seed):
    syms = symbols(4, seed)
    for t in range(steps):
        yield syms[t % 4]


def noise(steps, seed):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        yield torch.randn(DIM, generator=g)


class Agt:
    """one hidden layer: forward W (dir.a, softhebb) + backward E (dir.e,
    delta); both local."""

    def __init__(self, seed, dire=True, frozen=False, lr_a=0.03, lr_e=0.1):
        g = torch.Generator().manual_seed(seed)
        self.W = 3.0 * torch.randn(DIM, HIDDEN, generator=g) / DIM ** 0.5
        self.E = 0.01 * torch.randn(HIDDEN, DIM, generator=g)  # (pre, target), fc convention
        self.h = None
        self.dire, self.frozen, self.lr_a, self.lr_e = dire, frozen, lr_a, lr_e

    def guess(self):
        return torch.zeros(DIM) if self.h is None else self.h @ self.E

    def observe(self, a):
        x = (a - a.mean()) / (a.var() + 1e-5).sqrt()  # online norm
        u = x @ self.W
        h = fc.triangle(u, 0.7)
        if not self.frozen:
            gate = fc.softmax_wta(u, 1.0, signed=True)
            self.W = fc.lrn_oja_gated(x, self.W, gate, u, ss=self.lr_a)
            if self.dire and self.h is not None:
                self.E = fc.lrn_delta(self.h, self.E, a, ss=self.lr_e)
        self.h = h


class PC:
    """canonical temporal predictive coding (Rao-Ballard dynamic form):
    settling inference + error-hebbian learning on both maps."""

    def __init__(self, seed, settle=40, lr_inf=0.1, lr_w=0.1):
        g = torch.Generator().manual_seed(seed)
        self.W_out = torch.eye(DIM) + 0.01 * torch.randn(DIM, DIM, generator=g)
        self.W_dyn = 0.01 * torch.randn(DIM, DIM, generator=g)
        self.x = torch.zeros(DIM)
        self.settle, self.lr_inf, self.lr_w = settle, lr_inf, lr_w

    def guess(self):
        return self.W_out @ (self.W_dyn @ self.x)

    def observe(self, a):
        x_pred = self.W_dyn @ self.x
        x = x_pred.clone()
        for _ in range(self.settle):  # inference: descend obs error + prior error
            x = x + self.lr_inf * (self.W_out.T @ (a - self.W_out @ x) - (x - x_pred))
        self.W_out = self.W_out + self.lr_w * torch.outer(a - self.W_out @ x, x)
        self.W_dyn = self.W_dyn + self.lr_w * torch.outer(x - x_pred, self.x)
        self.x = x


class WakeSleep:
    """temporal helmholtz machine: one-pass recognition (no settling);
    generative + dynamics train in wake, recognition trains on dreams."""

    def __init__(self, seed, lr=0.1, dream=0.1):
        self.gen = torch.Generator().manual_seed(seed)
        g = self.gen
        self.R = torch.eye(DIM) + 0.01 * torch.randn(DIM, DIM, generator=g)
        self.G = torch.eye(DIM) + 0.01 * torch.randn(DIM, DIM, generator=g)
        self.D = 0.01 * torch.randn(DIM, DIM, generator=g)
        self.x = torch.zeros(DIM)
        self.lr, self.dream = lr, dream

    def guess(self):
        return self.G @ (self.D @ self.x)

    def observe(self, a):
        x = torch.tanh(self.R @ a)
        self.G = self.G + self.lr * torch.outer(a - self.G @ x, x) / (1 + x @ x)
        x_pred = self.D @ self.x
        self.D = self.D + self.lr * torch.outer(x - x_pred, self.x) / (1 + self.x @ self.x)
        z = torch.tanh(x_pred + self.dream * torch.randn(DIM, generator=self.gen))
        fantasy = self.G @ z
        self.R = self.R + self.lr * torch.outer(z - torch.tanh(self.R @ fantasy), fantasy) / (1 + fantasy @ fantasy)
        self.x = x


class CopyLast:
    def __init__(self, seed):
        self.last = torch.zeros(DIM)

    def guess(self):
        return self.last

    def observe(self, a):
        self.last = a.clone()


def run(make, stream):
    agt, scores = make, []
    for a in stream:
        gs = agt.guess()
        d = gs.norm() * a.norm()
        scores.append(float(gs @ a / d) if d > 0 else 0.0)
        agt.observe(a)
    return scores


def trail(xs):
    return sum(xs[-WINDOW:]) / WINDOW


def settled(xs):
    # first step the trailing mean is within 10% of the final plateau and stays
    plateau = trail(xs)
    if abs(plateau) < 0.1:
        return None
    run_len = 0
    for i in range(WINDOW, len(xs)):
        if abs(sum(xs[i - WINDOW:i]) / WINDOW - plateau) <= max(0.1 * abs(plateau), 0.02):
            run_len += 1
            if run_len >= 100:
                return i - 99
        else:
            run_len = 0
    return None


if __name__ == "__main__":
    subjects = {
        "dir.e on": lambda s: Agt(s),
        "pc": lambda s: PC(s),
        "wake-sleep": lambda s: WakeSleep(s),
        "dir.e off": lambda s: Agt(s, dire=False),
        "frozen twin": lambda s: Agt(s, frozen=True),
        "copy-last": lambda s: CopyLast(s),
    }
    print(f"{'subject':<13} {'cycle':>7} {'noise':>7} {'retention':>10} {'resettle':>16}")
    for name, make in subjects.items():
        cyc, nz, ret, res = [], [], [], []
        for seed in SEEDS:
            sub = derive_seed("subject", name, seed)
            cyc.append(trail(run(make(sub), cycle(N_STEPS, seed))))
            nz.append(trail(run(make(sub), noise(N_STEPS, seed))))
            agt = make(sub)
            pre = run(agt, cycle(SW[0], seed))
            run(agt, noise(SW[1], derive_seed("sw-noise", seed)))
            post = run(agt, cycle(SW[2], seed))
            if trail(pre) > 0.5:  # retention only means something for learners
                ret.append(sum(post[:WINDOW]) / WINDOW)
                s_pre, s_post = settled(pre), settled(post)
                if s_pre and s_post:
                    res.append((s_post, s_pre))
        r = f"{sum(ret)/len(ret):+10.2f}" if ret else f"{'n/a':>10}"
        s = (f"{sum(a for a, _ in res)/len(res):>7.0f} vs {sum(b for _, b in res)/len(res):.0f}"
             if res else f"{'n/a':>16}")
        print(f"{name:<13} {sum(cyc)/3:+7.2f} {sum(nz)/3:+7.2f} {r} {s:>16}")
