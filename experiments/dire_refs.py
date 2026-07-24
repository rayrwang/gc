"""
Reference arms for the Dir.E exam (RW-decided 2026-07-20): known-positive
comparator subjects from the predictive-coding family, one per corner of the
settling x error-coupling grid. Characterization comparators, never the gate.

PCReference: canonical temporal predictive coding (Rao-Ballard dynamic form):
    iterative settling inference + error-Hebbian learning. Both hinges on.
WakeSleepReference: temporal Helmholtz machine: one-pass recognition (no
    settling), generative/dynamics trained in wake, recognition trained in
    sleep on its own dreams. Coupled learning, no settling.

Both are local-rule implementations (every update an outer product of locally
available quantities; no autograd, no backprop). Standard recipes, maturity
disclosed as such. Free-play tuning of their constants is unrestricted until
a registered event declares them.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dire_exam import DirEInterface, SYMBOL_DIM


class PCReference(DirEInterface):
    """temporal predictive coding: latent x, observation map W_out, dynamics
    W_dyn. observe() settles x against the incoming actual (inference =
    error minimization), then updates both maps Hebbian-on-error."""

    def __init__(self, dim=SYMBOL_DIM, seed=0, latent=None, settle_steps=40,
                 lr_inf=0.1, lr_w=0.1):
        latent = latent or dim
        g = torch.Generator().manual_seed(seed)
        self.W_out = torch.eye(dim, latent) + 0.01 * torch.randn(dim, latent, generator=g)
        self.W_dyn = 0.01 * torch.randn(latent, latent, generator=g)
        self.x_prev = torch.zeros(latent)
        self.settle_steps, self.lr_inf, self.lr_w = settle_steps, lr_inf, lr_w

    def guess(self):
        return self.W_out @ (self.W_dyn @ self.x_prev)

    def observe(self, a):
        x_pred = self.W_dyn @ self.x_prev
        x = x_pred.clone()
        for _ in range(self.settle_steps):  # settle: descend obs error + prior error
            e_obs = a - self.W_out @ x
            x = x + self.lr_inf * (self.W_out.T @ e_obs - (x - x_pred))
        e_obs = a - self.W_out @ x
        self.W_out = self.W_out + self.lr_w * torch.outer(e_obs, x)
        self.W_dyn = self.W_dyn + self.lr_w * torch.outer(x - x_pred, self.x_prev)
        self.x_prev = x


class WakeSleepReference(DirEInterface):
    """temporal helmholtz machine. wake: one recognition pass, train generative
    G on (latent -> actual) and dynamics D on (prev latent -> latent). sleep:
    dream a latent, generate a fantasy, train recognition R to invert it."""

    def __init__(self, dim=SYMBOL_DIM, seed=0, latent=None, lr=0.1, dream_noise=0.1):
        latent = latent or dim
        self.gen = torch.Generator().manual_seed(seed)
        self.R = torch.eye(latent, dim) + 0.01 * torch.randn(latent, dim, generator=self.gen)
        self.G = torch.eye(dim, latent) + 0.01 * torch.randn(dim, latent, generator=self.gen)
        self.D = 0.01 * torch.randn(latent, latent, generator=self.gen)
        self.x_prev = torch.zeros(latent)
        self.lr, self.dream_noise = lr, dream_noise

    def _rec(self, a):
        return torch.tanh(self.R @ a)

    def guess(self):
        return self.G @ (self.D @ self.x_prev)

    def observe(self, a):
        # normalized-lms steps: update / (1 + ||input||^2) keeps the delta
        # rules stable on large-norm streams (noise fixtures)
        x = self._rec(a)  # wake: single pass, no settling
        self.G = self.G + self.lr * torch.outer(a - self.G @ x, x) / (1 + x @ x)
        x_pred = self.D @ self.x_prev
        self.D = self.D + self.lr * torch.outer(x - x_pred, self.x_prev) / (1 + self.x_prev @ self.x_prev)
        z = torch.tanh(x_pred + self.dream_noise * torch.randn(len(x), generator=self.gen))
        fantasy = self.G @ z  # sleep: recognition learns to invert the generator
        self.R = self.R + self.lr * torch.outer(z - self._rec(fantasy), fantasy) / (1 + fantasy @ fantasy)
        self.x_prev = x


REFS = {
    "ref:pc": lambda dim, seed: PCReference(dim, seed),
    "ref:wake_sleep": lambda dim, seed: WakeSleepReference(dim, seed),
}


if __name__ == "__main__":
    # free-play smoke: trailing means on the cycle (should be high) and on
    # noise (should be ~0); nulls sit at 0/chance for comparison
    from experiments import dire_exam as de

    for name, make in REFS.items():
        for fixture, stream_fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
            means = []
            for seed in (0, 1, 2):
                records = []
                de.run_stream(make(de.SYMBOL_DIM, de.derive_seed("subject", name, seed)),
                              stream_fn(400, seed=seed), records.append,
                              fixture=fixture, subject=name, seed=seed)
                xs = [r["score"] for r in records if r["score"] is not None]
                means.append(sum(xs[-50:]) / 50)
            print(f"{fixture} {name:<16} trailing mean per seed: "
                  + str([f"{m:+.3f}" for m in means]))
