"""
Generates the BCM stability phase diagram (assets/bcm_stability.png).

BCM is potentiation-dominated and self-stabilizes only through its sliding threshold
theta = EMA(y^2). Whether the recurrent substrate stays bounded or runs away is set
jointly by the learning-rate (ss) and the threshold decay (ALPHA): too large an ss
outruns the threshold, and ALPHA has an interior optimum (too low -> theta freezes and
stops braking; too high -> the homeostatic loop over-corrects and oscillates). This
script maps that plane.

Self-contained on purpose: it defines a minimal vectorized recurrent BCM network
(BCMAgt) instead of importing the Col/Agt framework, so it is decoupled from upcoming
BareAgt changes. The whole (decay, ss) grid is then run as one batched sweep, so a fine
phase diagram regenerates in a couple of seconds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

# Stability-optimal threshold-EMA rate for this network (the widest stable band in the
# diagram, the star). The optimum is architecture-dependent: too low -> theta freezes
# and stops braking; too high -> the homeostatic loop over-corrects. ~0.05 here.
OPTIMAL_DECAY = 0.05
N_UNITS = 256
DRIVE = 2.0


class BCMAgt:
    """Minimal vectorized recurrent BCM network (one cell of the phase diagram).

    Synchronous (Jacobi) update into a buffer, spike-thresholded recurrent
    propagation, and local BCM plasticity

        dW_ij = ss * a_i * a_j * (a_j - theta_j),     theta = EMA(a^2, decay)

    on a random recurrent weight matrix. No objective, no backprop. It diverges
    when potentiation outruns the sliding threshold. `divergence_grid` below is the
    batched form of this same step, run over every (decay, ss) pair at once.
    """

    def __init__(self, n_units=N_UNITS, decay=OPTIMAL_DECAY, ss=3e-5, drive=DRIVE, seed=0):
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.W = 2.0 * torch.randn(n_units, n_units, generator=g, device=DEV) / n_units**0.5
        self.W.fill_diagonal_(0.0)  # no self-connections
        self.a = torch.randn(n_units, generator=g, device=DEV)
        self.theta = torch.ones(n_units, device=DEV)
        self.decay, self.ss, self.drive, self.gen = decay, ss, drive, g

    @staticmethod
    def _spike(x):
        return torch.where(x < 1.0, 0.0, 1.0)

    def step(self):
        a = self.a
        post = a * (a - self.theta)                       # BCM post-synaptic factor (current snapshot)
        self.W = self.W + self.ss * torch.outer(a, post)  # local plasticity
        u = self._spike(a) @ self.W                       # synchronous spike-bounded propagation
        u = u + self.drive * torch.rand(a.shape[0], generator=self.gen, device=DEV)  # sensory drive
        self.theta = (1 - self.decay) * self.theta + self.decay * u**2  # slide the threshold
        self.a = u  # commit

    def diverged(self):
        return (not torch.isfinite(self.a).all()) or (self.a.abs().max() > 1e4)


def divergence_grid(decays, sss, n_units=N_UNITS, steps=200, drive=DRIVE, seed=0):
    """Batched form of BCMAgt.step over every (decay, ss) cell at once: one big recurrent
    BCM net per cell, all stepped together as batched matmuls so the whole grid is a
    single GPU sweep. Returns array[len(decays), len(sss)] of the step each cell diverged
    (NaN where it stayed bounded through `steps`)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nd, ns = len(decays), len(sss)
    B = nd * ns
    dec = torch.tensor(decays, device=DEV).repeat_interleave(ns)[:, None]  # [B,1]
    ss = torch.tensor(sss, device=DEV).repeat(nd)[:, None]                 # [B,1]
    W = 2.0 * torch.randn(B, n_units, n_units, generator=g, device=DEV) / n_units**0.5
    W *= ~torch.eye(n_units, dtype=torch.bool, device=DEV)  # zero diagonals
    a = torch.randn(B, n_units, generator=g, device=DEV)
    theta = torch.ones(B, n_units, device=DEV)
    div = torch.full((B,), -1, dtype=torch.long, device=DEV)  # -1 = still stable
    for t in range(1, steps + 1):
        post = a * (a - theta)                                          # [B,n]
        W = torch.baddbmm(W, (ss * a).unsqueeze(2), post.unsqueeze(1))  # W += ss * outer(a, post), per cell
        u = torch.bmm(torch.where(a < 1.0, 0.0, 1.0).unsqueeze(1), W).squeeze(1)  # spike-bounded propagation
        u = u + drive * torch.rand(B, n_units, generator=g, device=DEV)
        theta = (1 - dec) * theta + dec * u**2
        a = u
        bad = (~torch.isfinite(a).all(1)) | (a.abs().amax(1) > 1e4)  # [B]
        div = torch.where(bad & (div < 0), torch.full_like(div, t), div)
    Z = div.float().cpu().numpy()
    Z[Z < 0] = np.nan
    return Z.reshape(nd, ns)


def _edges_log(v):
    v = np.asarray(v, float)
    m = np.sqrt(v[:-1] * v[1:])
    return np.r_[v[0] ** 2 / m[0], m, v[-1] ** 2 / m[-1]]


def main():
    DECAYS = np.geomspace(3e-3, 0.95, 64)
    SS = np.geomspace(1e-5, 3e-2, 64)
    Z = divergence_grid(DECAYS.tolist(), SS.tolist())
    n_stable = int(np.isnan(Z).sum())
    print(f"swept {Z.size} cells ({len(DECAYS)} decays x {len(SS)} ss); {n_stable} stable", flush=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.YlOrRd_r.copy()
    cmap.set_bad("#2ca6a4")  # stable cells
    norm = mcolors.LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    pcm = ax.pcolormesh(_edges_log(SS), _edges_log(DECAYS), np.ma.masked_invalid(Z),
                        cmap=cmap, norm=norm, shading="flat")
    ax.scatter(3e-5, OPTIMAL_DECAY, s=300, marker="*", facecolor="cyan", edgecolor="black",
               zorder=6, label=f"BCMAgt default (decay={OPTIMAL_DECAY}, ss=3e-5)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate  ss")
    ax.set_ylabel("threshold decay  ALPHA")
    ax.set_title("BCM stability phase diagram\nteal = stable;  warm gradient = divergence speed (step at blow-up)")
    fig.colorbar(pcm).set_label("step at divergence  (smaller = faster runaway)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "bcm_stability.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
