"""
Generates the BCM stability phase diagram (assets/bcm_stability.png).

BCM is potentiation-dominated and self-stabilizes only through its sliding threshold
theta = EMA(y^2). Whether the recurrent substrate stays bounded or runs away is set
JOINTLY by the learning-rate (ss) and the threshold decay (ALPHA): too large an ss
outruns the threshold, and ALPHA has an interior optimum (too low -> theta freezes and
stops braking; too high -> the homeostatic loop over-corrects and oscillates). This
script maps that plane.

Self-contained on purpose: it defines a minimal vectorized recurrent BCM network
(BCMAgt) instead of importing the Col/Agt framework, so it is decoupled from upcoming
BareAgt changes. BCMAgt defaults to the stability-optimal decay.
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


class BCMAgt:
    """Minimal vectorized recurrent BCM network.

    Captures the substrate's stability behaviour without the agent framework:
    synchronous (Jacobi) update into a buffer, spike-thresholded recurrent
    propagation, and local BCM plasticity

        dW_ij = ss * a_i * a_j * (a_j - theta_j),     theta = EMA(a^2, decay)

    on a random recurrent weight matrix. No objective, no backprop. It diverges
    when potentiation outruns the sliding threshold.
    """

    def __init__(self, n_units=256, decay=OPTIMAL_DECAY, ss=3e-5, drive=2.0, seed=0):
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
        self.a = u                                        # commit

    def diverged(self):
        return (not torch.isfinite(self.a).all()) or (self.a.abs().max() > 1e4)


def divergence_step(decay, ss, n_units=256, steps=200):
    """Run until the network blows up; return the step it diverged, or None if stable."""
    net = BCMAgt(n_units=n_units, decay=decay, ss=ss)
    for t in range(1, steps + 1):
        net.step()
        if net.diverged():
            return t
    return None


def main():
    DECAYS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]
    SS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    Z = np.full((len(DECAYS), len(SS)), np.nan)  # NaN = stable
    for i, dec in enumerate(DECAYS):
        for j, ss in enumerate(SS):
            d = divergence_step(dec, ss)
            Z[i, j] = np.nan if d is None else d
            print(f"decay={dec:<6} ss={ss:.0e} -> {'stable' if d is None else 'div@' + str(d)}", flush=True)

    def _edges_log(v):
        v = np.asarray(v, float)
        m = np.sqrt(v[:-1] * v[1:])
        return np.r_[v[0] ** 2 / m[0], m, v[-1] ** 2 / m[-1]]

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.YlOrRd_r.copy()
    cmap.set_bad("#2ca6a4")  # stable cells
    norm = mcolors.LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    pcm = ax.pcolormesh(_edges_log(SS), _edges_log(DECAYS), np.ma.masked_invalid(Z),
                        cmap=cmap, norm=norm, shading="flat", edgecolors="white", lw=0.4)
    for i, dec in enumerate(DECAYS):
        for j, ss in enumerate(SS):
            stable = np.isnan(Z[i, j])
            ax.scatter(ss, dec, s=40, zorder=5, linewidths=1.3,
                       marker=("o" if stable else "x"),
                       c=("#0b3d3b" if stable else "black"))
    ax.scatter(3e-5, OPTIMAL_DECAY, s=280, marker="*", facecolor="cyan", edgecolor="black",
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
