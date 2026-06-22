"""Regenerates wta_geometry.png (used in the README): unsigned vs signed soft-WTA, on
discrete clusters vs a dominant-mode blob. Data points (soft) + learned prototypes (bold).
Run:  python assets/wta_geometry.py   (needs matplotlib + numpy)."""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(0)
N_PROTO, STEPS, LR, TEMP = 26, 12000, 0.04, 0.25
DATA_C, PROTO_C, GOOD, BAD = "#9db8d6", "#e63b46", "#1f9d55", "#c1121f"


def discrete():
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    centers = 3.0 * np.c_[np.cos(ang), np.sin(ang)]
    return np.vstack([c + 0.42 * rng.standard_normal((500, 2)) for c in centers])


def continuous():
    # a dominant central mode + a sparse halo -- the regime where unsigned piles onto the
    # mode and under-covers the rest (like the smooth-patch mode of natural images)
    return np.vstack([0.45 * rng.standard_normal((2200, 2)), 2.6 * rng.standard_normal((400, 2))])


def learn(data, signed):
    """Competitive instar with a soft winner-take-all gate; signed negates the losers."""
    w = rng.uniform(-4, 4, (N_PROTO, 2))
    for _ in range(STEPS):
        x = data[rng.integers(len(data))]
        d2 = ((x - w) ** 2).sum(1)
        resp = np.exp(-(d2 - d2.min()) / TEMP)                      # soft assignment
        resp /= resp.sum()
        g = -resp if signed else resp.copy()
        if signed:
            g[d2.argmin()] *= -1                                     # winner +, losers - (anti-Hebbian)
        w += LR * g[:, None] * (x - w)                               # move toward (or away from) x
    return w


def panel(ax, data, signed, title, verdict, good):
    ax.scatter(data[:, 0], data[:, 1], s=9, c=DATA_C, alpha=0.40, linewidths=0, zorder=1)
    w = learn(data, signed)
    ax.scatter(w[:, 0], w[:, 1], s=135, c=PROTO_C, marker="o",
               edgecolors="white", linewidths=1.6, zorder=4)
    ax.set_title(title, fontsize=11.5, color="#23272e", fontweight="bold", pad=9)
    ax.set_xlabel(verdict, fontsize=11, labelpad=9, color=(GOOD if good else BAD), fontweight="bold")
    ax.set_xlim(-5.6, 5.6)
    ax.set_ylim(-5.6, 5.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfcfd")
    for s in ax.spines.values():
        s.set_color("#d6dce4")


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.6), facecolor="white")
    disc, cont = discrete(), continuous()
    panel(axes[0, 0], disc, False, "discrete clusters  ·  unsigned WTA",
          "prototypes land ON the clusters  (k-means)  ✓", True)
    panel(axes[0, 1], disc, True, "discrete clusters  ·  signed WTA",
          "repelled into the empty gaps  ✗", False)
    panel(axes[1, 0], cont, False, "continuous blob  ·  unsigned WTA",
          "collapses onto the dense mode  ✗", False)
    panel(axes[1, 1], cont, True, "continuous blob  ·  signed WTA",
          "tiles the whole distribution  ✓", True)
    fig.suptitle("Competition sign must match data geometry",
                 fontsize=15, fontweight="bold", color="#1a1d22", y=0.985)
    fig.text(0.5, 0.945, "red = learned prototype     unsigned converges ONTO structure  ·  signed spreads to COVER it",
             ha="center", fontsize=10.5, color="#5b6675")
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.2, w_pad=2.0)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wta_geometry.png")
    fig.savefig(out, dpi=140)
    print("saved", out)
