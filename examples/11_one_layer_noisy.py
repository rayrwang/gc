"""
One-layer local-learning sweep on NOISY (overlapping) abstract clusters.

Identical sweep to 10_one_layer_clean but at LOW SNR -- the clusters heavily overlap, so a raw
random projection is near chance and the task is genuinely hard. Kept deliberately parallel to
10; sync changes across both. See 10 for the full method/finding writeup; this file documents
what changes when the signal gets buried.

DATA: same K=10 Gaussian clusters in a 16-dim signal subspace + isotropic noise over 256 dims,
but cluster radius R=2 (vs 4) -- centroids close together relative to the noise, so clusters
overlap. raw kNN only ~32 (chance 10), oracle (signal-subspace) ~59: lots of headroom, low SNR.

FINDINGS (chance 10%; raw 32.3 | frozen 26.5 | LEARNED-best 50.5 | oracle 58.7):
  - Same qualitative ranking as the clean case: softmax-WTA + UNSIGNED still wins; none/inhibit
    diverge; signed and bcm trail. The geometry verdict (discrete clusters -> unsigned) is
    SNR-independent -- noise changes the difficulty, not which sign is right.
  - The ONE place homeostasis helps the champion: at low SNR, per-activation |w|-clocked
    adaptive-LR edges ahead. Best cell = instar + softmax-WTA + UNSIGNED + adaptive-LR = 50.5
    (vs 47.7 without it), and instar overtakes oja here. The per-unit annealing buys robustness
    when each update is dominated by noise. (In the clean case 10, plain no-homeostasis wins.)
  - online-BN still HURTS (it equalizes the 16 signal dims with the 240 noise dims).

SIGNIFICANCE: the winning recipe (WTA + unsigned for clustered data) is robust across SNR; the
homeostasis knobs only earn their keep when the signal is weak -- adaptive-LR as a noise
stabilizer, BN as a liability. float32, 1 seed.
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before torch)

import torch
import torch.nn.functional as F

SEED = 0
D, K, WIDTH = 256, 10, 512        # input dim, n clusters, layer width (rep dim)
D_SIG = 16                        # cluster signal lives in this many dims; rest is noise
R = 2                             # cluster radius = SNR. R=2 = NOISY (overlapping)
N_TR, N_EVTR, N_EVTE = 30000, 3000, 1000
KNN, TEMP, INHIB, POWER = 20, 1.0, 5.0, 0.7
LR = {"oja": 0.03, "instar": 0.03, "bcm": 0.003}
HOMEO = [("none", False, False), ("bn", True, False), ("adaptLR", False, True), ("both", True, True)]


def knn(rtr, ytr, rte, yte):
    sim = F.normalize(rte, dim=1) @ F.normalize(rtr, dim=1).T
    return (torch.mode(ytr[sim.topk(KNN, dim=1).indices], dim=1).values == yte).float().mean().item() * 100


def standardize(rtr, rte):
    mu, sd = rtr.mean(0), rtr.std(0) + 1e-6
    return (rtr - mu) / sd, (rte - mu) / sd


def triangle(u):
    """Graded readout: relu(u - mean_units(u)) ** power -- the fixed probe code for every cell."""
    return F.relu(u - u.mean(1, keepdim=True)) ** POWER


def probe(rep_fn):  # kNN on a representation function rep_fn(X)
    rtr, rte = standardize(rep_fn(Xev), rep_fn(Xte))
    return knn(rtr, ytr_e, rte, yte_e)


class OneLayer:
    """A single dense layer u = xW, trained online by one local rule, with a configurable
    competition gate, sign, and homeostasis. Readout is always the Triangle of u."""
    def __init__(self, rule, comp, sign, bn, adaptlr, gen):
        self.W = 3.0 * torch.randn(D, WIDTH, generator=gen) / D ** 0.5
        self.rule, self.comp, self.sign, self.lr = rule, comp, sign, LR[rule]
        self.bn, self.adaptlr = bn, adaptlr
        self.theta = torch.ones(1, WIDTH)                    # BCM sliding threshold
        self.rm, self.rs, self.mom = torch.zeros(1, D), torch.ones(1, D), 0.01  # streaming input-BN stats

    def _bn(self, x, update):
        if not self.bn:
            return x
        if update:
            self.rm = (1 - self.mom) * self.rm + self.mom * x.mean(0, keepdim=True)
            self.rs = (1 - self.mom) * self.rs + self.mom * (x ** 2).mean(0, keepdim=True)
        return (x - self.rm) / ((self.rs - self.rm ** 2).clamp_min(0) + 1e-5).sqrt()

    def _gate(self, u):
        """Competition: turn pre-activations u into the per-unit learning credit g."""
        if self.comp == "wta":
            resp = torch.softmax(TEMP * u, dim=1)            # soft winner-take-all responsibilities
            if self.sign == "unsigned":
                return resp                                  # all positive: winner most, losers a little
            g = -resp                                        # signed: losers - (anti-Hebbian)...
            g[0, u.argmax(1)] *= -1                           # ...winner +
            return g
        if self.comp == "none":
            base = u
        elif self.comp == "triangle":
            base = u - u.mean(1, keepdim=True)
        elif self.comp == "inhibit":
            base = u - (INHIB / (WIDTH - 1)) * (u.sum(1, keepdim=True) - u)   # subtractive lateral inhibition
        p = POWER if self.comp == "triangle" else 1.0
        if self.sign == "unsigned":
            return F.relu(base) ** p
        return torch.sign(base) * base.abs() ** p            # signed: keep below-threshold negative

    def learn(self, x):
        x = self._bn(x.reshape(1, -1), update=True)
        u = x @ self.W
        g = self._gate(u)
        if self.rule == "instar":
            dW = g * (x.T - self.W)                           # g_j*(x_i - W_ij): move toward x
        elif self.rule == "oja":
            dW = x.T * g - self.W * (g * u)                   # g_j*(x_i - u_j*W_ij): Oja decay bounds |w|
        else:  # bcm
            self.theta = 0.8 * self.theta + 0.2 * g ** 2
            dW = x.T * (g * (g - self.theta))                 # x_i*g_j*(g_j - theta_j)
        if self.adaptlr:
            wn = self.W.norm(dim=0, keepdim=True)             # per-activation |w_i|
            dW = dW * (wn - 1.0).abs().clamp_min(1e-3) ** 0.5  # |w|-clocked per-unit learning rate
        self.W = self.W + self.lr * dW

    def rep(self, X):
        return triangle(self._bn(X, update=False) @ self.W)


def make_clusters(centroids, n_per, gen):
    labels = torch.arange(K).repeat_interleave(n_per)
    return centroids[labels] + torch.randn(len(labels), D, generator=gen), labels


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    gen = torch.Generator(device=device).manual_seed(SEED)
    centroids = torch.zeros(K, D)
    centroids[:, :D_SIG] = torch.randn(K, D_SIG, generator=gen)
    centroids = centroids / centroids.norm(dim=1, keepdim=True) * R     # radius R in the signal subspace
    Xev, ytr_e = make_clusters(centroids, N_EVTR // K, gen)             # eval-train gallery
    Xte, yte_e = make_clusters(centroids, N_EVTE // K, gen)             # eval-test
    Xtr, _ = make_clusters(centroids, N_TR // K, gen)                   # online training stream
    order = torch.randperm(len(Xtr), generator=gen)

    raw = probe(lambda X: X)                                            # do-nothing baseline
    oracle = probe(lambda X: X[:, :D_SIG])                              # signal subspace = headroom ceiling
    frozen = probe(OneLayer("oja", "none", "unsigned", False, False,
                            torch.Generator(device=device).manual_seed(1)).rep)  # random projection
    print(f"R={R} (noisy)  raw {raw:.1f} | oracle {oracle:.1f} | frozen {frozen:.1f}   [kNN %, chance 10]\n")

    results = []
    for hname, bn, adaptlr in HOMEO:
        for rule in ("oja", "instar", "bcm"):
            for comp in ("none", "triangle", "inhibit", "wta"):
                for sign in ("unsigned", "signed"):
                    agt = OneLayer(rule, comp, sign, bn, adaptlr,
                                   torch.Generator(device=device).manual_seed(1))
                    for s in order:
                        agt.learn(Xtr[s])
                    val = probe(agt.rep) if torch.isfinite(agt.rep(Xev)).all() else float("nan")
                    results.append((val, rule, comp, sign, hname))

    print(f"{'rank':<5}{'kNN':>6}  {'rule':<7}{'competition':<12}{'sign':<10}{'homeostasis'}")
    finite = sorted([r for r in results if r[0] == r[0]], key=lambda r: -r[0])
    for i, (v, rule, comp, sign, h) in enumerate(finite, 1):
        print(f"{i:<5}{v:>6.1f}  {rule:<7}{comp:<12}{sign:<10}{h}")
    print(f"\n(refs: oracle {oracle:.1f}, frozen {frozen:.1f}, raw {raw:.1f};  "
          f"{len(results) - len(finite)} cells diverged to nan)")
