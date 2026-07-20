"""
Expectations (Dir.E) through a deep feedforward stack of real substrate
parts: I_VectorCol input, BareCol hidden layers, framework conns. Dir.A runs
forward with the substrate's own rule (signed soft-WTA gated Oja); Dir.E
conns run backward, layer l+1 predicting layer l's next activity, layer 1
predicting the next input. The substrate transports expectations but leaves
their learning rule TODO; this example supplies the delta rule and reads the
result per column.

Two situations, one subject and its frozen twin, scored per column against a
persistence baseline (cosine of a column's activity with its own next
activity: what a copy-last predictor would score there):

Results (seeds 0/1/2, two hidden layers of 64, mean trailing score):

    cycle                 predict  persist      noise    predict  persist
    input   (0,0)          +1.00    -0.00                 +0.00    -0.00
    hidden  (1,0)          +1.00    +0.03                 -0.01    +0.01
    top     (2,0)          (none)   +0.47                 (none)   +0.50
    (frozen twin: ~0 everywhere except +0.16 on the cycle's input column:
     untrained output aligning with structured input, not learning)

Every predicted column of every depth learns the cycle to 1.00 (the delta
rule's credit assignment is layer-local, so depth costs nothing). The top
column has no predictor (no Dir.E conn targets it) and shows the second
finding: its activity is persistently self-similar at ~0.5 even on iid
noise. Two rounds of triangle-plus-norm dynamics turn white input into
activity dominated by its own stationary statistics: iid at the input is
not iid at depth, so honesty at depth means predicting no better than
persistence, not predicting nothing.
"""

import hashlib
import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import torch

import src.funcs as fc
from src.agents import (AgtBase, BareCol, BareColCfg, Dir, I_VectorCol,
                        I_VectorColCfg, conn)

DIM = 32
DIMS = (64, 64)        # hidden widths
N_STEPS = 400
SEEDS = (0, 1, 2)
WINDOW = 50


def derive_seed(*parts):
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


class DeepAgt(AgtBase):
    """input col (0,0) -> hidden BareCols (1,0), (2,0): dir.a conns forward,
    dir.e conns backward. same-tick sequential forward (matched learning);
    expectations emitted at t are scored against t+1."""

    def __init__(self, path, seed, lr_e=0.1, frozen=False):
        super().__init__(None, path)
        self.age = 0
        self.lr_e, self.frozen = lr_e, frozen
        torch.manual_seed(seed)  # conn()/activs() draw from the global rng
        col = I_VectorCol((0, 0), I_VectorColCfg(DIM))
        self.cols[(0, 0)] = col
        self.I_cols.append(col)
        for li, w in enumerate(DIMS):
            self.cols[(li + 1, 0)] = BareCol((li + 1, 0), BareColCfg(w))
        for li in range(len(DIMS)):
            lo, hi = self.cols[(li, 0)], self.cols[(li + 1, 0)]
            lo.conns[(hi.loc, Dir.A)] = conn(lo, hi, Dir.A, 3.0)
            hi.conns[(lo.loc, Dir.E)] = conn(hi, lo, Dir.E, 0.01)
        self.predicted = {loc for c in self.cols.values()
                          for (loc, d) in c.conns if d == Dir.E}
        self.prev = {loc: torch.zeros(c.d) for loc, c in self.cols.items()}
        self.create_directory()

    def cleanup(self):
        pass  # demo agent: never save on exit

    def step(self, ipt, use_lrn=True):
        use_lrn = use_lrn and not self.frozen
        self.cols[(0, 0)].ipt(ipt[0])
        self.cols[(0, 0)].update_activations()
        for li in range(len(DIMS)):  # forward, input sender raw, hidden triangled
            lo, hi = self.cols[(li, 0)], self.cols[(li + 1, 0)]
            w = lo.conns[(hi.loc, Dir.A)]
            hi.nr_1_.actual = lo.a_pre @ w if li == 0 else fc.atv_triangle(lo.a_pre, w, power=0.3)
            hi.update_activations()
        for li in range(len(DIMS)):  # dir.e: delta on last step's expectation error
            hi, lo = self.cols[(li + 1, 0)], self.cols[(li, 0)]
            if use_lrn:
                # lrn_delta recomputes the residual internally: pre @ w here
                # still equals last step's emitted expectation (same sender
                # activity, weights untouched since emission)
                hi.conns[(lo.loc, Dir.E)] = fc.lrn_delta(
                    self.prev[hi.loc], hi.conns[(lo.loc, Dir.E)],
                    lo.nr_1.actual, ss=self.lr_e)
            if use_lrn:  # dir.a: the substrate's own rule
                w = lo.conns[(hi.loc, Dir.A)]
                lo.conns[(hi.loc, Dir.A)] = fc.lrn_oja_gated(
                    lo.nr_1.actual, w,
                    fc.softmax_wta(hi.nr_1.actual, beta=0.5, signed=True),
                    hi.nr_1.actual, ss=1e-2)
        for li in range(len(DIMS)):  # emit expectations for t+1
            hi, lo = self.cols[(li + 1, 0)], self.cols[(li, 0)]
            lo.nr_1.expect = hi.nr_1.actual @ hi.conns[(lo.loc, Dir.E)]
        for loc, c in self.cols.items():
            self.prev[loc] = c.nr_1.actual.clone()
        self.age += 1
        return []


def run(agt, stream):
    """per-column prediction and persistence traces, tick-aligned: the guess
    emitted after step t targets the activity that materializes at t+1."""
    pred, persist, g_prev, a_prev = {}, {}, None, None
    for a in stream:
        agt.step([a])
        acts = {loc: c.nr_1.actual.clone() for loc, c in agt.cols.items()}
        for loc in acts:
            if g_prev is not None and loc in agt.predicted:
                g, x = g_prev[loc], acts[loc]
                d = g.norm() * x.norm()
                pred.setdefault(loc, []).append(float(g @ x / d) if d > 0 else 0.0)
            if a_prev is not None:
                p, x = a_prev[loc], acts[loc]
                d = p.norm() * x.norm()
                persist.setdefault(loc, []).append(float(p @ x / d) if d > 0 else 0.0)
        g_prev = {loc: agt.cols[loc].nr_1.expect.clone() for loc in agt.predicted}
        a_prev = acts
    t = lambda xs: sum(xs[-WINDOW:]) / WINDOW
    return {loc: t(xs) for loc, xs in pred.items()}, {loc: t(xs) for loc, xs in persist.items()}


if __name__ == "__main__":
    path = f"{project_root_path}/saves/expectations_deep"
    for frozen, title in ((False, "learner"), (True, "frozen twin")):
        print(f"{title}:  {'col':<6} " + " ".join(f"{s:>18}" for s in ("cycle", "noise")))
        rows = {}
        for name, fn in (("cycle", cycle), ("noise", noise)):
            pr, pe = {}, {}
            for seed in SEEDS:
                agt = DeepAgt(path, derive_seed("deep", title, seed), frozen=frozen)
                p, q = run(agt, fn(N_STEPS, seed))
                for loc in q:
                    pr.setdefault(loc, []).append(p.get(loc))
                    pe.setdefault(loc, []).append(q[loc])
            for loc in sorted(pe):
                m = lambda xs: sum(x for x in xs if x is not None) / len(xs) if any(x is not None for x in xs) else None
                rows.setdefault(loc, {})[name] = (m(pr[loc]), m(pe[loc]))
        for loc in sorted(rows):
            cells = []
            for name in ("cycle", "noise"):
                p, q = rows[loc][name]
                cells.append(f"{'(none)' if p is None else f'{p:+.2f}':>8} /{q:+.2f}")
            print(f"         {str(loc):<6} " + "  ".join(f"{c:>16}" for c in cells))
        print()
    print("cells: predict / persistence baseline")
