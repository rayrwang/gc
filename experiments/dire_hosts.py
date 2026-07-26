"""
gc-native hosts for the Dir.E exam: a configurable feedforward agent with
Dir.A running forward and an inductive Dir.E running backward, behind the
exam interface. The knob surface is the June CIFAR/MNIST arc's:

    dims       hidden layer widths, e.g. [64] or [64, 64]
    rule       dir.a learning rule: softhebb | bcm | oja | instar | basic
    act        relu | triangle(power) | spike(threshold)
    norm       per-layer input normalization: none | unit | bn (online, per
               feature: current stats while learning, running stats frozen)
    beta       soft-WTA gate inverse temperature (softhebb; signed gate)
    power      triangle exponent
    theta_decay  bcm sliding-threshold EMA rate
    dire       False disables the backward direction entirely (substrate-only
               arm); guesses then stay at the frozen E init
    dire_rule  delta (error-hebbian, nlms-normalized) | hebb
    frozen     True freezes all learning (the same-init frozen control)

Dir.E realization (foundation-inductive, predict-only rung): backward maps
E_l trained one-step-ahead, layer l activity at t-1 -> layer l-1 activity at
t; the exam-level guess is the input-level prediction E_1 @ h_1(t-1).
"""

import os
import sys
from collections import namedtuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.funcs as fc
from experiments.dire_exam import DirEInterface, SYMBOL_DIM

Activs = namedtuple("Activs", "actual expect avg avg_sq")  # duck-type for fc.lrn_adaptive


class OneLayerDirEHost(DirEInterface):
    """one hidden layer in exam effect: the guess reads layer 1, so the dims
    parameter is inert beyond it (kept for the depth-inertness demonstration).
    the framework-native multi-layer host is DeepFFAgt below."""

    def __init__(self, dim=SYMBOL_DIM, seed=0, dims=(64,), rule="softhebb",
                 act="triangle", norm="bn", beta=1.0, power=0.7, theta_decay=0.05,
                 lr_a=0.03, lr_e=0.1, init_scale=3.0, spike_threshold=1.0,
                 bn_mom=0.01, dire=True, dire_rule="delta", frozen=False):
        g = torch.Generator().manual_seed(seed)
        widths = [dim, *dims]
        self.W = [init_scale * torch.randn(i, o, generator=g) / i ** 0.5
                  for i, o in zip(widths[:-1], widths[1:])]
        self.E = [0.01 * torch.randn(i, o, generator=g)  # maps layer l (o) down to l-1 (i)
                  for i, o in zip(widths[:-1], widths[1:])]
        self.theta = [torch.ones(o) for o in widths[1:]]      # bcm sliding thresholds
        self.bn_m = [torch.zeros(i) for i in widths[:-1]]     # online-bn running stats
        self.bn_v = [torch.ones(i) for i in widths[:-1]]
        self.h_prev = None
        cfg = dict(rule=rule, act=act, norm=norm, beta=beta, power=power,
                   theta_decay=theta_decay, lr_a=lr_a, lr_e=lr_e,
                   spike_threshold=spike_threshold, bn_mom=bn_mom, dire=dire,
                   dire_rule=dire_rule, frozen=frozen)
        self.__dict__.update(cfg)
        self.cfg = cfg

    def _norm(self, x, li):
        if self.norm == "unit":
            return x / (x.norm() + 1e-6)
        if self.norm == "bn":
            if not self.frozen:  # current stats while learning, update running
                self.bn_m[li] = (1 - self.bn_mom) * self.bn_m[li] + self.bn_mom * x
                self.bn_v[li] = (1 - self.bn_mom) * self.bn_v[li] + self.bn_mom * (x - self.bn_m[li]) ** 2
                m, v = x.mean(), x.var()
                return (x - m) / (v + 1e-5).sqrt()
            return (x - self.bn_m[li]) / (self.bn_v[li] + 1e-5).sqrt()
        return x

    def _act(self, u):
        if self.act == "relu":
            return torch.relu(u)
        if self.act == "triangle":
            return fc.triangle(u, self.power)
        return fc.spike(u, self.spike_threshold)

    def _forward(self, a):
        h, us, xs = [a], [], []
        x = a
        for li, w in enumerate(self.W):
            x = self._norm(x, li)
            xs.append(x)
            u = x @ w
            us.append(u)
            x = self._act(u)
            h.append(x)
        return h, us, xs

    def _learn_a(self, h, us, xs):
        for li, w in enumerate(self.W):
            x, u, y = xs[li], us[li], h[li + 1]
            if self.rule == "softhebb":
                gate = fc.softmax_wta(u, self.beta, signed=True)
                self.W[li] = fc.lrn_oja_gated(x, w, gate, u, ss=self.lr_a)
            elif self.rule == "bcm":
                self.theta[li] = (1 - self.theta_decay) * self.theta[li] + self.theta_decay * y ** 2
                self.W[li] = fc.lrn_adaptive(Activs(x, None, None, None), w,
                                             Activs(y, None, None, self.theta[li]), ss=self.lr_a)
            elif self.rule == "oja":
                self.W[li] = fc.lrn_oja(x, w, y, ss=self.lr_a)
            elif self.rule == "instar":
                self.W[li] = fc.lrn_instar(x, w, y, ss=self.lr_a)
            else:
                self.W[li] = fc.lrn_basic(x, w, y, ss=self.lr_a)

    def _learn_e(self, h_now):
        for li, e in enumerate(self.E):
            pre, target = self.h_prev[li + 1], h_now[li]
            if self.dire_rule == "hebb":
                self.E[li] = e + self.lr_e * torch.outer(target, pre) / (1 + pre @ pre)
            else:  # delta, nlms-normalized
                self.E[li] = e + self.lr_e * torch.outer(target - e @ pre, pre) / (1 + pre @ pre)

    def guess(self):
        if self.h_prev is None:
            return torch.zeros(self.W[0].shape[0])
        return self.E[0] @ self.h_prev[1]  # input-level one-step-ahead prediction

    def observe(self, a):
        h, us, xs = self._forward(a)
        if not self.frozen:
            self._learn_a(h, us, xs)
            if self.dire and self.h_prev is not None:
                self._learn_e(h)
        self.h_prev = h


HOSTS = {
    "host:gc_ff": lambda dim, seed: OneLayerDirEHost(dim, seed),
    "host:gc_ff_dire_off": lambda dim, seed: OneLayerDirEHost(dim, seed, dire=False),
    "control:frozen": lambda dim, seed: OneLayerDirEHost(dim, seed, frozen=True),
}


if __name__ == "__main__":
    # free-play smoke: default recipe on cycle and noise, plus rule variants on kp1
    from experiments import dire_exam as de

    def trail(make, fixture, stream_fn, seed):
        records = []
        de.run_stream(make(de.SYMBOL_DIM, de.derive_seed("subject", "smoke", seed)),
                      stream_fn(400, seed=seed), records.append)
        xs = [r["score"] for r in records if r["score"] is not None]
        return sum(xs[-50:]) / 50

    for name, make in HOSTS.items():
        for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
            ms = [trail(make, fixture, fn, s) for s in (0, 1, 2)]
            print(f"{fixture} {name:<22} trailing mean per seed: " + str([f"{m:+.3f}" for m in ms]))
    print()
    for rule in ("softhebb", "bcm", "oja", "instar"):
        for act in ("triangle", "relu", "spike"):
            make = lambda dim, seed: OneLayerDirEHost(dim, seed, rule=rule, act=act)
            ms = [trail(make, "kp1", de.kp1_cycle, s) for s in (0, 1, 2)]
            print(f"kp1 {rule:<9}+{act:<9} trailing mean per seed: " + str([f"{m:+.3f}" for m in ms]))


# deeper feedforward on real framework parts (RW-directed 2026-07-20):
# AgtBase subclass, I_VectorCol input, BareCol hidden layers, framework conns.
# Dir.A forward uses BareAgt's own rule; Dir.E backward supplies the delta
# rule that src/agents.py:1255 leaves as TODO. predict-only rung: fc.inhibit
# (the affect-mode combiner 02_expectations demonstrates) stays off behind a
# flag.

from dataclasses import dataclass

from src import iotypes as T
from src.agents import (AgtBase, BareCol, BareColCfg, CfgBase, Dir,
                        I_VectorCol, I_VectorColCfg, conn)


@dataclass
class DeepFFCfg(CfgBase):
    ispec: list
    dims: tuple


class DeepFFAgt(AgtBase):
    """input col (0,0) -> hidden BareCols (1,0), (2,0), ... Dir.A conns run
    forward, Dir.E conns run backward (layer l+1 predicts layer l's next
    activity; layer 1 predicts the next input). same-tick sequential forward
    like MNISTAgt (matched learning); expectations are one-step-ahead by
    construction: emitted at t from t activity, scored against t+1."""

    def __init__(self, cfg, path, seed=0, lr_e=0.1, dire=True, frozen=False,
                 affect=False):
        super().__init__(cfg, path, False)
        self.age = 0
        self.lr_e, self.dire, self.frozen, self.affect = lr_e, dire, frozen, affect
        torch.manual_seed(seed)  # conn()/activs() draw from the global rng

        d_in = cfg.ispec[0].d
        col = I_VectorCol((0, 0), I_VectorColCfg(d_in))
        self.cols[(0, 0)] = col
        self.I_cols.append(col)
        widths = [d_in, *cfg.dims]
        for li, w in enumerate(cfg.dims):
            self.cols[(li + 1, 0)] = BareCol((li + 1, 0), BareColCfg(w))
        for li in range(len(widths) - 1):
            lo, hi = self.cols[(li, 0)], self.cols[(li + 1, 0)]
            lo.conns[(hi.loc, Dir.A)] = conn(lo, hi, Dir.A, 3.0)
            hi.conns[(lo.loc, Dir.E)] = conn(hi, lo, Dir.E, 0.01)
        self.prev_actual = {loc: torch.zeros(c.d) for loc, c in self.cols.items()}
        # columns something predicts: targets of dir.e conns (all but the top;
        # an unpredicted column emits no guess and never enters the aggregate)
        self.predicted = {loc for c in self.cols.values()
                          for (loc, d) in c.conns if d == Dir.E}
        self.create_directory()

    def cleanup(self):
        pass  # exam scaffold: never save on exit

    def step(self, ipt, use_lrn=True):
        use_lrn = use_lrn and not self.frozen
        # snapshot the expectations emitted last step: update_activations wipes
        # them (moves in the zeroed buffer) before the learning phase runs
        prev_expect = {loc: c.nr_1.expect.clone() for loc, c in self.cols.items()}
        col0 = self.cols[(0, 0)]
        col0.ipt(ipt[0])
        col0.update_activations()
        n_hidden = len(self.cols) - 1

        # forward (dir.a): sequential same-tick pass; input sender raw, hidden
        # senders triangle-transported (the framework's propagation forms)
        for li in range(n_hidden):
            lo, hi = self.cols[(li, 0)], self.cols[(li + 1, 0)]
            w = lo.conns[(hi.loc, Dir.A)]
            hi.nr_1_.actual = (lo.a_pre @ w if li == 0
                              else fc.atv_triangle(lo.a_pre, w, power=0.3))
            hi.update_activations()

        # dir.e learning (the agents.py:1255 TODO, delta form): the expectation
        # each col holds was emitted last step; its error against the activity
        # that just arrived trains the backward conn, pre = the sender activity
        # that emitted it
        if use_lrn and self.dire:
            for li in range(n_hidden):
                hi, lo = self.cols[(li + 1, 0)], self.cols[(li, 0)]
                w = hi.conns[(lo.loc, Dir.E)]
                err = lo.nr_1.actual - prev_expect[lo.loc]
                pre = self.prev_actual[hi.loc]
                hi.conns[(lo.loc, Dir.E)] = \
                    w + self.lr_e * torch.outer(pre, err) / (1 + pre @ pre)

        # dir.a learning: the framework's own rule (BareAgt step, matched acts)
        if use_lrn:
            for li in range(n_hidden):
                lo, hi = self.cols[(li, 0)], self.cols[(li + 1, 0)]
                w = lo.conns[(hi.loc, Dir.A)]
                lo.conns[(hi.loc, Dir.A)] = fc.lrn_oja_gated(
                    lo.nr_1.actual, w,
                    fc.softmax_wta(hi.nr_1.actual, beta=0.5, signed=True),
                    hi.nr_1.actual, ss=1e-2)

        # emit expectations for t+1 (linear transport: keeps the delta rule's
        # credit assignment exact; triangle-transported expectations are an
        # affect-rung question). affect=False leaves them recorded, never fed
        # back through fc.inhibit.
        for li in range(n_hidden):
            hi, lo = self.cols[(li + 1, 0)], self.cols[(li, 0)]
            lo.nr_1.expect = hi.nr_1.actual @ hi.conns[(lo.loc, Dir.E)]

        for loc, c in self.cols.items():
            self.prev_actual[loc] = c.nr_1.actual.clone()
        self.age += 1
        return []


class DeepFFHost(DirEInterface):
    """exam adapter: single-vector interface reads the input col's
    expectation (plugs into the existing exam unchanged); guesses()/actuals()
    expose every col for the per-column machinery."""

    def __init__(self, dim=SYMBOL_DIM, seed=0, dims=(64, 64), path=None, **kw):
        cfg = DeepFFCfg(ispec=[T.I_Vector(dim)], dims=tuple(dims))
        path = path or os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "saves", "dire_deep_ff")
        self.agt = DeepFFAgt(cfg, path, seed=seed, **kw)

    def guess(self):
        return self.agt.cols[(0, 0)].nr_1.expect.clone()

    def guesses(self):
        return {loc: c.nr_1.expect.clone() for loc, c in self.agt.cols.items()
                if loc in self.agt.predicted}

    def actuals(self):
        return {loc: c.nr_1.actual.clone() for loc, c in self.agt.cols.items()}

    def observe(self, a):
        self.agt.step([a])


# bareagt rung (RW-directed 2026-07-20): the real recurrent substrate behind
# the exam interface. two agents: BareAgtDirE wraps src.agents.BareAgt
# byte-untouched (the fidelity anchor: zero knobs, the Dir.E delta appended
# around the step per the synchronous-buffer boundary argument), and BareAgtX
# is the parameterized sibling (AgtBase child in the DeepFFAgt pattern) whose
# config surface is the sweep: connectivity law, init distribution, rule,
# transport, rates. X's "faithful" preset shadows stock's law statistically
# (same probabilities, different draws): stock stays the bit-faithful arm.

import contextlib
import io as _io
import math
import random

from src.agents import BareAgt, BareCfg


class BareAgtDirE(BareAgt):
    """stock BareAgt + the Dir.E learning the substrate's step leaves TODO
    (src/agents.py:1255). the synchronous double-buffer makes the step a
    clean t-1 -> t transition, so the delta composes as a wrapper: snapshot
    sender actuals at entry, let the untouched parent step run (dir.a
    learning, both transports, commit), then train each E conn on the
    committed expectation error. transport is atv_triangle(power 0.3), so
    the presynaptic factor is the triangled snapshot; a target with several
    incoming E conns shares one residual. no frozen mode: stock's step has
    no learning switch, so the frozen anchor lives in BareAgtX(faithful).
    exam sizes stay small: no col paging is exercised."""

    def __init__(self, cfg, path, seed=0, lr_e=0.1, tap=False):
        random.seed(seed)          # connectivity draws
        torch.manual_seed(seed)    # conn()/activs() draws
        with contextlib.redirect_stdout(_io.StringIO()), \
                contextlib.redirect_stderr(_io.StringIO()):  # tqdm uses stderr
            super().__init__(cfg, path)
        self.lr_e = lr_e
        self.tap = tap
        self.last_taps = None
        self.e_conns = [(col, tloc) for col in self.cols.values()
                        for (tloc, d) in col.conns if d == Dir.E]
        self.predicted = {tloc for _, tloc in self.e_conns}

    def cleanup(self):
        pass  # exam scaffold: never save on exit

    def step(self, ipt, disable_print=True):
        snap = {col.loc: col.nr_1.actual.clone() for col, _ in self.e_conns}
        out = super().step(ipt, disable_print=disable_print)
        if self.tap:
            # pre-sum tap, stock arm: recomputed from the snapshot with the
            # exact propagation form (e_pre = committed t-1 actual, triangle
            # transport) and the entry weights, which stock's step leaves
            # untouched (its dir.e learning is the TODO this class fills).
            # E only: A contributions form inside the untouched parent where
            # dir.a learning mutates weights mid-pass; X is the full-tap arm.
            self.last_taps = [
                (col.loc, tloc, Dir.E,
                 fc.atv_triangle(snap[col.loc], col.conns[(tloc, Dir.E)], power=0.3))
                for col, tloc in self.e_conns]
        for col, tloc in self.e_conns:
            target = self.cols[tloc]
            err = target.nr_1.actual - target.nr_1.expect  # shared residual
            pre = fc.triangle(snap[col.loc], power=0.3)    # what transport used
            w = col.conns[(tloc, Dir.E)]
            col.conns[(tloc, Dir.E)] = \
                w + self.lr_e * torch.outer(pre, err) / (1 + pre @ pre)
        return out


@dataclass
class BareXCfg(CfgBase):
    ispec: list
    n_cols: int = 16          # bulk grid cols
    d_col: int = 32           # bulk col width
    p_a: float = 0.3          # dir.a base probability (0.6 from io senders)
    p_e: float = 0.3          # dir.e base probability
    dist_exp: float = 2.0     # connectivity falloff exponent
    conn_scale: float = 2.0   # weight init scale
    conn_dist: str = "gauss"  # gauss | uniform
    single_conn: bool = True  # stock's at-most-one-conn-per-target break
    signed: bool = True       # soft-WTA gate signedness (competition knob)
    rule: str = "softhebb"    # softhebb | oja | instar | basic
    beta: float = 0.5
    ss: float = 1e-2
    transport: str = "triangle"  # triangle | relu | raw
    power: float = 0.3
    lr_e: float = 0.1


class BareAgtX(AgtBase):
    """parameterized recurrent grid: real BareCols, framework conn shapes,
    stock's phase structure (jacobi: read t-1, write buffers, commit), with
    the init and step knobs exposed as cfg and the dir.e delta native. the
    faithful preset (cfg defaults) shadows stock's law; sweeps move one knob
    at a time from there."""

    def __init__(self, cfg, path, seed=0, dire=True, frozen=False, tap=False,
                 a_frozen=False, dire_rule="delta"):
        super().__init__(cfg, path)
        self.age = 0
        self.dire, self.frozen = dire, frozen
        self.a_frozen = a_frozen  # freeze dir.a only: the E-vs-stationary-dynamics arm
        self.tap = tap
        self.last_taps = None
        random.seed(seed)
        torch.manual_seed(seed)
        g_in = cfg.ispec[0].d
        col = I_VectorCol((1, 0), I_VectorColCfg(g_in))  # stock's input loc:
        # the shadow property needs identical distances, so identical thresholds
        self.cols[(1, 0)] = col
        self.I_cols.append(col)
        width = math.ceil(cfg.n_cols ** 0.5)
        for i, (y, x) in enumerate((y, x) for y in range(width) for x in range(width)):
            if i < cfg.n_cols:
                loc = (x + 1, y + 1)
                self.cols[loc] = BareCol(loc, BareColCfg(cfg.d_col))
        for loc, col in self.cols.items():
            for other in self.cols:
                if other == loc:
                    continue
                dist = fc.dist(loc, other)
                base = 0.6 if self.is_io(loc) else cfg.p_a
                if self.is_i(other):
                    allowed = []          # nothing targets the input col
                elif self.is_io(loc):
                    allowed = [Dir.A]     # io senders provide actual only
                else:
                    allowed = [Dir.A, Dir.E]
                for direction in allowed:
                    p = (base if direction == Dir.A else cfg.p_e) \
                        / (dist + 1e-3) ** cfg.dist_exp
                    if random.random() < p:
                        w = conn(col, self.cols[other], direction, cfg.conn_scale)
                        if cfg.conn_dist == "uniform":
                            w = cfg.conn_scale * (2 * torch.rand_like(w) - 1) \
                                / w.shape[0] ** 0.5
                        col.conns[(other, direction)] = w
                        if cfg.single_conn:
                            break
        self.e_conns = [(c, tloc) for c in self.cols.values()
                        for (tloc, d) in c.conns if d == Dir.E]
        self.predicted = {tloc for _, tloc in self.e_conns}
        self.create_directory()

    def cleanup(self):
        pass  # exam scaffold: never save on exit

    def _transport(self, x, w):
        if self.cfg.transport == "raw":
            return x @ w
        if self.cfg.transport == "relu":
            return torch.relu(x) @ w
        return fc.atv_triangle(x, w, power=self.cfg.power)

    def step(self, ipt, disable_print=True):
        cfg = self.cfg
        for col, x in zip(self.I_cols, ipt, strict=True):
            col.ipt(x)
        snap = {loc: c.nr_1.actual.clone() for loc, c in self.cols.items()}
        taps = [] if self.tap else None
        for col in self.cols.values():  # pass 1: learn dir.a, propagate both
            if not self.frozen and not self.a_frozen:
                for (tloc, d), w in list(col.conns.items()):
                    if d != Dir.A:
                        continue
                    y = self.cols[tloc].nr_1.actual
                    if cfg.rule == "softhebb":
                        gate = fc.softmax_wta(y, beta=cfg.beta, signed=cfg.signed)
                        col.conns[(tloc, d)] = fc.lrn_oja_gated(
                            col.nr_1.actual, w, gate, y, ss=cfg.ss)
                    elif cfg.rule == "oja":
                        col.conns[(tloc, d)] = fc.lrn_oja(col.nr_1.actual, w, y, ss=cfg.ss)
                    elif cfg.rule == "instar":
                        col.conns[(tloc, d)] = fc.lrn_instar(col.nr_1.actual, w, y, ss=cfg.ss)
                    else:
                        col.conns[(tloc, d)] = fc.lrn_basic(col.nr_1.actual, w, y, ss=cfg.ss)
            for (tloc, d), w in col.conns.items():
                t = self.cols[tloc]
                contrib = (col.nr_1.actual @ w if self.is_i(col.loc)  # input sender: raw
                           else self._transport(col.nr_1.actual, w))
                if d == Dir.A:
                    t.nr_1_.actual = t.nr_1_.actual + contrib
                else:
                    t.nr_1_.expect = t.nr_1_.expect + contrib
                if taps is not None:  # pre-sum tap: the party-resolved fact
                    taps.append((col.loc, tloc, d, contrib))
        for col in self.cols.values():  # pass 2: commit
            col.update_activations()
        if self.dire and not self.frozen:  # dir.e delta on committed error
            for col, tloc in self.e_conns:
                target = self.cols[tloc]
                err = target.nr_1.actual - target.nr_1.expect
                pre = snap[col.loc] if cfg.transport == "raw" \
                    else (torch.relu(snap[col.loc]) if cfg.transport == "relu"
                          else fc.triangle(snap[col.loc], power=cfg.power))
                w = col.conns[(tloc, Dir.E)]
                # Rule-form sweep on the E channel. delta is the incumbent and
                # its branch is untouched, so registered runs stay identical.
                # The others drop the error term: they never subtract what they
                # already predict, so they have no least-squares optimum to
                # collapse onto (the AOFF26 degeneracy). Matched design: every
                # arm gets the SAME lr_e and the SAME NLMS denominator, so the
                # only thing varying is the rule's form. The library rules are
                # called at ss=1.0 and differenced to recover a unit-step dw.
                rule_e = getattr(self, "dire_rule", "delta")
                post = target.nr_1.actual
                if rule_e == "delta":
                    upd = torch.outer(pre, err)
                elif rule_e == "hebb":
                    upd = torch.outer(pre, post)
                elif rule_e == "oja":
                    upd = fc.lrn_oja(pre, w, post, ss=1.0) - w
                elif rule_e == "instar":
                    upd = fc.lrn_instar(pre, w, post, ss=1.0) - w
                elif rule_e == "softhebb":
                    gate = fc.softmax_wta(post, beta=cfg.beta, signed=cfg.signed)
                    upd = fc.lrn_oja_gated(pre, w, gate, post, ss=1.0) - w
                # no bcm here: lrn_adaptive takes Activs (it reads x.actual and
                # the running avg/avg_sq for its sliding threshold), which the
                # E channel does not maintain. The A side at this rung does not
                # offer it either (BareXCfg: softhebb | oja | instar | basic).
                # no basic here: lrn_basic is dw ~ xy, identical to hebb above.
                else:
                    raise ValueError(f"unknown dire_rule {rule_e!r}")
                col.conns[(tloc, Dir.E)] = \
                    w + cfg.lr_e * upd / (1 + pre @ pre)
        self.last_taps = taps
        self.age += 1
        return []


class BareHost(DirEInterface):
    """exam adapter over either bareagt: per-column guesses = committed
    expectations of the columns some E conn targets. stock bareagt's E conns
    never target the input col, so the predicted set is internal columns
    only, and can be empty for sparse draws: exposed for the manifest."""

    def __init__(self, agt):
        self.agt = agt

    def guess(self):  # single-vector interface: unused when guesses() exists
        return torch.zeros(self.agt.cfg.ispec[0].d)

    def guesses(self):
        return {loc: self.agt.cols[loc].nr_1.expect.clone()
                for loc in self.agt.predicted}

    def actuals(self):
        return {loc: c.nr_1.actual.clone() for loc, c in self.agt.cols.items()}

    def taps(self):
        """per-sender pre-sum contributions from the last step, or None when
        the tap is off. record facts, derive interpretations: sums, type
        splits, and own/other partitions are read-time arithmetic."""
        lt = self.agt.last_taps
        if lt is None:
            return None
        return [{"sender": f"{s[0]},{s[1]}", "target": f"{t[0]},{t[1]}",
                 "dir": d.name, "contribution": c.tolist()}
                for s, t, d, c in lt]

    def observe(self, a):
        self.agt.step([a])


def _bare_path(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "saves", name)


def stock_bare_host(dim, seed, n_cols=16, lr_e=0.1, tap=False):
    cfg = BareCfg(n_cols=n_cols, ispec=[T.I_Vector(dim)], ospec=[])
    return BareHost(BareAgtDirE(cfg, _bare_path("dire_bareagt_stock"), seed=seed,
                                lr_e=lr_e, tap=tap))


def x_bare_host(dim, seed, dire=True, frozen=False, tap=False, **knobs):
    cfg = BareXCfg(ispec=[T.I_Vector(dim)], **knobs)
    return BareHost(BareAgtX(cfg, _bare_path("dire_bareagt_x"), seed=seed,
                             dire=dire, frozen=frozen, tap=tap))


def tap_selftest():
    """pre-sum tap known answers: (1) tap-on and tap-off X runs are
    bit-identical, the tap only observes; (2) per target, the summed E
    contributions equal the committed expectation exactly (buffers zero each
    step, expect commits unnormalized); (3) the stock arm's recomputed E taps
    reproduce its committed expectations; (4) the host accessor labels
    parties and directions."""
    from experiments.dire_exam import derive_seed, kz_noise
    stream = list(kz_noise(40, seed=derive_seed("tap", "selftest", 0)))
    h_on = x_bare_host(SYMBOL_DIM, 7, tap=True)
    h_off = x_bare_host(SYMBOL_DIM, 7)
    n_e = n_a = 0
    for item in stream:
        a = item[1] if isinstance(item, tuple) else item
        h_on.observe(a)
        h_off.observe(a)
        for loc, c in h_on.agt.cols.items():
            other = h_off.agt.cols[loc]
            assert torch.equal(c.nr_1.actual, other.nr_1.actual), loc
            assert torch.equal(c.nr_1.expect, other.nr_1.expect), loc
        sums = {}
        for _s, t, d, contrib in h_on.agt.last_taps:
            if d == Dir.E:
                n_e += 1
                sums[t] = sums.get(t, 0) + contrib
            else:
                n_a += 1
        for t, sm in sums.items():
            assert torch.equal(sm, h_on.agt.cols[t].nr_1.expect), f"checksum {t}"
    assert h_off.taps() is None and h_on.taps() is not None
    rec = h_on.taps()[0]
    assert set(rec) == {"sender", "target", "dir", "contribution"}
    hs = stock_bare_host(SYMBOL_DIM, 7, tap=True)
    n_stock = 0
    for item in stream:
        a = item[1] if isinstance(item, tuple) else item
        hs.observe(a)
        sums = {}
        for _s, t, _d, contrib in hs.agt.last_taps:
            n_stock += 1
            sums[t] = sums.get(t, 0) + contrib
        for t, sm in sums.items():
            assert torch.allclose(sm, hs.agt.cols[t].nr_1.expect, atol=1e-6), \
                f"stock checksum {t}"
    print(f"tap selftest passed: x {n_e} E + {n_a} A contributions, "
          f"stock {n_stock} E recomputed, {len(stream)} steps, "
          f"checksums exact, tap-off bit-identical")
