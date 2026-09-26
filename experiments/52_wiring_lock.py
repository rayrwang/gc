"""
What about the wiring decides whether an activation locks, and do the
discontinuous activations ever lock? RW, 2026-09-26: "figure out what exaclty
about the wiring/archithecture makes trianlge lock or not lock. what exactly
changing between 16 and 45 makes triangle lock on one and not on the other?
also does teh discotionuous one slike wta always not lock? also throw in step
and other either consitnouous or disconsitnous activation fucntiosn for good
measure". Background: experiment 51 found tri0.3 locks on example 16's frozen
network (step-to-step correlation 0.92) and wta churns (0.05), whatever the
input; on run 39's network tri0.3 at the same settings does not lock.

Premises
  host    BareAgtXC (10_sweep_bare_x): bulk columns of 32 units on a square
          grid, one input column at (1,0). Edges and weights come from
          per-edge hashes of (seed, sender, target, direction), so a sparser
          graph is a subgraph of a denser one with the same weights on the
          shared edges, and the 49-column grid is the top-left 7x7 block of
          the 200-column grid (width 15). At most one conn per ordered pair,
          Dir.A drawn first. Dir.E conns write only to the expectation, which
          never feeds the activity, so they are inert here.
  fixed   learning frozen on both channels; weight scale 2.0; example 16's
          step settings unless the side grid says otherwise: keep 0 (the
          accumulator is zeroed each step) and the RMS norm on (floor 0.9);
          1000 steps, the last 300 read.
  varied  main grid, at keep 0 and norm on:
            n_cols    49, 200
            dist_exp  1, 2 (connection probability p_a / d^dist_exp)
            p_a       0.15, 0.3, 0.6 (density at a fixed falloff)
            input     none: no input edges
                      kp1: 32-dim 4-cycle symbols into 3 hash-chosen columns
                           (run 39's)
                      mnist: a random MNIST training image each step, 784
                           pixels in [0,1], input edges by the stock law
                           0.6 / d^dist_exp (example 16's, passive mode; the
                           same image sequence in every cell)
            activation, applied to what each bulk column sends (f(x) @ w);
            the input column sends raw:
              continuous     raw (linear), relu, tri0.3, tri0.7, tri1.0,
                             tanh, sig4_0, sig8_0
              discontinuous  step0, cstep, sign, wta, kwta4
            seeds 50 to 53
          side grid, at the two anchors only: keep {0, 0.9} x norm {on, off},
          so the step settings that differ between example 16 (keep 0, norm
          on) and experiment 45's default (keep 0.9, norm off) are covered.
  anchors example 16's network = (200 cols, dist_exp 2, p_a 0.3, mnist);
          run 39's network = (49 cols, dist_exp 1, p_a 0.3, kp1), identical
          to run 39's graph at seeds 50 and 51. Example 16's stock BareAgt
          (src.agents) is run directly at its anchor as the host check, on
          the CPU (51 ran on the GPU), seeds 0 to 3.

Measures (bulk columns, the last 300 steps)
  c1, c10  51's measure: per column, the correlation across its 32 units
           between the state at t and at t+1 (t+10), averaged over t; median
           over columns; a dead column counts 0
  lock     fraction of live columns whose state stops moving: mean over t of
           |x_{t+1} - x_t| / |x_t| below 0.01
  move     median over t of |X_{t+1} - X_t| / |X_t| over the whole bulk
  period   smallest p <= 50 with X_{t+p} = X_t (relative 1e-5) over the last
           100 steps; 1 = fixed point; None = no short cycle
  dead     fraction of bulk columns identically zero over the window
  static   per column, the share of its state's energy that is its average
           over the window, |mean_t x|^2 / mean_t |x|^2; median over live
           columns. 1 = the same vector every step; a pattern that looks the
           same step to step can be a large static part with churn on top
  fac1     per column, the lag-1 correlation of the fluctuation x_t - mean_t x
           (time-centred, summed over units and steps); median over live
           columns: whether what moves keeps its direction or is new each step
  ac1_39   run 39's own ac1 (_autocorr: time-centred, pooled over the bulk),
           for direct comparison with actdyn39.json
  sent     the sending side, f applied to each column's state: static, as
           above on what it sends; level, (mean of what it sends over units
           and steps)^2 / mean square, the share that is one constant level
Graph (per wiring and seed, activation-independent; Dir.A among bulk columns)
  kin      mean in-degree; the fractions with in-degree 0, 1, 2 and 3+
  cyc      fraction of bulk columns on a directed cycle (in a strongly
           connected component of 2 or more columns); gscc, the largest such
           component as a fraction of the bulk
  inreach  bulk columns the input reaches directly, and through any path
Per column, for the link between locking and the graph: in-degree, cycle
membership, input reach, locked, dead; kept in the JSON.

Expected before the run (Claude's guess, written 2026-09-26 before any
cell ran): the triangle's lock follows the recurrent graph, not the input
or the grid size as such. 1/d^2 at p_a 0.3 leaves most columns with in-degree
0 to 2 and the cycles small, and small cycles of a continuous map settle;
1/d gives a large strongly connected component and chaos. If so, raising p_a
at 1/d^2 should unlock it and lowering p_a at 1/d should lock it, and the
per-column lock rate should track cycle membership and in-degree. The
discontinuous activations should keep moving everywhere except where a
column has no input at all.

Smoke, 2026-09-26, before the grid: tri0.3 read c1 0.914 on example 16's
network (seed 50) and 0.913 on run 39's (seed 50), with lock 0.00 and move
0.42 on both. The premise of the question was wrong: "the triangle does not
lock on run 39's network" came from run 39's ac1 (-0.011 at keep 0, norm on),
a time-centred measure, set against 51's c1, which is not time-centred. By one
measure the two networks agree. Second guess, written after the smoke and
before the grid (Claude's): c1 is high when the activation's output has a
nonzero average (tri, relu, step0, cstep, sigmoids), because every column
then receives a fixed profile, the weights' column sums times that average,
under whatever churns; it is low when the output averages near zero (raw,
tanh, sign) or is almost all zero (wta). If so, static tracks the sign of
the activation's mean output, not continuity, and not the wiring.
Second smoke with static and fac1 (seed 50, example 16's network unless
noted): static equals c1 to two decimals in every cell and fac1 is within
0.02 of zero in every cell, tri0.3 included (tri0.3 c1 0.914 static 0.916
fac1 -0.011; run 39's network 0.913 / 0.910 / -0.007; step0 0.805 / 0.806 /
-0.001; wta 0.052 / 0.056; sign 0.005 / 0.004; tanh 0.009 / 0.019). The host
check: stock BareAgt on the CPU, seed 0, tri0.3 c1 0.921 (51 on the GPU,
seed 0: 0.922), wta 0.059; the harness matches the stock host.

Result, 2026-09-26 (wiringlock52.log, .json; tables in wiringlock52_report.txt):
2,200 cells, the only failures the 64 expected divergences (raw, relu, tri1.0,
kwta4 with the norm off). Host check: stock and harness within 0.01 for
tri0.3, wta, tanh and sign. Main grid, mean static over 144 cells each: sig4_0
0.997, tri0.7 0.961, sig8_0 0.959, tri1.0 0.931, relu 0.925, tri0.3 0.913,
step0 0.808, cstep 0.784; kwta4 0.201, wta 0.120, raw 0.102, tanh 0.031, sign
0.007. Static against the fixed share of what is sent r 0.999, against the
sent level r 0.913 (1,872 cells). The wiring does not move the triangle:
tri0.3 static 0.89 to 0.93 over mean in-degree 1.3 to 23, 27 to 100% of
columns on a cycle, both grid sizes and all three inputs, in-degree-1 columns
the same as 4+. Continuity: second order for static (sig4 > sig8 > step0;
tri0.3 > cstep); for real settling (half or more of the columns moving less
than 1% a step) decisive: at keep 0 in 53 of 144 sig4_0 cells, 1 of 144 wta
cells and no step0, cstep, sign or kwta4 cell; at keep 0.9 on the two
anchors 66 to 100% of columns for the smooth positive-level activations (the
triangles, relu, the sigmoids with the norm on), 0 to 1% for step0, cstep,
sign, wta and the sigmoids with the norm off (saturated, step-like). The norm
leaves sign, step0, cstep and wta unchanged, as predicted. Guess 1 (the graph
decides) failed; guess 2 (the sign of the activation's mean output) held.
Caveats: static matches c1 in the typical run, not in every run (c1 counts
dead columns as 0, and raw's linear dynamics flip); fac1 is near 0 in the
median run for every activation, but raw, relu, tri1.0 and sig4_0 often
alternate (fac1 negative).

Usage:
    venv/bin/python experiments/52_wiring_lock.py                 # everything, in parallel
    venv/bin/python experiments/52_wiring_lock.py one tri0.3 200 2 0.3 mnist 50   # one cell
    venv/bin/python experiments/52_wiring_lock.py stock tri0.3 0  # one stock-host cell
    venv/bin/python experiments/52_wiring_lock.py report          # tables from the JSON
"""

import importlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)  # the MNIST loader reads ./data

import torch

r39 = importlib.import_module("39_activation_dynamics")
sweep, de, T = r39.sweep, r39.de, r39.T

STEPS, WINDOW = 1000, 300
SEEDS = (50, 51, 52, 53)
STOCK_SEEDS = (0, 1, 2, 3)
CONT = ("raw", "relu", "tri0.3", "tri0.7", "tri1.0", "tanh", "sig4_0", "sig8_0")
DISC = ("step0", "cstep", "sign", "wta", "kwta4")
ACTS = CONT + DISC
ANCHORS = {"ex16": (200, 2.0, 0.3, "mnist"), "run39": (49, 1.0, 0.3, "kp1")}
LOG = os.path.join(HERE, "wiringlock52.log")
OUT = os.path.join(HERE, "wiringlock52.json")
_MNIST = None


def mnist_seq():
    """the same 1000 random training images for every mnist cell"""
    global _MNIST
    if _MNIST is None:
        from src.envs import MNISTDataset
        ds = MNISTDataset(train=True)
        rng = random.Random(20260926)
        _MNIST = [ds[rng.randrange(len(ds))][0] for _ in range(STEPS)]
    return _MNIST


def stream(inp, seed):
    if inp == "mnist":
        return mnist_seq()
    return list(de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", r39.NS, seed)))


def graph(cols, is_i, bulk):
    """Dir.A structure among bulk columns: in-degree, cycle membership
    (Kosaraju), and the input's direct and total reach."""
    idx = {loc: i for i, loc in enumerate(bulk)}
    out = [[] for _ in bulk]
    kin = [0] * len(bulk)
    direct = set()
    for loc, c in cols.items():
        for (t, d) in c.conns:
            if d.name != "A" or t not in idx:
                continue
            if is_i(loc):
                direct.add(idx[t])
            elif loc in idx:
                out[idx[loc]].append(idx[t])
                kin[idx[t]] += 1
    n = len(bulk)
    order, seen = [], [False] * n
    for s in range(n):  # iterative dfs, finish order
        if seen[s]:
            continue
        stack = [(s, 0)]
        seen[s] = True
        while stack:
            v, i = stack.pop()
            if i < len(out[v]):
                stack.append((v, i + 1))
                w = out[v][i]
                if not seen[w]:
                    seen[w] = True
                    stack.append((w, 0))
            else:
                order.append(v)
    rev = [[] for _ in bulk]
    for v in range(n):
        for w in out[v]:
            rev[w].append(v)
    comp, c = [-1] * n, 0
    for s in reversed(order):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            v = stack.pop()
            for w in rev[v]:
                if comp[w] < 0:
                    comp[w] = c
                    stack.append(w)
        c += 1
    size = {}
    for k in comp:
        size[k] = size.get(k, 0) + 1
    cyc = [size[comp[v]] >= 2 for v in range(n)]
    reach, frontier = set(direct), list(direct)
    while frontier:
        v = frontier.pop()
        for w in out[v]:
            if w not in reach:
                reach.add(w)
                frontier.append(w)
    return {"kin": kin, "cyc": cyc, "direct": sorted(direct), "reach": sorted(reach),
            "gscc": max(size.values()) / n if size else 0.0}


def measure(X, d_col, f):
    """X: steps x bulk-units, the read window; f: the activation, for the
    sending side"""
    S, n = X.shape[0], X.shape[1] // d_col
    C = X.view(S, n, d_col)
    Y = torch.stack([torch.stack([f(C[t, j]) for j in range(n)]) for t in range(S)])
    ybar = Y.mean(0)
    ysq = (Y ** 2).mean((0, 2)).clamp(min=1e-24)
    s_static = ybar.norm(dim=-1) ** 2 / (Y.norm(dim=-1) ** 2).mean(0).clamp(min=1e-24)
    s_level = Y.mean((0, 2)) ** 2 / ysq

    def corr(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)

    c1 = float(corr(C[:-1], C[1:]).mean(0).median())
    c10 = float(corr(C[:-10], C[10:]).mean(0).median())
    live = C.abs().amax(dim=(0, 2)) > 1e-9
    ac1_39, tau_39 = r39._autocorr(X)  # run 39's: time-centred, pooled
    xbar = C.mean(0)
    static = xbar.norm(dim=-1) ** 2 / (C.norm(dim=-1) ** 2).mean(0).clamp(min=1e-24)
    F = C - xbar
    fac1 = (F[:-1] * F[1:]).sum((0, 2)) / (F * F).sum((0, 2)).clamp(min=1e-24)
    colmove = ((C[1:] - C[:-1]).norm(dim=-1) / (C[:-1].norm(dim=-1) + 1e-12)).mean(0)
    locked = (colmove < 0.01) & live
    move = float(((X[1:] - X[:-1]).norm(dim=-1) / (X[:-1].norm(dim=-1) + 1e-12)).median())
    tail = X[-100:]
    period = None
    for p in range(1, 51):
        a, b = tail[p:], tail[:-p]
        if bool(((a - b).norm(dim=-1) <= 1e-5 * (b.norm(dim=-1) + 1e-12)).all()):
            period = p
            break
    nl = int(live.sum())
    med = (lambda v: float(v[live].median())) if nl else (lambda v: float("nan"))
    return {"c1": c1, "c10": c10, "move": move, "period": period,
            "dead": 1 - nl / n, "lock": float(locked.sum()) / nl if nl else float("nan"),
            "static": med(static), "fac1": med(fac1), "ac1_39": ac1_39, "tau_39": tau_39,
            "s_static": med(s_static), "s_level": med(s_level),
            "col_locked": locked.tolist(), "col_live": live.tolist(),
            "col_move": [round(float(v), 5) for v in colmove],
            "col_static": [round(float(v), 4) for v in static],
            "col_fac1": [round(float(v), 4) for v in fac1]}


def cell(act, n_cols, dist_exp, p_a, inp, seed, keep=0.0, norm=True):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    r39._patch(act, keep, norm)
    dim = 784 if inp == "mnist" else de.SYMBOL_DIM
    fan = {"none": 0, "kp1": 3, "mnist": -1}[inp]
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(dim)], n_cols=n_cols,
                          **dict(r39.arm_cfg(act), p_a=p_a, dist_exp=dist_exp, input_fanout=fan))
    tag = f"{act}|{n_cols}|{dist_exp}|{p_a}|{inp}|{seed}|{keep}|{int(norm)}"
    path = os.path.join(ROOT, "saves", "wiringlock52", f"w{os.getpid()}")
    agt = sweep.BareAgtXC(cfg, path, seed=seed, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    g = graph(agt.cols, agt.is_i, bulk)
    zero = torch.zeros(dim)
    xs = stream(inp, seed) if inp != "none" else None
    rows = []
    t0 = time.time()
    for t in range(STEPS):
        agt.step([xs[t] if xs is not None else zero])
        if t >= STEPS - WINDOW:
            s = r39.state_of(agt)
            if not torch.isfinite(s).all():
                return {"tag": tag, "status": "DIVERGED", "step": t, "graph": g}
            rows.append(s.clone())
    s = r39.state_of(agt)
    if not torch.isfinite(s).all():
        return {"tag": tag, "status": "DIVERGED", "graph": g}
    m = measure(torch.stack(rows).float(), cfg.d_col, r39.act_fn(act))
    return {"tag": tag, "status": "ok", "secs": round(time.time() - t0, 1), "graph": g, **m}


def stock_cell(act, seed):
    """example 16's own host (src BareAgt, stock wiring, output column
    included), learning off, on the CPU; passive MNIST as in the main grid"""
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    from src import funcs as fc
    from src.agents import BareAgt, BareCfg
    r39._patch("tri0.3", 0.0, True)  # restores stock update and norm
    fc.lrn_oja_gated = lambda x, w, gate, u, ss=1e-2: w
    f = r39.act_fn(act)
    fc.atv_triangle = lambda x, w, power=0.3: f(x) @ w
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = BareCfg(200, [T.I_Vector(784)], [T.O_Vector(10)])
    agt = BareAgt(cfg, os.path.join(ROOT, "saves", "wiringlock52", f"stock{os.getpid()}"))
    bulk = [loc for loc in agt.cols if not agt.is_io(loc)]
    g = graph(agt.cols, agt.is_i, bulk)
    xs = mnist_seq()
    rows, t0 = [], time.time()
    for t in range(STEPS):
        agt.step([xs[t]], disable_print=True)
        if t >= STEPS - WINDOW:
            rows.append(torch.cat([agt.cols[loc].nr_1.actual.flatten() for loc in bulk]).clone())
    m = measure(torch.stack(rows).float(), 32, f)
    return {"tag": f"STOCK {act}|seed {seed}", "status": "ok",
            "secs": round(time.time() - t0, 1), "graph": g, **m}


def jobs():
    out = []
    for n in (49, 200):
        for de_ in (1.0, 2.0):
            for pa in (0.15, 0.3, 0.6):
                for inp in ("none", "kp1", "mnist"):
                    for a in ACTS:
                        for s in SEEDS:
                            out.append(("x", (a, n, de_, pa, inp, s)))
    for name, (n, de_, pa, inp) in ANCHORS.items():
        for keep, norm in ((0.0, False), (0.9, True), (0.9, False)):
            for a in ACTS:
                for s in SEEDS:
                    out.append(("x", (a, n, de_, pa, inp, s, keep, norm)))
    for a in ("tri0.3", "wta", "tanh", "sign"):
        for s in STOCK_SEEDS:
            out.append(("stock", (a, s)))
    return out


def run(job):
    kind, args = job
    try:
        return cell(*args) if kind == "x" else stock_cell(*args)
    except Exception as e:  # report, never drop silently
        return {"tag": f"{kind} {args}", "status": f"ERROR {type(e).__name__}: {e}"}


def line(r):
    if r["status"] != "ok":
        return f"{r['tag']}: {r['status']}"
    g = r["graph"]
    kin = g["kin"]
    return (f"{r['tag']}: c1 {r['c1']:.3f} c10 {r['c10']:.3f} static {r['static']:.3f} "
            f"fac1 {r['fac1']:.3f} ac1_39 {r['ac1_39']:.3f} sent static {r['s_static']:.3f} "
            f"level {r['s_level']:.3f} lock {r['lock']:.2f} "
            f"move {r['move']:.3g} period {r['period']} dead {r['dead']:.2f} | "
            f"kin {sum(kin) / len(kin):.2f} cyc {sum(g['cyc']) / len(g['cyc']):.2f} "
            f"gscc {g['gscc']:.2f} in {len(g['direct'])}/{len(g['reach'])} | {r['secs']}s")


def report():
    """tables from wiringlock52.json; printed and written beside the log"""
    rs = json.load(open(OUT))
    ok = [r for r in rs if r["status"] == "ok"]
    bad = [r for r in rs if r["status"] != "ok"]
    out = []
    p = out.append

    def parse(tag):
        a, n, de_, pa, inp, s, keep, norm = tag.split("|")
        return dict(act=a, n=int(n), de=float(de_), pa=float(pa), inp=inp, seed=int(s),
                    keep=float(keep), norm=norm == "1")

    main_ = [dict(parse(r["tag"]), r=r) for r in ok if not r["tag"].startswith("STOCK")]
    grid = [c for c in main_ if c["keep"] == 0.0 and c["norm"]]
    mean = lambda v: sum(v) / len(v) if v else float("nan")

    p(f"cells {len(rs)}: ok {len(ok)}, not ok {len(bad)}")
    for r in bad:
        p(f"  {r['tag']}: {r['status']}")

    p("\nA. host check, example 16's network, seeds pooled (stock: src BareAgt seeds 0-3; "
      "harness: BareAgtXC seeds 50-53)")
    for a in ("tri0.3", "wta", "tanh", "sign"):
        st = [r["c1"] for r in ok if r["tag"].startswith(f"STOCK {a}|")]
        hx = [c["r"]["c1"] for c in grid if c["act"] == a and (c["n"], c["de"], c["pa"], c["inp"])
              == ANCHORS["ex16"]]
        p(f"  {a:7} stock c1 {min(st):.3f}..{max(st):.3f}   harness c1 {min(hx):.3f}..{max(hx):.3f}")

    p("\nB. per activation over the main grid (keep 0, norm on; 36 wirings x 3 inputs... "
      "i.e. 2 n_cols x 2 dist_exp x 3 p_a x 3 inputs x 4 seeds = 144 cells each)")
    p(f"  {'act':8}{'static mean':>12}{'min':>7}{'max':>7}{'|fac1| max':>11}{'sent static':>12}"
      f"{'sent level':>11}{'lock>=.5':>9}{'period':>8}{'dead':>6}")
    for a in ACTS:
        cs = [c["r"] for c in grid if c["act"] == a]
        if not cs:
            continue
        st = [r["static"] for r in cs]
        p(f"  {a:8}{mean(st):12.3f}{min(st):7.3f}{max(st):7.3f}"
          f"{max(abs(r['fac1']) for r in cs):11.3f}{mean([r['s_static'] for r in cs]):12.3f}"
          f"{mean([r['s_level'] for r in cs]):11.3f}{sum(r['lock'] >= 0.5 for r in cs):9d}"
          f"{sum(r['period'] is not None for r in cs):8d}{mean([r['dead'] for r in cs]):6.2f}")

    p("\nC. static by factor, each averaged over the other factors and seeds")
    heads = ([("n", v) for v in (49, 200)] + [("de", v) for v in (1.0, 2.0)]
             + [("pa", v) for v in (0.15, 0.3, 0.6)] + [("inp", v) for v in ("none", "kp1", "mnist")])
    p("  " + f"{'act':8}" + "".join(f"{k}={v}".rjust(10) for k, v in heads))
    for a in ACTS:
        row = []
        for k, v in heads:
            row.append(mean([c["r"]["static"] for c in grid if c["act"] == a and c[k] == v]))
        p("  " + f"{a:8}" + "".join(f"{x:10.3f}" for x in row))

    p("\nD. the graphs (per wiring, seeds pooled; Dir.A among bulk columns), with tri0.3's static")
    p(f"  {'n':>4}{'de':>5}{'pa':>6}{'kin':>6}{'kin0':>6}{'kin1':>6}{'kin2':>6}{'kin3+':>7}"
      f"{'cyc':>6}{'gscc':>6}{'tri0.3 static':>15}{'dead':>6}")
    for n in (49, 200):
        for de_ in (1.0, 2.0):
            for pa in (0.15, 0.3, 0.6):
                cs = [c["r"] for c in grid if (c["n"], c["de"], c["pa"]) == (n, de_, pa)
                      and c["act"] == "tri0.3"]
                ks = [k for r in cs for k in r["graph"]["kin"]]
                cy = [x for r in cs for x in r["graph"]["cyc"]]
                p(f"  {n:4d}{de_:5.0f}{pa:6.2f}{mean(ks):6.2f}"
                  + "".join(f"{mean([k == j for k in ks]):6.2f}" for j in (0, 1, 2))
                  + f"{mean([k >= 3 for k in ks]):7.2f}{mean(cy):6.2f}"
                  f"{mean([r['graph']['gscc'] for r in cs]):6.2f}"
                  f"{mean([r['static'] for r in cs]):15.3f}{mean([r['dead'] for r in cs]):6.2f}")

    p("\nE. per column, pooled over the main grid: static and locked share by in-degree and "
      "by cycle membership (live columns only)")
    for a in ("tri0.3", "step0", "relu", "tanh", "sign", "wta"):
        buckets = {}
        for c in grid:
            if c["act"] != a:
                continue
            r = c["r"]
            for k, cy, lv, st, lk in zip(r["graph"]["kin"], r["graph"]["cyc"], r["col_live"],
                                         r["col_static"], r["col_locked"]):
                if not lv:
                    continue
                for key in (f"kin {min(k, 4)}{'+' if k >= 4 else ''}", "on cycle" if cy else "off cycle"):
                    b = buckets.setdefault(key, [0, 0.0, 0])
                    b[0] += 1
                    b[1] += st
                    b[2] += lk
        keys = sorted(k for k in buckets if k.startswith("kin")) + ["on cycle", "off cycle"]
        p(f"  {a}: " + "; ".join(f"{k} n={buckets[k][0]} static {buckets[k][1] / buckets[k][0]:.3f} "
                                  f"locked {buckets[k][2] / buckets[k][0]:.2f}"
                                  for k in keys if k in buckets))

    p("\nF. the step settings at the two anchors (seeds pooled): static / fac1 / locked share / "
      "cells with a short period / diverged")
    for name, (n, de_, pa, inp) in ANCHORS.items():
        p(f"  {name} (n {n}, dist_exp {de_:.0f}, p_a {pa}, {inp})")
        p(f"    {'act':8}" + "".join(f"{f'keep {k} norm {int(nm)}':>30}" for k, nm in
                                     ((0.0, 1), (0.0, 0), (0.9, 1), (0.9, 0))))
        for a in ACTS:
            row = []
            for keep, nm in ((0.0, True), (0.0, False), (0.9, True), (0.9, False)):
                cs = [c["r"] for c in main_ if (c["act"], c["n"], c["de"], c["pa"], c["inp"],
                                                c["keep"], c["norm"]) == (a, n, de_, pa, inp, keep, nm)]
                dv = sum(1 for r in bad if r["tag"].startswith(f"{a}|{n}|{de_}|{pa}|{inp}|")
                         and r["tag"].endswith(f"|{keep}|{int(nm)}"))
                if cs:
                    row.append(f"{mean([r['static'] for r in cs]):.2f}/{mean([r['fac1'] for r in cs]):+.2f}/"
                               f"{mean([r['lock'] for r in cs]):.2f}/{sum(r['period'] is not None for r in cs)}"
                               f"/{dv}")
                else:
                    row.append(f"all {dv} diverged")
            p(f"    {a:8}" + "".join(f"{x:>30}" for x in row))

    p("\nG. what static tracks, across the main-grid cells: Pearson r with the sending side")
    for key in ("s_static", "s_level"):
        xs = [c["r"][key] for c in grid]
        ys = [c["r"]["static"] for c in grid]
        mx, my = mean(xs), mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        r_ = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        p(f"  static vs sent {key[2:]}: r = {r_:.3f} over {len(xs)} cells")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "wiringlock52_report.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        a, n, de_, pa, inp, s = sys.argv[2:8]
        print(line(cell(a, int(n), float(de_), float(pa), inp, int(s))))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "stock":
        print(line(stock_cell(sys.argv[2], int(sys.argv[3]))))
        return
    js = jobs()
    workers = int(os.environ.get("W52_WORKERS", "20"))
    print(f"{len(js)} cells on {workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(workers) as ex, open(LOG, "a") as fh:
        futs = [ex.submit(run, j) for j in js]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            fh.write(line(r) + "\n")
            fh.flush()
            if k % 100 == 0:
                print(f"{k}/{len(js)}", flush=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    print("done", flush=True)


if __name__ == "__main__":
    main()
