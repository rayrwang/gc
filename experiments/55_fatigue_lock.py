"""
Fatigue against the fixed-shape lock and against real settling (2026-09-28)

RW's ask, 2026-09-28 about 23:45 EDT: "ok see how all this ineteracts with the
fatigue./adaptiion. try multipel fatugie/adpation timescales & other configus like
how to impelment it idk what you know this better. cross it with the existing axes
to vary, liek activation function and carry and others i forgot. see how fatigue
affects both looking locked and really setting". And: "did all of these runs ...
include the other stats liek parition partio, etc...?" (52 to 54 did not record
the participation ratio or a chaos measure; this run does.)

Background (gc-project 4, 3.10 item 13): a frozen network looks locked through a
fixed shape, the activation's constant level (52); it really settles when a smooth
activation sits where its slope times the weights is small, by a shallow curve,
enough carry-over, or falling silent (53, 54). Fatigue, per-unit adaptation over
time (3.10 item 13), is the textbook learning-free brake on a locked pattern
(Kandel PDF p. 835: a half-centre "released from inhibition through some sort of
synaptic or neuronal fatigue").

Premises: 52's harness (BareAgtXC from 10_sweep_bare_x, learning frozen, the
activation applied on the sending side), at 52's two anchor networks: ex16 (200
columns, dist_exp 2, p_a 0.3, passive MNIST, the same 1000 images) and run39 (49
columns, dist_exp 1, p_a 0.3, kp1 into 3 columns). 1000 steps, the last 300 read.
Varied:
- activation: raw, relu, tri1.0, kwta4, tri0.3, sig4_0, tanh, sign, step0, step1
  (= fc.spike), wta;
- keep (carry-over): 0, 0.5, 0.9;
- the RMS norm: on for every arm; off as well for the scale-dependent ones (tanh,
  sig4_0, step1, tri0.3), as in 53, and for the unbounded ones (raw, relu, tri1.0,
  kwta4), which diverge without it at keep 0 (52), so that fatigue can be set
  against the norm as the bound (RW, 2026-09-28: "also do these runs still include
  the unbouned activaitons? since can also test rms norm vs fatigue on them too
  rihgt? though mabye fagiutie will be insufficinet to keep them bounded if the lag
  is too slow"); the order-only ones (sign, step0, wta) are unaffected by it;
- fatigue, per unit, applied in each bulk column's commit after the norm, so it
  changes what the unit sends and what is measured, and not the carried
  accumulator:
  none;
  sub, subtractive adaptation of the drive: f <- f + (u - f)/tau, u_sent = u - g f
    (a high-pass filter on each unit's input; g 0.5 or 1.0). The same as a threshold
    that rises with recent drive;
  div, divisive gain control: f <- f + (|u| - f)/tau, u_sent = u / (1 + g f) (g 0.5
    or 2.0);
  dep, synaptic depression on the sending side (Tsodyks-Markram, per unit):
    r <- r + (1 - r)/tau - U r min(|y|, 1), with y the unit's output, and every
    outgoing connection carries y r (U 0.1 or 0.5);
  with tau in steps 3, 10, 30, 100, 300;
- seeds 50 and 51.
Measured on the window: 52 and 53's lock measures (c1, c10, c50, static, fac1,
fac10, fac50, the share of live columns moving under 1% a step = real settling,
move, period, dead, sent level and sent static, frozen, nbin, run 39's ac1), plus:
pr, run 39's participation ratio (per column, averaged); sat, the share of sent
values at |y| >= 0.99 (bounded activations); osc, the most negative median
changing-part autocorrelation over lags 1 to 50 and its lag (oscillation or
alternation); sens10 and sens50, a finite-perturbation chaos check: at step 939 a
deep copy is displaced by 1e-4 of the state (and of the carried accumulator, as in
49) and both run 50 steps on the same input, log10 of the relative separation over
its start (above 0 grows, below shrinks; -99 when the twin rejoins the host exactly,
as wta and sign do at keep 0 when no winner flips); fat, the size of the adaptation
at the end, pooled over bulk columns (sub: sum |g f| over sum |u|; div: mean g f;
dep: mean r).
For dep the sent-side measures ignore r (they apply the activation to the state).

Expected (Claude's, written before the run):
1. sub removes the fixed shape: static for the positive-level activations falls
   toward the low-level ones' values when g = 1 and tau is long against the
   dynamics, less at g 0.5; at tau 3 it turns units into change detectors.
2. sub and dep stop real settling where it happened (tri0.3 at keep 0.9, relu and
   tri1.0 from 0.5, sig4 with the norm at keep 0), or turn it into slow oscillation
   or switching at tau near the tens (the half-centre); sens rises.
3. div rescales and keeps the shape, like a per-unit RMS norm over time: small
   effect on static, some on settling.
4. The discontinuous activations (wta, sign, steps) switch more under sub and dep
   (winners rotate), with osc showing a period tied to tau.
5. pr rises where the fixed shape falls, but not in lockstep, since pr removes the
   time average.
6. Unbounded activations with the norm off: div bounds them when tau is short
   against their growth (a unit's gain falls as its own recent size rises); sub and
   dep do not (a growing drive minus its own slow average still grows, and
   depression caps usage at 1 per step while the output is unbounded); a slow tau
   lets them escape before the adaptation catches up. The step of divergence is
   recorded.

Output: outputs/55_fatigue_lock.log and .json (the first launch was killed before its JSON
was written; 55r and 55f hold the full records); tables in 55_report.md.
Usage:
    venv/bin/python experiments/55_fatigue_lock.py           # all cells, in parallel
    venv/bin/python experiments/55_fatigue_lock.py smoke     # six cells, printed
    venv/bin/python experiments/55_fatigue_lock.py report    # tables from the JSON
"""

import copy
import importlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

m52 = importlib.import_module("52_wiring_lock")  # also sets ROOT on sys.path and chdir
r39, sweep, de, T = m52.r39, m52.sweep, m52.de, m52.T
ROOT = m52.ROOT

STEPS, WINDOW = m52.STEPS, m52.WINDOW
SEEDS = (50, 51)
KEEPS = (0.0, 0.5, 0.9)
TAUS = (3, 10, 30, 100, 300)
FATIGUE = ([("none", 0, 0.0)]
           + [("sub", tau, g) for tau in TAUS for g in (0.5, 1.0)]
           + [("div", tau, g) for tau in TAUS for g in (0.5, 2.0)]
           + [("dep", tau, u) for tau in TAUS for u in (0.1, 0.5)])
ARMS = ([(a, nm) for a in ("raw", "relu", "tri1.0", "kwta4") for nm in (True, False)]
        + [(a, True) for a in ("sign", "step0", "wta")]
        + [(a, nm) for a in ("tanh", "sig4_0", "step1", "tri0.3") for nm in (True, False)])
BOUNDED = ("tanh", "sig4_0", "sign", "step0", "step1")
EPS, SENS_AT, SENS_H = 1e-4, 939, 50
LOG = os.path.join(HERE, "outputs", "55_fatigue_lock.log")
OUT = os.path.join(HERE, "outputs", "55_fatigue_lock.json")


def install(act, kind, tau, g):
    """Wrap BareCol's commit (already reset to pristine or norm-off by r39._patch
    for this cell) with the fatigue, and for dep the transport as well. Re-done
    from scratch every cell, since the pool reuses its workers."""
    import src.agents as ag
    from experiments import dire_hosts as dh
    base = ag.BareCol.update_activations
    f = r39.act_fn(act)
    if kind == "none":
        return

    def commit(self, *a, **k):
        base(self, *a, **k)
        u = self.nr_1.actual
        if kind == "sub":
            fs = self.__dict__.get("_fat")
            fs = torch.zeros_like(u) if fs is None else fs
            fs = fs + (u - fs) / tau
            self._fat = fs
            self.nr_1.actual = u - g * fs
            self._fatsize = (float((g * fs).abs().sum()), float(u.abs().sum()))
        elif kind == "div":
            fs = self.__dict__.get("_fat")
            fs = torch.zeros_like(u) if fs is None else fs
            fs = fs + (u.abs() - fs) / tau
            self._fat = fs
            self.nr_1.actual = u / (1 + g * fs)
            self._fatsize = (float((g * fs).sum()), float(fs.numel()))
        elif kind == "dep":
            r = self.__dict__.get("_res")
            r = torch.ones_like(u) if r is None else r
            y = f(u).abs().clamp(max=1.0)
            r = (r + (1 - r) / tau - g * r * y).clamp(0.0, 1.0)
            self._res = r
            self.nr_1.actual._res = r  # the transport finds it on the sender's tensor
            self._fatsize = (float(r.sum()), float(r.numel()))

    ag.BareCol.update_activations = commit
    if kind == "dep":
        def transport(self, x, w):
            y = f(x)
            r = getattr(x, "_res", None)
            return (y if r is None else y * r) @ w
        dh.BareAgtX._transport = transport


def measure(X, d_col, f, act):
    m = m52.measure(X, d_col, f)
    S, n = X.shape[0], X.shape[1] // d_col
    C = X.view(S, n, d_col)
    live = C.abs().amax(dim=(0, 2)) > 1e-9

    def corr(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)

    def med(v):
        return float(v[live].median()) if bool(live.any()) else float("nan")

    m["c50"] = float(corr(C[:-50], C[50:]).mean(0).median())
    F = C - C.mean(0)
    den = (F * F).sum((0, 2)).clamp(min=1e-24)
    facs = [med((F[:-lag] * F[lag:]).sum((0, 2)) / den) for lag in range(1, 51)]
    m["fac10"], m["fac50"] = facs[9], facs[49]
    k = min(range(50), key=lambda i: facs[i] if facs[i] == facs[i] else 9.0)
    m["osc"], m["osc_lag"] = facs[k], k + 1
    B = X > 0.5
    m["frozen"] = float((B == B[:1]).all(0).float().mean())
    m["nbin"] = len({bytes(row.numpy().tobytes()) for row in B})
    m["pr"] = r39._participation(X, d_col)[0]
    if act in BOUNDED:
        m["sat"] = float((f(X).abs() >= 0.99).float().mean())
    else:
        m["sat"] = None
    for key in ("col_locked", "col_live", "col_move", "col_static", "col_fac1"):
        m.pop(key, None)  # per-column lists would make the JSON huge
    return m


def cell(net, act, norm, keep, kind, tau, g, seed):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    n_cols, dist_exp, p_a, inp = m52.ANCHORS[net]
    r39._patch(act, keep, norm)
    install(act, kind, tau, g)
    dim = 784 if inp == "mnist" else de.SYMBOL_DIM
    fan = {"none": 0, "kp1": 3, "mnist": -1}[inp]
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(dim)], n_cols=n_cols,
                          **dict(r39.arm_cfg(act), p_a=p_a, dist_exp=dist_exp, input_fanout=fan))
    tag = f"{net}|{act}|{int(norm)}|{keep}|{kind}|{tau}|{g}|{seed}"
    path = os.path.join(ROOT, "saves", "fatiguelock55", f"w{os.getpid()}")
    agt = sweep.BareAgtXC(cfg, path, seed=seed, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    xs = m52.stream(inp, seed)
    rows, twin, sep = [], None, {}
    t0 = time.time()
    for t in range(STEPS):
        agt.step([xs[t]])
        if twin is not None:
            twin.step([xs[t]])
            k = t - SENS_AT
            if k in (10, SENS_H):
                sa, sb = r39.state_of(agt), r39.state_of(twin)
                sep[k] = float((sa - sb).norm() / (sa.norm() + 1e-12))
                if k == SENS_H:
                    twin = None
        if t == SENS_AT:
            s = r39.state_of(agt)
            if torch.isfinite(s).all() and float(s.norm()) > 0:
                twin = copy.deepcopy(agt)
                gen = torch.Generator().manual_seed(seed)
                d = torch.randn(s.shape, generator=gen)
                d = d / d.norm() * EPS * s.norm()
                i = 0
                for loc in bulk:
                    c, hc = twin.cols[loc], agt.cols[loc]
                    nn_ = c.nr_1.actual.numel()
                    dd = d[i:i + nn_].view_as(c.nr_1.actual)
                    c.nr_1.actual = hc.nr_1.actual + dd
                    c.nr_1_.actual = hc.nr_1_.actual + keep * dd * hc.nr_1.rms_avg
                    if kind == "dep":
                        c.nr_1.actual._res = c._res
                    i += nn_
        if t >= STEPS - WINDOW or t % 10 == 0:
            s = r39.state_of(agt)
            if not torch.isfinite(s).all() or float(s.abs().max()) > 1e30:
                return {"tag": tag, "status": "DIVERGED", "step": t}
            if t >= STEPS - WINDOW:
                rows.append(s.clone())
    m = measure(torch.stack(rows).float(), cfg.d_col, r39.act_fn(act), act)
    for k in (10, SENS_H):
        v = sep.get(k)
        m[f"sens{k}"] = (math.log10(v / EPS) if v and v > 0 else (-99.0 if v == 0 else None))
    sizes = [c.__dict__.get("_fatsize") for loc, c in agt.cols.items() if loc in bulk]
    sizes = [v for v in sizes if v is not None]
    m["fat"] = (sum(a for a, _ in sizes) / max(sum(b for _, b in sizes), 1e-12)) if sizes else None
    return {"tag": tag, "status": "ok", "secs": round(time.time() - t0, 1), **m}


def run(job):
    try:
        return cell(*job)
    except Exception as e:  # report, never drop silently
        return {"tag": "|".join(map(str, job)), "status": f"ERROR {type(e).__name__}: {e}"}


def fmt(v, spec=".3f"):
    return "None" if v is None else format(v, spec)


def line(r):
    if r["status"] != "ok":
        return f"{r['tag']}: {r['status']}" + (f" at step {r['step']}" if "step" in r else "")
    return (f"{r['tag']}: static {fmt(r['static'])} sstatic {fmt(r['s_static'])} "
            f"c1 {fmt(r['c1'])} c50 {fmt(r['c50'])} fac1 {fmt(r['fac1'])} "
            f"fac50 {fmt(r['fac50'])} lock {fmt(r['lock'], '.2f')} move {fmt(r['move'], '.3g')} "
            f"period {r['period']} osc {fmt(r['osc'])}@{r['osc_lag']} pr {fmt(r['pr'], '.1f')} "
            f"sens10 {fmt(r['sens10'], '.2f')} sens50 {fmt(r['sens50'], '.2f')} "
            f"sat {fmt(r['sat'])} dead {fmt(r['dead'], '.2f')} frozen {fmt(r['frozen'], '.2f')} "
            f"nbin {r['nbin']} fat {fmt(r['fat'])} | {r['secs']}s")


def jobs():
    return [(net, a, nm, keep, kind, tau, g, s)
            for net in m52.ANCHORS for a, nm in ARMS for keep in KEEPS
            for kind, tau, g in FATIGUE for s in SEEDS]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for job in (("run39", "tri0.3", True, 0.9, "none", 0, 0.0, 50),
                    ("run39", "tri0.3", True, 0.9, "sub", 30, 1.0, 50),
                    ("run39", "wta", True, 0.0, "dep", 10, 0.5, 50),
                    ("run39", "sig4_0", True, 0.0, "div", 30, 2.0, 50),
                    ("ex16", "tri0.3", True, 0.0, "none", 0, 0.0, 50),
                    ("ex16", "tri0.3", True, 0.0, "sub", 100, 1.0, 50)):
            print(line(run(job)), flush=True)
        return
    js = jobs()
    workers = int(os.environ.get("W55_WORKERS", "20"))
    print(f"{len(js)} cells on {workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(workers) as ex, open(LOG, "a") as fh:
        futs = [ex.submit(run, j) for j in js]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            fh.write(line(r) + "\n")
            fh.flush()
            if k % 200 == 0:
                print(f"{k}/{len(js)}", flush=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    print("done", flush=True)


if __name__ == "__main__":
    main()
