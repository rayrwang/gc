"""
The keep sweep with the carry-over gain removed (2026-09-29)

RW, 2026-09-29 about 02:30 EDT, "ok" to: rerun the norm-off cells the audit found
affected, with the (1 - keep) factor, at keep 0, 0.25, 0.5, 0.75 and 0.9, with 49's
and 53's measures.

Why. The code carries keep * x into the next step and then adds the full new input,
so a column's state settles at input / (1 - keep): 2x at keep 0.5, 10x at 0.9. keep
sets loudness as well as memory. With the RMS norm on and activity above its 0.9
floor the extra scale divides out; with the norm off, or under the floor, it does
not, so every norm-off result at keep above 0 for an activation whose behaviour
depends on scale carries it (gc-project section 4, the 09-29 audit; 3.10 item 18):
run 39's norm-off polarization, 49's norm-off growth rates, 52's keep-0.9 norm-off
side grid, 53's norm-off cells. 55b and 56 cover keep 0.5 and 0.9 under fatigue and
for die-out; this is the plain sweep, both forms side by side in one instrument, so
no comparison crosses experiments (ledger #40).

Premises: 52's harness (BareAgtXC, learning frozen on both channels, weight scale
2.0, 1000 steps, the last 300 read), its two anchors:
  ex16    200 bulk columns, dist_exp 2, p_a 0.3, passive MNIST (example 16's network)
  run39   49 bulk columns, dist_exp 1, p_a 0.3, kp1 symbols into 3 columns
varied:
  keep and form: keep 0 (one form; the factor is 1), and keep 0.25, 0.5, 0.75, 0.9
          each as coded ("code": keep * x + the full input) and with the factor
          ("leak": keep * x + (1 - keep) * the input, the recurrent contributions
          and the input column's alike, so keep = e^(-dt/tau) and the steady level
          is the input's)
  arms:   scale-dependent, norm on and off: tanh, sig4_0, sig8_0, step1 (fc.spike),
          tri0.3, tri0.7
          unbounded, norm on and off: raw, relu, tri1.0, kwta4
          order-only, norm off, the control: sign, step0, cstep, wta
  seeds 50 to 53
24 arms x 9 keep-forms x 2 networks x 4 seeds = 1,728 cells.

Measures: 55's (52's c1, c10, static, fac1, the sent static and level, lock, move,
period, dead, run 39's ac1_39 and tau_39; 53's c50, fac10, fac50, frozen, nbin; the
participation ratio; osc; sat for the bounded activations, sig8_0 and cstep
included), plus 49's growth rate: a deepcopy twin of the whole agent at step 600,
its bulk activity displaced by a random vector of relative size 1e-6 and its
accumulator by keep times that (times each column's rms), both driven by the same
input; the rate is the least-squares slope of ln(relative distance) against step
over the first unbroken stretch with the distance between 1e-12 and 1e-2, after a
5-step settling window, from at least 3 points (49's rule; one direction per seed).
Also a finite-size rate, the same fit with the window's top at 0.1, and the
distances after 1, 10 and 50 steps (d1, d10, d50): on example 16's network the
smoke found tri0.3's distance jumping from 1e-6 to about 2e-3 in one step (the
triangle's slope is infinite at its cutoff), so 49's window catches nothing there.
The twin stops once the distance leaves (1e-12, 0.3).

Expected (Claude's, written before the run):
- order-only arms: an order-only activation gives the same output for x and c * x, so
  "leak" is "code" scaled by (1 - keep) at every step; every scale-free measure (c1,
  c10, c50, static, fac, lock, move, period, sent level, pr, growth rate) should
  match between the forms, likely exactly; frozen and nbin use a fixed 0.5 cut and
  may differ. A mismatch means the wrapper is wrong.
- norm on above the floor: the forms match in their statistics (not number for
  number: chaotic cells separate); under the floor they differ.
- scale-dependent, norm off, keep > 0: "leak" looks like keep 0 in loudness and like
  "code" in slowness: tanh and the sigmoids saturate less, step1 fires less, tri0.3
  and tri0.7 send less; frozen (a fixed 0.5 cut) may rise, since more units stay
  under 0.5 without ever crossing, so read it beside static and lock; tri0.3's growth
  rate at keep 0.9 stays near zero and positive (slow chaos), not sure.
- unbounded, norm off: diverge at every keep in both forms (without the gain, keep > 0
  has keep 0's steady gain, which already diverges); with the norm on they live.

Smoke, 2026-09-29 about 02:45 EDT (seven cells on run 39's network, one on 16's): wta's
code and leak forms match in statistics, not exactly (static 0.200 and 0.212,
c50 0.122 and 0.133): rounding differs, a near-tie winner flips and the runs
separate, so the first expectation above holds only in distribution; tri0.3 at
keep 0.9, norm off, grows at +0.034 coded and +0.030 with the factor (49: +0.03);
raw at keep 0.5 with the factor and the norm off diverges at step 70.

Output: outputs/57_keep_sweep_leak.log, .jsonl and .json; tables in 57_report.md.
Usage:
    venv/bin/python experiments/57_keep_sweep_leak.py          # all cells, 16 workers
    venv/bin/python experiments/57_keep_sweep_leak.py smoke    # a few cells, printed
"""

import copy
import importlib
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

m55 = importlib.import_module("55_fatigue_lock")
m52, r39, sweep, de, T = m55.m52, m55.r39, m55.sweep, m55.de, m55.T
ROOT = m55.ROOT
STEPS, WINDOW = m52.STEPS, m52.WINDOW
LOG = os.path.join(HERE, "outputs", "57_keep_sweep_leak.log")
OUT = os.path.join(HERE, "outputs", "57_keep_sweep_leak.json")
SEEDS = (50, 51, 52, 53)
KEEP_FORMS = [(0.0, "code")] + [(k, f) for k in (0.25, 0.5, 0.75, 0.9) for f in ("code", "leak")]
ARMS = ([(a, nm) for a in ("tanh", "sig4_0", "sig8_0", "step1", "tri0.3", "tri0.7") for nm in (True, False)]
        + [(a, nm) for a in ("raw", "relu", "tri1.0", "kwta4") for nm in (True, False)]
        + [(a, False) for a in ("sign", "step0", "cstep", "wta")])
BOUNDED = tuple(m55.BOUNDED) + ("sig8_0", "cstep")
KICK, EPS, SETTLE = 600, 1e-6, 5

_PART = r39._participation


def _safe_participation(X, d_col):
    try:
        return _PART(X, d_col)
    except Exception:  # eigh failed on near-blown-up states in 4 of 55's cells
        return float("nan"), float("nan")


def growth(cur, hi=1e-2):
    """49's fit: slope of ln(distance) over the first unbroken stretch in (1e-12, hi)."""
    run, prev = [], None
    for k, v in enumerate(cur):
        if k < SETTLE or not (1e-12 < v < hi):
            if run:
                break
            continue
        if prev is not None and k != prev + 1:
            break
        run.append((k, math.log(v)))
        prev = k
    if len(run) < 3:
        return None, len(run)
    xs, ys = [k for k, _ in run], [y for _, y in run]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx, len(run)


def cell(net, act, norm, keep, form, seed):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    r39._participation = _safe_participation
    n_cols, dist_exp, p_a, inp = m52.ANCHORS[net]
    r39._patch(act, keep, norm)  # resets the transport and the commit from pristine
    leak = form == "leak" and keep > 0
    if leak:
        from experiments import dire_hosts as dh
        cur = dh.BareAgtX._transport
        dh.BareAgtX._transport = lambda self, x, w, _c=cur, _k=keep: (1.0 - _k) * _c(self, x, w)
    dim = 784 if inp == "mnist" else de.SYMBOL_DIM
    fan = {"none": 0, "kp1": 3, "mnist": -1}[inp]
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(dim)], n_cols=n_cols,
                          **dict(r39.arm_cfg(act), p_a=p_a, dist_exp=dist_exp, input_fanout=fan))
    tag = f"{net}|{act}|{int(norm)}|{keep}|{form}|{seed}"
    path = os.path.join(ROOT, "saves", "keepleak57", f"w{os.getpid()}")
    agt = sweep.BareAgtXC(cfg, path, seed=seed, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    xs = m52.stream(inp, seed)
    if leak:  # the input column sends raw, outside the transport, so scale it here
        xs = [(1.0 - keep) * x for x in xs]
    rows, twin, dist = [], None, []
    t0 = time.time()
    for t in range(STEPS):
        agt.step([xs[t]])
        if twin is not None:
            twin.step([xs[t]])
            sa, sb = r39.state_of(agt), r39.state_of(twin)
            v = float((sa - sb).norm() / (sa.norm() + 1e-12))
            dist.append(v)
            if len(dist) > SETTLE and not (1e-12 < v < 0.3):
                twin = None  # past both fit windows: nothing more to learn from it
        if t == KICK:
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
                    i += nn_
        if t >= STEPS - WINDOW or t % 10 == 0:
            s = r39.state_of(agt)
            if not torch.isfinite(s).all() or float(s.abs().max()) > 1e30:
                return {"tag": tag, "status": "DIVERGED", "step": t}
            if t >= STEPS - WINDOW:
                rows.append(s.clone())
    X = torch.stack(rows).float()
    f = r39.act_fn(act)
    m = m55.measure(X, cfg.d_col, f, act)
    if act in BOUNDED:
        m["sat"] = float((f(X).abs() >= 0.99).float().mean())
    m["rate"], m["rate_pts"] = growth(dist)
    m["rate_fs"], m["rate_fs_pts"] = growth(dist, 0.1)
    for k in (1, 10, 50):
        m[f"d{k}"] = dist[k - 1] if len(dist) >= k else None
    m["dist_final"] = dist[-1] if dist else None
    m["level"] = float(X.abs().mean())  # the received state's size, to see the gain
    return {"tag": tag, "status": "ok", "secs": round(time.time() - t0, 1), **m}


def run(job):
    try:
        return cell(*job)
    except Exception as e:  # report, never drop silently
        return {"tag": "|".join(map(str, job)), "status": f"ERROR {type(e).__name__}: {e}"}


def fmt(v, spec=".3f"):
    return "None" if v is None or (isinstance(v, float) and v != v) else format(v, spec)


def line(r):
    if r["status"] != "ok":
        return f"{r['tag']}: {r['status']}" + (f" at step {r['step']}" if "step" in r else "")
    return (f"{r['tag']}: level {fmt(r['level'], '.3g')} slevel {fmt(r.get('s_level'))} "
            f"static {fmt(r['static'])} c1 {fmt(r['c1'])} c50 {fmt(r['c50'])} "
            f"fac1 {fmt(r['fac1'])} fac50 {fmt(r['fac50'])} lock {fmt(r['lock'], '.2f')} "
            f"move {fmt(r['move'], '.3g')} period {r['period']} frozen {fmt(r['frozen'], '.2f')} "
            f"pr {fmt(r['pr'], '.1f')} sat {fmt(r['sat'])} dead {fmt(r['dead'], '.2f')} "
            f"rate {fmt(r['rate'], '+.3f')} ({r['rate_pts']} pts) rate_fs {fmt(r['rate_fs'], '+.3f')} "
            f"({r['rate_fs_pts']} pts) d1 {fmt(r['d1'], '.1e')} | {r['secs']}s")


def jobs():
    # the two networks interleaved, so every chunk mixes slow and fast cells
    return [(net, a, nm, k, f, s) for s in SEEDS for a, nm in ARMS for k, f in KEEP_FORMS
            for net in m52.ANCHORS]


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for job in (("run39", "tri0.3", False, 0.9, "code", 50),
                    ("run39", "tri0.3", False, 0.9, "leak", 50),
                    ("run39", "wta", False, 0.9, "code", 50),
                    ("run39", "wta", False, 0.9, "leak", 50),
                    ("run39", "sig8_0", True, 0.5, "leak", 50),
                    ("run39", "cstep", False, 0.25, "leak", 50),
                    ("run39", "raw", False, 0.5, "leak", 50)):
            print(line(run(job)), flush=True)
        return
    import pool_runner
    pool_runner.run_jobs(jobs(), run, OUT.replace(".json", ".jsonl"), LOG, line, 16, 40, OUT)


if __name__ == "__main__":
    main()
