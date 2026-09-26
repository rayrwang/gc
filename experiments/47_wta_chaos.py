"""
Is wta's activity at keep 0.9 chaotic, or does it settle? (RW, 2026-09-26:
"also itnresting that the acti ity is sitll cahoatic even with high kepep".)
Run 39 recorded lyap = NaN for wta at every carry-over (never measured),
period None and 1600 distinct binarised states in 1600 steps.

Premises (run 39 cells):
  host        BareAgtXC (10_sweep_bare_x), 49 bulk columns of 32 units plus
              one input column; p_a 0.3, p_e 0.6, falloff 1/d, input_fanout 3,
              conn_scale 2.0
  learning    frozen on both channels
  input       kp1 (4-cycle of 32-dim symbols), ns comp25, seed 50; the input
              column sends the previous step's symbol
  cells       wta and tri0.3, keep 0.9, norm off; 8000 steps
  test 1      finite perturbations: at step 400 a twin (same weights) is put
              at the host's full state, bulk activity displaced by a random
              vector of relative size eps in {0 (the known-zero check), 1e-6,
              1e-3, 1e-1}, then both are driven by the same input; reported:
              relative state distance and the fraction of columns whose
              argmax differs. Chaos: small differences grow. Contraction:
              they shrink. Multistability: a large kick moves to another
              state and stays there without growing further.
  test 2      whether the host's argmax sequence is periodic over the last
              3000 steps, and with what period (checked up to 1500)
  also        how often a column's argmax changes

Sync: 39's twin_sync copies the bulk columns only. The input column holds
the previous step's symbol, which a never-stepped twin lacks, so a twin
synced with twin_sync alone gets a one-step input kick (6.7% relative for
this wta cell, whatever eps). full_sync below also copies the input column.
Run 39's own Benettin loop re-syncs every step, so there the kick enters
once, at launch: one log-ratio term in 1600, about +0.007 on the exponent.

Usage:
    venv/bin/python experiments/47_wta_chaos.py              # all cells in parallel
    venv/bin/python experiments/47_wta_chaos.py wta 1e-3     # one act and eps, norm off
    venv/bin/python experiments/47_wta_chaos.py tri0.3 1e-6 on   # norm on
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

r39 = importlib.import_module("39_activation_dynamics")
sweep, de, T = r39.sweep, r39.de, r39.T

SEED, KEEP, NORM, STEPS, LAUNCH = 50, 0.9, False, 8000, 400
EPS = (0.0, 1e-6, 1e-3, 1e-1)
ACTS = ("wta", "tri0.3")
REPORT = {401, 402, 405, 410, 420, 450, 500, 1000, 2000, 4000, STEPS - 1}
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wtachaos47.log")


def full_sync(host, twin, bulk, vec):
    """39's twin_sync plus the input column, which it leaves out."""
    r39.twin_sync(host, twin, bulk, vec)
    for loc in host.cols:
        if host.is_i(loc):
            for name in ("nr_1", "nr_1_"):
                d, s = getattr(twin.cols[loc], name), getattr(host.cols[loc], name)
                d.actual, d.expect = s.actual.clone(), s.expect.clone()
                d.avg, d.avg_sq, d.rms_avg = s.avg.clone(), s.avg_sq.clone(), s.rms_avg


def run(act, eps, norm=NORM):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    r39._patch(act, KEEP, norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **r39.arm_cfg(act))
    base = os.path.join(ROOT, "saves", "wtachaos47", f"{act}_{eps}_{norm}")
    A = sweep.BareAgtXC(cfg, base + "_a", seed=SEED, dire=True, frozen=True)
    B = sweep.BareAgtXC(cfg, base + "_b", seed=SEED, dire=True, frozen=True)
    bulk = [loc for loc in A.cols if not A.is_i(loc)]

    def win(g):
        return torch.stack([g.cols[loc].nr_1.actual.argmax() for loc in bulk])

    out, WA = [], []
    for t, a in enumerate(de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", r39.NS, SEED))):
        A.step([a])
        if t == LAUNCH:
            s = r39.state_of(A)
            d = torch.randn_like(s)
            full_sync(A, B, bulk, s + d / d.norm() * eps * s.norm())
        elif t > LAUNCH:
            B.step([a])
        WA.append(win(A))
        if t in REPORT:
            sa, sb = r39.state_of(A), r39.state_of(B)
            out.append(f"{act:6} norm {'on' if norm else 'off'} eps {eps:g} step {t}: rel. distance "
                       f"{float((sa - sb).norm() / sa.norm()):.2e}, columns with a different "
                       f"argmax {float((win(A) != win(B)).float().mean()):.3f}")
    W = torch.stack(WA)
    tail = W[-3000:]
    per = next((p for p in range(1, 1501) if bool((tail[p:] == tail[:-p]).all())), None)
    switch = float((W[LAUNCH + 1:] != W[LAUNCH:-1]).float().mean())
    out.append(f"{act:6} norm {'on' if norm else 'off'} eps {eps:g}: host argmax changes on {switch:.3f} of column-steps; "
               f"argmax sequence over the last 3000 steps: period {per}")
    return "\n".join(out)


def main():
    if len(sys.argv) in (3, 4):
        print(run(sys.argv[1], float(sys.argv[2]), len(sys.argv) == 4 and sys.argv[3] == "on"))
        return
    jobs = [(a, e) for a in ACTS for e in EPS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        blocks = list(ex.map(run, *zip(*jobs)))
    with open(LOG, "a") as f:
        for b in blocks:
            print(b)
            f.write(b + "\n")


if __name__ == "__main__":
    main()
