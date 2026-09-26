"""
Does wta need a larger weight init to stay alive? (RW, 2026-09-26: "does
the scale of the weight inits have to be hgiher to compsnesate for the fact
that only 1 qacitaviton i sactive at a time, to avoid the activity dyning
out?")

Premises (every cell is a run 39 cell except for conn_scale):
  host        BareAgtXC (10_sweep_bare_x), 49 bulk columns of 32 units plus
              one input column; p_a 0.3, p_e 0.6, falloff 1/d, input_fanout 3
  learning    frozen on both channels
  input       kp1 (4-cycle of 32-dim symbols), ns comp25, seed 50
  cell        keep 0.9, norm off; 2000 steps, the first 400 discarded
  varied      conn_scale (weight-init scale) in {0.25, 2.0 (default), 16.0},
              for wta and, as the contrast, tri0.3; and input_fanout 3 (run
              39's) against 0, where the input reaches no column (RW,
              2026-09-26: "also the not dying out at smal weight salces mgiht
              be due to teh alwyas active input")
  measures    participation ratio of the summed input per column (run 39's
              estimator); mean |summed input|; the old density (summed input
              > 0.5, the measure behind the superseded 07-28 verdict); how
              often a column's argmax changes; a hash of the argmax sequence

Why wta should not care: argmax only depends on the order within a column,
and with the norm off the whole update (carried-over state, recurrent sum,
input sum) is linear in the weights, so scaling every weight scales every
summed input and keeps the winners.

Result, 2026-09-26 (wtascale46.log): wta PR 16.00 / 15.97 / 15.79, argmax
changes on 11.5-11.6% of steps at all three scales, |state| 0.54 / 4.29 /
34.4 (linear); winner sequences not bit-identical (hashes differ; most likely
rounding differences amplified over 2000 steps, not checked). tri0.3 PR
8.73 / 7.79 / 6.58, |state| growing about 19x per 8x in scale, which is
scale^(1/(1 - 0.3)). The old >0.5 density fell from 0.46 to 0.22 for wta at
scale 0.25 with the dynamics unchanged.

Usage:
    venv/bin/python experiments/46_wta_scale.py              # the grid, 12 cells in parallel
    venv/bin/python experiments/46_wta_scale.py wta 0.9 off 2.0 0   # one cell (last: input_fanout)
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

SEED = 50
GRID = [(act, 0.9, False, s, f) for f in (3, 0) for act in ("wta", "tri0.3")
        for s in (0.25, 2.0, 16.0)]
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wtascale46.log")


def cell(act, keep, norm, scale, fanout=3):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    r39._patch(act, keep, norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **dict(r39.arm_cfg(act), conn_scale=scale, input_fanout=fanout))
    agt = sweep.BareAgtXC(cfg, os.path.join(ROOT, "saves", "wtascale46", f"{act}_{scale}_f{fanout}"),
                          seed=SEED, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    rows, wins = [], []
    for step, a in enumerate(de.kp1_cycle(r39.STEPS, seed=de.derive_seed("kp1", r39.NS, SEED))):
        agt.step([a])
        s = r39.state_of(agt)
        if not torch.isfinite(s).all():
            return f"{act:6} scale {scale:5} fanout {fanout}: DIVERGED at step {step}"
        if step >= r39.BURN:
            rows.append(s.clone())
            wins.append(torch.stack([agt.cols[loc].nr_1.actual.argmax() for loc in bulk]))
    X = torch.stack(rows).float()
    pr, _ = r39._participation(X, cfg.d_col)
    W = torch.stack(wins)
    switch = float((W[1:] != W[:-1]).float().mean())
    return (f"{act:6} keep {keep} norm {'on' if norm else 'off'} fanout {fanout} scale {scale:5}: "
            f"PR {pr:6.2f}  mean|state| {float(X.abs().mean()):9.4f}  "
            f"summed>0.5 {float((X > 0.5).float().mean()):.3f}  argmax switch {switch:.3f}  "
            f"winner-seq hash {hash(tuple(W.flatten().tolist())) % 10**8}")


def main():
    if len(sys.argv) in (5, 6):
        print(cell(sys.argv[1], float(sys.argv[2]), sys.argv[3] == "on", float(sys.argv[4]),
                   int(sys.argv[5]) if len(sys.argv) == 6 else 3))
        return
    with ProcessPoolExecutor(len(GRID)) as ex:
        lines = list(ex.map(cell, *zip(*GRID)))
    with open(LOG, "a") as f:
        for line in lines:
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
