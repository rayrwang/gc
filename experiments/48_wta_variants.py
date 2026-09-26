"""
Four variants of wta against wta itself (RW, 2026-09-26: "i mean what if when
all zero it sends zero, toherwise sends 1 at the argmax" and "also try sending
the argmax itself rather than 1, with bothteh raw argmax, and clipped to 0 and
1").

Activations, per 32-unit column, applied to what each unit sends:
  wta      1.0 at the argmax, 0 elsewhere (run 39's; an all-equal column,
           e.g. all zero, sends 1.0 from unit 0, torch's first index)
  wta0     as wta, but an all-zero column sends nothing
  wtapos   as wta, but nothing unless the column's maximum is positive:
           the version that can fall silent
  wtaraw   the winner's own summed input at the argmax (unbounded, signed)
  wtaclip  the winner's summed input clipped to [0, 1] at the argmax

Premises (run 39 cells otherwise):
  host        BareAgtXC (10_sweep_bare_x), 49 bulk columns of 32 units plus
              one input column; p_a 0.3, p_e 0.6, falloff 1/d, input_fanout 3,
              conn_scale 2.0
  learning    frozen on both channels
  input       kp1 (4-cycle of 32-dim symbols), ns comp25
  varied      activation (5) x keep {0, 0.5, 0.9} x norm {off, on} x seed
              {50, 51}; 2000 steps, the first 400 discarded
  measures    pr_in: participation ratio of the summed input per column (run
              39's estimator); pr_out: the same on what units send; send:
              fraction of column-steps that send anything; nz: fraction of
              units sending a nonzero value; switch: how often a column's
              argmax changes; |state|: mean absolute summed input; status
              (DIVERGED if anything goes non-finite)

Usage:
    venv/bin/python experiments/48_wta_variants.py            # the grid, in parallel
    venv/bin/python experiments/48_wta_variants.py wtapos 0.9 off 50   # one cell
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

ACTS = ("wta", "wta0", "wtapos", "wtaraw", "wtaclip")
KEEPS = (0.0, 0.5, 0.9)
NORMS = (False, True)
SEEDS = (50, 51)
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wtavar48.log")


def variant(name):
    def at_argmax(x, value):
        o = torch.zeros_like(x)
        o[int(torch.argmax(x))] = value
        return o
    if name == "wta":
        return lambda x: at_argmax(x, 1.0)
    if name == "wta0":
        return lambda x: torch.zeros_like(x) if bool((x == 0).all()) else at_argmax(x, 1.0)
    if name == "wtapos":
        return lambda x: at_argmax(x, 1.0) if float(x.max()) > 0 else torch.zeros_like(x)
    if name == "wtaraw":
        return lambda x: at_argmax(x, float(x.max()))
    if name == "wtaclip":
        return lambda x: at_argmax(x, min(max(float(x.max()), 0.0), 1.0))
    raise ValueError(name)


def cell(act, keep, norm, seed):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    f = variant(act)
    r39.act_fn = lambda name, f=f: f       # _patch installs f as the transport
    r39._patch(act, keep, norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **r39.arm_cfg("wta"))
    agt = sweep.BareAgtXC(cfg, os.path.join(ROOT, "saves", "wtavar48", f"{act}_{keep}_{norm}_{seed}"),
                          seed=seed, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    xin, xout, wins = [], [], []
    tag = f"{act:7} keep {keep:<4} norm {'on ' if norm else 'off'} seed {seed}"
    for step, a in enumerate(de.kp1_cycle(r39.STEPS, seed=de.derive_seed("kp1", r39.NS, seed))):
        agt.step([a])
        s = r39.state_of(agt)
        if not torch.isfinite(s).all():
            return f"{tag}: DIVERGED at step {step}"
        if step >= r39.BURN:
            xin.append(s.clone())
            xout.append(torch.cat([f(agt.cols[loc].nr_1.actual) for loc in bulk]))
            wins.append(torch.stack([agt.cols[loc].nr_1.actual.argmax() for loc in bulk]))
    Xi, Xo, W = torch.stack(xin).float(), torch.stack(xout).float(), torch.stack(wins)
    pr_in, _ = r39._participation(Xi, cfg.d_col)
    pr_out, _ = r39._participation(Xo, cfg.d_col)
    per_col = Xo.view(Xo.shape[0], len(bulk), cfg.d_col)
    send = float((per_col != 0).any(-1).float().mean())
    nz = float((Xo != 0).float().mean())
    switch = float((W[1:] != W[:-1]).float().mean())
    return (f"{tag}: pr_in {pr_in:6.2f}  pr_out {pr_out:6.2f}  send {send:.3f}  nz {nz:.3f}  "
            f"switch {switch:.3f}  |state| {float(Xi.abs().mean()):9.4f}")


def main():
    if len(sys.argv) == 5:
        print(cell(sys.argv[1], float(sys.argv[2]), sys.argv[3] == "on", int(sys.argv[4])))
        return
    jobs = [(a, k, n, s) for a in ACTS for n in NORMS for k in KEEPS for s in SEEDS]
    with ProcessPoolExecutor(20) as ex:
        lines = list(ex.map(cell, *zip(*jobs)))
    with open(LOG, "a") as fh:
        for line in lines:
            print(line)
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
