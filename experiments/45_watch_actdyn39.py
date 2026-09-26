"""
Watch one cell of run 39 (39_activation_dynamics.py) in the pygame debugger
(RW, 2026-09-26: "write up keep 0.9, norm off ... so i can see it").
Default cell: wta, keep 0.9, norm off, seed 50.

The network is built by run 39's own code (act_fn through _patch, arm_cfg,
the kp1 stream and namespace), so the cell watched is the cell measured:
  host        BareAgtXC (10_sweep_bare_x), 49 bulk columns of 32 units plus
              one input column
  wiring      p_a 0.3, p_e 0.6, falloff 1/d, input_fanout 3 (the input
              reaches exactly 3 bulk columns), weight scale 2.0
  learning    frozen on both channels
  input       kp1: a 4-cycle of near-orthogonal 32-dim symbols, ns comp25,
              repeating
  activation  applied to what each unit sends; wta = one-hot argmax per
              32-unit column
  keep        fraction of the summed input carried into the next step
  norm        the RMS norm in BareCol.update_activations, on or off

The debugger draws each column's committed state (nr_1.actual), the summed
input before the activation; for wta the unit that sends is the column's
largest value. The terminal prints, every 200 steps, the mean |state| and
how often each column's argmax changed.

--headless skips the debugger, runs run 39's 2000 steps and prints the
participation ratio the same way run 39 does; for the default cell it gives
15.97 (seed 50) and 16.33 (seed 51), matching actdyn39.json.

Display note, as for 24_watch_fan3.py: pygame needs a display (a local
session on the box, or ssh -X).

Usage:
    venv/bin/python experiments/45_watch_actdyn39.py
    venv/bin/python experiments/45_watch_actdyn39.py --act tri0.3 --keep 0
    venv/bin/python experiments/45_watch_actdyn39.py --norm --seed 51 --delay 0.1
    venv/bin/python experiments/45_watch_actdyn39.py --headless
"""

import argparse
import importlib
import itertools
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

r39 = importlib.import_module("39_activation_dynamics")
sweep, de, T = r39.sweep, r39.de, r39.T


class WatchableX(sweep.BareAgtXC):
    """run 39's host plus the debugger feed, the same hook as
    24_watch_fan3.py: BareAgtXC.step replaces the stock loop that carried
    the per-column debugger updates."""

    def step(self, ipt, disable_print=True):
        if self.use_debug and self.pipes["overview"][0].poll():
            print("\ndebugger window closed, exiting")
            sys.exit(0)
        out = super().step(ipt, disable_print=disable_print)
        if self.use_debug:
            self.debug_update()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="wta", help="any run 39 activation name")
    ap.add_argument("--keep", type=float, default=0.9)
    ap.add_argument("--norm", action="store_true", help="RMS norm on (default off)")
    ap.add_argument("--seed", type=int, default=50, help="run 39 used 50 and 51")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between steps")
    ap.add_argument("--headless", action="store_true",
                    help="no debugger: run 39's 2000 steps, print the participation ratio")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    r39._patch(args.act, args.keep, args.norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **r39.arm_cfg(args.act))
    tag = f"{args.act}_keep{args.keep}_norm{'on' if args.norm else 'off'}_s{args.seed}"
    agt = WatchableX(cfg, os.path.join(ROOT, "saves", f"watch39_{tag}"),
                     seed=args.seed, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    print(f"run 39 cell: act {args.act}, keep {args.keep}, norm {'on' if args.norm else 'off'}, "
          f"seed {args.seed}; {len(bulk)} bulk columns, learning frozen, kp1 input")

    def stream():
        while True:
            yield from de.kp1_cycle(r39.STEPS, seed=de.derive_seed("kp1", r39.NS, args.seed))

    if args.headless:
        rows = []
        for step, a in enumerate(itertools.islice(stream(), r39.STEPS)):
            agt.step([a])
            s = r39.state_of(agt)
            if not torch.isfinite(s).all():
                print(f"DIVERGED at step {step}")
                return
            if step >= r39.BURN:
                rows.append(s.clone())
        pr, _ = r39._participation(torch.stack(rows).float(), cfg.d_col)
        print(f"participation ratio {pr:.2f} of {cfg.d_col} "
              f"(summed input, per column, steps {r39.BURN} to {r39.STEPS})")
        return

    agt.debug_init()
    prev, switches, n = None, 0, 0
    for step, a in enumerate(stream()):
        agt.step([a])
        win = torch.stack([agt.cols[loc].nr_1.actual.argmax() for loc in bulk])
        if prev is not None:
            switches += int((win != prev).sum())
            n += 1
        prev = win
        if step % 200 == 0 and step > 0:
            s = r39.state_of(agt)
            print(f"step {step}: mean |state| {float(s.abs().mean()):.3f}, "
                  f"column argmax changed on {switches / (n * len(bulk)):.0%} of steps", flush=True)
            switches, n = 0, 0
        if args.delay:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
