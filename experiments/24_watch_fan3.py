"""
Hook the pygame debugger to a representative instance of the substrate the
last two days of experiments actually ran: default = fan3 (the fanout-3
topology family from R-029..R-032) on its best registered seed by R-031
differential, driven by an endless ts-style stream (cycle -> new cycle
forever) so world-switches are watchable live.

The debugger process and pipe protocol are stock (src/debugger.py via
AgtBase.debug_init); the only addition is a step-granularity feed hook,
since BareAgtXC.step overrides the stock loop that carried the per-column
hooks. Display note: pygame needs a display: run under `ssh -X` (usable at
Starlink latency; the debugger's cooldowns keep traffic modest) or from a
local session on the box.

Usage:7
    venv/bin/python experiments/24_watch_fan3.py
    venv/bin/python experiments/24_watch_fan3.py --preset faithful --fixture kp1
    venv/bin/python experiments/24_watch_fan3.py --seed 4634 --segment 2000
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

sweep = importlib.import_module("10_sweep_bare_x")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from src import iotypes as T

PRESETS = {
    "fan3": dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
                 rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1),
    "faithful": dict(rule="softhebb", transport="triangle", ss=1e-2, lr_e=0.1),
}


class WatchableX(sweep.BareAgtXC):
    """BareAgtXC + the debugger feed the stock step loop provides per column,
    here at step granularity (49 small cols at ~20ms/step comfortably outruns
    the debugger's 0.2s overview cooldown)."""

    def step(self, ipt, disable_print=True):
        if self.use_debug and self.pipes["overview"][0].poll():
            print("\ndebugger window closed, exiting")
            sys.exit(0)
        out = super().step(ipt, disable_print=disable_print)
        if self.use_debug:
            self.debug_update()
        return out


def best_registered_seed():
    """argmax fan3 differential from the R-031 verdict ledger; the
    'representative/better instance' request made reproducible."""
    try:
        store = D3Store(sweep.STORE_ROOT)
        v = next(e for e in store.ledger()
                 if e.get("kind") == "r031-verdicts" and not e.get("smoke"))
        seed, diff = max(v["fan3_diffs"].items(), key=lambda kv: kv[1])
        return int(seed), float(diff)
    except (StopIteration, OSError, KeyError):
        return 4634, float("nan")


def endless_stream(fixture, segment, seed0):
    n = 0
    while True:
        if fixture == "kz":
            yield from de.kz_noise(segment, seed=de.derive_seed("kz", "watch24", seed0 + n))
        elif fixture == "kp1":
            yield from de.kp1_cycle(segment, seed=de.derive_seed("kp1", "watch24", seed0))
        else:  # ts: a new cycle every segment
            yield from de.kp1_cycle(segment,
                                    seed=de.derive_seed("ts", "watch24", seed0 + n))
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fan3")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fixture", choices=("ts", "kp1", "kz"), default="ts")
    ap.add_argument("--segment", type=int, default=2000,
                    help="steps per world before a ts switch")
    args = ap.parse_args()

    seed, diff = (args.seed, float("nan")) if args.seed is not None \
        else best_registered_seed()
    knobs = dict(PRESETS[args.preset])
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS,
                          **knobs)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "saves", f"watch24_{args.preset}_s{seed}")
    agt = WatchableX(cfg, path, seed=seed)
    print(f"watching {args.preset} seed {seed}"
          + ("" if diff != diff else f" (R-031 differential {diff:+.3f})")
          + f", fixture {args.fixture}, segment {args.segment}")
    agt.debug_init()

    stream = endless_stream(args.fixture, args.segment, seed)
    for step, item in enumerate(stream):
        a = item[1] if isinstance(item, tuple) else item
        agt.step([a])
        if step % args.segment == 0 and step > 0 and args.fixture == "ts":
            print(f"step {step}: world switched (segment {step // args.segment})", flush=True)
        elif step % 1000 == 0 and step > 0:
            print(f"step {step}", flush=True)


if __name__ == "__main__":
    main()
