"""
Does a smooth activation settle without fc.inhibit? (2026-09-28)

RW's question: example 01 found every smooth sigmoid settles to a fixed point on the
Col/Agt path (the one main.py uses: fc.inhibit on, learning off, no carry-over), with the
caveat "inhibition does this work; with inhibit off, a steep enough sigmoid (b40) also
stays alive". Is fc.inhibit needed for the settling at all?

Premises: example 01's setup exactly (Agt at its defaults, 100 cols on the GPU, the fixed
default GridEnv input every step, fc.lrn replaced by a no-op, fc.update resetting to zero
so keep 0, seed 0 so every condition gets the same weights). Varied: the activation
(spike, sigmoid gain 1, 4, 40, centred on the threshold 1.0) x fc.inhibit (on, or replaced
by a pass-through). Measured per step: mean |a_t - a_t-1| over every nr_*.actual; for the
inhibit-on runs also the mean |change| the inhibition adds, to show it fires.

Output: outputs/54_inhibit_settle.log and .json; summary in 54_report.md.
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import random

import torch

import src.funcs as fc

# isort: off
from src.agents import Cfg, Agt
from src.envs import GridEnvCfg, GridEnv, get_default  # also sets default dtype

SEED = 0
STEPS = 300
HERE = os.path.dirname(os.path.abspath(__file__))


def no_lrn(x, w, y, *args, **kwargs):
    return w


def state(agt):
    parts = []
    for col in agt.cols.values():
        for name, val in vars(col).items():
            if name.startswith("nr_") and not name.endswith("_"):
                parts.append(val.actual.flatten().float())
    return torch.cat(parts)


def build(n_cols, ispec, ospec, name):
    random.seed(SEED)
    torch.manual_seed(SEED)
    return Agt(Cfg(n_cols, ispec, ospec), f"{project_root_path}/saves/inhibit54_{name}")


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    n_cols = 100 if torch.cuda.is_available() else 30
    ispec, ospec = GridEnv.get_specs(GridEnvCfg(width=4))
    ipt = get_default(ispec)

    fc.lrn = no_lrn
    real_spike = fc.spike
    real_inhibit = fc.inhibit
    inhib_size = []

    def counting_inhibit(x):
        out = real_inhibit(x)
        inhib_size.append((out.actual - x.actual).abs().mean().item())
        return out

    def pass_through(x):
        return x

    activations = [
        ("spike", real_spike),
        ("sig_b1", lambda x, threshold=1.0: torch.sigmoid(1.0 * (x - threshold))),
        ("sig_b4", lambda x, threshold=1.0: torch.sigmoid(4.0 * (x - threshold))),
        ("sig_b40", lambda x, threshold=1.0: torch.sigmoid(40.0 * (x - threshold))),
    ]
    conditions = [(f"{a}_inhib{'on' if on else 'off'}", act, on)
                  for a, act in activations for on in (True, False)]

    agents = {name: build(n_cols, ispec, ospec, name) for name, _, _ in conditions}
    prev = {name: state(agents[name]) for name, _, _ in conditions}
    changes = {name: [] for name, _, _ in conditions}
    inhib = {name: [] for name, _, _ in conditions}

    log = open(os.path.join(HERE, "outputs", "54_inhibit_settle.log"), "w")
    print(f"n_cols {n_cols}, steps {STEPS}, device {torch.get_default_device()}", file=log, flush=True)
    for step in range(STEPS):
        for name, activation, on in conditions:
            fc.spike = activation
            fc.inhibit = counting_inhibit if on else pass_through
            inhib_size.clear()
            agt = agents[name]
            agt.step(ipt, disable_print=True)
            a = state(agt)
            changes[name].append((a - prev[name]).abs().mean().item())
            inhib[name].append(sum(inhib_size) / len(inhib_size) if inhib_size else 0.0)
            prev[name] = a
        if step % 10 == 0 or step == STEPS - 1:
            print(f"{step:>4} " + " ".join(f"{n}={changes[n][-1]:.2e}" for n, _, _ in conditions),
                  file=log, flush=True)
    fc.spike, fc.inhibit = real_spike, real_inhibit

    summary = {}
    for name, _, on in conditions:
        c = changes[name]
        settled = next((t for t in range(len(c)) if max(c[t:]) < 1e-5), None)
        a = prev[name]
        summary[name] = {
            "final_change_mean_last50": sum(c[-50:]) / 50,
            "settled_step_below_1e-5": settled,
            "active_fraction_final": (a >= 1).float().mean().item(),
            "mean_abs_a_final": a.abs().mean().item(),
            "inhib_mean_abs_last50": (sum(inhib[name][-50:]) / 50) if on else None,
        }
    print("\nSUMMARY", file=log)
    for name, s in summary.items():
        print(name, json.dumps(s), file=log)
    log.close()
    with open(os.path.join(HERE, "outputs", "54_inhibit_settle.json"), "w") as f:
        json.dump({"changes": changes, "summary": summary, "n_cols": n_cols, "steps": STEPS}, f)
