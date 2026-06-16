
"""
Compare activation functions in the recurrent network.

Fix the weights (learning off) and feed the same input every step. The network
is a fixed map applied over and over, a_t+1 = F(a_t). The activation F decides
whether the activations settle or keep moving.

spike    step to step change stays large, never decays to 0 (limit cycle, alive).
sigmoid  change decays by a roughly constant factor each step and goes to 0
         (settles to a fixed point, dies). A steeper sigmoid does not fix this,
         even at high gain a smooth map still contracts. The discontinuity is
         what keeps it off the fixed point, not the steepness.

Each activation has its own agent. Step them round robin (one step each per
loop, swapping the activation before each, since it is a module global) so they
stay on the same step. Runs until killed (ctrl-c), like the main net. Each logs
two scalars per step to tensorboard:

  mean_change      mean |a_t - a_t-1|, does it keep moving
  active_fraction  fraction of a >= threshold, how sparse / how alive

  tensorboard --logdir runs/activation_functions
"""

import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import itertools
import random

import torch
from torch.utils.tensorboard import SummaryWriter

import src.funcs as fc
from src.agents import Cfg, Agt
from src.envs import GridEnvCfg, GridEnv, get_default  # also sets default dtype

SEED = 0


# Pure activation dynamics, leave the weights alone
def no_lrn(x, w, y, *args, **kwargs):
    return w


# All actual activations (nr_*[0]) across every col, flattened to one vector
def state(agt):
    parts = []
    for col in agt.cols.values():
        for name, val in vars(col).items():
            if name.startswith("nr_") and not name.endswith("_"):
                parts.append(val[0].flatten().float())
    return torch.cat(parts)


# Same seed every condition, so only the activation differs
def build(n_cols, ispec, ospec, name):
    random.seed(SEED)
    torch.manual_seed(SEED)
    return Agt(Cfg(n_cols, ispec, ospec), f"{project_root_path}/saves/activation_{name}")


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    n_cols = 100 if torch.cuda.is_available() else 30
    ispec, ospec = GridEnv.get_specs(GridEnvCfg(width=4))
    ipt = get_default(ispec)

    fc.lrn = no_lrn
    real_spike = fc.spike  # hard threshold, what the main net uses

    # name, activation (sigmoid centered on the threshold, gentle to steep)
    conditions = [
        ("spike", real_spike),
        ("sigmoid_b1", lambda x, threshold=1.0: torch.sigmoid(1.0 * (x - threshold))),
        ("sigmoid_b4", lambda x, threshold=1.0: torch.sigmoid(4.0 * (x - threshold))),
        ("sigmoid_b40", lambda x, threshold=1.0: torch.sigmoid(40.0 * (x - threshold))),
    ]

    agents = {name: build(n_cols, ispec, ospec, name) for name, _ in conditions}
    writers = {name: SummaryWriter(f"runs/activation_functions/{name}", flush_secs=10)
               for name, _ in conditions}
    prev = {name: state(agents[name]) for name, _ in conditions}

    print("\ntensorboard --logdir runs/activation_functions\n")
    print(f"{'step':>5} " + " ".join(f"{name:>11}" for name, _ in conditions) + "   (mean_change)")

    # Round robin, runs until ctrl-c
    try:
        for step in itertools.count():
            changes = {}
            for name, activation in conditions:
                fc.spike = activation
                agt = agents[name]
                agt.step(ipt, disable_print=True)
                a = state(agt)
                change = (a - prev[name]).abs().mean().item()
                active = (a >= 1).float().mean().item()
                writers[name].add_scalar("mean_change", change, step)
                writers[name].add_scalar("active_fraction", active, step)
                prev[name] = a
                changes[name] = change
            print(f"{step:>5} " + " ".join(f"{changes[name]:>11.4f}" for name, _ in conditions))
    finally:
        for writer in writers.values():
            writer.close()
