
"""
Demonstrate combining expected and actual activations,
currently via lateral inhibition.

        0           1    
  .-----------.-----------.
  |           |           |
  | expected  | unexpected|
0 | (less     | (more     |
  | activity) | activity) |
  |           |           |
  .-----------.-----------.
"""

import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import itertools
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import src.funcs as fc
from src.agents import BareColCfg, BareCol
from src.agents import AgtBase


class ExpectationsAgt(AgtBase):
    def __init__(self, d: int, path):
        self.d = d
        self.path = path

        self.cols = {}

        # Expectations align with actual
        col = BareCol((0, 0), BareColCfg(d))
        self.cols[(0, 0)] = col

        # Expectations different from actual
        col = BareCol((1, 0), BareColCfg(d))
        self.cols[(1, 0)] = col

        self.create_directory()

    def step(self):
        actual = torch.randn(self.d)

        col_align = self.cols[0, 0]
        col_diff = self.cols[1, 0]

        # Expectations align with actual
        col_align.nr_1_[0] = actual
        col_align.nr_1_[1] = actual

        # Expectations different from actual
        col_diff.nr_1_[0] = actual
        col_diff.nr_1_[1] = torch.randn(self.d)

        for col in self.cols.values():
            col.nr_1_ = fc.inhibit(col.nr_1_)
            col.update_activations()

        self.debug_update()
        

if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    agt = ExpectationsAgt(
        1024 if torch.cuda.is_available() else 128,
        f"{project_root_path}/saves/expectations_agt")
    agt.debug_init()

    print("Running testing expectations...")
    MIN_ITER_SECS = 0.2
    t_prev = time.perf_counter()
    for step in itertools.count():
        agt.step()

        density_expected = fc.density(agt.cols[0, 0].nr_1[0])
        density_unexpected = fc.density(agt.cols[1, 0].nr_1[0])
        print(f"\nStep {step}")
        print(f"Expected activity: {density_expected:.4f}")
        print(f"Unexpected activity: {density_unexpected:.4f}")
        with SummaryWriter("runs/expectations") as writer:
            writer.add_scalars("Density (level of activity)", {
                "Expected": density_expected,
                "Unexpected": density_unexpected
            }, global_step=step)

        if agt.pipes["overview"][0].poll():
            sys.exit(0)

        while (time.perf_counter() - t_prev) < MIN_ITER_SECS:
            pass
        t_prev = time.perf_counter()
