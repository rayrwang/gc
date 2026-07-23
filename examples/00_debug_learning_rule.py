
"""
See the effect of the learning rule on reference situations:

      0       1       2       3       4
  ┌───────┬───────┬───────┬───────┬───────┐        ─┬─
  │ randn │ randn │       │       │       │         │
0 │refresh│ fixed │ zeros │  ones │ -ones │         │ Inputs
  │       │       │       │       │       │         │
  └───────┴───────┴───────┴───────┴───────┘        ─┴─
      
    
    
  ┌───────┬───────┬───────┬───────┬───────┐        ─┬─
  │ randn │ randn │       │       │       │         │ Passive
2 │refresh│ fixed │ zeros │  ones │ -ones │         │ outputs
  │       │       │       │       │       │         │
  ├───────┼───────┼───────┼───────┼───────┤        ─┼─           ─┬─              
  │ (0,0) │ (1,0) │ zeros │  ones │ -ones │         │             │ Without       
3 │ times │ times │ times │ times │ times │         │             │ activation    
  │weights│weights│weights│weights│weights│         │ Active      │ function      
  ├───────┼───────┼───────┼───────┼───────┤         │ Outputs    ─┼─              
  │ (0,0) │ (1,0) │ zeros │  ones │ -ones │         │             │ With          
4 │ times │ times │ times │ times │ times │         │             │ activation    
  │weights│weights│weights│weights│weights│         │             │ function      
  └───────┴───────┴───────┴───────┴───────┘        ─┴─           ─┴─              

For example to see what happens when input is all -1
and output is all 0, see the connection from (4, 0) to (2, 2).

(coordinates are in screen space: right then down) TODO change?
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import itertools

import torch
from tqdm import tqdm

import src.funcs as fc

# isort: off
from src.agents import Dir, conn
from src.agents import BareColCfg, BareCol
from src.agents import AgtBase


# Generic agents are in src/agents.py,
# but specific agents such as this one are in each example file.
class DebugLearningRuleAgt(AgtBase):
    def __init__(self, d: int, path):
        self.d = d
        self.path = path
        self.age: int = 0

        inits = (
            torch.randn(d),
            torch.randn(d),
            torch.zeros(d),
            torch.ones(d),
            -torch.ones(d)
        )

        self.cols = {}

        # Inputs
        self.input_locs = [(i, 0) for i in range(4+1)]
        for loc, init in zip(self.input_locs, inits, strict=True):
            col = BareCol(loc, BareColCfg(d))
            col.nr_1.actual = init
            self.cols[loc] = col

        # Passive outputs
        passive_output_locs = [(i, 2) for i in range(4+1)]
        for loc, init in zip(passive_output_locs, inits, strict=True):
            col = BareCol(loc, BareColCfg(d))
            col.nr_1.actual = init
            self.cols[loc] = col

        # Active outputs (without and with activation function)
        active_output_locs = \
            [(i, 3) for i in range(4+1)] \
            + [(i, 4) for i in range(4+1)]
        for loc in active_output_locs:
            col = BareCol(loc, BareColCfg(d))
            col.nr_1.actual = torch.zeros(d)
            self.cols[loc] = col

        # Conns
        self.output_locs = passive_output_locs + active_output_locs
        for input_loc in self.input_locs:
            for output_loc in self.output_locs:
                self.cols[input_loc].conns[output_loc, Dir.A] = \
                    conn(self.cols[input_loc], self.cols[output_loc], Dir.A, 1)

        self.create_directory()

    def step(self):
        # Refresh randns
        for loc in ((0, 0), (0, 2)):
            self.cols[loc].nr_1_.actual = torch.randn(self.d)

        # Preserve non changing
        for loc_x in range(1, 4+1):
            self.cols[(loc_x, 0)].nr_1_.actual = self.cols[(loc_x, 0)].nr_1.actual
            self.cols[(loc_x, 2)].nr_1_.actual = self.cols[(loc_x, 2)].nr_1.actual

        # Compute new active outputs
        for loc_x in range(4+1):
            input_loc = (loc_x, 0)
            input_layer = self.cols[input_loc].nr_1.actual

            # Don't apply activation function
            output_loc = (loc_x, 3)
            weights = self.cols[input_loc].conns[(output_loc, Dir.A)]
            self.cols[output_loc].nr_1_.actual = input_layer @ weights

            # Apply activation function
            func_output_loc = (loc_x, 4)
            weights = self.cols[input_loc].conns[(func_output_loc, Dir.A)]
            self.cols[func_output_loc].nr_1_.actual = fc.atv(input_layer, weights)

        # Update activations
        for col in self.cols.values():
            col.update_activations()

        # Apply learning rule
        for input_loc in self.input_locs:
            for output_loc in self.output_locs:
                self.cols[input_loc].conns[(output_loc, Dir.A)] = fc.lrn(
                    self.cols[input_loc].nr_1.actual,
                    self.cols[input_loc].conns[(output_loc, Dir.A)],
                    self.cols[output_loc].nr_1.actual)

        self.debug_update()
        self.age += 1


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    agt = DebugLearningRuleAgt(
        1024 if torch.cuda.is_available() else 128,
        f"{project_root_path}/saves/debug_lrn_agt")
    agt.debug_init()
    for _ in (bar := tqdm(itertools.count(), desc="Running debug learning rule")):
        agt.step()

        if agt.pipes["overview"][0].poll():
            bar.close()
            sys.exit(0)
