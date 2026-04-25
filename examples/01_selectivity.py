
"""
(WIP) Demonstrate learning of selectivity of output to one-hot input
"""

import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import itertools
import random
import time

import torch
from tqdm import tqdm

import src.funcs as fc
from src.agents import Dir, conn
from src.agents import BareColCfg, BareCol
from src.agents import AgtBase


class SelectivityAgt(AgtBase):
    def __init__(self, d: int, path):
        self.d = d
        self.path = path

        self.cols = {}

        col1 = BareCol((0, 0), BareColCfg(d))
        self.cols[col1.loc] = col1

        col2 = BareCol((1, 0), BareColCfg(d))
        self.cols[col2.loc] = col2

        self.cols[col1.loc].conns[col2.loc, Dir.A] = conn(
            col1, col2, Dir.A, 6
        )

        self.create_directory()

    def step(self):
        loc1 = (0, 0)
        loc2 = (1, 0)

        # Refresh input
        self.cols[loc1].a_pre = torch.zeros(self.d)
        self.cols[loc1].a_pre[random.randrange(self.d)] = 1

        # Activity rule
        self.cols[loc2].a_post = fc.atv(
            self.cols[loc1].a_pre,
            self.cols[loc1].conns[loc2, Dir.A]
        )

        # Learning rule
        self.cols[loc1].conns[loc2, Dir.A] = fc.lrn(
            self.cols[loc1].a_pre,
            self.cols[loc1].conns[loc2, Dir.A],
            self.cols[loc2].a_post,
            reg_width=float("inf"),  # Disalbe regulation
        )

        agt.debug_update()


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    agt = SelectivityAgt(128, f"{project_root_path}/saves/selectivity_agt")
    agt.debug_init()

    MIN_ITER_SECS = 0.005
    t_prev = time.perf_counter()
    for _ in (bar := tqdm(itertools.count(), desc="Running learning selectivity")):
        agt.step()

        if agt.pipes["overview"][0].poll():
            bar.close()
            sys.exit()

        while (time.perf_counter() - t_prev) < MIN_ITER_SECS:
            pass
        t_prev = time.perf_counter()
