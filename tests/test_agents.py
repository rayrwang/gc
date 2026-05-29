
import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import torch

from src.agents import Cfg, Agt

N_COLS = 4

def test_agent_init(tmp_path):
    agt = Agt(Cfg(N_COLS, [], []), tmp_path)
    agt.verify()

def test_agent_step(tmp_path):
    agt = Agt(Cfg(N_COLS, [], []), tmp_path)
    agt.step([])

def test_agent_save_and_load(tmp_path):
    agt1 = Agt(Cfg(N_COLS, [], []), tmp_path)
    agt1.save()
    agt2 = Agt.load(tmp_path)
    for loc1, col1 in agt1.cols.items():
        assert loc1 in agt2.cols
        col2 = agt2.cols[loc1]
        for name1, value1 in vars(col1).items():
            assert name1 in vars(col2)
            if name1.startswith("nr_"):
                for a1, a2 in zip(value1, getattr(col2, name1)):
                    assert torch.allclose(a1, a2)
            elif name1.startswith("is_"):
                assert torch.allclose(value1, getattr(col2, name1))
