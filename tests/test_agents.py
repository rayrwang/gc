
import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import pytest

import torch

from src.iotypes import I_Vector, O_Vector
from src.agents import Cfg, Agt, BareCfg, BareAgt, MNISTCfg, MNISTAgt

N_COLS = 4
cases = [
    pytest.param(Cfg, (N_COLS, [], []), Agt, id=Agt.__name__),
    pytest.param(BareCfg, (N_COLS, [], []), BareAgt, id=BareAgt.__name__),
    pytest.param(MNISTCfg, ([I_Vector(784)], [O_Vector(10)]), MNISTAgt, id=MNISTAgt.__name__),
]


@pytest.mark.parametrize("cfg_cls, cfg_args, agt_cls", cases)
def test_agent_init(tmp_path, cfg_cls, cfg_args, agt_cls):
    agt = agt_cls(cfg_cls(*cfg_args), tmp_path)
    if hasattr(agt, "verify"):
        agt.verify()


@pytest.mark.parametrize("cfg_cls, cfg_args, agt_cls", cases)
def test_agent_step(tmp_path, cfg_cls, cfg_args, agt_cls):
    agt = Agt(Cfg(N_COLS, [], []), tmp_path)
    agt.step([])


@pytest.mark.parametrize("cfg_cls, cfg_args, agt_cls", cases)
def test_agent_save_and_load(tmp_path, cfg_cls, cfg_args, agt_cls):
    agt1 = Agt(Cfg(N_COLS, [], []), tmp_path)
    agt1.save()
    agt2 = Agt.load(tmp_path)
    assert len(agt1.cols) == len(agt2.cols)
    for loc1, col1 in agt1.cols.items():
        assert loc1 in agt2.cols
        col2 = agt2.cols[loc1]
        assert len(vars(col1)) == len(vars(col2))
        for name1, value1 in vars(col1).items():
            assert name1 in vars(col2)
            if name1.startswith("nr_"):
                for a1, a2 in zip(value1, getattr(col2, name1)):
                    assert torch.allclose(a1, a2)
            elif name1.startswith("is_"):
                assert torch.allclose(value1, getattr(col2, name1))
        assert len(col1.conns) == len(col2.conns)
        for address, weight in col1.conns.items():
            assert address in col2.conns
            assert torch.allclose(weight, col2.conns[address])
