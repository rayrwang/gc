
from dataclasses import dataclass

import pytest
import torch

# isort: off
from src.iotypes import I_Vector, O_Vector
from src.agents import Cfg, Agt, BareCfg, BareAgt, MNISTCfg, MNISTAgt

N_COLS = 4

@dataclass
class Case:
    cfg_cls: type
    cfg_args: tuple
    agt_cls: type
    step_input: list
cases = [
    pytest.param(Case(Cfg,      (N_COLS, [], []),                  Agt,      []),                 id="Agt"),
    pytest.param(Case(BareCfg,  (N_COLS, [], []),                  BareAgt,  []),                 id="BareAgt"),
    pytest.param(Case(MNISTCfg, ([I_Vector(784)], [O_Vector(10)]), MNISTAgt, [torch.randn(784)]), id="MNISTAgt"),
]


@pytest.mark.parametrize("case", cases)
def test_agent_init(tmp_path, case):
    agt = case.agt_cls(case.cfg_cls(*case.cfg_args), tmp_path)
    if hasattr(agt, "verify"):
        agt.verify()


@pytest.mark.parametrize("case", cases)
def test_agent_step(tmp_path, case):
    agt = case.agt_cls(case.cfg_cls(*case.cfg_args), tmp_path)
    agt.step(case.step_input)


@pytest.mark.parametrize("case", cases)
def test_agent_save_and_load(tmp_path, case):
    agt1 = case.agt_cls(case.cfg_cls(*case.cfg_args), tmp_path)
    agt1.save()
    agt2 = case.agt_cls.load(tmp_path)
    assert len(agt1.cols) == len(agt2.cols)
    for loc1, col1 in agt1.cols.items():
        assert loc1 in agt2.cols
        col2 = agt2.cols[loc1]
        assert len(vars(col1)) == len(vars(col2))
        assert col1.cfg == col2.cfg
        for name1, value1 in vars(col1).items():
            assert name1 in vars(col2)
            if name1.startswith("nr_"):
                assert len(value1) == len(getattr(col2, name1))
                for a1, a2 in zip(value1, getattr(col2, name1), strict=True):
                    assert torch.allclose(a1, a2)
            elif name1.startswith("is_"):
                assert torch.allclose(value1, getattr(col2, name1))
        assert len(col1.conns) == len(col2.conns)
        for address, weight in col1.conns.items():
            assert address in col2.conns
            assert torch.allclose(weight, col2.conns[address])
