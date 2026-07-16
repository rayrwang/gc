
from dataclasses import dataclass, fields

import pytest
import torch

# isort: off
from src.iotypes import I_Vector, O_Vector
from src.agents import Cfg, Agt, BareCfg, BareAgt, MNISTCfg, MNISTAgt, CIFARAgt, Dir

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
    assert agt.age == 0
    agt.step(case.step_input)
    assert agt.age == 1


@pytest.mark.parametrize("case", cases)
def test_agent_save_and_load(tmp_path, case):
    agt1 = case.agt_cls(case.cfg_cls(*case.cfg_args), tmp_path)
    agt1.save()
    agt2 = case.agt_cls.load(tmp_path)
    assert agt1.age == agt2.age
    assert agt1.cfg == agt2.cfg
    assert agt1.path == agt2.path
    assert len(agt1.cols) == len(agt2.cols)
    for loc1, col1 in agt1.cols.items():
        assert loc1 in agt2.cols
        col2 = agt2.cols[loc1]
        assert type(col1) is type(col2)  # Load must reconstruct the same col subclass
        assert len(vars(col1)) == len(vars(col2))
        assert col1.cfg == col2.cfg
        for name1, value1 in vars(col1).items():
            assert name1 in vars(col2)
            if name1.startswith("nr_"):
                activs2 = getattr(col2, name1)  # Activs dataclass: compare field-by-field
                for f in fields(value1):
                    v1, v2 = getattr(value1, f.name), getattr(activs2, f.name)
                    if isinstance(v1, torch.Tensor):
                        assert torch.allclose(v1, v2)
                    else:
                        assert v1 == v2  # non-Tensor Activs fields, e.g. rms_avg: float
            elif name1.startswith("is_"):
                assert torch.allclose(value1, getattr(col2, name1))
        assert len(col1.conns) == len(col2.conns)
        for address, weight in col1.conns.items():
            assert address in col2.conns
            assert torch.allclose(weight, col2.conns[address])


# CIFARAgt is standalone (deep oja-signed conv, no Cfg / save-load), so it gets its own tests
def test_cifar_agt_smoke():
    """Deep oja-signed conv forward + learning on synthetic input: rep is the right
    shape, finite, the dummy output is 10-dim, and learning moves the weights."""
    torch.manual_seed(0)
    agt = CIFARAgt(layers=[(3, 8, 5, "max"), (8, 16, 3, "avg")])  # tiny 2-layer net for speed
    w0 = [w.clone() for w in agt.W]
    out = None
    for _ in range(5):
        out = agt.step([torch.rand(3, 32, 32)], use_lrn=True, disable_print=True)
    assert out is not None
    assert len(out) == 1
    assert out[0].shape == (10,)
    rep = agt.get_representations()
    assert rep.shape == (16 * 4 * 4,)   # last layer 16 ch, final adaptive-pool 4x4 (default)
    assert torch.isfinite(rep).all()
    assert any(not torch.allclose(w, w_old) for w, w_old in zip(agt.W, w0, strict=True))  # learning moved weights


def test_cifar_agt_bn_modes():
    """Online BN: a no-learning warmup (training=True) moves the running stats; eval
    (training defaults to use_lrn=False) uses them, stays finite, and learns nothing."""
    torch.manual_seed(0)
    agt = CIFARAgt(layers=[(3, 8, 5, "max"), (8, 16, 3, "avg")])
    m0 = [m.clone() for m in agt.bn_m]
    for _ in range(5):  # warmup: BN stats update, weights do not learn
        agt.step([torch.rand(3, 32, 32)], use_lrn=False, training=True)
    assert any(not torch.allclose(m, m_old) for m, m_old in zip(agt.bn_m, m0, strict=True))
    w_before = [w.clone() for w in agt.W]
    agt.step([torch.rand(3, 32, 32)], use_lrn=False)  # eval: training defaults to use_lrn=False
    assert torch.isfinite(agt.get_representations()).all()
    assert all(torch.allclose(w, w_old) for w, w_old in zip(agt.W, w_before, strict=True))  # no learning in eval


# MNISTAgt is the ONLY thing that exercises the Col/conns/activs/lrn_adaptive path
# (CIFARAgt and the abstract examples are standalone). This is a bit-identical golden
# of that path, captured pre-refactor on CPU/float32/seed-0, so the planned structural
# refactor (activations -> dataclass, conns -> endpoint addresses) can prove it changed
# shape, not behavior. If these numbers move, the refactor altered computation. To
# legitimately re-baseline (an intended math change), re-capture all values from one run.
def test_mnist_agt_learning_golden(tmp_path):
    torch.manual_seed(0)  # seeds the weight init inside __init__
    agt = MNISTAgt(MNISTCfg([I_Vector(784)], [O_Vector(10)]), str(tmp_path))
    gen = torch.Generator().manual_seed(0)  # inputs: independent of init's RNG draws
    for _ in range(10):
        agt.step([torch.rand(784, generator=gen)], use_lrn=True, disable_print=True)
    rep = agt.cols[1, 0].nr_1.actual  # post-step hidden actual activation

    col_in = agt.cols[0, 0]
    assert col_in.conns is not None  # ColBase types conns as dict|None; narrow it
    w = col_in.conns[(1, 0), Dir.A]  # the learned input->hidden connection
    assert w.dtype is torch.float32          # a float16 leak would silently break the golden
    assert tuple(w.shape) == (784, 128)
    # rel=1e-5 tolerates the literals' rounding; a real behavior change moves these by O(1)
    assert w.sum().item() == pytest.approx(-54.4567222595, rel=1e-5)
    assert w.abs().sum().item() == pytest.approx(8587.6699218750, rel=1e-5)
    assert w[0, 0].item() == pytest.approx(0.0038963521, rel=1e-5)
    assert w[500, 64].item() == pytest.approx(0.1359269172, rel=1e-5)
    assert w[783, 127].item() == pytest.approx(-0.0241337400, rel=1e-5)
    # forward path: pin the resulting hidden representation too
    assert tuple(rep.shape) == (128,)
    assert rep.sum().item() == pytest.approx(57.5112113953, rel=1e-5)
    assert rep.max().item() == pytest.approx(2.9827098846, rel=1e-5)
