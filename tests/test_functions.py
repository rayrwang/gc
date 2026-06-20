
import inspect

import pytest
import torch

import src.funcs as fc
from src.agents import activs


def test_activity_rule_shapes():
    D_X = 321
    D_Y = 257
    x = torch.randn(D_X)
    w = torch.randn(D_X, D_Y)
    d_y_actual, = fc.atv(x, w).shape
    assert d_y_actual == D_Y


@pytest.mark.parametrize("func", [
    func
    for name, func in inspect.getmembers(fc, inspect.isfunction)
    if name.startswith("lrn") and func.__module__ == fc.__name__ \
        and not name.startswith("lrn_adaptive")  # TODO make signature consistent
])
def test_learning_rule_shapes_basic(func):
    D_X = 233
    D_Y = 391
    x = torch.randn(D_X)
    w = torch.randn(D_X, D_Y)
    y = torch.randn(D_Y)
    w_shape_new = func(x, w, y).shape
    assert tuple(w_shape_new) == (D_X, D_Y)


@pytest.mark.parametrize("func", [
    # TODO make signatures consistent
    fc.lrn_adaptive,
    fc.lrn_adaptive_d
])
def test_learning_rule_shapes_rich(func):
    D_X = 233
    D_Y = 391
    x = activs(D_X)
    w = torch.randn(D_X, D_Y)
    y = activs(D_Y)
    w_shape_new = func(x, w, y).shape
    assert tuple(w_shape_new) == (D_X, D_Y)


def test_lrn_basic_behavior():
    """
    `lrn_basic` behaviour, not just shape: plain Hebb, Δw = ss * outer(x, y).
    Co-active pairs strengthen, anti-correlated pairs weaken, and an inactive
    presynaptic unit leaves its whole row unchanged.
    """
    ss = 0.1
    x = torch.tensor([2.0, 0.0, 1.0])  # unit at index 1 is presynaptically inactive
    y = torch.tensor([1.0, -1.0])      # output 0 positive, output 1 negative
    w = torch.zeros(3, 2)

    dw = fc.lrn_basic(x, w, y, ss=ss) - w

    # Exact rule: Δw == ss * outer(x, y)
    assert torch.allclose(dw, ss * torch.outer(x, y))

    # Co-active (x > 0, y > 0) strengthens the weight
    assert dw[0, 0] > 0
    # Anti-correlated (x > 0, y < 0) weakens the weight
    assert dw[0, 1] < 0
    # Inactive presynaptic unit -> its row is unchanged
    assert torch.all(dw[1] == 0)


def test_lrn_instar_behavior():
    """
    `lrn_instar` (Grossberg): Δw = ss * (x - w) * y. For an active output the
    weights move *toward* x; an inactive output leaves its column unchanged;
    w == x is a fixed point (zero update).
    """
    ss = 0.5
    x = torch.tensor([1.0, 0.0])
    y = torch.tensor([1.0, 0.0])    # output 0 active, output 1 inactive
    w = torch.tensor([[0.0, 9.0],   # w[0,0]=0 < x[0]=1 -> should increase
                      [3.0, 9.0]])  # w[1,0]=3 > x[1]=0 -> should decrease

    dw = fc.lrn_instar(x, w, y, ss=ss) - w

    # Exact rule
    assert torch.allclose(dw, ss * (x[:, None] - w) * y[None, :])
    # Active output: weights move toward x (up if below, down if above)
    assert dw[0, 0] > 0
    assert dw[1, 0] < 0
    # Inactive output -> its column is unchanged
    assert torch.all(dw[:, 1] == 0)
    # Fixed point: w == x (broadcast across outputs) gives zero update
    w_fp = x[:, None].repeat(1, 2)
    assert torch.allclose(fc.lrn_instar(x, w_fp, y, ss=ss) - w_fp, torch.zeros(2, 2))


def test_lrn_oja_behavior():
    """
    `lrn_oja`: Δw = ss * (x - w*y) * y -- Hebbian growth with a normalizing
    decay, so a large weight is *decreased* where plain Hebb would grow it;
    w*y == x is a fixed point.
    """
    ss = 0.1
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 1.0])
    w = torch.tensor([[1.0, 0.0],
                      [0.5, 0.5]])

    dw = fc.lrn_oja(x, w, y, ss=ss) - w

    # Exact rule
    assert torch.allclose(dw, ss * (x[:, None] - w * y[None, :]) * y[None, :])
    # Normalization: w[0,0]=1, x=1, y=2 -> Oja decreases it, whereas plain Hebb
    # (x*y = 2 > 0) would increase it
    assert dw[0, 0] < 0
    # Fixed point: w[i,j] = x[i] / y[j] gives zero update
    w_fp = x[:, None] / y[None, :]
    assert torch.allclose(fc.lrn_oja(x, w_fp, y, ss=ss) - w_fp,
                          torch.zeros(2, 2), atol=1e-6)


def test_lrn_adaptive_behavior():
    """
    `lrn_adaptive` (BCM): Δw = ss * x * y * (y - θ), sliding threshold θ = y[3].
    A response above threshold potentiates (Δw > 0), below threshold depresses
    (Δw < 0). Takes the full activations lists.
    """
    ss = 0.1
    x0 = torch.tensor([1.0, 1.0])
    theta = torch.tensor([1.0, 1.0])       # the sliding threshold (y[3])
    y0 = torch.tensor([2.0, 0.5])          # output 0 above theta, output 1 below
    x = [x0, torch.zeros(2), x0, x0 ** 2]  # activs layout: [actual, exp, avg, avg_sq]
    y = [y0, torch.zeros(2), y0, theta]    # y[3] = threshold

    w = torch.zeros(2, 2)
    dw = fc.lrn_adaptive(x, w, y, ss=ss) - w

    # Exact rule
    assert torch.allclose(
        dw, ss * x0[:, None] * y0[None, :] * (y0[None, :] - theta[None, :]))
    # Above threshold (y=2 > theta=1) potentiates
    assert dw[0, 0] > 0
    # Below threshold (y=0.5 < theta=1) depresses
    assert dw[0, 1] < 0


def test_spike_threshold():
    """`spike` binarizes at the threshold: x >= threshold -> 1, x < threshold -> 0."""
    x = torch.tensor([0.5, 0.99, 1.0, 1.5, -3.0])
    assert torch.equal(fc.spike(x), torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0]))   # default thr=1.0
    assert torch.equal(fc.spike(x, threshold=0.5), torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]))


def test_atv_discretizes_then_projects():
    """Activity rule is spike-then-matmul: the *input* is binarized at the threshold
    before the projection, so sub-threshold inputs contribute nothing."""
    x = torch.tensor([0.5, 2.0])          # unit 0 sub-threshold, unit 1 active
    w = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])
    # spike(x) = [0, 1] -> only row 1 of w survives
    assert torch.allclose(fc.atv(x, w), torch.tensor([3.0, 4.0]))
    assert torch.allclose(fc.atv(x, w), fc.spike(x) @ w)


def test_inhibit_equals_rank1_matmul():
    """Lateral inhibition's O(d) scale-and-sum form must equal the full O(d^2) matmul
    against W = A/(d-1) * (I - 11^T) (the optimization the inhibit() comment claims).
    Only the actual activations (x[0]) change; expectations/averages pass through."""
    torch.manual_seed(0)
    d, A = 8, 5.0
    x0 = torch.randn(d) + 1.0                          # straddles the spike threshold
    x1 = torch.rand(d) + 0.3                           # straddles the 0.8 expectation gate
    x = [x0, x1, torch.randn(d), torch.randn(d)]       # activs: [actual, expectation, avg, avg_sq]

    out = fc.inhibit(x)

    expected = fc.spike(x0) * torch.where(x1 < 0.8, 0.0, 1.0)
    W = (A / (d - 1)) * (torch.eye(d) - torch.ones(d, d))
    assert torch.allclose(out[0], x0 + expected @ W, atol=1e-5)   # rank-1 form == full matmul
    assert torch.equal(out[1], x1)        # expectation passes through
    assert torch.equal(out[2], x[2])      # average passes through
    assert torch.equal(out[3], x[3])      # average-of-squares passes through


def test_lrn_discrete_truth_table():
    """`lrn_discrete` (in-band weight, so the regulation gate ~= 1): when the presynaptic
    unit fires (x >= 1), y >= 1 potentiates, y <= -1 depresses, |y| < 1 decays toward 0;
    a silent presynaptic unit (x < 1) leaves the weight unchanged regardless of y."""
    x_on, x_off = torch.tensor([2.0]), torch.tensor([0.5])
    w = torch.tensor([[0.01]])                         # small -> inside the regulation band

    assert (fc.lrn_discrete(x_on, w, torch.tensor([2.0])) - w).item() > 0     # y>=1 -> up
    assert (fc.lrn_discrete(x_on, w, torch.tensor([-2.0])) - w).item() < 0    # y<=-1 -> down
    assert (fc.lrn_discrete(x_on, w, torch.tensor([0.0])) - w).item() < 0     # |y|<1 -> decay to 0
    assert torch.allclose(fc.lrn_discrete(x_off, w, torch.tensor([2.0])), w)  # x<1 -> no change


def test_lrn_discrete_regulation_gate_attenuates_large_weights():
    """The Gaussian regulation gate exp(-(w/band)^2) suppresses updates for weights well
    outside the band, so an out-of-band weight barely moves where an in-band one would."""
    x, y = torch.tensor([2.0]), torch.tensor([2.0])    # x active, y>=1 -> would potentiate
    in_band = (fc.lrn_discrete(x, torch.tensor([[0.05]]), y) - 0.05).abs().item()
    out_band = (fc.lrn_discrete(x, torch.tensor([[1.0]]), y) - 1.0).abs().item()
    assert in_band > 1e-3            # in-band weight gets a real update
    assert out_band < 1e-6           # out-of-band weight is essentially frozen by the gate
    assert out_band < in_band / 1000


def test_density_and_dist():
    """`density` = fraction of entries >= threshold; `dist` = Euclidean distance."""
    assert fc.density(torch.tensor([0.5, 1.5, 2.0, 0.9])) == 0.5
    assert fc.density(torch.tensor([2.0, 2.0]), 1.0) == 1.0
    assert fc.dist([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)
