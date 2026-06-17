
import pytest
import inspect

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
