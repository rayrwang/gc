
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
