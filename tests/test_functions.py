
import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import torch

import src.funcs as fc

def test_activity_rule_shapes():
    D_X = 321
    D_Y = 257
    x = torch.randn(D_X)
    w = torch.randn(D_X, D_Y)
    d_y_actual, = fc.atv(x, w).shape
    assert d_y_actual == D_Y
