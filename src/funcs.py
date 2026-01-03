
import math
import platform

import torch

"""
NOTE:

atv returns change,
while all others return new value,
since multiple atv's need to happen at the same time,
while the others don't have special requirements
and returning new value is easier to use
"""

disable_compile = (platform.system() == "Windows")

def spike(x, threshold=1.0):
    """
    `d, () -> d`

    activation function
    """
    return torch.where(x < threshold, 0.0, 1.0)


def atv(x, w, y, threshold=1.0):
    """
    `d_x, (d_x d_y), d_y, () -> d_y` 
    
    activity rule
    """
    return spike(x, threshold=threshold) @ w


inhibit_weights = {}
def inhibit(x):
    """
    `[d, d] -> [d, d]`

    lateral inhibition for winner take all behavior
    """
    d = x[0].shape[0]
    weights = inhibit_weights.get(d)
    if weights is None:
        # 0's down diagonal, -A / (d-1) everywhere else
        A = 5.0
        weights = (0 + A/(d-1))*torch.eye(d, device=torch.get_default_device(), dtype=torch.get_default_dtype()) \
            - torch.full((d, d), A/(d-1), device=torch.get_default_device(), dtype=torch.get_default_dtype())
    THRESHOLD = 0.8
    # Where both activations AND expectations are above threshold
    expected = spike(x[0]) * torch.where(x[1] < THRESHOLD, 0.0, 1.0)
    return [x[0] + expected @ weights, x[1]]


@torch.compile(disable=disable_compile)
def lrn(x, w, y, ss=1e-2):
    """
    `d_x, (d_x d_y), d_y, () -> (d_x d_y)`

    learning rule

       x  y             Δw  
     < 1, any        -> 0  
    >= 1, < -1       -> -
    >= 1, [-1, 1)    -> towards 0
    >= 1, >= 1       -> +
    """

    d_x, = x.shape
    d_y, = y.shape

    assert w.shape[0:] == (d_x, d_y), \
        f"Expected weights of shape {(d_x, d_y)} but got shape {w.shape[0:]}!"

    xr = x.repeat(d_y, 1).T  # (d_x d_y)
    yr = y.repeat(d_x, 1)    # (d_x d_y)

    # Calculate the changes to the weights in the 3 cases

    # Case 1: y >= 1, add ss to weight
    excite = torch.where(yr >= 1, ss, 0)

    # Case 2: -1 <= y < 1, decay weight
    weaken = torch.where(torch.logical_and(-1 <= yr, yr < 1), 1.0, 0.0) \
        * (0.9*w - w)

    # Case 3: y < -1, subtract ss from weight
    inhibit = torch.where(yr < -1, -ss, 0.0)

    xr = spike(xr)  # Only change weights when x >= 1
    changes = xr * (excite + weaken + inhibit)

    # Scale changes for regulation
    changes = changes * torch.exp(-(w / (0.1 * d_x**-0.5))**2)

    return w + changes


def update(x, threshold=1.0):
    """
    `d_x, () -> d_x`
    
    reset or decay activations after applying activity rule
    """
    return torch.zeros(x.shape)


def update_e(x):
    """
    `d_x -> d_x`
    
    Update expectations
    """
    return torch.zeros(x.shape)


def dist(x, y, /):
    """`d, d -> ()`"""
    assert len(x) == len(y)
    return math.sqrt(sum([(x_i-y_i)**2 for x_i, y_i in zip(x,y)]))

