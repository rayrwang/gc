
import math
import platform

import torch

disable_compile = (platform.system() == "Windows")  # Issues

# atv returns change, while all others return new value,
# since multiple atv's need to happen at the same time,
# while the others don't have special requirements
# and returning new value is easier to use.


def spike(x, threshold=1.0):
    """
    `d, () -> d`

    Activation function
    """
    return torch.where(x < threshold, 0.0, 1.0)


def atv(x, w, y=None, threshold=1.0):
    """
    `d_x, (d_x d_y), d_y | None, () -> d_y` 
    
    Activity rule

    TODO possible changes:
    - Adaptive thresholds by taking into account average values of activations
    """
    if y is not None:
        d_x, = x.shape
        d_y, = y.shape
        assert (d_x, d_y) == w.shape, (
            f"Shape mismatch in activity rule:\n"
            f"|-- Activations: {d_x} to {d_y}\n"
            f"|-- Weights: {tuple(w.shape)}"
        )
    return spike(x, threshold=threshold) @ w


inhibit_weights = {}
def inhibit(x, disable=False):
    """
    `Activs, bool -> Activs`

    Lateral inhibition for winner take all behavior,
    to have less (more) activity for (un)expected
    """
    if disable:
        return x

    d = x[0].shape[0]
    weights = inhibit_weights.get(d)
    if weights is None:
        # 0's down diagonal, -A / (d-1) everywhere else
        A = 5.0
        weights = (0 + A/(d-1))*torch.eye(d) - torch.full((d, d), A/(d-1))
        inhibit_weights[d] = weights
    THRESHOLD = 0.8
    # Where both activations AND expectations are above threshold
    expected = spike(x[0]) * torch.where(x[1] < THRESHOLD, 0.0, 1.0)
    return [x[0] + expected @ weights, x[1], x[2]]


@torch.compile(disable=disable_compile)
def lrn(x, w, y, ss=1e-2, decay=0.9, reg_width=0.1, disable=False):
    """
    `d_x, (d_x d_y), d_y, (), (), (), bool -> (d_x d_y)`

    (Discrete) learning rule

       x  y             Δw  
     < 1, any        -> 0  
    >= 1, <= -1      -> -
    >= 1, (-1, 1)    -> towards 0
    >= 1, >= 1       -> +

    TODO possible changes:
    - Unify strengthening and decaying: e.g. one hyperparameter, consider adding vs. multiplying
    - Less restrictive regulation: e.g. take into account more than just value of weight
    """
    if disable:
        return w

    d_x, = x.shape
    d_y, = y.shape

    assert (d_x, d_y) == w.shape, (
        f"Shape mismatch in learning rule:\n"
        f"|-- Activations: {d_x} to {d_y}\n"
        f"|-- Weights: {tuple(w.shape)}"
    )
    
    xr = x.repeat(d_y, 1).T  # (d_x d_y)
    yr = y.repeat(d_x, 1)    # (d_x d_y)

    # Calculate the changes to the weights in the 3 cases

    # Case 1: y >= 1, add ss to weight
    excite = torch.where(yr >= 1, ss, 0)

    # Case 2: -1 < y < 1, decay weight
    weaken = torch.where(torch.logical_and(-1 < yr, yr < 1), 1.0, 0.0) \
        * (decay*w - w)

    # Case 3: y <= -1, subtract ss from weight
    inhibit = torch.where(yr <= -1, -ss, 0.0)

    # Only change weights when x >= 1
    changes = spike(xr) * (excite + weaken + inhibit)

    # Scale changes for regulation
    changes = changes * torch.exp(-(w / (reg_width * d_x**-0.5))**2)

    return w + changes


def lrn_adaptive(x, w, y, ss=1e-2, disable=False):
    """
    `Activs, (d_x d_y), Activs, (), bool -> (d_x d_y)`

    Adaptive learning rule that takes into account average values of activations
    """
    if disable:
        return w

    d_x, = x[0].shape
    d_y, = y[0].shape

    assert (d_x, d_y) == w.shape, (
        f"Shape mismatch in adaptive learning rule:\n"
        f"|-- Activations: {d_x} to {d_y}\n"
        f"|-- Weights: {tuple(w.shape)}"
    )

    return w  # TODO


def update(x, threshold=1.0):
    """
    `d_x, () -> d_x`
    
    Reset activations after applying activity rule
    """
    return torch.zeros(x.shape)


def update_e(x):
    """
    `d_x -> d_x`
    
    Reset expectations
    """
    return torch.zeros(x.shape)


def dist(x, y, /):
    """`d, d -> ()`"""
    assert len(x) == len(y)
    return math.sqrt(sum([(x_i-y_i)**2 for x_i, y_i in zip(x,y)]))

