
import inspect
import math
import platform

import torch

disable_compile = (platform.system() == "Windows")  # Issues

# atv returns change, while all others return new value,
# since multiple atv's need to happen at the same time,
# while the others don't have special requirements
# and returning new value is easier to use.


def check_shapes(
        x_shape: int,
        w_shape: tuple[int, int],
        y_shape: int,
        name: str | None = None) -> None:
    if name is None:
        # Fallback to function name
        name = f"`{inspect.currentframe().f_back.f_code.co_name}`"  # ty: ignore[unresolved-attribute]
    assert (x_shape, y_shape) == w_shape, (
        f"Shape mismatch in {name}:\n"
        f"|-- Activations: {x_shape} to {y_shape}\n"
        f"|-- Weights: {w_shape}"
    )


def spike(x, threshold=1.0):
    """
    `d, () -> d`

    Heaviside step activation function
    """
    return torch.where(x < threshold, 0.0, 1.0)


def atv(x, w, y=None, threshold=1.0):
    """
    `d_x, (d_x d_y), d_y | None, () -> d_y`

    Activity rule using step activation function

    TODO possible changes:
    - Adaptive thresholds by taking into account average values of activations
    """
    if y is not None:
        check_shapes(x.shape[0], tuple(w.shape), y.shape[0], "activity rule")
    return spike(x, threshold=threshold) @ w


def inhibit(x):
    """
    `Activs d -> Activs d`

    Lateral inhibition for winner take all behavior,
    to have less (more) activity & change for (un)expected

    Global subtractive normalization: each unit is suppressed by A/(d-1) times
    the total expected activity of all other units. The inhibition matrix is
    `W = A/(d-1) * (I - 11^T)` (0 on the diagonal, -A/(d-1) elsewhere), which is
    a diagonal + rank-1 map, so `expected @ W == A/(d-1) * (expected - sum)`,
    an O(d) scale-and-sum rather than an O(d^2) matmul.

    TODO possible changes:
    - Other ways of integrating expectations e.g. Outstar, predictive coding
    - Structured (distance-tuned) surround: sum over neighbours instead of all
    """
    from src.agents import Activs  # Avoid circular import
    A = 5.0
    THRESHOLD = 0.8
    d = x.actual.shape[0]
    # Where both activations AND expectations are above threshold
    expected = spike(x.actual) * torch.where(x.expect < THRESHOLD, 0.0, 1.0)
    inhib = (A / (d - 1)) * (expected - expected.sum())
    return Activs(x.actual + inhib, x.expect, x.avg, x.avg_sq, x.rms_avg)


def triangle(u, power=0.7):
    """
    `d, () -> d`

    Triangle activation function

    A graded soft competition: units above the mean fire
    """
    return torch.relu(u - u.mean()) ** power


def triangle_batched(u, power=0.7):
    """
    `(n d), () -> (n d)`
    
    See `triangle`
    """
    return torch.relu(u - u.mean(-1, keepdim=True)) ** power


def atv_triangle(x, w, power=0.7):
    """
    `d_x, (d_x d_y), () -> d_y`

    Activity rule using triangle activation function
    """
    return triangle(x, power=power) @ w


def softmax_wta(u, beta=1.0, signed=False):
    """
    `d, (), () -> d`

    Soft winner-take-all gate: computes the learning credit each unit gets.
    - `signed=False`: all non-negative (winner most, losers a little).
    - `signed=True` (SoftHebb): the winner keeps +softmax,
        every loser is negated -> anti-Hebbian repulsion.

    Pair with `lrn_oja_gated` which uses the gate to learn.
    """
    resp = torch.softmax(beta * u, dim=-1)
    if not signed:
        return resp
    gate = -resp
    gate[u.argmax()] *= -1
    return gate


def softmax_wta_batched(u, beta=1.0, signed=False):
    """`(n d), (), () -> (n d)`"""
    resp = torch.softmax(beta * u, dim=-1)
    if not signed:
        return resp
    gate = -resp
    gate[torch.arange(u.shape[0]), u.argmax(-1)] *= -1
    return gate


@torch.compile(disable=disable_compile)
def lrn_discrete(x, w, y, ss=1e-2, decay=0.9, reg_width=0.1):
    """
    `d_x, (d_x d_y), d_y, (), (), () -> (d_x d_y)`

    TODO old, mostly inert, placeholder (discrete) learning rule

       x  y             Δw
     < 1, any        -> 0
    >= 1, <= -1      -> -
    >= 1, (-1, 1)    -> towards 0
    >= 1, >= 1       -> +

    TODO possible changes:
    - Unify strengthening and decaying: e.g. one hyperparameter, consider adding vs. multiplying
    - Less restrictive regulation: e.g. take into account more than just value of weight
    """
    d_x, = x.shape
    d_y, = y.shape

    check_shapes(d_x, tuple(w.shape), d_y, "learning rule")

    xu = x[:, None]  # (d_x 1)
    yu = y[None, :]  # (1 d_y)

    # Calculate the changes to the weights in the 3 cases

    # Case 1: y >= 1, add ss to weight
    excite = torch.where(yu >= 1, ss, 0)

    # Case 2: -1 < y < 1, decay weight
    weaken = torch.where(torch.logical_and(yu > -1, yu < 1), 1.0, 0.0) \
        * (decay*w - w)

    # Case 3: y <= -1, subtract ss from weight
    inhibit = torch.where(yu <= -1, -ss, 0.0)

    # Only change weights when x >= 1
    changes = spike(xu) * (excite + weaken + inhibit)

    # Scale changes for regulation
    changes = changes * torch.exp(-(w / (reg_width * d_x**-0.5))**2)

    return w + changes


@torch.compile(disable=disable_compile)
def lrn_instar(x, w, y, ss=1e-2):
    """
    `d_x, (d_x d_y), d_y, () -> (d_x d_y)`

    Grossberg's Instar rule

    Δw ∝ (x-w)y
    """
    d_x, = x.shape
    d_y, = y.shape

    check_shapes(d_x, tuple(w.shape), d_y, "instar learning rule")

    xu = x[:, None]  # (d_x 1)
    yu = y[None, :]  # (1 d_y)

    return w + ss * (xu-w) * yu
@torch.compile(disable=disable_compile)
def lrn_instar_d(x, w, y, ss=1e-2):
    return lrn_instar(spike(x), w, y, ss=ss)


@torch.compile(disable=disable_compile)
def lrn_oja(x, w, y, ss=1e-2):
    """
    `d_x, (d_x d_y), d_y, () -> (d_x d_y)`

    Oja rule

    Δw ∝ (x-wy)y
    """
    d_x, = x.shape
    d_y, = y.shape

    check_shapes(d_x, tuple(w.shape), d_y, "oja learning rule")

    xu = x[:, None]  # (d_x 1)
    yu = y[None, :]  # (1 d_y)

    return w + ss * (xu-w*y) * yu
@torch.compile(disable=disable_compile)
def lrn_oja_d(x, w, y, ss=1e-2):
    return lrn_oja(spike(x), w, y, ss=ss)


def lrn_oja_gated(x, w, gate, u, ss=1e-2):
    """
    `d_x, (d_x d_y), d_y, d_y, () -> (d_x d_y)`

    Gated Oja rule (SoftHebb), the gate carries the sign

    Δw ∝ outer(x, gate) - w*(gate*u)

    Pair with `softmax_wta` which provides the gate
    """
    return w + ss * (torch.outer(x, gate) - w * (gate * u))


def lrn_oja_gated_batched(x, w, gate, u, ss=1e-2):
    """
    `(n d_x), (d_x d_y), (n d_y), (n d_y), () -> (d_x d_y)`
    
    See `lrn_oja_gated`
    """
    n = x.shape[0]
    return w + ss * (x.T @ gate - w * (gate * u).sum(0, keepdim=True)) / n


@torch.compile(disable=disable_compile)
def lrn_adaptive(x, w, y, ss=1e-2):
    """
    `Activs d_x, (d_x d_y), Activs d_y, () -> (d_x d_y)`

    BCM learning rule which takes into account average values of activations

    Δw ∝ x * y * (y-y_avg)
    """
    d_x, = x.actual.shape
    d_y, = y.actual.shape

    check_shapes(d_x, tuple(w.shape), d_y, "adaptive learning rule")

    xu = x.actual[:, None]       # (d_x 1)
    yu = y.actual[None, :]       # (1 d_y)
    y_avg_u = y.avg_sq[None, :]  # (1 d_y)

    return w + ss * xu * yu * (yu-y_avg_u)
@torch.compile(disable=disable_compile)
def lrn_adaptive_d(x, w, y, ss=1e-2):
    from src.agents import Activs  # Avoid circular import
    return lrn_adaptive(Activs(spike(x.actual), x.expect, x.avg, x.avg_sq), w, y, ss=ss)


# Default learning rule to expose
# TODO remove?
lrn = lrn_discrete


def update(x, threshold=1.0):
    """
    `d_x, () -> d_x`

    Reset activations after applying activity rule
    """
    return torch.zeros_like(x)


def update_e(x):
    """
    `d_x -> d_x`

    Reset expectations
    """
    return torch.zeros_like(x)


def dist(x, y, /):
    """`d, d -> ()`"""
    assert len(x) == len(y)
    return math.sqrt(sum([(x_i-y_i)**2 for x_i, y_i in zip(x, y, strict=True)]))


def density(x: torch.Tensor, threshold: float = 1.0, /) -> float:
    """Proportion of elements of x over threshold; NaN if x has any NaN"""
    if torch.isnan(x).any():
        return float("NaN")
    return ((x >= threshold).sum() / x.numel()).item()


# Archive #####################################################################
@torch.compile(disable=disable_compile)
def lrn_basic(x, w, y, ss=1e-4):
    """
    `d_x, (d_x d_y), d_y, () -> (d_x d_y)`

    Simplest possible learning rule

    Δw ∝ xy
    """
    d_x, = x.shape
    d_y, = y.shape

    check_shapes(d_x, tuple(w.shape), d_y, "basic learning rule")

    return w + ss*torch.outer(x, y)
@torch.compile(disable=disable_compile)
def lrn_basic_d(x, w, y, ss=1e-4):
    """
    Same as above but discrete in inputs
    """
    return lrn_basic(spike(x), w, y, ss=ss)
