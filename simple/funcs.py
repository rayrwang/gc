
import math

from src.funcs import dist

THRESHOLD = 1

def spike(x):
    return 0 if x < THRESHOLD else 1

def update(x):
    return math.tanh(x) if x < THRESHOLD else 0

def atv(x, w):
    return spike(x) * w

def lrn(x, w, y, ss=0.1):
    ss = ss * min(math.exp(3*(y-1)), 1)  # Smaller ss for more negative y (unchanged ss for y > 1)
    ss = ss * math.exp(-w**2 / 0.01)  # Smaller ss for larger absolute value of weight

    if ss < 1e-2:
        return w

    x = spike(x)
    y = spike(y)

    correl = 2*x*y - x
    return w + ss*correl
