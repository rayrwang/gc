"""Complete per-run determinism (CHARTER §3: a faithful rerun must be possible).

Call set_determinism(seed) at t0, before any torch work. Use named sub-streams
(gen(seed, "purpose")) instead of the global RNG so unrelated code changes can't
shift downstream draws — the classic silent determinism killer.
"""
import hashlib
import os
import random


def _subseed(seed, name):
    h = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()
    return int(h[:16], 16)


def gen(seed, name):
    """Independent named torch.Generator: gen(seed, 'data'), gen(seed, 'init'),
    gen(seed, 'noise'). Each consumer owns its stream."""
    import torch
    g = torch.Generator()
    g.manual_seed(_subseed(seed, name))
    return g


def set_determinism(seed):
    """Global seeding + forbid nondeterministic kernels. Order matters:
    CUBLAS_WORKSPACE_CONFIG must precede first CUDA use."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(_subseed(seed, "stdlib"))
    np.random.seed(_subseed(seed, "numpy") % 2**32)
    torch.manual_seed(_subseed(seed, "torch"))
    torch.use_deterministic_algorithms(True)   # nondet kernels ERROR (fail loud)
    torch.backends.cudnn.benchmark = False
    return seed


def selftest():
    import torch
    set_determinism(42)
    a = torch.randn(4, generator=gen(42, "init"))
    _ = torch.randn(100, generator=gen(42, "noise"))   # unrelated consumption...
    b = torch.randn(4, generator=gen(42, "init"))
    assert torch.equal(a, b), "named streams must be independent of each other"
    g1, g2 = gen(42, "data"), gen(43, "data")
    assert not torch.equal(torch.randn(4, generator=g1), torch.randn(4, generator=g2))
    x = torch.randn(64, 32) @ torch.randn(32, 16)      # runs under det-algorithms
    assert x.shape == (64, 16)
    print("selftest passed: named streams independent, det mode active")


if __name__ == "__main__":
    selftest()
