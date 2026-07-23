
"""Run the examples that fit a CI machine (CPU-only, ~4 cores, no display)
end to end as subprocesses and check they exit clean with their expected
final output.

Included and why:
    08_bcm_stability        batched phase-diagram sweep; ~1 min per CI core
                            budget (2s on a GPU), regenerates a committed
                            asset, so it doubles as a regression check
    13_wta_geometry         seconds everywhere
    14_expectations_one_layer   pure CPU small tensors, ~7s
    15_expectations_deep        same, ~5s

Excluded and why (deliberate, not oversight):
    00-03, 16   endless interactive loops with the live pygame debugger:
                no completion to assert; smoke-testing them means
                kill-after-timeout, which CI turns into flakes
    04, 06, 09-12   GPU-scale batch runs (minutes to an hour on a
                    workstation GPU; hours on CI CPU)
    05, 07      complete in seconds locally but pull the MNIST/FashionMNIST
                downloads; kept out so CI stays hermetic and fast

The examples are executed unmodified; anything needing a smaller CI-shaped
configuration should get it as an explicit constant in the example itself,
not a patch here.
"""

import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "examples")

CASES = [
    pytest.param("08_bcm_stability.py", "saved", 900, id="08_bcm_stability"),
    pytest.param("13_wta_geometry.py", "saved", 120, id="13_wta_geometry"),
    pytest.param("14_expectations_one_layer.py", "dir.e on", 300,
                 id="14_expectations_one_layer"),
    pytest.param("15_expectations_deep.py", "cells: predict / persistence baseline",
                 300, id="15_expectations_deep"),
]


@pytest.mark.parametrize(("script", "marker", "timeout"), CASES)
def test_example_runs_clean(script, marker, timeout):
    result = subprocess.run(
        [sys.executable, os.path.join(EXAMPLES, script)],
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, \
        f"{script} exited {result.returncode}\nstderr tail:\n{result.stderr[-2000:]}"
    assert marker in result.stdout, (
        f"{script} finished but its expected output marker is missing\n"
        f"stdout tail:\n{result.stdout[-2000:]}"
    )
