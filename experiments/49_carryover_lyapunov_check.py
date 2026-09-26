"""
Re-measure run 39's carry-over Lyapunov exponents without its per-step twin
re-sync (2026-09-26, found while answering RW's "also itnresting that the
acti ity is sitll cahoatic even with high kepep").

The suspicion. Run 39's Benettin loop re-syncs the twin every step with
twin_sync, which sets the twin's committed activity to the host's plus a
small displacement but copies the host's accumulator nr_1_ unchanged. With
keep > 0 the accumulator is what carries the state into the next step
(committed S, then nr_1_ = keep * S), so the displacement never enters the
carried part: the twin's next state is keep*S_host + contribution(f(S+d))
instead of keep*(S+d) + contribution(f(S+d)). The measured exponent then
follows the contribution path alone and loses the keep*I term of the
Jacobian, which biases it downward more the larger keep is; at keep 0 the
accumulator is zeroed every step and nothing is lost. Separately, at keep
0.9 with the norm off the per-unit renormalised separation (RENORM/sqrt(n),
2.5e-6) is below float32 resolution on about half the units (states near 76),
and PERT 1e-8 on all of them.

The check. No re-sync: one twin per direction, put at the host's full state
(bulk and input columns; see 47's full_sync) at step 400 with the bulk
activity displaced by a random vector of relative size 1e-6 and the
accumulator displaced consistently (keep times the displacement times the
column's rms; without that, a piecewise-constant activation such as sign or
wta loses the displacement in one step at any keep, which is what the first
run of this script and 47's small-eps rows showed), then both are driven by
the same input. The growth rate is the least-squares slope of
log(relative distance) against step while the distance is between 1e-12
and 1e-2 relative (the linear regime), after a 5-step settling window,
from at least 3 consecutive points; three
directions per cell. Compared against run 39's committed exponent for the
same cell (actdyn39.json, seed 50).

Premises (run 39 cells): BareAgtXC, 49 bulk columns of 32 units plus one
input column; p_a 0.3, p_e 0.6, falloff 1/d, input_fanout 3, conn_scale 2.0;
learning frozen; kp1 input, ns comp25, seed 50; tri0.3, sign and wta; keep
{0, 0.25, 0.5, 0.75, 0.9}; norm off and on.

Expected for wta with the norm off: its output is piecewise constant, so a
displacement that flips no winner shrinks by exactly keep per step, a rate
of ln(keep) (-0.105 at 0.9); at keep 0 it vanishes in one step.

Usage:
    venv/bin/python experiments/49_carryover_lyapunov_check.py
    venv/bin/python experiments/49_carryover_lyapunov_check.py tri0.3 0.9 off
"""

import importlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch

r39 = importlib.import_module("39_activation_dynamics")
c47 = importlib.import_module("47_wta_chaos")
sweep, de, T = r39.sweep, r39.de, r39.T

SEED, LAUNCH, HORIZON, EPS, NDIR, SETTLE = 50, 400, 600, 1e-6, 3, 5
ACTS = ("tri0.3", "sign", "wta")
KEEPS = (0.0, 0.25, 0.5, 0.75, 0.9)
LOG = os.path.join(HERE, "lyapcheck49.log")


def slope(ts, ys):
    n = len(ts)
    mt, my = sum(ts) / n, sum(ys) / n
    den = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / den if den else float("nan")


def cell(act, keep, norm):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    torch.manual_seed(11)
    r39._patch(act, keep, norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=sweep.N_COLS, **r39.arm_cfg(act))
    base = os.path.join(ROOT, "saves", "lyapcheck49", f"{act}_{keep}_{norm}")
    A = sweep.BareAgtXC(cfg, base + "_a", seed=SEED, dire=True, frozen=True)
    twins = [sweep.BareAgtXC(cfg, base + f"_b{i}", seed=SEED, dire=True, frozen=True) for i in range(NDIR)]
    bulk = [loc for loc in A.cols if not A.is_i(loc)]
    stream = list(de.kp1_cycle(LAUNCH + HORIZON + 1, seed=de.derive_seed("kp1", r39.NS, SEED)))
    for t in range(LAUNCH + 1):
        A.step([stream[t]])
    s = r39.state_of(A)
    if not torch.isfinite(s).all():
        return f"{act:6} keep {keep:<4} norm {'on ' if norm else 'off'}: host DIVERGED"
    for B in twins:
        d = torch.randn_like(s)
        d = d / d.norm() * EPS * s.norm()
        c47.full_sync(A, B, bulk, s + d)
        # carry the displacement into the accumulator too: after a commit the
        # accumulator holds keep * (the committed activity * that column's rms)
        i = 0
        for loc in bulk:
            c, hc = B.cols[loc], A.cols[loc]
            n = c.nr_1.actual.numel()
            c.nr_1_.actual = hc.nr_1_.actual + keep * d[i:i + n].view_as(c.nr_1.actual) * hc.nr_1.rms_avg
            i += n
    curves = [[] for _ in twins]
    for t in range(LAUNCH + 1, LAUNCH + HORIZON + 1):
        A.step([stream[t]])
        sa = r39.state_of(A)
        for B, cur in zip(twins, curves):
            B.step([stream[t]])
            cur.append(float((sa - r39.state_of(B)).norm() / sa.norm()))
    rates, notes = [], []
    for cur in curves:
        pts = [(k, math.log(v)) for k, v in enumerate(cur) if k >= SETTLE and 1e-12 < v < 1e-2]
        # stop at the first exit from the window, so saturation is not averaged in
        run, prev = [], None
        for k, y in pts:
            if prev is not None and k != prev + 1:
                break
            run.append((k, y)); prev = k
        zero_at = next((k for k, v in enumerate(cur) if v == 0.0), None)
        if len(run) >= 3:
            rates.append(slope([k for k, _ in run], [y for _, y in run]))
        notes.append(f"window {run[0][0] if run else '-'}-{run[-1][0] if run else '-'}"
                     + (f", distance exactly 0 from step {zero_at}" if zero_at is not None else "")
                     + f", final {cur[-1]:.1e}")
    ref = next((r.get("lyap") for r in json.load(open(os.path.join(HERE, "actdyn39.json")))
                if r["act"] == act and r["keep"] == keep and r["norm"] == norm and r["seed"] == SEED), None)
    rate = sum(rates) / len(rates) if rates else float("nan")
    return (f"{act:6} keep {keep:<4} norm {'on ' if norm else 'off'}: growth rate {rate:+.3f}/step "
            f"(directions: {', '.join(f'{x:+.3f}' for x in rates) or 'none in window'}); "
            f"run 39 committed {ref if ref is None else f'{ref:+.3f}'}; " + " | ".join(notes))


def main():
    if len(sys.argv) == 4:
        print(cell(sys.argv[1], float(sys.argv[2]), sys.argv[3] == "on"))
        return
    jobs = [(a, k, n) for a in ACTS for n in (False, True) for k in KEEPS]
    with ProcessPoolExecutor(20) as ex:
        lines = list(ex.map(cell, *zip(*jobs)))
    with open(LOG, "a") as fh:
        for line in lines:
            print(line)
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
