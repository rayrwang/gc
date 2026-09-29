"""
When does activity die out? (2026-09-29)

RW's ask, 2026-09-29 about 00:25 EDT: "basically under what conditions does the
activity die out or not (e.g. due to the weights magnitudes (scale) being too small,
alognside no input, and toher possible factors)? since liek you said the wta doenst
die out sindce always sends 1 but for example others liek the spike or mabey sigmoid
would die out? run an expeirmnt to find out", beside the criticality diagnostic of
gc-project 4 (activity decaying from the start = subcritical, growing then flattening
= supercritical and self-limited, flat = at the edge), which he read as related.

On record so far (09-26, experiments 46 and 48): wta cannot die (every column sends
one 1.0 whatever its input, and the winner does not depend on weight scale); its
all-zero and positive-max variants never went silent either; tri0.3 did not die at
small scales (x^0.3 amplifies small values); Claude's claim then, untested beyond
those: "Only activations with a threshold above zero can die from small weights".

Premises: 52's harness (BareAgtXC from 10_sweep_bare_x, learning frozen, the
activation on the sending side, the RMS norm's divisor floored at 0.9 so it never
amplifies low activity). The network starts from random activity (activs(): a
standard normal draw per unit, in the committed state and the accumulator alike), so
every run begins with a kick. Main grid on run 39's network (49 columns, dist_exp 1,
p_a 0.3): activation {raw, relu, rrelu1 = relu(x - 1), tri0.3, tri1.0, tanh,
sig4_0, sig4_1 = sigmoid(4(x - 1)), step0, step0.5, step1 = fc.spike, step2, cstep,
sign, wta, kwta4} x conn_scale {0.1, 0.3, 1, 2 (default), 4, 16} x input {none; kp1
into 3 columns} x keep {0, 0.5, 0.9} x norm {on, off} x seeds {50, 51}, with keep 0.5 and 0.9 run twice (below): 3,840 runs.
Side grid on example 16's network (200 columns, dist_exp 2, p_a 0.3): the same 16
activations x the 6 scales x input {none; passive MNIST into every column} x keep {0,
0.9, 0.9 leak}, norm on, seed 50: 576 runs. 2000 steps each.

Carry-over arm (added before launch, 2026-09-29 about 00:50 EDT, on RW's question
"woudl it affect 55 and 56?"): the code's carry-over adds the full new input on top
of keep x, so its accumulator is the leaky average times 1/(1 - keep), 10 times at
0.9; with the norm on and activity above the norm's 0.9 floor that scale divides out
exactly, but with the norm off, or when activity is quiet and the floor binds, keep
also acts as a gain. So keep 0.5 and 0.9 run twice: as the code has it ("code") and
with the (1 - keep) factor on everything entering the accumulator, the recurrent
contributions and the input alike ("leak", a true leaky integrator, keep = e^(-dt/tau)).
Expected: for the scale-dependent activations with the norm off, "code" at keep k and
scale s behaves like "leak" at scale s / (1 - k) for the slow part of the activity;
the scale-free ones (wta, sign, step0, cstep) are unaffected.

Measured: the size of what is sent (the RMS of the activation applied to the state)
and of the state itself at steps 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000;
the outcome: blown up (not finite or above 1e30, with its step), dead (the state, the summed input, below 1e-4 of its largest size over the
first 10 steps, or below 1e-9, with the first step it happened; judged on the state
since a resting sigmoid still sends f(0), so dead means no drive left), fixed point (not dead, and the state changing by under 1e-6
of its size per step over the last 100 steps), cycle (an exact repeat with period up
to 50 in the last 100 steps), alive otherwise; the shape (Claude's reading of the
gc-project 4 diagnostic, on the sent size): decaying = the sent size at step 2000
below a tenth of its step-1 value, growing = above ten times, flat otherwise; the
share of units that sent exactly nothing over the last 100 steps (silent units) and
the share whose summed input stayed under 1e-3 of the early state size (resting
units, no drive); and the active share, the share of unit-steps in the last 100 whose
output differs by more than 0.05 from what the unit sends at zero input (a sigmoid
at its floor counts as inactive); and for
runs still moving, 52's lock measures and the participation ratio on the last 300
steps.

Expected (Claude's, written before the run):
1. The positive-threshold activations (step0.5, step1, step2, sig4_1, rrelu1) die at
   small scales with no input; with kp1 only the few columns the input reaches stay
   awake, so most units fall silent; at large scales they live.
2. The scale-free ones (sign, step0, cstep, wta, and sig4_0, whose output is 0.5 at
   0) never die at any scale; the zero-threshold, scale-dependent ones (raw, relu,
   tri1.0, tanh near 0, kwta4) decay to zero below a critical scale when there is no
   input, and blow up above it with the norm off; tri0.3 never dies (degree 0.3
   below 1).
3. Carry-over slows dying and, since the code adds the full input on top of keep x,
   moves the critical scale down (raw at keep 0.9 grows where keep 0 decays).
4. The norm cannot rescue dying (its divisor never falls below 0.9), only curb
   growth.
5. Input keeps the reached columns alive and does little for the rest when the
   fan-out is 3 (the June input-starvation finding); MNIST into every column keeps
   more alive.

Output: dieout56.log, dieout56.json beside this file.
Usage:
    venv/bin/python experiments/56_dieout.py            # all runs, in parallel
    venv/bin/python experiments/56_dieout.py smoke      # a few runs, printed
"""

import importlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

m52 = importlib.import_module("52_wiring_lock")  # sets ROOT on sys.path and chdir
r39, sweep, de, T = m52.r39, m52.sweep, m52.de, m52.T
ROOT = m52.ROOT

STEPS, TAIL, WINDOW = 2000, 100, 300
MARKS = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
ACTS = ("raw", "relu", "rrelu1", "tri0.3", "tri1.0", "tanh", "sig4_0", "sig4_1",
        "step0", "step0.5", "step1", "step2", "cstep", "sign", "wta", "kwta4")
SCALES = (0.1, 0.3, 1.0, 2.0, 4.0, 16.0)
NETS = {"run39": (49, 1.0, 0.3), "ex16": (200, 2.0, 0.3)}
LOG = os.path.join(HERE, "dieout56.log")
OUT = os.path.join(HERE, "dieout56.json")
_MNIST = None


def mnist_seq():
    global _MNIST
    if _MNIST is None:
        from src.envs import MNISTDataset
        ds = MNISTDataset(train=True)
        rng = random.Random(20260929)
        _MNIST = [ds[rng.randrange(len(ds))][0] for _ in range(STEPS)]
    return _MNIST


def sent_rms(agt, bulk, f):
    parts = [f(agt.cols[loc].nr_1.actual) for loc in bulk]
    y = torch.cat(parts)
    return float(y.pow(2).mean().sqrt()), y


def cell(net, act, scale, inp, keep, norm, seed, leak=False):
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    n_cols, dist_exp, p_a = NETS[net]
    r39._patch(act, keep, norm)
    f = r39.act_fn(act)
    if leak:  # the (1 - keep) factor on everything entering the accumulator
        from experiments import dire_hosts as dh
        dh.BareAgtX._transport = lambda self, x, w, _f=f, _k=keep: (1.0 - _k) * (_f(x) @ w)
    dim = 784 if inp == "mnist" else de.SYMBOL_DIM
    fan = {"none": 0, "kp1": 3, "mnist": -1}[inp]
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(dim)], n_cols=n_cols,
                          **dict(r39.arm_cfg(act), p_a=p_a, dist_exp=dist_exp,
                                 input_fanout=fan, conn_scale=scale))
    tag = f"{net}|{act}|{scale}|{inp}|{keep}|{int(norm)}|{seed}|{'leak' if leak else 'code'}"
    path = os.path.join(ROOT, "saves", "dieout56", f"w{os.getpid()}")
    agt = sweep.BareAgtXC(cfg, path, seed=seed, dire=True, frozen=True)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    zero = torch.zeros(dim)
    if inp == "kp1":
        xs = list(de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", r39.NS, seed)))
    elif inp == "mnist":
        xs = mnist_seq()
    else:
        xs = None
    if leak and xs is not None:
        xs = [(1.0 - keep) * x for x in xs]
    t0 = time.time()
    traj_sent, traj_state = {}, {}
    s0, _ = sent_rms(agt, bulk, f)
    traj_sent[0] = s0
    traj_state[0] = float(r39.state_of(agt).pow(2).mean().sqrt())
    x0 = traj_state[0]
    early_max, early_max_x, dead_at = s0, x0, None
    rows, prev, moves, active = [], None, [], []
    f0 = torch.cat([f(torch.zeros(cfg.d_col)) for _ in bulk])  # what a unit sends at zero input
    tail_silent = None
    for t in range(1, STEPS + 1):
        agt.step([xs[t - 1] if xs is not None else zero])
        s = r39.state_of(agt)
        if not torch.isfinite(s).all() or float(s.abs().max()) > 1e30:
            return {"tag": tag, "status": "ok", "outcome": "blowup", "blowup_step": t,
                    "sent": traj_sent, "state": traj_state, "secs": round(time.time() - t0, 1)}
        srms, y = sent_rms(agt, bulk, f)
        xrms = float(s.pow(2).mean().sqrt())
        if t <= 10:
            early_max = max(early_max, srms)
            early_max_x = max(early_max_x, xrms)
        if dead_at is None and (xrms < 1e-9 or (t > 10 and xrms < 1e-4 * early_max_x)):
            dead_at = t
        if t in MARKS:
            traj_sent[t] = srms
            traj_state[t] = float(s.pow(2).mean().sqrt())
        if t > STEPS - TAIL:
            if prev is not None:
                moves.append(float((s - prev).norm() / (prev.norm() + 1e-30)))
            prev = s.clone()
            nz = (y.abs() > 0)
            tail_silent = nz if tail_silent is None else (tail_silent | nz)
            act_on = ((y - f0).abs() > 0.05).float().mean().item()
            active.append(act_on)
            awake = s.abs() >= 1e-3 * early_max_x
            tail_awake = awake if t == STEPS - TAIL + 1 else (tail_awake | awake)
        if t > STEPS - WINDOW:
            rows.append(s.clone())
    end_sent = traj_sent[STEPS]
    end_x = traj_state[STEPS]
    dead = end_x < 1e-9 or end_x < 1e-4 * early_max_x
    X = torch.stack(rows).float()
    tail = X[-TAIL:]
    period = None
    for p in range(1, 51):
        a, b = tail[p:], tail[:-p]
        if bool(((a - b).norm(dim=-1) <= 1e-6 * (b.norm(dim=-1) + 1e-30)).all()):
            period = p
            break
    if dead:
        outcome = "dead"
    elif max(moves) < 1e-6:
        outcome = "fixed"
    elif period is not None:
        outcome = "cycle"
    else:
        outcome = "alive"
    s1 = traj_sent.get(1, s0)
    ratio = end_sent / s1 if s1 > 0 else float("nan")
    shape = "decaying" if ratio < 0.1 else ("growing" if ratio > 10 else "flat")
    r = {"tag": tag, "status": "ok", "outcome": outcome, "dead_at": dead_at if dead else None,
         "period": period, "shape": shape, "sent_ratio": ratio,
         "silent_units": float(1 - tail_silent.float().mean()),
         "resting_units": float(1 - tail_awake.float().mean()),
         "active_share": sum(active) / len(active),
         "move_tail": sum(moves) / len(moves), "sent": traj_sent, "state": traj_state,
         "secs": round(time.time() - t0, 1)}
    if outcome in ("alive", "cycle"):
        m = m52.measure(X, cfg.d_col, f)
        for key in ("c1", "static", "fac1", "lock", "s_level"):
            r[key] = m[key]
        try:
            r["pr"] = r39._participation(X, cfg.d_col)[0]
        except Exception:  # eigh can fail on near-blown-up states; keep the run
            r["pr"] = float("nan")
    return r


def run(job):
    try:
        return cell(*job)
    except Exception as e:  # report, never drop silently
        return {"tag": "|".join(map(str, job)), "status": f"ERROR {type(e).__name__}: {e}"}


def line(r):
    if r["status"] != "ok":
        return f"{r['tag']}: {r['status']}"
    s = r["sent"]
    traj = " ".join(f"{k}:{s[k]:.2g}" for k in (1, 10, 100, 1000, 2000) if k in s)
    extra = ""
    if r["outcome"] == "blowup":
        extra = f" at step {r['blowup_step']}"
    elif r["outcome"] == "dead":
        extra = f" at step {r['dead_at']}"
    elif r["outcome"] == "cycle":
        extra = f" period {r['period']}"
    more = ""
    if "lock" in r:
        more = f" lock {r['lock']:.2f} static {r['static']:.3f} pr {r['pr']:.1f}"
    if r["outcome"] == "blowup":
        return f"{r['tag']}: blowup{extra} | sent {traj} | {r['secs']}s"
    return (f"{r['tag']}: {r['outcome']}{extra} shape {r['shape']} ratio {r['sent_ratio']:.3g} "
            f"silent {r['silent_units']:.2f} resting {r['resting_units']:.2f} active {r['active_share']:.2f} move {r['move_tail']:.3g}{more} | sent {traj} | {r['secs']}s")


def jobs():
    out = [("run39", a, sc, inp, keep, nm, s, lk)
           for a in ACTS for sc in SCALES for inp in ("none", "kp1")
           for keep, lk in ((0.0, False), (0.5, False), (0.9, False), (0.5, True), (0.9, True))
           for nm in (True, False) for s in (50, 51)]
    out += [("ex16", a, sc, inp, keep, True, 50, lk)
            for a in ACTS for sc in SCALES for inp in ("none", "mnist")
            for keep, lk in ((0.0, False), (0.9, False), (0.9, True))]
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for job in (("run39", "step1", 0.3, "none", 0.0, True, 50),
                    ("run39", "step1", 16.0, "none", 0.0, True, 50),
                    ("run39", "wta", 0.1, "none", 0.0, True, 50),
                    ("run39", "raw", 0.3, "none", 0.0, False, 50),
                    ("run39", "raw", 4.0, "none", 0.9, False, 50),
                    ("run39", "tri0.3", 0.1, "none", 0.0, True, 50),
                    ("run39", "sig4_1", 1.0, "kp1", 0.0, True, 50)):
            print(line(run(job)), flush=True)
        return
    js = jobs()
    # capped at 16 (2026-09-29 01:28 EDT): 55's 22 workers held 53 to 57 GB of the
    # box's 91 GB with swap full; 16 leaves headroom for the desktop
    workers = min(int(os.environ.get("W56_WORKERS", "22")), 16)
    import pool_runner
    pool_runner.run_jobs(js, run, OUT.replace(".json", ".jsonl"), LOG, line, workers, 40, OUT)


if __name__ == "__main__":
    main()
