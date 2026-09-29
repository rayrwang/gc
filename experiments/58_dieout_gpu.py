"""Experiment 58: experiment 56 (when activity dies out) rerun batched on the GPU, as a
proof of concept for batching frozen-network sweeps (2026-09-29).

RW 2026-09-29, verbatim: "ok lets do the gpu batchign expiemrnt. this will be nice to
both test out the new run tooling and the gpu batchign to see how it would work how
hard it is to make it ideitical or at least as lcose as possible tiwht diffenrt floats
and how much fatester it woudl be. also nice that ithis is purely reproduct ion so no
stakes ha fun". Queued in gc-project 8.7 at 02:35 EDT the same day ("to do a batched
run reporduxe one of the them mahbe the longest one, as a proof of conceot"); 56 is
the longest of that night's runs and has the simplest measures.

What 56 did: 4,416 cells, each a stock host (52's harness: BareAgtXC, frozen) stepped
2000 times on one CPU thread, 16 cells at a time. What this does: every cell that
shares a network (net, seed, weight scale, input) runs in one batch. The network's
Dir.A weights are copied out of the stock host, not re-sampled, into one dense
bulk-to-bulk matrix and one input-to-bulk matrix, together with the host's initial
states, its initial accumulators (random, not zero: BareCol starts nr_1_ from
activs()) and the input column's initial value (the input column sends the previous
step's input, so step 1 sends that random start). A step is then the input's and the
bulk's contributions added to the accumulator, and BareCol.update_activations for
every column at once: the RMS norm (a 0.2 running average in float64, as the stock's
.item() arithmetic does, floored at 0.9; off per cell), the state = accumulator /
that average, and the accumulator kept as keep times itself; the "leak" cells take
(1 - keep) on everything entering it, as 56 did. Each cell's activation is applied per
32-unit column. Dir.E is left out: with learning frozen its expectations never reach
the activity. The per-step bookkeeping and the outcome logic are 56's, rewritten for a
batch; the end-of-run measures are 52's measure() and 39's per-column participation
ratio, called on the same 300-step windows on the CPU (39's global block, which 56
computed and discarded, is skipped).

Premises (56's): run 39's network (49 columns of 32, dist_exp 1, p_a 0.3) and example
16's (200 columns, dist_exp 2, p_a 0.3), frozen; 16 activations x weight scale {0.1,
0.3, 1, 2, 4, 16} x input {none; kp1 into 3 columns | MNIST into every column} x keep
{0; 0.5, 0.9 as coded and with the (1 - keep) factor} x norm x seed, started from the
host's random activity, 2000 steps; outcomes blowup, dead, fixed, cycle, alive.

Parts, each with its own results file and launch record in outputs/:
  check        16 cells (CHECK below) stepped by the stock host itself and by the
               batched code in five variants: dense matrix on the GPU in float32 and
               float64, dense on the CPU in float32, and "edge", the stock's own order
               (every connection added in turn, as the host does, but for the whole
               batch at once) on the CPU and the GPU in float32; the distance between
               each variant's state and the stock's after every step; the stock path
               rerun through 56's own cell() against 56's logged record; and the
               bookkeeping run on the stock trajectory against that record.
  full f32/f64 all 4,416 cells, dense on the GPU, in 56's record format, timed.
  full exact   all 4,416 cells in the stock's order on the CPU in float32, one thread,
               batched per group: whether a batch can reproduce 56 bit for bit.
58_report.py compares them with 56 cell by cell.

Usage:
    venv/bin/python experiments/58_dieout_gpu.py smoke      # 2 check cells, 2 groups; dirty allowed
    venv/bin/python experiments/58_dieout_gpu.py check
    venv/bin/python experiments/58_dieout_gpu.py full f32
    venv/bin/python experiments/58_dieout_gpu.py full f64
    venv/bin/python experiments/58_dieout_gpu.py full exact
"""

import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

m56 = importlib.import_module("56_dieout")  # via 52: sets ROOT on sys.path and chdir
m52, r39, sweep, de, T = m56.m52, m56.r39, m56.sweep, m56.de, m56.T
import src.agents as ag  # noqa: E402
import launch  # noqa: E402

STEPS, TAIL, WINDOW, MARKS = m56.STEPS, m56.TAIL, m56.WINDOW, m56.MARKS
D_COL = 32
OUTDIR = os.path.join(HERE, "outputs")
BASE = os.path.join(OUTDIR, "58_dieout_gpu")
GPU = torch.device("cuda")
CPU = torch.device("cpu")
torch.backends.cuda.matmul.allow_tf32 = False  # full float32 matmuls, never TF32
torch.backends.cudnn.allow_tf32 = False

# net, act, scale, input, keep, norm, seed, leak: 56's job tuples, one or two per behaviour
CHECK = [
    ("run39", "tri0.3", 2.0, "none", 0.0, True, 50, False),   # chaotic
    ("run39", "tri0.3", 2.0, "kp1", 0.9, False, 50, True),    # slow chaos, leak form
    ("run39", "wta", 1.0, "none", 0.9, True, 50, False),      # order-only, carry-over
    ("run39", "sign", 2.0, "kp1", 0.0, False, 51, False),
    ("run39", "sig4_0", 1.0, "none", 0.0, True, 50, False),   # fixed point
    ("run39", "raw", 0.3, "none", 0.0, False, 50, False),     # dies
    ("run39", "relu", 1.0, "none", 0.0, False, 50, False),    # blows up
    ("run39", "step1", 2.0, "none", 0.0, True, 51, False),
    ("run39", "tanh", 4.0, "kp1", 0.9, True, 50, True),
    ("run39", "kwta4", 2.0, "none", 0.5, True, 51, False),
    ("run39", "cstep", 16.0, "kp1", 0.5, False, 50, True),
    ("run39", "step2", 4.0, "none", 0.0, True, 50, False),    # dies under the norm
    ("ex16", "tri0.3", 2.0, "none", 0.0, True, 50, False),
    ("ex16", "wta", 2.0, "mnist", 0.9, True, 50, False),
    ("ex16", "sig4_1", 1.0, "mnist", 0.9, True, 50, True),
    ("ex16", "relu", 0.3, "none", 0.0, True, 50, False),
]


def tag(job):
    net, act, scale, inp, keep, norm, seed, leak = job
    return f"{net}|{act}|{scale}|{inp}|{keep}|{int(norm)}|{seed}|{'leak' if leak else 'code'}"


def groups():
    by = {}
    for j in m56.jobs():
        by.setdefault((j[0], j[6], j[2], j[3]), []).append(j)
    return by


# -- the stock host, and what is copied out of it ---------------------------------

def build(net, seed, scale, inp):
    """The stock host exactly as 56's cell() builds it (arm_cfg is the same for every
    activation, so one host serves a whole group)."""
    n_cols, dist_exp, p_a = m56.NETS[net]
    dim = 784 if inp == "mnist" else de.SYMBOL_DIM
    fan = {"none": 0, "kp1": 3, "mnist": -1}[inp]
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(dim)], n_cols=n_cols,
                          **dict(r39.arm_cfg("raw"), p_a=p_a, dist_exp=dist_exp,
                                 input_fanout=fan, conn_scale=scale))
    path = os.path.join(m56.ROOT, "saves", "gpu58", f"w{os.getpid()}")
    return sweep.BareAgtXC(cfg, path, seed=seed, dire=True, frozen=True), dim


def host(net, seed, scale, inp):
    agt, dim = build(net, seed, scale, inp)
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]
    pos = {loc: i for i, loc in enumerate(bulk)}
    N = len(bulk) * D_COL
    W = torch.zeros(N, N, dtype=torch.float64)
    Win = torch.zeros(dim, N, dtype=torch.float64)
    edges = []  # the stock's order: senders in cols order, then each sender's conns
    for loc, c in agt.cols.items():
        for (tloc, d), w in c.conns.items():
            if d.name != "A":
                continue
            t0 = pos[tloc] * D_COL
            if agt.is_i(loc):
                Win[:, t0:t0 + D_COL] = w.double()
                edges.append((-1, pos[tloc], w.clone()))
            else:
                s0 = pos[loc] * D_COL
                W[s0:s0 + D_COL, t0:t0 + D_COL] = w.double()
                edges.append((pos[loc], pos[tloc], w.clone()))
    icol = agt.cols[(1, 0)]
    assert all(agt.cols[l].nr_1.rms_avg == 1.0 for l in bulk)
    return {"n": len(bulk), "N": N, "dim": dim, "W": W, "Win": Win, "edges": edges,
            "X0": torch.cat([agt.cols[l].nr_1.actual for l in bulk]).clone(),
            "A0": torch.cat([agt.cols[l].nr_1_.actual for l in bulk]).clone(),
            "u0": icol.nr_1.actual.clone(), "has_input": bool(Win.abs().sum() > 0)}


def inputs(inp, seed):
    if inp == "kp1":
        return torch.stack(list(de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", r39.NS, seed))))
    if inp == "mnist":
        return torch.stack([x.reshape(-1).float() for x in m56.mnist_seq()])
    return None


# -- activations for a batch, per 32-unit column (the last dim) --------------------

def bact(name):
    if name == "raw":
        return lambda x: x
    if name == "relu":
        return torch.relu
    if name == "rrelu1":
        return lambda x: torch.relu(x - 1.0)
    if name == "tanh":
        return torch.tanh
    if name == "sign":
        return torch.sign
    if name.startswith("sig"):
        k, c = (float(v) for v in name[3:].split("_"))
        return lambda x, k=k, c=c: torch.sigmoid(k * (x - c))
    if name == "cstep":
        return lambda x: (x > x.mean(-1, keepdim=True)).to(x.dtype)
    if name.startswith("step"):
        t = float(name[4:])
        return lambda x, t=t: (x >= t).to(x.dtype)
    if name.startswith("tri"):
        p = float(name[3:])
        return lambda x, p=p: torch.relu(x - x.mean(-1, keepdim=True)) ** p
    if name == "wta":
        return lambda x: torch.nn.functional.one_hot(x.argmax(-1), x.shape[-1]).to(x.dtype)
    if name.startswith("kwta"):
        k = int(name[4:])

        def _kwta(x, k=k):
            idx = x.topk(k, -1).indices
            return torch.zeros_like(x).scatter(-1, idx, x.gather(-1, idx))
        return _kwta
    raise ValueError(name)


# -- 56's per-step bookkeeping, for a batch ------------------------------------------

def _rms(X):
    return X.pow(2).mean(1).sqrt()


class Book:
    def __init__(self, X, Y, f0):
        B, N = X.shape
        self.f0 = f0
        s0, x0 = _rms(Y), _rms(X)
        self.sent, self.state = {0: s0}, {0: x0}
        self.early, self.early_x = s0.clone(), x0.clone()
        self.dead_at = torch.full((B,), -1, dtype=torch.long, device=X.device)
        self.blow = torch.full((B,), -1, dtype=torch.long, device=X.device)
        self.moves = torch.zeros(B, TAIL - 1, dtype=torch.float64, device=X.device)
        self.active = torch.zeros(B, TAIL, dtype=torch.float32, device=X.device)
        self.silent = torch.zeros(B, N, dtype=torch.bool, device=X.device)
        self.awake = torch.zeros(B, N, dtype=torch.bool, device=X.device)
        self.win = torch.empty(B, WINDOW, N, dtype=torch.float32, device=X.device)
        self.prev = None

    def step(self, t, X, Y):
        bad = ~(torch.isfinite(X).all(1) & (X.abs().amax(1) <= 1e30))
        self.blow[bad & (self.blow < 0)] = t
        s, x = _rms(Y), _rms(X)
        if t <= 10:
            self.early = torch.maximum(self.early, s)
            self.early_x = torch.maximum(self.early_x, x)
        dead = (self.dead_at < 0) & ((x < 1e-9) | ((t > 10) & (x < 1e-4 * self.early_x)))
        self.dead_at[dead] = t
        if t in MARKS:
            self.sent[t], self.state[t] = s, x
        if t > STEPS - TAIL:
            i = t - (STEPS - TAIL) - 1
            if self.prev is not None:
                self.moves[:, i - 1] = ((X - self.prev).norm(dim=1)
                                        / (self.prev.norm(dim=1) + 1e-30)).double()
            self.prev = X.clone()
            self.silent |= Y.abs() > 0
            # 56: a float32 mean per step, then a Python average of those
            self.active[:, i] = ((Y - self.f0).abs() > 0.05).float().mean(1)
            aw = X.abs() >= 1e-3 * self.early_x[:, None]
            self.awake = aw if i == 0 else (self.awake | aw)
        if t > STEPS - WINDOW:
            self.win[:, t - (STEPS - WINDOW) - 1] = X.float()

    def records(self, jobs):
        """56's end-of-run logic per cell; the windows needing 52's measures are
        returned beside the records."""
        cpu = {k: v.cpu().tolist() for k, v in self.sent.items()}
        cpx = {k: v.cpu().tolist() for k, v in self.state.items()}
        blow, dead_at = self.blow.cpu().tolist(), self.dead_at.cpu().tolist()
        early_x = self.early_x.cpu().tolist()
        out, need = [], []
        for i, job in enumerate(jobs):
            if blow[i] >= 0:
                out.append({"tag": tag(job), "status": "ok", "outcome": "blowup",
                            "blowup_step": blow[i],
                            "sent": {k: v[i] for k, v in cpu.items() if k < blow[i]},
                            "state": {k: v[i] for k, v in cpx.items() if k < blow[i]}})
                continue
            end_sent, end_x = cpu[STEPS][i], cpx[STEPS][i]
            dead = end_x < 1e-9 or end_x < 1e-4 * early_x[i]
            X = self.win[i].cpu()
            tail = X[-TAIL:]
            period = None
            for p in range(1, 51):
                a, b = tail[p:], tail[:-p]
                if bool(((a - b).norm(dim=-1) <= 1e-6 * (b.norm(dim=-1) + 1e-30)).all()):
                    period = p
                    break
            moves = self.moves[i].cpu().tolist()
            if dead:
                outcome = "dead"
            elif max(moves) < 1e-6:
                outcome = "fixed"
            elif period is not None:
                outcome = "cycle"
            else:
                outcome = "alive"
            s1 = cpu[1][i]
            ratio = end_sent / s1 if s1 > 0 else float("nan")
            shape = "decaying" if ratio < 0.1 else ("growing" if ratio > 10 else "flat")
            r = {"tag": tag(job), "status": "ok", "outcome": outcome,
                 "dead_at": dead_at[i] if dead else None, "period": period, "shape": shape,
                 "sent_ratio": ratio, "silent_units": float(1 - self.silent[i].float().mean()),
                 "resting_units": float(1 - self.awake[i].float().mean()),
                 "active_share": (lambda a: sum(a) / len(a))(self.active[i].cpu().tolist()),
                 "move_tail": sum(moves) / len(moves),
                 "sent": {k: v[i] for k, v in cpu.items()}, "state": {k: v[i] for k, v in cpx.items()}}
            out.append(r)
            if outcome in ("alive", "cycle"):
                need.append((len(out) - 1, job[1], X))
        return out, need


def measures(act, X):
    """52's measure() and 39's per-column participation ratio, as 56 took them."""
    torch.set_num_threads(1)
    m = m52.measure(X, D_COL, r39.act_fn(act))
    r = {k: m[k] for k in ("c1", "static", "fac1", "lock", "s_level")}
    try:
        per = [r39._pr_block(X[:, i * D_COL:(i + 1) * D_COL]) for i in range(X.shape[1] // D_COL)]
        per = [v for v in per if v == v]
        r["pr"] = sum(per) / len(per) if per else float("nan")
    except Exception:  # as 56: eigh can fail on near-blown-up states
        r["pr"] = float("nan")
    return r


# -- the batched step ---------------------------------------------------------------

def simulate(h, jobs, dtype, device, mode="dense", xs=None, ref=None):
    """Runs a batch of cells that share one network. mode "dense": one matrix
    multiply per step; "edge": the stock's order, connection by connection. ref: the
    stock trajectory of a single cell, to measure the distance after every step."""
    B, n, N = len(jobs), h["n"], h["N"]
    kw = dict(dtype=dtype, device=device)
    W, Win = h["W"].to(**kw), h["Win"].to(**kw)
    edges = [(s, t, w.to(**kw)) for s, t, w in h["edges"]] if mode == "edge" else None
    keep = torch.tensor([[j[4]] for j in jobs], **kw)
    lf = torch.tensor([[1.0 - j[4] if j[7] else 1.0] for j in jobs], **kw)
    norm = torch.tensor([[j[5]] for j in jobs], dtype=torch.bool, device=device)
    X = h["X0"].to(**kw).repeat(B, 1)
    A = h["A0"].to(**kw).repeat(B, 1)
    R = torch.ones(B, n, dtype=torch.float64, device=device)
    u0 = h["u0"].to(**kw).repeat(B, 1)
    xs = xs.to(**kw) if xs is not None else None
    fns = {}
    for i, j in enumerate(jobs):
        fns.setdefault(j[1], []).append(i)
    idx = {a: torch.tensor(ii, device=device) for a, ii in fns.items()}
    funcs = {a: bact(a) for a in fns}

    def act(Z):
        Yz = torch.empty_like(Z)
        for a, ii in idx.items():
            Yz[ii] = funcs[a](Z[ii].view(-1, n, D_COL)).reshape(-1, N)
        return Yz

    f0 = act(torch.zeros(B, N, **kw))
    Y = act(X)
    book = Book(X, Y, f0)
    dist = [0.0] if ref is not None else None
    if ref is not None:
        r0 = ref[0].double()
        dist[0] = float((X[0].double().cpu() - r0).norm() / (r0.norm() + 1e-30))
    alpha = ag.ALPHA_RMS
    for t in range(1, STEPS + 1):
        if h["has_input"]:
            U = u0 if t == 1 else xs[t - 2].unsqueeze(0) * lf  # 56 scaled the leak cells' inputs
        if mode == "dense":
            if h["has_input"]:
                A = A + U @ Win
            A = A + (Y @ W) * lf
        else:
            for s, tt, w in edges:
                b = slice(tt * D_COL, (tt + 1) * D_COL)
                if s < 0:
                    A[:, b] = A[:, b] + U @ w
                else:
                    A[:, b] = A[:, b] + (Y[:, s * D_COL:(s + 1) * D_COL] @ w) * lf
        Av = A.view(B, n, D_COL)
        rms = Av.pow(2).mean(-1).sqrt().double()
        new = alpha * rms + (1 - alpha) * R
        new = torch.where(norm, new.clamp(min=0.9), torch.ones_like(new))
        X = (Av / new.to(dtype).unsqueeze(-1)).reshape(B, N)
        R = new
        A = A * keep
        Y = act(X)
        book.step(t, X, Y)
        if ref is not None:
            rt = ref[t].double()
            dist.append(float((X[0].double().cpu() - rt).norm() / (rt.norm() + 1e-30)))
    return book, dist


# -- the stock reference, for the check -----------------------------------------------

def stock_traj(job):
    """56's cell() set-up exactly, stepped by the stock host; the bulk state after
    every step."""
    net, act, scale, inp, keep, norm, seed, leak = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    r39._patch(act, keep, norm)
    f = r39.act_fn(act)
    if leak:
        from experiments import dire_hosts as dh
        dh.BareAgtX._transport = lambda self, x, w, _f=f, _k=keep: (1.0 - _k) * (_f(x) @ w)
    agt, dim = build(net, seed, scale, inp)
    xs = None
    if inp == "kp1":
        xs = list(de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", r39.NS, seed)))
    elif inp == "mnist":
        xs = m56.mnist_seq()
    if leak and xs is not None:
        xs = [(1.0 - keep) * x for x in xs]
    zero = torch.zeros(dim)
    s0 = r39.state_of(agt)
    traj = torch.empty(STEPS + 1, s0.numel())
    traj[0] = s0
    for t in range(1, STEPS + 1):
        agt.step([xs[t - 1] if xs is not None else zero])
        traj[t] = r39.state_of(agt)
    return traj


def logged56():
    p = os.path.join(OUTDIR, "56_dieout.jsonl")
    return {json.loads(s)["tag"]: json.loads(s) for s in open(p)}


def same_record(a, b):
    """Field-by-field comparison of two 56-format records, ignoring timing; returns the
    fields that differ."""
    keys = sorted((set(a) | set(b)) - {"secs", "job_key", "src", "group_secs", "cells_in_group", "build_secs"})
    diff = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            va = {str(x): y for x, y in va.items()}
            vb = {str(x): y for x, y in vb.items()}
        if va != vb and not (isinstance(va, float) and isinstance(vb, float) and va != va and vb != vb):
            diff.append(k)
    return diff


VARIANTS = [("dense", "cuda", torch.float32), ("dense", "cuda", torch.float64),
            ("dense", "cpu", torch.float32), ("edge", "cpu", torch.float32),
            ("edge", "cuda", torch.float32)]


def check(cells, out_path, rec):
    log = logged56()
    results = []
    for job in cells:
        t0 = time.time()
        r56_now = m56.cell(*job)            # the stock path, today
        r56_log = log[tag(job)]             # the stock path, 09-29 small hours
        ref = stock_traj(job)
        net, act, scale, inp, keep, norm, seed, leak = job
        h = host(net, seed, scale, inp)
        xs = inputs(inp, seed)
        # the bookkeeping, run on the stock trajectory itself
        bk = Book(ref[0:1].clone(), bact(act)(ref[0:1].view(1, -1, D_COL)).reshape(1, -1),
                  bact(act)(torch.zeros(1, h["n"], D_COL)).reshape(1, -1))
        for t in range(1, STEPS + 1):
            Xt = ref[t:t + 1]
            bk.step(t, Xt, bact(act)(Xt.view(1, -1, D_COL)).reshape(1, -1))
        rb, need = bk.records([job])
        for i, a, X in need:
            rb[i].update(measures(a, X))
        row = {"tag": tag(job), "stock_now_vs_log": same_record(r56_now, r56_log),
               "book_on_stock_vs_log": same_record(rb[0], r56_log),
               "outcome_log": r56_log["outcome"], "variants": {}}
        for mode, dev, dt in VARIANTS:
            t1 = time.time()
            book, dist = simulate(h, [job], dt, torch.device(dev), mode, xs, ref)
            rv, need = book.records([job])
            for i, a, X in need:
                rv[i].update(measures(a, X))
            first = {thr: next((t for t, d in enumerate(dist) if d > thr), None)
                     for thr in (0.0, 1e-6, 1e-3, 1e-1)}
            row["variants"][f"{mode}_{dev}_{str(dt)[6:]}"] = {
                "outcome": rv[0]["outcome"], "differs_from_log": same_record(rv[0], r56_log),
                "first_step_over": {str(k): v for k, v in first.items()},
                "dist_at": {str(t): dist[t] for t in (1, 2, 5, 10, 100, 1000, 2000) if t < len(dist)},
                "dist": [float(f"{d:.3g}") for d in dist], "secs": round(time.time() - t1, 2)}
        row["secs"] = round(time.time() - t0, 1)
        results.append(row)
        print(f"{row['tag']}: log {row['outcome_log']}; stock now vs log {row['stock_now_vs_log'] or 'same'}; "
              f"bookkeeping on stock {row['book_on_stock_vs_log'] or 'same'}; "
              + "; ".join(f"{k} {v['outcome']} first>0 {v['first_step_over']['0.0']} >1e-3 "
                          f"{v['first_step_over']['0.001']}" for k, v in row["variants"].items()), flush=True)
        with open(out_path, "w") as fh:
            json.dump(results, fh)
    rec.finish("done", finished=len(results))


def full(dtype, jl_path, workers=12, only=None, rec=None, device=GPU, mode="dense"):
    done = set()
    if os.path.exists(jl_path):
        done = {json.loads(s)["tag"] for s in open(jl_path)}
    gs = groups()
    keys = [k for k in gs if only is None or k in only]
    pool = ProcessPoolExecutor(workers)
    pending = []  # (records, [(index, future)], group key, group secs, cells)
    n_done = 0
    t_all = time.time()

    def flush(block):
        nonlocal n_done
        keep_ = []
        with open(jl_path, "a") as fh:
            for recs, futs, k, gsecs, cells in pending:
                if block or all(f.done() for _, f in futs):
                    for i, f in futs:
                        recs[i].update(f.result())
                    for r, j in zip(recs, cells):
                        if r["tag"] in done:
                            continue
                        r.update(job_key=repr(j), group_secs=gsecs, cells_in_group=len(cells))
                        fh.write(json.dumps(r) + "\n")
                        n_done += 1
                    print(f"group {k}: {len(cells)} cells, {gsecs:.1f}s stepping, "
                          f"{n_done} written, {time.time() - t_all:.0f}s so far", flush=True)
                else:
                    keep_.append((recs, futs, k, gsecs, cells))
        pending[:] = keep_

    for k in keys:
        cells = gs[k]
        if all(tag(j) in done for j in cells):
            continue
        net, seed, scale, inp = k
        t0 = time.time()
        h = host(net, seed, scale, inp)
        xs = inputs(inp, seed)
        t1 = time.time()
        book, _ = simulate(h, cells, dtype, device, mode, xs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gsecs = time.time() - t1
        recs, need = book.records(cells)
        futs = [(i, pool.submit(measures, a, X)) for i, a, X in need]
        for r in recs:
            r["build_secs"] = round(t1 - t0, 2)
        pending.append((recs, futs, k, round(gsecs, 2), cells))
        del book
        torch.cuda.empty_cache()
        flush(False)
    flush(True)
    pool.shutdown()
    print(f"all done in {time.time() - t_all:.0f}s", flush=True)
    if rec is not None:
        rec.finish("done", finished=n_done, wall_secs=round(time.time() - t_all, 1))


def main():
    torch.set_grad_enabled(False)
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(OUTDIR, exist_ok=True)
    if what == "smoke":
        sm = BASE + "_smoke"
        rec = launch.start(sm + "_check", {"cells": 2}, allow_dirty=True)
        check([CHECK[0], CHECK[5]], sm + "_check.json", rec)
        only = [("run39", 50, 2.0, "none"), ("ex16", 50, 0.3, "none")]
        if os.path.exists(sm + "_f32.jsonl"):
            os.remove(sm + "_f32.jsonl")
        rec = launch.start(sm + "_f32", {"groups": len(only)}, allow_dirty=True)
        full(torch.float32, sm + "_f32.jsonl", only=only, rec=rec)
        if os.path.exists(sm + "_exact.jsonl"):
            os.remove(sm + "_exact.jsonl")
        torch.set_num_threads(1)
        rec = launch.start(sm + "_exact", {"groups": 1}, allow_dirty=True)
        full(torch.float32, sm + "_exact.jsonl", only=only[:1], rec=rec, device=CPU, mode="edge")
        log = logged56()
        for name in ("f32", "exact"):
            rs = [json.loads(s) for s in open(f"{sm}_{name}.jsonl")]
            agree = sum(1 for r in rs if r["outcome"] == log[r["tag"]]["outcome"])
            ident = sum(1 for r in rs if not same_record(r, log[r["tag"]]))
            print(f"smoke {name}: {agree} of {len(rs)} outcomes agree with 56, {ident} records identical")
        return
    if what == "check":
        rec = launch.start(BASE + "_check", {"cells": len(CHECK), "variants": len(VARIANTS)})
        check(CHECK, BASE + "_check.json", rec)
        return
    if what == "full":
        name = sys.argv[2]
        dtype, device, mode = {"f32": (torch.float32, GPU, "dense"), "f64": (torch.float64, GPU, "dense"),
                               "exact": (torch.float32, CPU, "edge")}[name]
        if name == "exact":
            torch.set_num_threads(1)  # as 56's cells ran
        jl = f"{BASE}_{name}.jsonl"
        rec = launch.start(f"{BASE}_{name}", {"cells": len(m56.jobs()), "groups": len(groups()),
                                              "dtype": str(dtype)[6:], "device": device.type, "mode": mode},
                           resuming=os.path.exists(jl))
        full(dtype, jl, rec=rec, device=device, mode=mode)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
