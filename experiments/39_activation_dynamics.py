"""
Activation form x reset policy, both weight channels frozen (RW, 2026-07-27).

Question. Every run from 25 to 38 varied a learning knob and read a learning
outcome. This one freezes both channels and asks what the substrate does as a
dynamical system: given fixed random weights, what activity does the grid
produce, and how do the activation nonlinearity and the reset policy shape it.
Nothing here can teach the substrate anything. That is the point: with
`frozen=True` there is no update to either Dir.A or Dir.E, so every difference
between arms is a property of the dynamics alone.

Why these two knobs. Both are pinned at exactly one setting in every prior run
and neither has ever been moved.

  Activation. `_transport` applies the nonlinearity to the sender before the
  matmul: raw is `x @ w`, relu is `relu(x) @ w`, triangle is
  `relu(x - x.mean())**power @ w`. The shipped default is triangle at power
  0.3. Triangle subtracts the mean, so it is the only one of the three that
  makes a unit's output depend on its neighbours; that coupling is a plausible
  source of the synchrony the A-rule work kept running into, and it has never
  been isolated from the learning.

  Reset. `fc.update` returns `zeros_like`, so the accumulator `nr_1_` is hard
  zeroed after every commit. The step is therefore memoryless in the buffer:
  whatever arrives in one step is fully consumed by it. A leak instead of a
  zero turns the same grid into a leaky integrator, which is a different
  dynamical system with the same weights. `keep` is the fraction retained, so
  keep=0.0 is the incumbent.

  Norm, carried as a third axis because it is free and is the obvious
  confounder. `BareCol.update_activations(use_norm)` divides the committed
  activity by a running RMS floored at 0.9. That is a divisive normalisation,
  the classic stabiliser, and if it dominates then the other two axes are
  being measured through it rather than around it.

Instruments. The prior runs measure weights; there is nothing to measure here
but activity, so the panel is wide and deliberately includes three independent
ways of asking the same question, because a dynamical claim from one statistic
is not worth having.

  rate, dens        mean |a| and fraction of units above threshold
  dead, sat         units never / always active over the measured window
  pr                participation ratio of the activity covariance,
                    (sum lam)^2 / sum lam^2: effective dimensionality, and the
                    activity-side analogue of run 37's rank-1 detector
  maj, npat         the existing degeneracy pair, kept for comparability
  ac1, act          lag-1 autocorrelation and the lag at which it falls below
                    1/e: how long the state remembers itself
  nstate, period    distinct binarised states visited, and the cycle length if
                    the trajectory closes; a fixed point is period 1
  lyap              divergence rate of a twin perturbed by 1e-6 at the end of
                    the transient. positive means chaotic, ~0 marginal,
                    negative means contracting to a point or a cycle. this is
                    the instrument that actually classifies the dynamics; the
                    rest describe it
  fdom, fpow        dominant frequency of the mean-activity series and its
                    share of total power: oscillation detector
  sync              mean pairwise correlation between columns' activity
  drift             log |a| slope over the measured window: is the state
                    growing, shrinking or stationary

Reading, decided before the run. There is no "good" arm here because nothing
is being learned; the output is a map, not a winner. What would count as a
finding:

  a reset or activation choice that moves `lyap` across zero   -> the knob
      selects the dynamical regime, and the learning work has been running in
      whichever regime the pinned defaults happen to give
  `pr` collapsing to ~1 with weights frozen                    -> the rank-1
      story from 37/38 is not caused by learning at all
  norm dominating both other axes                              -> the two
      knobs are not independently meaningful and the grid is the wrong shape
  everything flat                                              -> the dynamics
      are insensitive here and the substrate's behaviour is set by the weights

Vitals. Any arm whose activity overflows or goes non-finite is reported
DIVERGED and its statistics are dropped, not clipped; clipping would turn a
finding about the dynamics into an artefact of the harness. Arms that go
identically zero are reported silent for the same reason.

Summaries only (kind actdyn39).
"""

import importlib
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
rep = importlib.import_module("11_replicate_instar_relu")
arule = importlib.import_module("31_a_rule_degeneracy")
from experiments import dire_exam as de
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("ACT_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51)
STEPS = 300 if SMOKE else 2000
BURN = 60 if SMOKE else 400          # transient discarded before measuring
WORKERS = 2 if SMOKE else int(os.environ.get("ACT_WORKERS", "5"))
NS = "comp25"
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", lr_e=0.1, signed=True)

# activation applied to the sender inside _transport.
# grouped by family so the report reads as a map rather than a list:
#   linear        raw
#   rectifier     relu, lrelu, rrelu1 (relu at a raised threshold)
#   even          abs, square
#   triangle      relu(x - mean)^p, the shipped family. p=0.3 is the incumbent
#   step          Heaviside at four thresholds, plus the mean-subtracted one.
#                 cstep is to step what triangle is to relu, so the pair
#                 isolates mean-subtraction from the power
#   bipolar       sign, tanh, htanh, softsign
#   sigmoid       the smooth step. sigk_c is logistic with steepness k about
#                 centre c, so sig8_0 approximates step0 and sig1_0 is nearly
#                 linear near the origin: the family interpolates between the
#                 step arms and the linear one, which tells us whether any
#                 difference the step arms show is about the discontinuity or
#                 only about the saturation
#   smooth relu   softplus, elu, gelu, silu
#   sparse        shrink, the soft-threshold from sparse coding
#   competitive   wta, kwta4: the gate shape used as an activation
#   geometric     l2norm, direction only, magnitude discarded
ACTS = (
    "raw",
    "relu", "lrelu", "rrelu1",
    "abs", "square",
    "tri0.3", "tri0.7", "tri1.0",
    "step0", "step0.5", "step1", "step2", "cstep",
    "sign", "tanh", "htanh", "softsign",
    "sig1_0", "sig4_0", "sig8_0", "sig4_1",
    "softplus", "elu", "gelu", "silu",
    "shrink0.5",
    "wta", "kwta4",
    "l2norm",
)
# fraction of the accumulator retained across a step. 0.0 is the incumbent
KEEPS = (0.0, 0.25, 0.5, 0.75, 0.9)
NORMS = (True, False)
PERT = 1e-8           # twin perturbation for the divergence estimate
RENORM = 1e-4         # rescale the twin separation back to this each step
MAXLAG = 40


def act_fn(name):
    """sender-side nonlinearity. `raw` is identity, i.e. no activation."""
    if name == "raw":
        return lambda x: x
    if name == "relu":
        return torch.relu
    if name == "lrelu":
        return lambda x: torch.nn.functional.leaky_relu(x, 0.1)
    if name == "rrelu1":
        return lambda x: torch.relu(x - 1.0)
    if name == "abs":
        return torch.abs
    if name == "square":
        return lambda x: x * x
    if name == "tanh":
        return torch.tanh
    if name == "htanh":
        return lambda x: x.clamp(-1.0, 1.0)
    if name == "softsign":
        return lambda x: x / (1.0 + x.abs())
    if name == "sign":
        return torch.sign
    if name == "softplus":
        return torch.nn.functional.softplus
    if name == "elu":
        return torch.nn.functional.elu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "silu":
        return torch.nn.functional.silu
    if name.startswith("sig"):            # sig{steepness}_{centre}
        k, c = name[3:].split("_")
        k, c = float(k), float(c)
        return lambda x, k=k, c=c: torch.sigmoid(k * (x - c))
    if name == "cstep":                      # Heaviside about the mean
        return lambda x: (x > x.mean()).to(x.dtype)
    if name == "l2norm":
        return lambda x: x / x.norm().clamp(min=1e-9)
    if name.startswith("step"):
        t = float(name[4:])
        return lambda x, t=t: (x >= t).to(x.dtype)
    if name.startswith("tri"):
        p = float(name[3:])
        return lambda x, p=p: torch.relu(x - x.mean()) ** p
    if name.startswith("shrink"):
        t = float(name[6:])
        return lambda x, t=t: torch.sign(x) * torch.relu(x.abs() - t)
    if name == "wta":
        def _wta(x):
            o = torch.zeros_like(x)
            o[int(torch.argmax(x))] = 1.0
            return o
        return _wta
    if name.startswith("kwta"):
        k = int(name[4:])

        def _kwta(x, k=k):
            o = torch.zeros_like(x)
            idx = torch.topk(x, min(k, x.numel())).indices
            o[idx] = x[idx]
            return o
        return _kwta
    raise ValueError(name)


def arm_cfg(act):
    """transport is set to raw so `_transport` is the identity, and the
    nonlinearity is applied by the patched function instead. this keeps one
    code path for every arm, including the incumbent, so `tri0.3` is a
    reproduction of the shipped setting rather than a different branch."""
    return dict(BASE, transport="raw", power=0.3)


def cfg_hash(act, keep, norm):
    return rep.arm_hash(f"actdyn39:{act}",
                        dict(arm_cfg(act), keep=keep, use_norm=norm), STEPS)


_PRISTINE_UPDATE = None


def _patch(act, keep, norm):
    """Install the arm. A process pool reuses its workers, so every patch here
    has to be re-applied from the pristine function on each call, not layered
    on whatever the previous cell left behind.

    The norm patch used to be applied only in the norm=False branch, over
    whatever `update_activations` currently was. Once a worker ran one
    norm=False cell the override stayed installed, and every norm=True cell
    that worker picked up afterwards silently ran with the norm off. That made
    the norm axis of runs 39 and 41 report a null it had not measured: 12 of
    300 pairs in run 39 and 12 of 144 in run 41 were the only ones where the
    norm-on cell reached a worker that had not yet seen a norm-off cell."""
    import src.agents as ag
    from experiments import dire_hosts as dh

    global _PRISTINE_UPDATE
    if _PRISTINE_UPDATE is None:
        _PRISTINE_UPDATE = ag.BareCol.update_activations

    f = act_fn(act)
    dh.BareAgtX._transport = lambda self, x, w: f(x) @ w

    if keep > 0.0:
        fc.update = lambda x, threshold=1.0: keep * x
        fc.update_e = lambda x: keep * x
    else:
        fc.update = lambda x, threshold=1.0: torch.zeros_like(x)
        fc.update_e = lambda x: torch.zeros_like(x)

    if norm:
        ag.BareCol.update_activations = _PRISTINE_UPDATE
    else:
        ag.BareCol.update_activations = \
            lambda self, use_norm=True, _b=_PRISTINE_UPDATE: _b(self,
                                                                use_norm=False)


def twin_sync(agt, twin_agt, bulk, vec):
    """Put the twin in the host's state, displaced to `vec` in the observed
    coordinates. Every other per-column variable is copied across.

    A column carries more state than the activity vector this run measures:
    `Activs` also holds avg, avg_sq and rms_avg, and the accumulator nr_1_
    holds whatever has arrived but not yet been committed. An earlier version
    of this function wrote only nr_1.actual, so the twin kept its own running
    RMS and its own accumulator, and the separation being measured was those
    relaxing rather than the trajectories diverging.

    That mattered wherever the extra variables are live. With the norm off and
    keep 0 they are not: rms_avg is pinned to 1.0 and the accumulator is zeroed
    every step, so the old function happened to be correct. Turn either knob on
    and it stopped being. Measured on tri0.3 at conn_scale 2: keep 0.9 read
    +10.07 with the partial copy and +0.38 with the full one, and the median
    twin separation sat at 2.35 against a renormalisation target of 1e-4, four
    orders of magnitude adrift. Under the partial copy step1 also produced a
    number where it should have produced none."""
    i = 0
    for loc in bulk:
        c, hc = twin_agt.cols[loc], agt.cols[loc]
        for dst, src in ((c.nr_1, hc.nr_1), (c.nr_1_, hc.nr_1_)):
            dst.actual = src.actual.clone()
            dst.expect = src.expect.clone()
            dst.avg = src.avg.clone()
            dst.avg_sq = src.avg_sq.clone()
            dst.rms_avg = src.rms_avg
        n = c.nr_1.actual.numel()
        c.nr_1.actual = vec[i:i + n].view_as(c.nr_1.actual).clone()
        i += n


def state_of(agt):
    """the bulk activity as one flat vector, input column excluded."""
    return torch.cat([c.nr_1.actual.flatten() for loc, c in agt.cols.items()
                      if not agt.is_i(loc)])


def _pr_block(X):
    """(sum lam)^2 / sum lam^2 of one block's activity covariance: the number
    of directions the activity actually uses. 1 is a line, d is isotropic."""
    Xc = X - X.mean(0, keepdim=True)
    if Xc.shape[0] < 2:
        return float("nan")
    C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    lam = torch.linalg.eigvalsh(C.double()).clamp(min=0)
    s1, s2 = float(lam.sum()), float((lam ** 2).sum())
    return s1 * s1 / s2 if s2 > 0 else float("nan")


def _participation(X, d_col):
    """Participation ratio per column, averaged. computed per column deliberately: the
    estimator is bounded above by the sample count, and the concatenated
    state is 16*32 = 512 wide, so a global PR on a few thousand steps would
    report the number of samples rather than the dimensionality of the
    activity. one column is 32 wide against thousands of steps, which is
    properly sampled. the global figure is returned alongside and is only
    meaningful as an upper bound."""
    n_blocks = X.shape[1] // d_col
    per = [_pr_block(X[:, i * d_col:(i + 1) * d_col]) for i in range(n_blocks)]
    per = [v for v in per if v == v]
    return (sum(per) / len(per) if per else float("nan")), _pr_block(X)


def _autocorr(X):
    """lag-1 correlation of the state vector with itself, and the lag where
    it first drops below 1/e."""
    Xc = X - X.mean(0, keepdim=True)
    n = Xc.shape[0]
    den = float((Xc * Xc).sum())
    if den <= 0:
        return float("nan"), float("nan")
    ac = []
    for k in range(1, min(MAXLAG, n - 1) + 1):
        ac.append(float((Xc[:-k] * Xc[k:]).sum()) / den * n / (n - k))
    tau = next((k for k, v in enumerate(ac, 1) if v < math.exp(-1)), float("nan"))
    return ac[0], tau


def _spectrum(v):
    """dominant non-DC frequency of the mean-activity series and its share of
    total power. a clean oscillation puts most of the power in one bin."""
    x = v - v.mean()
    if float(x.abs().sum()) == 0:
        return float("nan"), float("nan")
    p = torch.fft.rfft(x).abs() ** 2
    p[0] = 0.0
    tot = float(p.sum())
    if tot <= 0:
        return float("nan"), float("nan")
    i = int(torch.argmax(p))
    return i / len(x), float(p[i]) / tot


def _cycle(keys):
    """period of the binarised trajectory if it closes, else None."""
    seen = {}
    for i, k in enumerate(keys):
        if k in seen:
            return i - seen[k]
        seen[k] = i
    return None


def run_one(job):
    seed, act, keep, norm = job
    torch.set_grad_enabled(False)
    # one thread per worker. the tensors here are 32 wide, so torch's intra-op
    # threading buys nothing and costs cores: measured 5.7s at one thread
    # against 5.5s at four for the same cell, while holding 3-4 cores instead
    # of one. pinning to one lets the pool run three times as many workers.
    torch.set_num_threads(1)
    _patch(act, keep, norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                          n_cols=sweep.N_COLS, **arm_cfg(act))
    host = BareHost(sweep.BareAgtXC(
        cfg, _bare_path(f"actdyn39/a{os.getpid()}"), seed=seed,
        dire=True, frozen=True))          # frozen: both channels, no learning
    agt = host.agt

    # Benettin: a second host, identical weights and init because the seed is
    # the same, perturbed once at the end of the transient and then advanced
    # on the identical input stream. the separation is renormalised every step
    # so it stays in the linear regime, and the exponent is the mean log
    # growth per step. this needs two real trajectories; comparing one
    # trajectory to its own delayed image measures velocity, not divergence.
    twin_host = BareHost(sweep.BareAgtXC(
        cfg, _bare_path(f"actdyn39/b{os.getpid()}"), seed=seed,
        dire=True, frozen=True))
    twin_agt = twin_host.agt
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]

    def twin_set(vec):
        return twin_sync(agt, twin_agt, bulk, vec)

    rows, keys = [], []
    dlog, status, launched = [], "ok", False
    stream = de.kp1_cycle(STEPS, seed=de.derive_seed("kp1", NS, seed))
    for step, a in enumerate(stream):
        host.observe(a)
        s = state_of(agt)
        if not torch.isfinite(s).all():
            status = "DIVERGED"
            break
        if not launched and step == BURN:
            twin_set(s + PERT * torch.randn_like(s))
            launched = True
        elif launched:
            twin_host.observe(a)          # same input, both trajectories
            t = state_of(twin_agt)
            if not torch.isfinite(t).all():
                launched = False          # twin blew up; drop the estimate
            else:
                d = float((s - t).norm())
                if d > 0:
                    dlog.append(math.log(d / RENORM))
                    # pull the twin back to a fixed separation along the same
                    # direction, so the next step measures growth afresh
                    twin_set(s + (t - s) * (RENORM / d))
                else:
                    twin_set(s + RENORM * torch.randn_like(s) / s.numel() ** 0.5)
        if step >= BURN:
            rows.append(s.clone())
            keys.append(hash(tuple((s > 0.5).tolist())))

    if status != "ok" or len(rows) < 32:
        return {"seed": seed, "act": act, "keep": keep, "norm": norm,
                "status": status if status != "ok" else "SHORT",
                "config_hash": cfg_hash(act, keep, norm)}

    X = torch.stack(rows).float()
    if float(X.abs().max()) == 0.0:
        status = "SILENT"
    rate = float(X.abs().mean())
    dens = float((X > 0.5).float().mean())
    per_unit = (X > 0.5).float().mean(0)
    dead = float((per_unit == 0).float().mean())
    sat = float((per_unit == 1).float().mean())
    pr, pr_all = _participation(X, cfg.d_col)
    ac1, tau = _autocorr(X)
    fdom, fpow = _spectrum(X.mean(1))
    Xc = X - X.mean(0, keepdim=True)
    sd = Xc.std(0)
    ok = sd > 1e-9
    if int(ok.sum()) > 1:
        Z = Xc[:, ok] / sd[ok]
        C = (Z.T @ Z) / (Z.shape[0] - 1)
        k = C.shape[0]
        sync = float((C.sum() - k) / (k * (k - 1)))
    else:
        sync = float("nan")
    mag = X.norm(dim=1).clamp(min=1e-12).log()
    t = torch.arange(len(mag), dtype=torch.float32)
    drift = float(((t - t.mean()) * (mag - mag.mean())).sum()
                  / ((t - t.mean()) ** 2).sum())
    # each entry is log(separation / RENORM) for one step of a twin that was
    # reset to exactly RENORM at the end of the previous step, so the mean is
    # the growth rate per step directly.
    lyap = sum(dlog) / len(dlog) if len(dlog) > 8 else float("nan")
    maj, _, npat = arule.degeneracy(agt)
    period = _cycle(keys)

    return {"seed": seed, "act": act, "keep": keep, "norm": norm,
            "status": status, "rate": rate, "dens": dens, "dead": dead,
            "sat": sat, "pr": pr, "pr_all": pr_all,
            "maj": maj, "npat": npat, "ac1": ac1,
            "act_tau": tau, "nstate": len(set(keys)), "period": period,
            "lyap": lyap, "fdom": fdom, "fpow": fpow, "sync": sync,
            "drift": drift, "config_hash": cfg_hash(act, keep, norm)}


def main():
    jobs = [(s, a, k, n) for a in ACTS for k in KEEPS
            for n in NORMS for s in SEEDS]
    print(f"{len(jobs)} cells "
          f"({len(ACTS)} acts x {len(KEEPS)} keeps x {len(NORMS)} norms "
          f"x {len(SEEDS)} seeds), {STEPS} steps, burn {BURN}", flush=True)
    out = []
    # spawn, not fork: forking a process that has already imported torch and
    # started its thread pool deadlocks the workers before they reach any of
    # this file's code. the extra ~1.5s of import per worker is nothing
    # against a cell that runs for minutes.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            print(f"  {i}/{len(jobs)} s{r['seed']} {r['act']:>7} "
                  f"keep{r['keep']:<5} norm{int(r['norm'])} {r['status']}",
                  flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "actdyn39.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    report(out)


def report(out):
    def agg(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))
             and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    print("\nactivation x reset, both channels frozen, seeds", list(SEEDS))
    print("lyap > 0 chaotic, ~0 marginal, < 0 contracting. "
          "pr is effective dimensionality (1 = rank-1 activity).")
    for norm in NORMS:
        print(f"\n--- use_norm={norm} ---")
        print(f"{'act':>7}{'keep':>6}{'rate':>8}{'dens':>7}{'dead':>7}"
              f"{'pr':>7}{'maj':>7}{'ac1':>7}{'tau':>6}{'nstate':>7}"
              f"{'per':>5}{'lyap':>8}{'fpow':>6}{'sync':>7}{'drift':>9}  st")
        for a in ACTS:
            for k in KEEPS:
                rs = [r for r in out if r["act"] == a and r["keep"] == k
                      and r["norm"] == norm]
                bad = [r["status"] for r in rs if r["status"] != "ok"]
                st = bad[0] if bad else "ok"
                print(f"{a:>7}{k:>6}{agg(rs,'rate'):>8.3f}{agg(rs,'dens'):>7.3f}"
                      f"{agg(rs,'dead'):>7.3f}{agg(rs,'pr'):>7.2f}"
                      f"{agg(rs,'maj'):>7.3f}{agg(rs,'ac1'):>7.3f}"
                      f"{agg(rs,'act_tau'):>6.1f}{agg(rs,'nstate'):>7.0f}"
                      f"{agg(rs,'period'):>5.0f}{agg(rs,'lyap'):>8.4f}"
                      f"{agg(rs,'fpow'):>6.2f}{agg(rs,'sync'):>7.3f}"
                      f"{agg(rs,'drift'):>9.2e}  {st}")


if __name__ == "__main__":
    main()
