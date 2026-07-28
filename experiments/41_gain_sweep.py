"""
Weight scale x fan-in: is there one gain parameter (RW, 2026-07-27).

Hypothesis, stated before the run. `conn` initialises each weight matrix as

    scale * randn(d1, d2) / sqrt(d1)

so one connection contributes variance that does not depend on the sender's
width. A column with K incoming connections sums K of those, so the standard
deviation of what arrives goes as scale * sqrt(K). That is the gain parameter
of a random recurrent network, and it predicts that conn_scale and fan-in are
not two knobs but one:

    g = conn_scale * sqrt(mean in-degree)

The line carries a TODO saying the number of incoming connections is not taken
into account, so this has never been controlled for. If the hypothesis holds,
every arm below should collapse onto a single curve in g, and the pair
(conn_scale, p_a) should be unidentifiable except through their product.

Why run 40's answer on this axis is not usable. Run 40 swept conn_scale over a
sixteenfold range and reported no effect at all, but every cell had the RMS
normalisation on, and a divisive normalisation by running RMS is a device for
removing exactly this factor. So that arm measured whether the norm cancels a
constant, which it does. The norm is off in the primary arms here and on in a
matched control, which is the comparison run 40 could not make.

What run 40 did establish, and why it points here: the divergence rate tracked
total connection count across every arm that changed it, from -1.44 at 8
connections to +1.01 at 784. That is the gain acting through fan-in while the
scale axis was masked.

Design. A factorial on the two things the hypothesis says are one thing, plus
two axes that should behave differently if it is right:

  scale x p_a     the main grid. if g is the parameter, arms with equal
                  scale*sqrt(K) should agree regardless of how they got there
  n_cols          changes K as a side effect of size, so it should act through
                  g and add nothing of its own
  d_col           changes width but NOT g, because the 1/sqrt(d1) in the init
                  already removes it. so width should move the number of
                  patterns used and leave the divergence rate alone. this is
                  the discriminating axis: if d_col moves the divergence rate,
                  the hypothesis is wrong

Both activations are run because they fail differently. tri0.3 is unbounded so
it can diverge, and the boundary is itself the measurement. step1 is bounded so
it cannot, and should show the same transition as a smooth change rather than
as an overflow.

Reading, decided before the run:
  arms collapse onto g                      -> one parameter, and conn_scale
      has never been an independent knob
  d_col moves pr but not lyap               -> the init's normalisation does
      what it claims and width is free
  d_col moves lyap                          -> hypothesis is wrong and the
      init is not doing what the formula says
  norm-on flattens everything               -> confirms run 40's null was an
      artefact of the norm rather than a fact about the substrate

Vitals: mean in-degree is measured per cell rather than assumed, because p_a
and n_cols set it only in expectation, and the realised graph is what the
dynamics see. g is computed from the measured value.

Summaries only (kind gain41).
"""

import importlib
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

act39 = importlib.import_module("39_activation_dynamics")
sweep = importlib.import_module("10_sweep_bare_x")
rep = importlib.import_module("11_replicate_instar_relu")
arule = importlib.import_module("31_a_rule_degeneracy")
from experiments import dire_exam as de
from experiments.dire_hosts import BareHost, _bare_path
from src import iotypes as T

SMOKE = bool(os.environ.get("GAIN_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51)
STEPS = 300 if SMOKE else 2000
BURN = 60 if SMOKE else 400
WORKERS = 2 if SMOKE else int(os.environ.get("GAIN_WORKERS", "10"))
NS = "comp25"
KEEP = 0.0
PERT, RENORM = act39.PERT, act39.RENORM

BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3, rule="softhebb",
            lr_e=0.1, signed=True, transport="raw", power=0.3)
BASE_NCOLS = sweep.N_COLS

SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) if not SMOKE else (0.5, 4.0)
PAS = (0.1, 0.3, 0.6, 1.0) if not SMOKE else (0.3,)
ACTS = ("tri0.3", "step1") if not SMOKE else ("tri0.3",)
NORMS = (False, True)

ARMS = []
for s in SCALES:                       # main grid: scale x fan-in
    for pa in PAS:
        ARMS.append((f"s{s:g}_pa{pa:g}", {"conn_scale": s, "p_a": pa}))
for n in (16, 100):                    # size, should act only through g
    for s in (0.5, 2.0, 8.0):
        ARMS.append((f"n{n}_s{s:g}", {"n_cols": n, "conn_scale": s}))
for dc in (16, 64):                    # width, should NOT move g
    for s in (0.5, 2.0, 8.0):
        ARMS.append((f"d{dc}_s{s:g}", {"d_col": dc, "conn_scale": s}))
ARMS = tuple(ARMS)


def arm_cfg(over):
    d = dict(BASE)
    d.update({k: v for k, v in over.items() if k != "n_cols"})
    return d


def run_one(job):
    seed, label, over, act, norm = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    act39._patch(act, KEEP, norm)
    n_cols = over.get("n_cols", BASE_NCOLS)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                          n_cols=n_cols, **arm_cfg(over))
    mk = lambda t: BareHost(sweep.BareAgtXC(          # noqa: E731
        cfg, _bare_path(f"gain41/{t}{os.getpid()}"), seed=seed,
        dire=True, frozen=True))
    host, twin_host = mk("a"), mk("b")
    agt, twin_agt = host.agt, twin_host.agt
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]

    # measured in-degree: how many Dir.A connections each bulk column RECEIVES
    indeg = {loc: 0 for loc in bulk}
    for c in agt.cols.values():
        for (tloc, d) in c.conns:
            if d.name == "A" and tloc in indeg:
                indeg[tloc] += 1
    k_mean = sum(indeg.values()) / max(1, len(indeg))
    g = cfg.conn_scale * k_mean ** 0.5

    def twin_set(vec):
        act39.twin_sync(agt, twin_agt, bulk, vec)

    rows, keys, dlog = [], [], []
    status, launched = "ok", False
    for step, a in enumerate(de.kp1_cycle(
            STEPS, seed=de.derive_seed("kp1", NS, seed))):
        host.observe(a)
        s = act39.state_of(agt)
        if not torch.isfinite(s).all():
            status = "DIVERGED"
            break
        if not launched and step == BURN:
            twin_set(s + PERT * torch.randn_like(s))
            launched = True
        elif launched:
            twin_host.observe(a)
            t = act39.state_of(twin_agt)
            if not torch.isfinite(t).all():
                launched = False
            else:
                dd = float((s - t).norm())
                if dd > 0:
                    dlog.append(torch.log(torch.tensor(dd / RENORM)).item())
                    twin_set(s + (t - s) * (RENORM / dd))
        if step >= BURN:
            rows.append(s.clone())
            keys.append(hash(tuple((s > 0.5).tolist())))

    base = {"seed": seed, "arm": label, "act": act, "norm": norm,
            "scale": cfg.conn_scale, "n_cols": n_cols, "d_col": cfg.d_col,
            "k_mean": k_mean, "g": g,
            "config_hash": rep.arm_hash(f"gain41:{label}:{act}:{norm}",
                                        arm_cfg(over), STEPS)}
    if status != "ok" or len(rows) < 32:
        return dict(base, status=status if status != "ok" else "SHORT")

    X = torch.stack(rows).float()
    pr, pr_all = act39._participation(X, cfg.d_col)
    ac1, tau = act39._autocorr(X)
    mag = X.norm(dim=1).clamp(min=1e-12).log()
    tt = torch.arange(len(mag), dtype=torch.float32)
    drift = float(((tt - tt.mean()) * (mag - mag.mean())).sum()
                  / ((tt - tt.mean()) ** 2).sum())
    maj, _, npat = arule.degeneracy(agt)
    return dict(base, status=status, rate=float(X.abs().mean()),
                dens=float((X > 0.5).float().mean()), pr=pr, pr_all=pr_all,
                maj=maj, npat=npat, ac1=ac1, act_tau=tau,
                nstate=len(set(keys)),
                lyap=sum(dlog) / len(dlog) if len(dlog) > 8 else float("nan"),
                drift=drift)


def main():
    jobs = [(s, lbl, ov, a, n) for lbl, ov in ARMS for a in ACTS
            for n in NORMS for s in SEEDS]
    print(f"{len(jobs)} cells ({len(ARMS)} arms x {len(ACTS)} acts x "
          f"{len(NORMS)} norms x {len(SEEDS)} seeds), {STEPS} steps", flush=True)
    out = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            print(f"  {i}/{len(jobs)} s{r['seed']} {r['arm']:>12} {r['act']:>7} "
                  f"norm{int(r['norm'])} g={r['g']:6.2f} {r['status']}",
                  flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gain41.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    report(out)


def report(out):
    def m(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))
             and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    for act in ACTS:
        for norm in NORMS:
            rs0 = [r for r in out if r["act"] == act and r["norm"] == norm]
            if not rs0:
                continue
            print(f"\n--- act={act} norm={norm} --- sorted by g")
            print(f"{'arm':>12}{'scale':>7}{'k':>6}{'g':>7}{'pr':>7}"
                  f"{'lyap':>8}{'rate':>8}{'drift':>10}  st")
            for r in sorted(rs0, key=lambda r: r["g"]):
                if r["seed"] != SEEDS[0]:
                    continue
                print(f"{r['arm']:>12}{r['scale']:>7.2f}{r['k_mean']:>6.1f}"
                      f"{r['g']:>7.2f}{r.get('pr', float('nan')):>7.1f}"
                      f"{r.get('lyap', float('nan')):>8.3f}"
                      f"{r.get('rate', float('nan')):>8.3f}"
                      f"{r.get('drift', float('nan')):>10.2e}  {r['status']}")


if __name__ == "__main__":
    main()
