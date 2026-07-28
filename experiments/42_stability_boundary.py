"""
Is "17 of 30 activations are stable" a fact about activations or about gain?
(RW, 2026-07-28.)

WHY. Run 39 swept 30 activations at five carry-over settings and found that
ten of them diverge everywhere: lrelu, rrelu1, square, tri1.0, softplus, elu,
gelu, silu, shrink0.5, kwta4. Every one of those is unbounded. Every survivor
is either bounded (step, sign, tanh, sigmoid, softsign, l2norm, wta) or
sublinear (tri0.3 and tri0.7 live, tri1.0 dies at exactly the point where the
triangle stops compressing). That reads as a statement about how loud the
network is, not about which function it applies.

Run 39 only ever ran at one loudness. conn_scale sat at its default of 2.0 in
all 600 cells. Run 41 then measured that conn_scale moves mean activity by
three orders of magnitude, from 0.06 at scale 0.25 to 61.7 at scale 8, without
changing how many patterns the network uses. So the divergent set may simply
be the set of activations whose amplification exceeds one at scale 2, and the
stable set may be everything that saturates or compresses before it gets there.

The registered reading:
  the ten dead activations come back at low scale
      -> the activation axis and the gain axis are one axis, and run 39's
         headline is really "at conn_scale 2.0". The claim gets rewritten.
  they still diverge at a sixty-fourth of the gain, and at the lowest fan-in
      -> divergence is a property of the function's shape, not of amplitude.
         The claim gets stronger and unbounded activations are ruled out for
         this substrate on their own merits.
  the boundary is at a consistent value of measured activity across acts
      -> there is a single amplitude threshold, and it can be reported as one
         number rather than as a list of survivors.

Design.
  main      30 acts x 7 scales x 2 seeds, norm off. The scale ladder reaches
            2.0 so run 39's cells are reproduced at the top rung.
  norm      30 acts x 2 scales x 2 seeds, norm ON. Run 39's norm axis never
            actually ran: its _patch installed the norm-off override only in
            the norm=False branch and never restored it, so once a pool worker
            had run one norm-off cell every later norm-on cell on that worker
            silently ran with the norm off too. 12 of 300 pairs in run 39 and
            12 of 144 in run 41 were the only real ones. The patch is fixed;
            this is the first honest measurement of that axis.
  sparse    the ten dead acts at the lowest scale and the lowest fan-in
            (p_a 0.1, mean in-degree 1.7). The most favourable setting the
            substrate has. If they diverge here they diverge anywhere.

Vitals: div_step records when the state stopped being finite, and peak records
the largest mean |activity| seen while it was still finite, so a cell that
diverges is not just a missing row. Both are needed to tell a slow runaway
from an immediate overflow.

Summaries only (kind stab42).
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

SMOKE = bool(os.environ.get("STAB_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51)
STEPS = 300 if SMOKE else 2000
BURN = 60 if SMOKE else 400
WORKERS = 2 if SMOKE else int(os.environ.get("STAB_WORKERS", "8"))
NS = "comp25"
KEEP = 0.0
PERT, RENORM = act39.PERT, act39.RENORM

BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3, rule="softhebb",
            lr_e=0.1, signed=True, transport="raw", power=0.3)

ACTS = act39.ACTS if not SMOKE else ("tri0.3", "relu", "gelu")
SCALES = (0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0) if not SMOKE \
    else (0.0625, 2.0)
NORM_SCALES = (0.25, 2.0) if not SMOKE else (2.0,)
DEAD = ("lrelu", "rrelu1", "square", "tri1.0", "softplus", "elu", "gelu",
        "silu", "shrink0.5", "kwta4") if not SMOKE else ("gelu",)

# (label, act, conn_scale, p_a, norm)
ARMS = []
for a in ACTS:
    for s in SCALES:
        ARMS.append((f"{a}@s{s:g}", a, s, 0.3, False))
for a in ACTS:
    for s in NORM_SCALES:
        ARMS.append((f"{a}@s{s:g}+n", a, s, 0.3, True))
for a in DEAD:
    ARMS.append((f"{a}@sparse", a, SCALES[0], 0.1, False))
ARMS = tuple(ARMS)


def run_one(job):
    seed, label, act, scale, p_a, norm = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    act39._patch(act, KEEP, norm)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                          n_cols=sweep.N_COLS,
                          **dict(BASE, conn_scale=scale, p_a=p_a))
    mk = lambda t: BareHost(sweep.BareAgtXC(          # noqa: E731
        cfg, _bare_path(f"stab42/{t}{os.getpid()}"), seed=seed,
        dire=True, frozen=True))
    host, twin_host = mk("a"), mk("b")
    agt, twin_agt = host.agt, twin_host.agt
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]

    indeg = {loc: 0 for loc in bulk}
    for c in agt.cols.values():
        for (tloc, d) in c.conns:
            if d.name == "A" and tloc in indeg:
                indeg[tloc] += 1
    k_mean = sum(indeg.values()) / max(1, len(indeg))

    def twin_set(vec):
        i = 0
        for loc in bulk:
            c = twin_agt.cols[loc]
            n = c.nr_1.actual.numel()
            c.nr_1.actual = vec[i:i + n].view_as(c.nr_1.actual).clone()
            i += n

    rows, dlog = [], []
    status, launched, div_step, peak = "ok", False, None, 0.0
    for step, a in enumerate(de.kp1_cycle(
            STEPS, seed=de.derive_seed("kp1", NS, seed))):
        host.observe(a)
        s = act39.state_of(agt)
        if not torch.isfinite(s).all():
            status, div_step = "DIVERGED", step
            break
        peak = max(peak, float(s.abs().mean()))
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

    base = {"seed": seed, "arm": label, "act": act, "scale": scale,
            "p_a": p_a, "norm": norm, "k_mean": k_mean, "peak": peak,
            "div_step": div_step,
            "config_hash": rep.arm_hash(f"stab42:{label}", dict(
                BASE, conn_scale=scale, p_a=p_a, act=act, norm=norm,
                keep=KEEP), STEPS)}
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
                lyap=sum(dlog) / len(dlog) if len(dlog) > 8 else float("nan"),
                drift=drift)


def main():
    jobs = [(s, lbl, a, sc, pa, nm) for (lbl, a, sc, pa, nm) in ARMS
            for s in SEEDS]
    print(f"stab42: {len(jobs)} cells ({len(ARMS)} arms x {len(SEEDS)} seeds), "
          f"{STEPS} steps, keep={KEEP}", flush=True)
    out = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            print(f"  {i}/{len(jobs)} s{r['seed']} {r['arm']:>16} "
                  f"k={r['k_mean']:4.1f} peak={r['peak']:9.3g} {r['status']}",
                  flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "stab42.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    report(out)


def report(out):
    ok = {(r["act"], r["scale"], r["norm"], r["p_a"]): 0 for r in out}
    for r in out:
        ok[(r["act"], r["scale"], r["norm"], r["p_a"])] += r["status"] == "ok"
    print("\n--- survival by act x scale (norm off, p_a 0.3), "
          f"{len(SEEDS)} seeds ---")
    print(f"{'act':<10}" + "".join(f"{s:>9g}" for s in SCALES))
    for a in ACTS:
        print(f"{a:<10}" + "".join(
            f"{ok.get((a, s, False, 0.3), 0):>9}" for s in SCALES))
    print(f"\n--- norm ON, scales {NORM_SCALES} ---")
    print(f"{'act':<10}" + "".join(f"{s:>9g}" for s in NORM_SCALES))
    for a in ACTS:
        print(f"{a:<10}" + "".join(
            f"{ok.get((a, s, True, 0.3), 0):>9}" for s in NORM_SCALES))
    print("\n--- the ten dead acts at lowest scale and lowest fan-in ---")
    for a in DEAD:
        rs = [r for r in out if r["act"] == a and r["p_a"] == 0.1]
        if rs:
            print(f"{a:<10} {sum(r['status'] == 'ok' for r in rs)}"
                  f"/{len(rs)} survived  peak={max(r['peak'] for r in rs):.3g}")


if __name__ == "__main__":
    main()
