"""
Was the gate family's collapse about the rule or about the reset? (RW, 2026-07-28.)

WHY. Every Dir.A rule experiment in this arc (29, 31, 34, 36, 38) ran with the
activation reset hard on. src.funcs.update returns zeros_like(x) every step, so
carry-over is zero. Run 39 then measured what that costs: at keep 0 the
network's memory is one step. Activity is a function of the current input and
nothing else.

That matters for a Hebbian rule specifically. Gated Oja strengthens the
connection between a sender and a receiver that are active together. With no
carry-over, two units are only ever active together because the same input
drove them both in the same step. There is no internal state for the rule to
find structure in, so every column's weights are being pulled toward the same
thing: the input. Rank-1 collapse is what that looks like from the outside.

So run 38's conclusion, that no gate shape avoids the collapse and the whole
family is exhausted, may be a conclusion about the reset policy rather than
about gated Oja. Nobody has checked. This checks.

Two things are added that run 38 could not test.

  keep      0.0 reproduces run 38 exactly and is the built-in check on this
            harness. 0.5 and 0.9 give the network 2 and 31 steps of memory
            respectively, measured in run 39.
  topk4z    a hard-zero gate. Run 38's topk arms set losers to -resp, so they
            were signed, not sparse; every unit still moved every step. A unit
            that receives exactly zero cannot be dragged toward the common
            direction at all. This is the "zeros not soup" item and it has
            never been run.

Reading, fixed before the run:
  collapse survives at keep 0.9 in every arm
      -> run 38's verdict stands on its own, the reset was not the cause, and
         the ground is clear for the three-factor experiment.
  some arm holds |cos| near the frozen 0.142 at keep > 0 while moving |W|
      -> five experiments were measuring the reset policy. Everything from 29
         onward is reopened at nonzero carry-over before anything else runs.
  topk4z holds where topk4 collapses
      -> the mechanism is that a nonzero gate on losers is what does the
         dragging, and sparsity is the fix, independent of carry-over.
  every arm goes unstable at keep 0.9
      -> carry-over and plastic weights together run away, which is itself the
         answer to whether the substrate needs a third factor for stability.

The inert trap from run 38 applies unchanged: an arm that keeps |cos| low while
|W| never leaves its frozen value has not solved anything, it has stopped
learning. w_norm is reported next to cos on every line for that reason.

Summaries plus a 25-point trace per cell (kind gate43).
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

sweep = importlib.import_module("10_sweep_bare_x")
rep = importlib.import_module("11_replicate_instar_relu")
arule = importlib.import_module("31_a_rule_degeneracy")
wdir = importlib.import_module("37_weight_direction")
gate38 = importlib.import_module("38_gate_family")
from experiments import dire_exam as de
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("GATE43_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS = 240 if SMOKE else 12000
TRACE_PTS = 25
WORKERS = 2 if SMOKE else int(os.environ.get("GATE43_WORKERS", "8"))
NS = "comp25"
BETA = 0.5
BASE = dict(gate38.BASE)

KEEPS = (0.0, 0.5) if SMOKE else (0.0, 0.5, 0.9)

# (name, kind, param, a_frozen). subset of run 38 plus the hard-zero gate.
ARMS = (("frozen", "argmax", 0.0, True),
        ("argmax", "argmax", 0.0, False),
        ("centred", "centred", 0.0, False),
        ("shift100", "shift", 1.00, False),
        ("topk4", "topk", 4, False),
        ("topk4z", "topkzero", 4, False),
        ("unsigned", "unsigned", 0.0, False))

_ORIG_WTA = fc.softmax_wta
_ORIG_UPD = fc.update
_ORIG_UPD_E = fc.update_e
GSUM = []


def install(kind, param, keep):
    """Reinstall every patch from the pristine originals on each call. A pool
    reuses its workers, so a patch layered on whatever the previous cell left
    behind leaks into later cells: that is the bug that silently disabled the
    norm axis of runs 39 and 41."""
    fc.softmax_wta = _ORIG_WTA
    fc.update, fc.update_e = _ORIG_UPD, _ORIG_UPD_E

    if keep > 0.0:
        fc.update = lambda x, threshold=1.0: keep * x
        fc.update_e = lambda x: keep * x

    gate38.GSUM = GSUM        # so run 38's arms record into this list too
    if kind == "topkzero":
        def f(u, beta=1.0, signed=False):
            resp = torch.softmax(beta * u, dim=-1)
            if not signed:
                return _rec(resp)
            g = torch.zeros_like(resp)        # losers get exactly zero
            idx = torch.topk(u, int(param)).indices
            g[idx] = resp[idx]
            return _rec(g)
        fc.softmax_wta = f
    else:
        gate38.install(kind, param)


def _rec(g):
    GSUM.append(float(g.sum()))
    return g


def cfg_hash(name, kind, param, a_frozen, keep):
    return rep.arm_hash(f"gate43:{name}",
                        dict(BASE, gate=kind, gparam=param,
                             a_frozen=a_frozen, keep=keep), STEPS)


def run_one(job):
    seed, name, kind, param, a_frozen, keep = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    install(kind, param, keep)
    GSUM.clear()
    gate38.GSUM.clear()
    try:
        cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                              n_cols=sweep.N_COLS, **BASE)
        host = BareHost(sweep.BareAgtXC(
            cfg, _bare_path(f"gate43/w{os.getpid()}"), seed=seed,
            dire=False, a_frozen=a_frozen))
        agt = host.agt
        W0 = wdir.snapshot(agt)
        ci0, cc0, w0 = wdir.wstats(agt, W0)

        every = max(STEPS // TRACE_PTS, 1)
        trace, cross = [], {"t_maj75": None, "t_cos95": None, "t_cos90": None}
        prev, ac_num, ac_den, status = None, 0.0, 0.0, "ok"
        for step, a in enumerate(de.kp1_cycle(
                STEPS, seed=de.derive_seed("kp1", NS, seed))):
            host.observe(a)
            s = torch.cat([c.nr_1.actual.flatten() for loc, c in
                           agt.cols.items() if not agt.is_i(loc)])
            if not torch.isfinite(s).all():
                status = "DIVERGED"
                cross["div_step"] = step
                break
            if prev is not None:                 # lag-1 autocorrelation, so
                ac_num += float((s * prev).sum())    # the carry-over can be
                ac_den += float((s * s).sum())       # confirmed to have landed
            prev = s
            if step % every and step != STEPS - 1:
                continue
            maj, _, npat = arule.degeneracy(agt)
            ci, cc, wn = wdir.wstats(agt, W0)
            trace.append([step, round(maj, 5), round(wn, 4),
                          round(ci, 6), round(cc, 6)])
            if cross["t_maj75"] is None and maj > 0.75:
                cross["t_maj75"] = step
            if cross["t_cos95"] is None and ci < 0.95:
                cross["t_cos95"] = step
            if cross["t_cos90"] is None and cc > 0.90:
                cross["t_cos90"] = step

        gs = GSUM or gate38.GSUM
        out = {"seed": seed, "arm": name, "gate": kind, "gparam": param,
               "keep": keep, "status": status,
               "w0": w0, "mean_abs_cos0": cc0,
               "net_mass": (sum(gs) / len(gs)) if gs else float("nan"),
               "ac1": ac_num / ac_den if ac_den > 0 else float("nan"),
               "trace": trace, **cross,
               "config_hash": cfg_hash(name, kind, param, a_frozen, keep)}
        if status != "ok":
            return out
        maj, minority, npat = arule.degeneracy(agt)
        ci, cc, wn = wdir.wstats(agt, W0)
        norms = [float(c.nr_1.actual.norm()) for loc, c in agt.cols.items()
                 if not agt.is_i(loc)]
        return dict(out, maj=maj, minority=minority, n_patterns=npat,
                    act_norm=sum(norms) / max(len(norms), 1),
                    w_norm=wn, cos_init=ci, mean_abs_cos=cc)
    finally:
        fc.softmax_wta = _ORIG_WTA
        fc.update, fc.update_e = _ORIG_UPD, _ORIG_UPD_E


def main():
    jobs = [(s, n, k, p, fz, kp) for (n, k, p, fz) in ARMS for kp in KEEPS
            for s in SEEDS]
    print(f"gate43: {len(jobs)} cells ({len(ARMS)} arms x {len(KEEPS)} keeps "
          f"x {len(SEEDS)} seeds), Dir.E frozen, beta {BETA}, {STEPS} steps",
          flush=True)
    out = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            print(f"  {i}/{len(jobs)} s{r['seed']} {r['arm']:>9} "
                  f"keep{r['keep']:<4} |cos| "
                  f"{r.get('mean_abs_cos', float('nan')):.3f} |W| "
                  f"{r.get('w_norm', float('nan')):.2f} maj "
                  f"{r.get('maj', float('nan')):.3f} ac1 "
                  f"{r.get('ac1', float('nan')):+.3f} {r['status']}",
                  flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gate43.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    report(out)


def report(out):
    def m(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))
             and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    for keep in KEEPS:
        print(f"\n--- keep={keep} (mean over {len(SEEDS)} seeds) ---")
        print(f"{'arm':>9}{'|cos|':>8}{'cos_init':>10}{'|W|':>9}{'maj':>8}"
              f"{'npat':>7}{'ac1':>8}{'netmass':>9}  ok")
        for name, _, _, _ in ARMS:
            rs = [r for r in out if r["arm"] == name and r["keep"] == keep]
            if not rs:
                continue
            print(f"{name:>9}{m(rs, 'mean_abs_cos'):>8.3f}"
                  f"{m(rs, 'cos_init'):>10.3f}{m(rs, 'w_norm'):>9.2f}"
                  f"{m(rs, 'maj'):>8.3f}{m(rs, 'n_patterns'):>7.1f}"
                  f"{m(rs, 'ac1'):>+8.3f}{m(rs, 'net_mass'):>9.3f}"
                  f"  {sum(r['status'] == 'ok' for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    main()
