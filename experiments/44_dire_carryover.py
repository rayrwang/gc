"""
Was Dir.E's fixture-blindness about the rule or about the reset? (RW, 2026-07-28.)

WHY. aoff26 found delta converges to a near-perfect predictor that is exactly
fixture-blind, and the phrase that closed the Dir.E rule-form axis was "the
failure is AT the least-squares optimum, not short of it". Run 27 then swept
five rule forms and three step sizes and nothing recovered the differential.

That is a claim about what the optimum IS, so it depends entirely on what the
regressors are. And the regressors were crippled by a setting nobody in that
arc varied. src.funcs.update returns zeros_like, so carry-over is zero, and
run 39 measured the resulting memory at one step: the network's state is a
function of the current input alone. A least-squares predictor built on that
state can only be a function of the current input. Fixture-blindness is close
to a tautology under those conditions, because the two fixtures kp1 and kz are
distinguished by their temporal structure and the state carries no temporal
structure to regress on.

Raise the carry-over and the state contains its own history, so the optimum is
a different function and it is not obvious it stays blind.

Reading, decided before the run:
  delta's differential moves off zero at keep > 0
      -> the aoff26 verdict was about the reset policy, and the whole Dir.E
         rule-form axis (26, 27, 28) is reopened at nonzero carry-over. Confirm
         on fresh seeds before believing it, per run 28's protocol.
  delta stays exactly fixture-blind at keep 0.9
      -> the original finding gets much stronger, because it now holds when
         the regressors DO carry temporal structure, which is the only reading
         under which "the failure is at the optimum" says something about
         least squares rather than about the buffer.
  oja's 3/3 win at keep 0 does not survive keep > 0
      -> consistent with run 28, where it died on eight fresh seeds; the win
         was seed-specific and carry-over is not what rescues it.
  every rule diverges at keep 0.9
      -> carry-over plus plastic E weights runs away, which is a stability
         finding in its own right and matches the question run 43 asks on the
         other channel.

Design. Three rules, not run 27's five: frozen as the reference, delta as the
incumbent least-squares learner, oja as the only error-free rule run 27 found
bounded. hebb, instar and softhebb diverged at every step size there and
carry-over can only make that worse. Two step sizes, because a rule that is
stable at keep 0 has no guarantee of being stable when the state persists.

Each fixture gets its own host AND its own save path. Run 27 gave both
fixtures the same path within a worker; that is the shape of the shared-host
artefact that once produced a false world-imprint result, and there is no
reason to leave it in place.

The keep=0 cells reproduce run 27's lr_e 0.1 row and are the harness check.

Summaries only (kind dire44).
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
frz = importlib.import_module("14_registered_bare_freeze")
rep = importlib.import_module("11_replicate_instar_relu")
probe27 = importlib.import_module("27_hebb_e_probe")
from experiments import dire_exam as de
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T

SMOKE = bool(os.environ.get("DIRE44_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
WORKERS = 2 if SMOKE else int(os.environ.get("DIRE44_WORKERS", "8"))
NS = "comp25"
BASE = dict(probe27.BASE)

RULES = ("frozen", "delta", "oja")
KEEPS = (0.0, 0.5) if SMOKE else (0.0, 0.5, 0.9)
LR_ES = (0.1,) if SMOKE else (0.01, 0.1)

_ORIG_UPD = fc.update
_ORIG_UPD_E = fc.update_e


def install(keep):
    """Reinstall from the pristine originals every call: a pool reuses its
    workers, and a patch layered on the previous cell's leftovers is the bug
    that silently disabled the norm axis of runs 39 and 41."""
    fc.update, fc.update_e = _ORIG_UPD, _ORIG_UPD_E
    if keep > 0.0:
        fc.update = lambda x, threshold=1.0: keep * x
        fc.update_e = lambda x: keep * x


def cfg_hash(rule, lr_e, keep):
    return rep.arm_hash(f"dire44:{rule}",
                        dict(BASE, lr_e=lr_e, keep=keep), STEPS)


def run_one(job):
    seed, rule, lr_e, keep = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    install(keep)
    try:
        out = {"seed": seed, "rule": rule, "lr_e": lr_e, "keep": keep}
        deg_sum = deg_n = 0
        for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
            cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                                  n_cols=sweep.N_COLS,
                                  **dict(BASE, lr_e=lr_e))
            kw = dict(dire=False) if rule == "frozen" else dict(dire_rule=rule)
            host = BareHost(sweep.BareAgtXC(
                cfg, _bare_path(f"dire44/{fixture}{os.getpid()}"),
                seed=seed, **kw))          # own path per fixture, never shared
            agt = host.agt
            e0, _ = probe27.e_norm_stats(agt)
            r = frz.run_light(
                host, fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                WINDOW)
            out[fixture] = r["margin"]
            out[f"{fixture}_nan"] = r["nan"]
            if fixture == "kp1":
                e1, e1max = probe27.e_norm_stats(agt)
                out["e_norm_growth"] = e1 / max(e0, 1e-9)
                out["e_norm_max"] = e1max
                acts = []
                for loc, c in agt.cols.items():
                    if agt.is_i(loc):
                        continue
                    x = c.nr_1.actual
                    pos = int((x > 0).sum())
                    deg_sum += max(pos, x.numel() - pos) / x.numel()
                    deg_n += 1
                    acts.append(float(x.abs().mean()))
                out["act_rate"] = sum(acts) / max(1, len(acts))
        out["deg_final"] = deg_sum / max(1, deg_n)
        out["differential"] = (out["kp1"] - out["kz"]
                               if None not in (out["kp1"], out["kz"]) else None)
        out["config_hash"] = cfg_hash(rule, lr_e, keep)
        return out
    finally:
        fc.update, fc.update_e = _ORIG_UPD, _ORIG_UPD_E


def main():
    jobs = [(s, r, lr, kp) for kp in KEEPS for lr in LR_ES for r in RULES
            for s in SEEDS]
    print(f"dire44: {len(jobs)} cells ({len(RULES)} rules x {len(KEEPS)} keeps "
          f"x {len(LR_ES)} lr_e x {len(SEEDS)} seeds), 2 fixtures each, "
          f"{STEPS} steps", flush=True)
    out = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            d = r.get("differential")
            print(f"  {i}/{len(jobs)} s{r['seed']} {r['rule']:>8} "
                  f"keep{r['keep']:<4} lr{r['lr_e']:<6} diff "
                  f"{d if d is None else round(d, 4)} egrow "
                  f"{r.get('e_norm_growth', float('nan')):.2f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dire44.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    report(out)


def report(out):
    def m(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))
             and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    for keep in KEEPS:
        for lr in LR_ES:
            rs0 = [r for r in out if r["keep"] == keep and r["lr_e"] == lr]
            if not rs0:
                continue
            print(f"\n--- keep={keep} lr_e={lr} (mean over {len(SEEDS)} "
                  f"seeds) ---")
            print(f"{'rule':>8}{'kp1':>9}{'kz':>9}{'diff':>9}{'egrow':>9}"
                  f"{'emax':>10}{'deg':>7}{'rate':>9}  nan")
            for rule in RULES:
                rs = [r for r in rs0 if r["rule"] == rule]
                if not rs:
                    continue
                nan = sum(bool(r.get("kp1_nan")) or bool(r.get("kz_nan"))
                          for r in rs)
                print(f"{rule:>8}{m(rs, 'kp1'):>9.4f}{m(rs, 'kz'):>9.4f}"
                      f"{m(rs, 'differential'):>9.4f}"
                      f"{m(rs, 'e_norm_growth'):>9.2f}"
                      f"{m(rs, 'e_norm_max'):>10.3g}{m(rs, 'deg_final'):>7.3f}"
                      f"{m(rs, 'act_rate'):>9.3f}  {nan}/{len(rs)}")


if __name__ == "__main__":
    main()
