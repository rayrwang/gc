"""
Topology x activation, both weight channels frozen (RW, 2026-07-27).

Companion to 39. That run asked what the nonlinearity and the reset do to the
dynamics with the wiring held at the shipped values; this one holds the
nonlinearity at three representative points and moves the wiring.

Every knob here is a field of BareXCfg that every prior run has left at its
default, so nothing below has ever been varied on this substrate:

  n_cols      how many bulk columns. the grid is ceil(sqrt(n_cols)) on a side
              and connection probability falls off with distance, so changing
              the count changes the diameter as well as the size
  d_col       width of one column
  p_a, p_e    base connection probability for the two channels, before the
              distance falloff
  dist_exp    the falloff exponent. 0 makes connectivity distance-independent,
              i.e. an unstructured random graph; the shipped 1.0 is a spatial
              network; higher is more local
  single_conn stock's at-most-one-connection-per-target break. turning it off
              is the only knob here that changes the graph's multiplicity
              rather than its density
  conn_scale  weight init scale, which is not topology but sets the loop gain,
              and any dynamical claim about connectivity is confounded with it
              unless it is swept alongside
  input_fanout how many bulk columns the input column reaches

Design. One knob at a time from the run-39 baseline, which is the same
baseline the whole 25-38 arc used. A full grid over eight axes is not
affordable and would not be readable; a star design answers "does this knob
move the dynamics" for each knob, which is the question being asked, and
leaves interactions for a follow-up on whichever knobs turn out to matter.

The three activations are chosen to span the shape space rather than to be
representative: `raw` is linear, `tri0.3` is the shipped mean-subtracting
rectifier, `step1` is the shipped spike threshold and is discontinuous and
binary. If a topology effect appears under all three it is a property of the
wiring; if it appears under one it is an interaction and the wiring knob is
not independently meaningful.

Why the RMS normalisation's own constants are in here. Reading the history
back, the norm is not a design choice, it is a patch, and it has a date and a
cause:

    until 2026-06-24  activation is `fc.atv` = spike(x) @ w, a step function.
                      bounded and binary, so the recurrent loop cannot run away
    2026-06-24 1d5a211  switched to atv_triangle = relu(x - mean)^p @ w.
                      unbounded. this is the change that made divergence
                      possible
    2026-06-24 dfa2fe9  init scale 1.0 -> 1.1, raising loop gain again
    2026-06-26 94a1cba  gated Oja
    2026-06-27 35e50b7  "Add bare agent RMS norm"
    2026-06-29 462f48c  triangle power -> 0.3, gate signed, beta 0.5

Three days after the activation stopped being bounded, a divisive
normalisation appeared. Run 39's early cells reproduce the condition that
forced it: every `use_norm=False` cell diverges under every unbounded
activation tested so far, and the only survivors are `use_norm=True`.

So ALPHA_RMS and the 0.9 floor were written in one commit to solve one
problem, have not been touched since, and are now load-bearing for the whole
substrate. That is exactly the kind of constant worth sweeping, and the loop
gain they were fighting has meanwhile risen further: conn_scale is 2.0 today
against the 1.1 in force when the norm was added, which is why conn_scale is
swept in the same run rather than a later one.

Note that run 39's `step1` arm is the pre-2026-06-24 substrate. If the step
arms survive with the norm off, the history is confirmed by measurement
rather than inferred from a commit message.

Two confounds found while checking the arms construct, both left in the design
deliberately and both made visible by reporting the realised connection counts
`nA` and `nE` next to every row:

  Density. `dist_exp` is nominally a structure knob but it is also a density
  knob: at the baseline the graph has nA=240, nE=455, and at dist_exp=0 the
  same nominal probabilities give nA=697, nE=1006. Comparing those two rows
  compares an unstructured graph against a spatial one and a dense graph
  against a sparse one. The census is the only way to tell which did the work,
  and `p_a` is swept alongside precisely so density can be matched after the
  fact rather than assumed away.

  Channel competition. `single_conn` breaks out of the loop after the first
  successful connection to a target, and Dir.A is offered first, so raising
  `p_a` crowds out Dir.E: p_a=1.0 gives nA=833 against the baseline's 240
  while nE falls from 455 to 283. The two channels are competing for one slot
  per target. That is a property of the shipped wiring nobody has stated, and
  it means p_a and p_e are not independent axes on this substrate.

Instruments are 39's, imported rather than reimplemented so the two runs are
directly comparable. The Lyapunov twin is the one that classifies the regime;
the rest describe it. See 39's docstring for the panel.

Reading, decided before the run:
  a knob that moves `lyap` across zero        -> it selects the regime, and
      the substrate's behaviour has been set by an unexamined default
  `pr` scaling with d_col but not n_cols      -> dimensionality is a within-
      column property and adding columns adds copies, not capacity
  `dist_exp`=0 differing from the rest        -> spatial structure matters,
      which would be the first evidence that the grid geometry does anything
  conn_scale dominating                       -> the connectivity knobs are
      being read through loop gain and the design needs gain held fixed

Summaries only (kind topo40).
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

SMOKE = bool(os.environ.get("TOPO_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51)
STEPS = 300 if SMOKE else 2000
BURN = 60 if SMOKE else 400
WORKERS = 2 if SMOKE else int(os.environ.get("TOPO_WORKERS", "5"))
NS = "comp25"
ACTS = ("tri0.3",) if SMOKE else ("raw", "tri0.3", "step1")
KEEP, NORM = 0.0, True          # 39's incumbent reset and norm
PERT, RENORM = act39.PERT, act39.RENORM

# the 25-38 baseline. n_cols is not in BASE because it is passed separately
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            rule="softhebb", lr_e=0.1, signed=True,
            transport="raw", power=0.3)
BASE_NCOLS = sweep.N_COLS

# star design: (label, overrides). the empty override is the baseline itself
ARMS = [("base", {})]
# the baseline is sweep.N_COLS = 49, so the ladder has to span it in both
# directions. square values only: the grid side is ceil(sqrt(n_cols)) and a
# non-square count leaves a ragged edge, which is a different perturbation
# from changing the size
for v in (4, 9, 16, 25, 36, 64, 81, 100):
    ARMS.append((f"ncols{v}", {"n_cols": v}))
for v in (8, 16, 64, 128):
    ARMS.append((f"dcol{v}", {"d_col": v}))
for v in (0.1, 0.2, 0.5, 0.8, 1.0):
    ARMS.append((f"pa{v:g}", {"p_a": v}))
for v in (0.2, 0.4, 0.8, 1.0):
    ARMS.append((f"pe{v:g}", {"p_e": v}))
for v in (0.0, 0.5, 2.0, 4.0):
    ARMS.append((f"dexp{v:g}", {"dist_exp": v}))
ARMS.append(("multiconn", {"single_conn": False}))
for v in (0.5, 1.0, 4.0, 8.0):
    ARMS.append((f"scale{v:g}", {"conn_scale": v}))
for v in (1, 6, 12):
    ARMS.append((f"fanout{v}", {"input_fanout": v}))
# the RMS normalisation's own constants. `agents.ALPHA_RMS` is the EMA decay
# on the running root-mean-square and the 0.9 in `max(0.9, new_rms_avg)` is a
# floor that stops the norm amplifying quiet layers. Both arrived together in
# 35e50b7 (2026-06-27, "Add bare agent RMS norm") and neither has been moved
# since. Run 39 sweeps the norm on and off; this sweeps its shape.
#   alpha 1.0   no smoothing, divide by the current step's RMS
#   alpha 0.05  heavy smoothing, the divisor is nearly a constant
#   floor 0.0   always normalise, including quiet layers (amplifies)
#   floor 1.0   never amplify, only ever damp
# these are prefixed `norm_` and handled outside arm_cfg because they patch
# module constants rather than the agent config.
for v in (0.05, 0.5, 1.0):
    ARMS.append((f"alpharms{v:g}", {"norm_alpha": v}))
for v in (0.0, 0.5, 1.0):
    ARMS.append((f"rmsfloor{v:g}", {"norm_floor": v}))
ARMS = tuple(ARMS)


NORM_KEYS = ("n_cols", "norm_alpha", "norm_floor")


def arm_cfg(over):
    """cfg fields only. n_cols is passed separately and the norm_* keys patch
    module constants rather than the config."""
    d = dict(BASE)
    d.update({k: v for k, v in over.items() if k not in NORM_KEYS})
    return d


def patch_norm(over):
    """move the RMS normalisation's own constants. `update_activations` reads
    ALPHA_RMS off the module and has the floor written inline, so the floor is
    changed by rebinding the whole method rather than a constant."""
    import src.agents as ag
    if "norm_alpha" in over:
        ag.ALPHA_RMS = float(over["norm_alpha"])
    if "norm_floor" in over:
        fl = float(over["norm_floor"])
        import torch as _t

        def _upd(self, use_norm=True):
            a = ag.ALPHA
            new_avg = a * self.nr_1_.actual + (1 - a) * self.nr_1.avg
            new_avg_sq = a * self.nr_1_.actual ** 2 + (1 - a) * self.nr_1.avg_sq
            if use_norm:
                r = ag.ALPHA_RMS * _t.sqrt(
                    _t.mean(self.nr_1_.actual ** 2)).item() \
                    + (1 - ag.ALPHA_RMS) * self.nr_1.rms_avg
                r = max(fl, r)          # the swept floor, 0.9 in the shipped code
            else:
                r = 1.0
            self.nr_1.actual = self.nr_1_.actual / r if r else self.nr_1_.actual
            self.nr_1.expect = self.nr_1_.expect
            self.nr_1.avg, self.nr_1.avg_sq, self.nr_1.rms_avg = \
                new_avg, new_avg_sq, r
            self.nr_1_.actual = fc_update(self.nr_1_.actual)
            self.nr_1_.expect = fc_update_e(self.nr_1_.expect)

        from src import funcs as _fc
        fc_update, fc_update_e = _fc.update, _fc.update_e
        ag.BareCol.update_activations = _upd


def cfg_hash(label, over, act):
    return rep.arm_hash(f"topo40:{label}:{act}",
                        dict(arm_cfg(over), n_cols=over.get("n_cols",
                                                            BASE_NCOLS)), STEPS)


def run_one(job):
    seed, label, over, act = job
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)   # see 39: threading buys nothing at d_col=32
    act39._patch(act, KEEP, NORM)
    patch_norm(over)              # after _patch: it may rebind the same method
    n_cols = over.get("n_cols", BASE_NCOLS)
    cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                          n_cols=n_cols, **arm_cfg(over))
    mk = lambda tag: BareHost(sweep.BareAgtXC(   # noqa: E731
        cfg, _bare_path(f"topo40/{tag}{os.getpid()}"), seed=seed,
        dire=True, frozen=True))
    host, twin_host = mk("a"), mk("b")
    agt, twin_agt = host.agt, twin_host.agt
    bulk = [loc for loc in agt.cols if not agt.is_i(loc)]

    # connection census: the thing the topology knobs are supposed to move
    n_a = sum(1 for c in agt.cols.values() for (_, d) in c.conns
              if d.name == "A")
    n_e = sum(1 for c in agt.cols.values() for (_, d) in c.conns
              if d.name == "E")

    def twin_set(vec):
        i = 0
        for loc in bulk:
            c = twin_agt.cols[loc]
            n = c.nr_1.actual.numel()
            c.nr_1.actual = vec[i:i + n].view_as(c.nr_1.actual).clone()
            i += n

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
                d = float((s - t).norm())
                if d > 0:
                    dlog.append(torch.log(torch.tensor(d / RENORM)).item())
                    twin_set(s + (t - s) * (RENORM / d))
                else:
                    twin_set(s + RENORM * torch.randn_like(s)
                             / s.numel() ** 0.5)
        if step >= BURN:
            rows.append(s.clone())
            keys.append(hash(tuple((s > 0.5).tolist())))

    base = {"seed": seed, "arm": label, "act": act, "n_cols": n_cols,
            "d_col": cfg.d_col, "n_a": n_a, "n_e": n_e,
            "config_hash": cfg_hash(label, over, act)}
    if status != "ok" or len(rows) < 32:
        return dict(base, status=status if status != "ok" else "SHORT")

    X = torch.stack(rows).float()
    if float(X.abs().max()) == 0.0:
        status = "SILENT"
    per_unit = (X > 0.5).float().mean(0)
    pr, pr_all = act39._participation(X, cfg.d_col)
    ac1, tau = act39._autocorr(X)
    fdom, fpow = act39._spectrum(X.mean(1))
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
    maj, _, npat = arule.degeneracy(agt)
    return dict(base, status=status,
                rate=float(X.abs().mean()), dens=float((X > 0.5).float().mean()),
                dead=float((per_unit == 0).float().mean()),
                sat=float((per_unit == 1).float().mean()),
                pr=pr, pr_all=pr_all, maj=maj, npat=npat, ac1=ac1,
                act_tau=tau, nstate=len(set(keys)), period=act39._cycle(keys),
                lyap=sum(dlog) / len(dlog) if len(dlog) > 8 else float("nan"),
                fdom=fdom, fpow=fpow, sync=sync, drift=drift)


def main():
    jobs = [(s, lbl, ov, a) for lbl, ov in ARMS for a in ACTS for s in SEEDS]
    print(f"{len(jobs)} cells ({len(ARMS)} topologies x {len(ACTS)} acts "
          f"x {len(SEEDS)} seeds), {STEPS} steps, burn {BURN}", flush=True)
    out = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            out.append(r)
            print(f"  {i}/{len(jobs)} s{r['seed']} {r['arm']:>10} "
                  f"{r['act']:>7} nA{r['n_a']:<5} {r['status']}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "topo40.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    report(out)


def report(out):
    def agg(rs, k):
        v = [r[k] for r in rs if isinstance(r.get(k), (int, float))
             and r[k] == r[k]]
        return sum(v) / len(v) if v else float("nan")

    print("\ntopology star sweep, both channels frozen, seeds", list(SEEDS))
    print("nA/nE are the realised connection counts: the knobs are nominal, "
          "these are what the graph actually got.")
    for a in ACTS:
        print(f"\n--- act={a} ---")
        print(f"{'arm':>10}{'nA':>6}{'nE':>6}{'rate':>8}{'dens':>7}"
              f"{'pr':>7}{'maj':>7}{'ac1':>7}{'tau':>6}{'nstate':>7}"
              f"{'lyap':>8}{'fpow':>6}{'sync':>7}{'drift':>9}  st")
        for lbl, _ in ARMS:
            rs = [r for r in out if r["arm"] == lbl and r["act"] == a]
            if not rs:
                continue
            bad = [r["status"] for r in rs if r["status"] != "ok"]
            print(f"{lbl:>10}{agg(rs,'n_a'):>6.0f}{agg(rs,'n_e'):>6.0f}"
                  f"{agg(rs,'rate'):>8.3f}{agg(rs,'dens'):>7.3f}"
                  f"{agg(rs,'pr'):>7.2f}{agg(rs,'maj'):>7.3f}"
                  f"{agg(rs,'ac1'):>7.3f}{agg(rs,'act_tau'):>6.1f}"
                  f"{agg(rs,'nstate'):>7.0f}{agg(rs,'lyap'):>8.4f}"
                  f"{agg(rs,'fpow'):>6.2f}{agg(rs,'sync'):>7.3f}"
                  f"{agg(rs,'drift'):>9.2e}  {bad[0] if bad else 'ok'}")


if __name__ == "__main__":
    main()
