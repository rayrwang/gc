"""
Free-play sweep over BareAgtX's config surface, under the draw-luck rules
(RW, 2026-07-20): fixed graph for non-topological knobs, common random
numbers for density knobs, input fanout as an explicit knob, multi-seed on
top. Selection is threshold-shaped, never margin-maximal: the output is the
set of configs clearing the alive/honest bars, with margins and the kp1-kz
differential reported for reading, not ranking.

Two tiers, breadth-first, each config run to completion:
    screen   400 steps (09's first-contact length): alive + honest bars only
    confirm  3000 steps (09's convergence horizon): trailing margins + the
             kp1-kz world-imprint differential, on the passing set

Tier A sweeps non-topological knobs (rule/beta/transport/ss/lr_e) on the one
graph each seed draws. Tier B sweeps density knobs (p_a/p_e/dist_exp) and
input_fanout with per-edge hash-derived draws, so raising a density adds
edges to the same base graph and every shared edge keeps its weights.
input_fanout=0 is a built-in known answer: the world never enters, so any
config showing a real kp1-kz differential there flags broken machinery.

Storage: summaries only, one ledger entry per (tier, config, seed): runs are
deterministic given (config, seed) and cost seconds, so full traces are
regenerated on demand rather than stored. Resume = rerun this script: ledger
entries already present are skipped.
"""

import hashlib
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.funcs as fc
from experiments import dire_exam as de
from experiments.d3 import D3Store, canonical_bytes
from experiments.dire_hosts import BareAgtX, BareHost, BareXCfg, _bare_path
from src import iotypes as T
from src.agents import AgtBase, BareCol, BareColCfg, Dir, I_VectorCol, I_VectorColCfg

STORE_ROOT = os.environ.get("SWEEP_STORE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runs", "d3_store")
SMOKE = bool(os.environ.get("SWEEP_SMOKE"))
N_COLS = 49
SEEDS = (0,) if SMOKE else (0, 1, 2)
SCREEN_STEPS, SCREEN_WINDOW = (60, 20) if SMOKE else (400, 100)
CONFIRM_STEPS, CONFIRM_WINDOW = (120, 40) if SMOKE else (3000, 500)
WORKERS = 4 if SMOKE else 12
FIXTURES = (("kp1", de.kp1_cycle), ("kz", de.kz_noise))


@dataclass
class XSweepCfg(BareXCfg):
    input_fanout: int = -1  # -1 = law-default draw for the input col


def _u(edge_seed, s, t, d):
    """per-edge uniform in [0,1): the common-random-numbers stream."""
    h = hashlib.sha256(f"edge|{edge_seed}|{s}|{t}|{d}".encode()).digest()
    return int.from_bytes(h[:7], "big") / 2 ** 56


def _wgen(edge_seed, s, t, d):
    h = hashlib.sha256(f"w|{edge_seed}|{s}|{t}|{d}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "big") % 2 ** 63)


class BareAgtXC(BareAgtX):
    """common-random-numbers sibling of BareAgtX: same cols, same step, but
    edge existence and weights come from per-edge hashes of (seed, sender,
    target, dir), so densities nest monotonically across configs and shared
    edges keep identical weights; input fanout is explicit when >= 0. col
    activation init stays on the global rng, so all configs at one seed
    share identical columns."""

    def __init__(self, cfg, path, seed=0, dire=True, frozen=False, tap=False,
                 a_frozen=False, dire_rule="delta"):
        AgtBase.__init__(self, cfg, path)
        self.age = 0
        self.dire, self.frozen, self.tap = dire, frozen, tap
        self.a_frozen = a_frozen  # freeze dir.a only (additive, default off)
        self.dire_rule = dire_rule  # delta | hebb (additive, default = incumbent)
        self.last_taps = None
        random.seed(seed)
        torch.manual_seed(seed)
        g_in = cfg.ispec[0].d
        col = I_VectorCol((1, 0), I_VectorColCfg(g_in))
        self.cols[(1, 0)] = col
        self.I_cols.append(col)
        width = math.ceil(cfg.n_cols ** 0.5)
        for i, (y, x) in enumerate((y, x) for y in range(width) for x in range(width)):
            if i < cfg.n_cols:
                loc = (x + 1, y + 1)
                self.cols[loc] = BareCol(loc, BareColCfg(cfg.d_col))

        def draw(sender, other, direction):
            w = cfg.conn_scale * torch.randn(
                self.cols[sender].nr_1.actual.shape[0], self.cols[other].nr_1.actual.shape[0],
                generator=_wgen(seed, sender, other, direction.name)) \
                / self.cols[sender].nr_1.actual.shape[0] ** 0.5
            if cfg.conn_dist == "uniform":
                w = cfg.conn_scale * (2 * torch.rand(w.shape, generator=_wgen(
                    seed, sender, other, direction.name)) - 1) / w.shape[0] ** 0.5
            self.cols[sender].conns[(other, direction)] = w

        internal = [loc for loc in self.cols if not self.is_i(loc)]
        iloc = (1, 0)
        if cfg.input_fanout >= 0:  # explicit knob: k smallest per-edge u's,
            # so fanouts nest monotonically like the densities
            ranked = sorted(internal, key=lambda o: _u(seed, iloc, o, "A"))
            for other in ranked[:cfg.input_fanout]:
                draw(iloc, other, Dir.A)
        else:  # law default, crn form
            for other in internal:
                p = 0.6 / (fc.dist(iloc, other) + 1e-3) ** cfg.dist_exp
                if _u(seed, iloc, other, "A") < p:
                    draw(iloc, other, Dir.A)
        for loc in internal:
            for other in self.cols:
                if other == loc or self.is_i(other):
                    continue
                dist = fc.dist(loc, other)
                got_a = _u(seed, loc, other, "A") < cfg.p_a / (dist + 1e-3) ** cfg.dist_exp
                if got_a:
                    draw(loc, other, Dir.A)
                if got_a and cfg.single_conn:
                    continue
                if _u(seed, loc, other, "E") < cfg.p_e / (dist + 1e-3) ** cfg.dist_exp:
                    draw(loc, other, Dir.E)
        self.e_conns = [(c, tloc) for c in self.cols.values()
                        for (tloc, d) in c.conns if d == Dir.E]
        self.predicted = {tloc for _, tloc in self.e_conns}
        self.create_directory()


def tier_a_grid():
    for rule in ("softhebb", "oja", "instar", "basic"):
        for beta in ((0.25, 0.5, 1.0) if rule == "softhebb" else (0.5,)):
            for transport in ("triangle", "relu", "raw"):
                for ss in (1e-2, 3e-3):
                    for lr_e in (0.03, 0.1, 0.3):
                        yield dict(rule=rule, beta=beta, transport=transport,
                                   ss=ss, lr_e=lr_e)


def tier_b_grid():
    for p_a in (0.15, 0.3, 0.6):
        for p_e in (0.15, 0.3, 0.6):
            for dist_exp in (1.0, 2.0, 3.0):
                for input_fanout in (0, 1, 3, 6):
                    yield dict(p_a=p_a, p_e=p_e, dist_exp=dist_exp,
                               input_fanout=input_fanout)


def grids():
    a = list(tier_a_grid())
    b = list(tier_b_grid())
    if SMOKE:
        a, b = a[:2], b[:2]
    return [("a", k) for k in a] + [("b", k) for k in b]


def config_hash(tier, knobs, steps):
    return hashlib.sha256(canonical_bytes(
        {"tier": tier, "n_cols": N_COLS, "steps": steps, **knobs})).hexdigest()[:16]


def run_one(job):
    """one (tier, config, seed): both fixtures to completion, summary only."""
    tier, knobs, seed, steps, window = job
    cfg = XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)], n_cols=N_COLS, **knobs)
    # per-worker save path: create_directory clears its target, so concurrent
    # workers sharing one path race (the observed rmtree pileup shape)
    host = BareHost(BareAgtXC(cfg, _bare_path(f"dire_sweep_x/w{os.getpid()}"),
                              seed=seed))
    records = []
    for fixture, fn in FIXTURES:
        de.run_stream_multi(host, fn(steps, seed=de.derive_seed(fixture, "sweep10", seed)),
                            records.append, fixture=fixture, subject=f"x:{tier}", seed=seed)
    nan = any(x != x for r in records for x in r["actual"])
    final_nan = any(bool(torch.isnan(c.nr_1.actual).any())
                    for c in host.agt.cols.values())
    tables = de.column_tables(records, window)
    margins, exists = {}, {}
    for fixture, _ in FIXTURES:
        cells = [(v["trail"], v["persist"]) for (f, _s, _c), v in tables.items()
                 if f == fixture and v["trail"] is not None and v["persist"] is not None]
        exists[fixture] = bool(cells)
        margins[fixture] = (sum(t - p for t, p in cells) / len(cells)) if cells else None
    # silence bar counts only driven columns: a col with no incoming dir.a is
    # zero by graph shape, which is topology (visible in the summary), not a
    # dynamics death
    driven = {f"{t[0]},{t[1]}" for c in host.agt.cols.values()
              for (t, d) in c.conns if d == Dir.A}
    trail_amp = {}  # peak |actual| per step per predicted driven col, kp1 stream
    for r in records:
        if r["fixture"] == "kp1" and r["column"] in driven:
            trail_amp.setdefault(r["column"], []).append(max(abs(x) for x in r["actual"]))
    # silent (active early, zero at the end) and n_zero (never active) are
    # recorded readings, not bars: init activity makes structural settling
    # and dynamical death inseparable here, and the activity floor is a
    # pending freeze-page ruling, so the screen must not invent one
    silent = sum(1 for vs in trail_amp.values()
                 if max(vs[:window]) > 1e-9 and max(vs[-window:]) < 1e-9)
    n_zero = sum(1 for vs in trail_amp.values() if max(vs) < 1e-9)
    n_pred = len(host.agt.predicted)
    alive = (not nan) and (not final_nan) and n_pred > 0
    honest = all(exists.values())
    return {"tier": tier, "config": knobs, "config_hash": config_hash(tier, knobs, steps),
            "seed": seed, "n_pred": n_pred, "n_driven": len(driven),
            "n_zero": n_zero, "nan": nan, "silent": silent,
            "alive": alive, "honest": honest, "passed": alive and honest,
            "margin_kp1": margins["kp1"], "margin_kz": margins["kz"],
            "differential": (margins["kp1"] - margins["kz"]
                             if None not in margins.values() else None)}


def stage(store, kind, jobs):
    """run jobs not already in the ledger; append one entry per completion."""
    done = {(e["kind"], e["config_hash"], e["seed"]) for e in store.ledger()
            if e.get("kind") == kind}
    todo = [j for j in jobs
            if (kind, config_hash(j[0], j[1], j[3]), j[2]) not in done]
    print(f"{kind}: {len(jobs)} jobs, {len(jobs) - len(todo)} already in ledger, "
          f"{len(todo)} to run", flush=True)
    out = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(run_one, j): j for j in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": kind, "event_id":
                                 f"{kind}:{s['config_hash']}:s{s['seed']}", **s})
            out.append(s)
            if n % 25 == 0 or n == len(todo):
                print(f"  {kind} {n}/{len(todo)}", flush=True)
    for e in store.ledger():  # resumed entries join the report
        if e.get("kind") == kind:
            out.append(e)
    seen, uniq = set(), []
    for s in out:
        k = (s["config_hash"], s["seed"])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def main():
    store = D3Store(STORE_ROOT)
    jobs = [(t, k, s, SCREEN_STEPS, SCREEN_WINDOW) for t, k in grids() for s in SEEDS]
    screens = stage(store, "sweep-screen", jobs)
    by_cfg = {}
    for s in screens:
        by_cfg.setdefault(s["config_hash"], []).append(s)
    passers = [(v[0]["tier"], v[0]["config"]) for v in by_cfg.values()
               if all(x["passed"] for x in v)]
    print(f"screen: {len(passers)}/{len(by_cfg)} configs passed on all seeds", flush=True)

    jobs = [(t, k, s, CONFIRM_STEPS, CONFIRM_WINDOW) for t, k in passers for s in SEEDS]
    confirms = stage(store, "sweep-confirm", jobs)
    by_cfg = {}
    for s in confirms:
        by_cfg.setdefault(s["config_hash"], []).append(s)

    print("\nconfirm results (mean over seeds; reading, not ranking):", flush=True)
    print(f"{'tier':<5}{'config':<58}{'kp1':>8}{'kz':>8}{'diff':>8}")
    rows = []
    for g in by_cfg.values():
        vals = [x for x in g if x["differential"] is not None]
        if not vals:
            continue
        m1 = sum(x["margin_kp1"] for x in vals) / len(vals)
        mz = sum(x["margin_kz"] for x in vals) / len(vals)
        rows.append((vals[0]["tier"], vals[0]["config"], m1, mz, m1 - mz))
    for tier, knobs, m1, mz, d in sorted(rows, key=lambda r: -r[4]):
        desc = ",".join(f"{k}={v}" for k, v in knobs.items())
        print(f"{tier:<5}{desc:<58}{m1:>+8.3f}{mz:>+8.3f}{d:>+8.3f}")
    bad = [(k, d) for t, k, _m, _z, d in rows
           if t == "b" and k.get("input_fanout") == 0 and abs(d) > 0.05]
    print(f"\nfanout-0 known answer: {'VIOLATED: ' + str(bad) if bad else 'clean'} "
          f"(closed world must show ~zero differential)", flush=True)


if __name__ == "__main__":
    main()
