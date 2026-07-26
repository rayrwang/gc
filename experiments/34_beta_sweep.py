"""
Is there a beta where the recurrent gate is active AND uncollapsed?
(RW, 2026-07-26.) Free play, seeds 50-52, fan3, Dir.E FROZEN throughout.

WHY THIS IS THE FIRST RUN OF THE PER-CLASS ARC. The plan is a per-connection-
class gate: CIFAR-regime on conns outgoing from input cols, something
non-degenerate on the rest. Two facts block writing that arm today.

  1. There is no measured CIFAR-regime beta at d=32. CIFAR's net mass +0.305
     is 96 channels at beta 1.0. Net mass is 2*p_win - 1 and p_win depends on
     d and beta jointly, so "use the CIFAR gate" is not yet a runnable
     setting on a 32-unit column. This sweep produces that number.
  2. The split has never been shown to be necessary. beta was pinned at 0.5
     for the whole recurrent arc (comp25, gate29, arule31). If one global
     beta gives an active uncollapsed substrate, the two-class design is
     unmotivated complexity and dies here, cheap.

So: one axis, matched design, and the curve it produces is exactly the
calibration needed to choose both betas if the split survives.

PRIOR. gate29's docstring records comp25 at two temperatures: 94% majority-
sign / 89% near-one-vs-rest at beta 0.5, easing to 87%/50% at beta 2.0.
So beta does move the collapse. Unknown is whether any beta moves it to the
58% chance floor without killing the activity, and what the net mass is
doing along the way.

ARMS (rule=softhebb throughout, Dir.E frozen, only the gate changes):
  frozen                a_frozen=True. Nothing learns. The dynamics null at
                        these seeds, so maj/|a| have an in-run reference
                        instead of a quoted one.
  b0.5 .. b32           signed argmax gate at beta 0.5/1/2/4/8/16/32.
                        b0.5 reproduces arule31's softhebb row.
  unsigned              signed=False at beta 1.0: plain softmax, every unit
                        gets non-negative credit, net mass exactly +1. The
                        far end of the axis, and already a supported flag.

METRICS. arule31's threshold-free set is reused, not reimplemented:
  maj / minority        max(pos,neg)/d and min(pos,neg)/d. 58% = chance at
                        d=32, 94-95% = collapsed.
  n_patterns            distinct sign patterns / n columns. k-agnostic: it
                        cannot be fooled by changing how many units win,
                        which is what killed the topk4 statistic.
  maj_sd                spread of maj across seeds. Near zero means the state
                        is set by the rule, not the data.
  act_norm              mean activation norm. Near 0 is dead, and a dead
                        network is trivially uncollapsed.
Plus three the gate question needs:
  net_mass              mean gate row sum over every gate call during kp1
                        training. This is the quantity run 33 found to be
                        load-bearing on CIFAR (it sustains |W| against the
                        -w*u decay term in gated-Oja).
  net_in / net_rec      net mass split by SENDER CLASS, measured after
                        training with weights frozen. The gate is a function
                        of the TARGET column's activity, and input-fed
                        targets see raw input while recurrent targets see
                        triangle-transported activity, so the two classes may
                        already sit in different regimes with no code change.
  w_norm                mean |W| over Dir.A conns. Run 33's tell: net mass
                        near zero let decay win and |W| fell 52.6 -> 9.8.

READING, fixed before the run:
  some beta has maj well under 95%, |a| alive, net mass at or above 0
        -> GLOBAL FIX. The two-class design is unmotivated; kill it.
  maj drops only where |a| dies
        -> no active uncollapsed cell at any beta. The split, or a different
           rule, is motivated. Report which beta kills the activity.
  net_in and net_rec differ materially at the same beta
        -> the two regimes already exist in the current network. The
           per-class idea is reading a real distinction rather than imposing
           one, and the sweep hands over both target betas.
  non-monotone in beta  -> report it, do not smooth it.

CAVEAT ON SCOPE, recorded so it is not rediscovered later. "softhebb" names
different algorithms on the two substrates. Verified against the code
2026-07-26, not against the paper (an earlier version of this note claimed
CIFARAgt has an adaptive learning rate; it does not, `ss=self.base_lr` is a
constant 0.03. The adaptive rate is in Moraitis et al., not in this repo):

                    CIFARAgt                     BareAgtX
  gate              signed soft-WTA, beta 1.0,   signed soft-WTA, beta 0.5,
                    96/384/1536 channels         32 units
  rule              lrn_oja_gated_batched        lrn_oja_gated
  step size         ss 0.03, constant            ss 0.01, constant
  soft weight-norm  yes (the rule's own decay)   yes (same rule)
  Triangle          ACTIVATION on the post-      TRANSPORT on the pre-
                    synaptic drive:              synaptic activity:
                    relu(u - mean u)^0.7         relu(x - mean x)^0.3 @ W
  output atv        Triangle: non-negative,      NONE. BareCol.update_
                    sparse by construction       activations is a running-RMS
                                                 scalar divide, agents.py:384
  online BN         yes, before every layer      no
  whitening         no                           no
  topology          feedforward, 3 conv layers   recurrent, 49 columns

The Triangle row is the larger gap, not the BN row. In CIFAR the Triangle is
what makes the layer competitive (units above the mean fire); on the bare
rung it rectifies the SENDER and the receiver's drive is a plain linear sum
with no competition shaping the activity at all. That is why 69.6% of
postsynaptic activity is negative here (the fact that killed instar in
arule31) and why it structurally cannot be in CIFAR.

Nothing in THIS run crosses substrates: every arm is the same substrate with
only the gate changing, so the sweep is internally matched. But two
inferences are NOT licensed by it. First, "SoftHebb works on CIFAR so it
should work here once the gate is fixed" assumes a parity that does not
hold. Second, matching net mass does not transplant the CIFAR regime: net
mass is 2*p_win - 1 and p_win is a function of the activity distribution
feeding the softmax, which is non-negative and sparse there and signed and
dense here. The number this sweep produces is a bare-rung calibration.

Summaries only (kind beta34).
"""

import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sweep = importlib.import_module("10_sweep_bare_x")
frz = importlib.import_module("14_registered_bare_freeze")
rep = importlib.import_module("11_replicate_instar_relu")
probe = importlib.import_module("27_hebb_e_probe")
arule = importlib.import_module("31_a_rule_degeneracy")
from experiments import dire_exam as de
from experiments.d3 import D3Store
from experiments.dire_hosts import BareHost, _bare_path
from src import funcs as fc
from src import iotypes as T
from src.agents import Dir

SMOKE = bool(os.environ.get("BETA_SMOKE"))
SEEDS = (50,) if SMOKE else (50, 51, 52)
STEPS, WINDOW = (240, 60) if SMOKE else (12000, 500)
SPLIT_STEPS = 60 if SMOKE else 400   # frozen-weight pass for the class split
WORKERS = 2 if SMOKE else 12
NS = "comp25"
BASE = dict(p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3,
            transport="triangle", ss=1e-2, lr_e=0.1, rule="softhebb")

# (name, beta, signed, a_frozen). Resolution is concentrated in beta 1-4:
# the smoke bracketed net mass at -0.786 (b0.5), -0.255 (b1), +0.874 (b2), so
# the CIFAR-equivalent regime (+0.305 at 96ch/beta 1.0) lands near beta 1.2-1.4
# on a 32-unit column. Above beta 8 the softmax is a hard argmax (net +1.000 to
# 3dp, no repulsion left), so one saturated arm is enough.
ARMS = (("frozen", 0.5, True, True),
        ("b0.5", 0.5, True, False),
        ("b1", 1.0, True, False),
        ("b1.25", 1.25, True, False),
        ("b1.5", 1.5, True, False),
        ("b1.75", 1.75, True, False),
        ("b2", 2.0, True, False),
        ("b3", 3.0, True, False),
        ("b4", 4.0, True, False),
        ("b8", 8.0, True, False),
        ("unsigned", 1.0, False, False))

_ORIG_GATE = fc.softmax_wta
GATE_SUMS = []


def arm_cfg(beta, signed):
    return dict(BASE, beta=beta, signed=signed)


def cfg_hash(name, beta, signed, a_frozen):
    return rep.arm_hash(f"beta34:{name}",
                        dict(arm_cfg(beta, signed), a_frozen=a_frozen), STEPS)


def install_recorder():
    """count every gate call during training. process-local, gate29's pattern."""
    def g(u, beta=1.0, signed=False):
        out = _ORIG_GATE(u, beta, signed)
        GATE_SUMS.append(float(out.sum()))
        return out
    fc.softmax_wta = g


def class_split(host, cfg, seed):
    """net gate mass by SENDER class, weights frozen.

    The gate is computed from the TARGET column's activity, so this asks
    whether the targets reached by input senders already sit at a different
    p_win than the targets reached by recurrent senders. Uses the unpatched
    gate so the training counter is not polluted.
    """
    agt = host.agt
    was_a, was_e = agt.a_frozen, agt.dire
    agt.a_frozen, agt.dire = True, False
    ins, recs = [], []
    stream = de.kp1_cycle(SPLIT_STEPS,
                          seed=de.derive_seed("split34", NS, seed))
    for a in stream:
        host.observe(a)
        for loc, col in agt.cols.items():
            for (tloc, d) in col.conns:
                if d != Dir.A:
                    continue
                y = agt.cols[tloc].nr_1.actual
                s = float(_ORIG_GATE(y, cfg["beta"], cfg["signed"]).sum())
                (ins if agt.is_i(loc) else recs).append(s)
    agt.a_frozen, agt.dire = was_a, was_e
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return mean(ins), mean(recs), len(ins), len(recs)


def w_norm(agt):
    ws = [float(w.norm()) for c in agt.cols.values()
          for (tloc, d), w in c.conns.items() if d == Dir.A]
    return sum(ws) / len(ws) if ws else float("nan")


def run_one(job):
    seed, name, beta, signed, a_frozen = job
    cfg_d = arm_cfg(beta, signed)
    out = {"seed": seed, "arm": name, "beta": beta, "signed": signed}
    install_recorder()
    try:
        for fixture, fn in (("kp1", de.kp1_cycle), ("kz", de.kz_noise)):
            GATE_SUMS.clear()
            cfg = sweep.XSweepCfg(ispec=[T.I_Vector(de.SYMBOL_DIM)],
                                  n_cols=sweep.N_COLS, **cfg_d)
            host = BareHost(sweep.BareAgtXC(
                cfg, _bare_path(f"beta34/w{os.getpid()}"), seed=seed,
                dire=False, a_frozen=a_frozen))
            agt = host.agt
            r = frz.run_light(host,
                              fn(STEPS, seed=de.derive_seed(fixture, NS, seed)),
                              WINDOW)
            out[fixture] = r["margin"]
            out[f"{fixture}_nan"] = r["nan"]
            if fixture == "kp1":
                out["maj"], out["minority"], out["n_patterns"] = \
                    arule.degeneracy(agt)
                norms = [float(c.nr_1.actual.norm()) for l, c in agt.cols.items()
                         if not agt.is_i(l)]
                out["act_norm"] = sum(norms) / max(len(norms), 1)
                out["w_norm"] = w_norm(agt)
                out["net_mass"] = (sum(GATE_SUMS) / len(GATE_SUMS)
                                   if GATE_SUMS else float("nan"))
                (out["net_in"], out["net_rec"],
                 out["n_in"], out["n_rec"]) = class_split(host, cfg_d, seed)
    finally:
        fc.softmax_wta = _ORIG_GATE
    out["differential"] = (out["kp1"] - out["kz"]
                           if None not in (out.get("kp1"), out.get("kz"))
                           else None)
    out["config_hash"] = cfg_hash(name, beta, signed, a_frozen)
    return out


def main():
    store = D3Store(os.environ.get("SWEEP_STORE") or sweep.STORE_ROOT)
    led = [e for e in store.ledger() if e.get("kind") == "beta34"]
    done = {(e.get("config_hash"), e["seed"]) for e in led}
    todo = [(s, n, b, g, f) for (n, b, g, f) in ARMS for s in SEEDS
            if (cfg_hash(n, b, g, f), s) not in done]
    print(f"beta34: {len(ARMS)*len(SEEDS)} cells, {len(todo)} to run "
          f"(Dir.E FROZEN in all, rule=softhebb in all)", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(run_one, j) for j in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            s = fut.result()
            store.append_ledger({"kind": "beta34", "event_id":
                                 f"beta34:{s['config_hash']}:s{s['seed']}", **s})
            print(f"  {n}/{len(todo)} s{s['seed']} {s['arm']}: "
                  f"maj {s['maj']:.1%} patterns {s['n_patterns']:.2f} "
                  f"|a| {s['act_norm']:.2f} net {s['net_mass']:+.3f} "
                  f"(in {s['net_in']:+.3f} rec {s['net_rec']:+.3f}) "
                  f"|W| {s['w_norm']:.1f}", flush=True)

    led = [e for e in store.ledger() if e.get("kind") == "beta34"]
    print(f"\ngate temperature x collapse, Dir.E frozen, rule=softhebb, "
          f"seeds {list(SEEDS)}, {STEPS} steps")
    print("refs: no-learning twin ~58% maj = chance for d=32; arule31 softhebb "
          "@ beta 0.5 = 95.3% maj / 0.70 patterns; comp25 @ beta 2.0 = 87% maj")
    ns = next((e for e in led if e.get("n_in") is not None), None)
    if ns:
        print(f"class-split sample per seed: {ns['n_in']} input-sender gate "
              f"calls, {ns['n_rec']} recurrent-sender, over {SPLIT_STEPS} "
              f"frozen-weight steps")
    print("frozen arm's net_mass is nan by construction: a_frozen skips the "
          "learning block so the gate is never called during training. Its "
          "net_in/net_rec come from the split pass and give the untrained "
          "baseline (comp25 reference at beta 0.5 is -0.845).\n")
    hdr = (f"{'arm':>9} {'maj':>7} {'maj_sd':>8} {'patterns':>9} {'|a|':>7} "
           f"{'net':>8} {'net_in':>8} {'net_rec':>8} {'|W|':>12} {'diff':>9}  status")
    print(hdr)
    rows = []
    fz = [e for e in led if e.get("arm") == "frozen"]
    w_ref = probe.avg(probe.finite([e.get("w_norm") for e in fz])) if fz else None
    for (name, beta, signed, a_frozen) in ARMS:
        g = sorted([e for e in led if e.get("config_hash")
                    == cfg_hash(name, beta, signed, a_frozen)],
                   key=lambda x: x["seed"])
        if not g:
            continue
        majs = probe.finite([e.get("maj") for e in g])
        f = lambda k: probe.avg(probe.finite([e.get(k) for e in g]))
        row = {k: f(k) for k in ("maj", "minority", "n_patterns", "act_norm",
                                 "net_mass", "net_in", "net_rec", "w_norm",
                                 "differential")}
        row["arm"], row["maj_sd"] = name, arule.sd(majs)
        # run-27 vitals convention: a diverged arm's statistics are a finding
        # about STABILITY, not about the question asked. Flag before reading.
        flags = []
        if row["w_norm"] != row["w_norm"]:
            flags.append("NAN")
        elif w_ref and row["w_norm"] > 100 * w_ref:
            flags.append(f"RUNAWAY-W(x{row['w_norm'] / w_ref:.0f})")
        elif w_ref and row["w_norm"] < 1.5 * w_ref and name != "frozen":
            # uncollapsed because INERT, not because the gate was fixed. The
            # activity floor cannot catch this: |a| is fine, the weights just
            # never moved. Same trap as oja/basic in arule31.
            flags.append(f"INERT-W(x{row['w_norm'] / w_ref:.2f})")
        if any(e.get("kp1_nan") or e.get("kz_nan") for e in g):
            flags.append("NAN-MARGIN")
        row["status"] = " ".join(flags) or "ok"
        rows.append(row)
        print(f"{name:>9} {row['maj']:7.1%} {row['maj_sd']:8.4f} "
              f"{row['n_patterns']:9.2f} {row['act_norm']:7.2f} "
              f"{row['net_mass']:+8.3f} {row['net_in']:+8.3f} "
              f"{row['net_rec']:+8.3f} {row['w_norm']:12.4g} "
              f"{row['differential']:+9.4f}  {row['status']}")

    ref = next((r for r in rows if r["arm"] == "frozen"), None)
    live = [r for r in rows if r["arm"] != "frozen"]
    if SMOKE:
        print(f"\nSMOKE at {STEPS} steps: the reading below is NOT VALID. "
              "Collapse is cumulative (arule31 softhebb @ beta 0.5 reads 73% "
              "maj at 240 steps and 95.3% at 12000). Smoke shows the code "
              "paths execute; it must not be read for direction. This is the "
              "run-33 failure and it is disabled here on purpose.")
        return
    print("\nreading:")
    if ref:
        print(f"  no-learning reference this run: maj {ref['maj']:.1%} "
              f"|a| {ref['act_norm']:.2f} |W| {ref['w_norm']:.1f}")
        floor = ref["act_norm"] * 0.5
        good = [r for r in live if r["maj"] < 0.85 and r["status"] == "ok"
                and r["act_norm"] > floor and r["net_mass"] >= -0.05]
        bad = [r["arm"] for r in live if r["status"] != "ok"]
        if bad:
            print(f"  UNREADABLE (stability, not the gate question): {bad}")
            print("  their maj/net_mass are properties of a diverging network. "
                  "Excluded from the candidate test below, per run 27.")
        if good:
            print("  GLOBAL FIX CANDIDATES (maj < 85%, |a| alive, net >= 0): "
                  + ", ".join(f"{r['arm']} (maj {r['maj']:.1%}, "
                              f"net {r['net_mass']:+.3f})" for r in good))
            print("  -> if one holds up, the per-class design is unmotivated. "
                  "Kill it before building it.")
        else:
            dead = [r["arm"] for r in live if r["act_norm"] <= floor]
            print("  NO global cell is both uncollapsed and alive.")
            print(f"  arms below half the twin's activity: {dead or 'none'}")
            print("  -> the split or a different rule is motivated.")
    gaps = [(r["arm"], r["net_in"] - r["net_rec"]) for r in live]
    gaps = [(a, g) for a, g in gaps if g == g]
    if gaps:
        big = max(gaps, key=lambda t: abs(t[1]))
        print(f"  largest sender-class net-mass gap: {big[0]} at "
              f"{big[1]:+.3f} (net_in - net_rec)")
        print("  -> a material gap means the two regimes already exist here "
              "and the per-class split reads a real distinction.")
    print("\nnet mass is 2*p_win - 1 for the signed gate. Run 33 found it "
          "load-bearing on CIFAR: it sustains |W| against the -w*u decay in "
          "gated-Oja. Read |W| alongside it, never net mass alone.")


if __name__ == "__main__":
    main()
