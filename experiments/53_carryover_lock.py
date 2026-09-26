"""
Carry-over as the third axis of the lock (RW, 2026-09-26: "ok so the curent
exeirmtn is run on carry over 0 right? isn't that a third axes, the hypothesis
beign that higher carry leads to more "locked in" and vice versa", then, on the
norm, "since the boudned acitvaitons dontne end it right, only teh unboudned
need it to avoid blwoing up", then "ok queue that").

Premises: experiment 52's harness and measures (BareAgtXC, frozen, 1000
steps, the last 300 read; weight scale 2.0), at its two anchor networks:
  ex16    200 bulk columns, dist_exp 2, p_a 0.3, passive MNIST by the stock
          input law (example 16's network)
  run39   49 bulk columns, dist_exp 1, p_a 0.3, kp1 symbols into 3 columns
          (run 39's network)
varied: keep (the share of the accumulator carried into the next step) in
{0, 0.25, 0.5, 0.75, 0.9}; activation with the norm set by kind:
  unbounded, norm on only (they diverge without it): raw, relu, tri1.0, kwta4
  order-only, norm off only (their output depends only on the order or sign
  of a column's values, and the norm divides only the committed copy, not
  the carried accumulator, so it cannot change their dynamics at any keep):
  sign, step0, cstep, wta
  scale-dependent, norm on and off (a fixed threshold or a saturating range,
  so the state's scale sets what they do): tanh, sig4_0, sig8_0, step1 (=
  fc.spike, a step at 1.0); and tri0.3, tri0.7 (output grows as the power of
  the scale)
seeds 50 to 53. 2 networks x 5 keeps x 20 arms x 4 seeds = 800 cells.

Measures: 52's (c1, c10, static, fac1, sent static and level, lock, move,
period, dead), plus
  c50          52's c1 at lag 50
  fac10, fac50 the fluctuation's correlation (each unit's window average
               removed) at lags 10 and 50: carry-over should make the changing
               part persist, decaying about as keep^lag, while a fixed shape
               shows in static at every lag
  frozen       share of units whose side of 0.5 never changes over the window
               (run 39's polarization: 65 to 70% for tri0.3 at keep 0.9)
  nbin         distinct binarised bulk states (x > 0.5) over the 300 steps

Expected (Claude's, written before the run): carry-over raises fac1 toward 1
for every activation (slow drift), while static follows the activation's
constant level as at keep 0; frozen is high only where both hold (the
triangle at high keep); for wta at keep 0.9, c1 high but c50 low; for the
order-only arms nothing changes between norms (checked in 52's side grid).

Result, 2026-09-26 (carrylock53.log, .json; tables in carrylock53_report.txt;
lock and nbin tables printed from the JSON in the session): 800 cells, none
failed. (1) Carry-over makes the changing part persist for every activation:
fac1 about 0 at keep 0, then 0.33 to 0.98 at 0.25, 0.72 to 0.99 at 0.5, 0.95
to 0.995 at 0.75, 0.99 at 0.9; at 0.25 the discontinuous ones persist least
(wta 0.33, step0 and cstep 0.53, sign 0.61) and the smooth ones most (raw,
relu, tanh with the norm 0.93 to 0.98). So c1 goes to 1 for everything (wta
0.05 to 0.995, sign 0.00 to 0.998). (2) Not at lag 50: fac50 stays within
0.1 of zero at every keep for the discontinuous activations and for tri0.3;
only smooth ones at keep 0.9 keep some (raw 0.41 to 0.43, tri0.7 0.22 to 0.34,
tri1.0 0.34 to 0.35, tanh with the norm 0.20 to 0.23). c50 therefore stays
low for the low-level activations (wta 0.15 to 0.20, sign 0.24 to 0.31 at
keep 0.9) and high for the positive-level ones through their fixed shape
(tri0.3 0.97, step0 0.91 to 0.92). (3) Real settling (share of columns moving
under 1% a step) needs a smooth activation with a positive level plus enough
carry or a shallow slope: tri0.3 0 at keep 0 to 0.75, 0.66 to 0.85 at 0.9;
tri0.7, tri1.0 and relu from keep 0.5; sig8 with the norm from 0.75; sig4 with
the norm already at keep 0 on example 16's network. The discontinuous ones
(sign, step0, cstep, step1, wta, kwta4), tanh, and the sigmoids with the norm
off (saturated): 0 to 1% at every keep. (4) The norm moves the scale-dependent
ones as predicted: step1 (fc.spike) with the norm off fires more (static 0.55
to 0.59 against 0.27 at keep 0); the sigmoids and tanh with it off saturate,
look more step-like and never settle; tri0.3 barely moves. (5) static also
rises with keep for the low-level ones (wta 0.05 to 0.22 to 0.26, tanh with
the norm 0.03 to 0.41 to 0.56 at 0.9): partly slow drift counted by a
300-step window average, not a fixed shape. Expected, from the docstring:
held, except that static is not independent of keep.

Usage:
    venv/bin/python experiments/53_carryover_lock.py            # all cells, in parallel
    venv/bin/python experiments/53_carryover_lock.py report     # tables from the JSON
"""

import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch

m52 = importlib.import_module("52_wiring_lock")
KEEPS = (0.0, 0.25, 0.5, 0.75, 0.9)
SEEDS = m52.SEEDS
ARMS = ([(a, True) for a in ("raw", "relu", "tri1.0", "kwta4")]
        + [(a, False) for a in ("sign", "step0", "cstep", "wta")]
        + [(a, nm) for a in ("tanh", "sig4_0", "sig8_0", "step1", "tri0.3", "tri0.7")
           for nm in (True, False)])
LOG = os.path.join(HERE, "carrylock53.log")
OUT = os.path.join(HERE, "carrylock53.json")
_measure52 = m52.measure


def measure(X, d_col, f):
    m = _measure52(X, d_col, f)
    S, n = X.shape[0], X.shape[1] // d_col
    C = X.view(S, n, d_col)
    live = C.abs().amax(dim=(0, 2)) > 1e-9

    def med(v):
        return float(v[live].median()) if bool(live.any()) else float("nan")

    def corr(a, b):
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)

    m["c50"] = float(corr(C[:-50], C[50:]).mean(0).median())
    F = C - C.mean(0)
    den = (F * F).sum((0, 2)).clamp(min=1e-24)
    for lag in (10, 50):
        m[f"fac{lag}"] = med((F[:-lag] * F[lag:]).sum((0, 2)) / den)
    B = X > 0.5
    m["frozen"] = float((B == B[:1]).all(0).float().mean())
    m["nbin"] = len({bytes(row.numpy().tobytes()) for row in B})
    return m


m52.measure = measure  # 52's cell looks the name up at call time


def run(job):
    net, act, norm, keep, seed = job
    n, de_, pa, inp = m52.ANCHORS[net]
    try:
        r = m52.cell(act, n, de_, pa, inp, seed, keep, norm)
    except Exception as e:
        r = {"tag": f"{act}|{n}|{de_}|{pa}|{inp}|{seed}|{keep}|{int(norm)}",
             "status": f"ERROR {type(e).__name__}: {e}"}
    r["net"] = net
    return r


def line(r):
    if r["status"] != "ok":
        return f"{r['net']} {r['tag']}: {r['status']}"
    return (f"{r['net']} {r['tag']}: c1 {r['c1']:.3f} c10 {r['c10']:.3f} c50 {r['c50']:.3f} "
            f"static {r['static']:.3f} fac1 {r['fac1']:.3f} fac10 {r['fac10']:.3f} "
            f"fac50 {r['fac50']:.3f} frozen {r['frozen']:.2f} nbin {r['nbin']} "
            f"sent level {r['s_level']:.3f} lock {r['lock']:.2f} period {r['period']} "
            f"dead {r['dead']:.2f} | {r['secs']}s")


def report():
    rs = json.load(open(OUT))
    out = [f"cells {len(rs)}, not ok {sum(r['status'] != 'ok' for r in rs)}"]
    for r in rs:
        if r["status"] != "ok":
            out.append(f"  {r['net']} {r['tag']}: {r['status']}")
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    for key, title in (("static", "static (fixed-shape share)"),
                       ("fac1", "fac1 (the changing part, lag 1)"),
                       ("fac50", "fac50 (the changing part, lag 50)"),
                       ("c1", "c1 (51's pattern correlation, lag 1)"),
                       ("c50", "c50 (pattern correlation, lag 50)"),
                       ("frozen", "frozen (units never crossing 0.5)")):
        out.append(f"\n{title}, mean over seeds (DIV = all seeds diverged)")
        for net in m52.ANCHORS:
            out.append(f"  {net}: {'act':8}{'norm':>5}" + "".join(f"{k:>8}" for k in KEEPS))
            for act, nm in ARMS:
                row = []
                for keep in KEEPS:
                    cs = [r for r in rs if r.get("net") == net and r["status"] == "ok"
                          and r["tag"].split("|")[0] == act and r["tag"].endswith(f"|{keep}|{int(nm)}")]
                    row.append(f"{mean([r[key] for r in cs]):8.3f}" if cs else f"{'DIV':>8}")
                out.append(f"  {'':{len(net) + 2}}{act:8}{'on' if nm else 'off':>5}" + "".join(row))
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "carrylock53_report.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
        return
    jobs = [(net, a, nm, k, s) for net in m52.ANCHORS for k in KEEPS for a, nm in ARMS
            for s in SEEDS]
    workers = int(os.environ.get("W53_WORKERS", "20"))
    print(f"{len(jobs)} cells on {workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(workers) as ex, open(LOG, "a") as fh:
        futs = [ex.submit(run, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            fh.write(line(r) + "\n")
            fh.flush()
            if k % 100 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh)
    print("done", flush=True)


if __name__ == "__main__":
    main()
