"""05: A1 commissioning — R-013 attempt #1, R-015 attempt #2 (relearning-
savings instrument on known arms).

R-013 FAIL (R-014 verdict): gates clean except (a) — 4/32 seed-medians tied
at exactly 0, the pre-registered quantization mode: ALL crossings live in the
first 50 steps, the 10-step grid had 2-5 gradations. R-015 delta: 1-step grid
through the crossing zone + sustained-crossing rule; nothing else changed
(training trajectories identical by construction — probes are read-only).

NOT retroactive: the R-013 DRAFT's premise (banked comm4 data = 2 exposures)
was FALSE — 04's make_schedule plays each task once ("5x2" = tasks x digits).
A1 needs re-exposures, so this is a NEW-RUNS commissioning on the same arms,
orders, seeds and task streams as R-011/R-012. Design registered as R-013.

Delta from the C1 harness driver (deliberate, harness.py untouched):
  - re-exposure curriculum: each frozen R-007 order played TWICE (uniform
    4-phase lag); exposure 2 streams FRESH samples (slice [1600:3200] of the
    same single-pass stream) so savings measures the task, not the samples
  - A1 side-channels logged on a dense offset grid instead of the probe
    battery (whose fresh refits are the R-006 read-time-repair construct):
      native  arm's own head, task-restricted accuracy (SGD only)
      a1knn   kNN(k=5, raw reps) fit on the reps of the samples streamed SO
              FAR in the current exposure (buffer resets each phase; onset
              with empty buffer reads 0.0 by convention), evaluated on the
              task's fixed probe-test samples
  - rdm_corr/rdm_euclid logged every 400 steps (A1-geometry secondary)
Batched rep equivalence: reps for eval batches are computed as body(X) /
relu(X@W) directly — identical by construction to per-sample step() for
these two arms (pure feedforward, deterministic), registered in R-013.
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.determinism import set_determinism
from experiments.harness import SGDArm, RandomProjArm, Task, rdm_profiles
from experiments.provenance import load_stamped, snapshot_code, stamp, write_stamped
from experiments.tasks import class_group_tasks, probe_sets

SEEDS = list(range(8))
GROUPS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
ORDERS = {"forward": (0, 1, 2, 3, 4), "reverse": (4, 3, 2, 1, 0),
          "perm3": (4, 1, 0, 3, 2), "perm4": (3, 4, 2, 1, 0)}   # frozen in R-007
GATED = ("sgd", "randproj")
RUNS_DIR = "experiments/runs"
PHASE = 1600
# A1 offset grid: dense early (crossings are fast), coarser late. 0 = onset,
# evaluated BEFORE any learning in the phase.
# R-015 grid (attempt #2): R-013's 10-step floor quantized away the signal —
# ALL crossings live in the first 50 steps (t1 in 10..50, t2 in {10,20}).
OFFSETS = (list(range(0, 101, 1)) + list(range(102, 201, 2))
           + list(range(210, 401, 10)) + list(range(450, 1601, 50)))
OFFSET_SET = frozenset(OFFSETS)
PLATEAU_OFFS = [o for o in OFFSETS if o > PHASE * 3 // 4]        # final quarter
RDM_EVERY = 400
KNN_K = 5
THREADS = 8                       # fixed: thread count changes reduction order


ALIAS_PREFIX = "comm_a1b"          # attempt #2 (R-015); R-013 evidence = comm_a1_*
VERDICT_FILE = f"{RUNS_DIR}/commissioning_a1b_verdict.txt"

def alias(arm, oname, seed, rerun=False):
    tag = "_rerun" if rerun else ""
    return f"{RUNS_DIR}/{ALIAS_PREFIX}_{arm}_{oname}_s{seed}{tag}.jsonl"


def make_schedule(oname, seed):
    """Each frozen order twice; exposure e streams slice [(e-1)*PHASE : e*PHASE]
    of the task's single-pass stream (fresh samples on re-exposure)."""
    tasks = class_group_tasks(seed, GROUPS)
    sched = []
    for exp in (1, 2):
        for i in ORDERS[oname]:
            t = tasks[i]
            sched.append((Task(t.name, t.samples[(exp - 1) * PHASE: exp * PHASE]),
                          exp))
    return sched


def factory_for(arm):
    if arm == "sgd":
        return lambda s: SGDArm(784, 10, hidden=(256, 256), lr=1e-2, seed=s)
    return lambda s: RandomProjArm(784, hidden=256, seed=s)


def rep_batch(arm, X):
    """Batched frozen reps; equivalent-by-construction to per-sample step()."""
    with torch.no_grad():
        if isinstance(arm, SGDArm):
            return arm.body(X)
        return torch.relu(X @ arm.W)


def native_acc(arm, X, y):
    """Arm's own head on task samples (registered side-channel; SGD only)."""
    if not isinstance(arm, SGDArm):
        return None
    with torch.no_grad():
        pred = arm.head(arm.body(X)).argmax(1)
    return round(float((pred == y).float().mean()), 6)


def knn_acc(buf_R, buf_y, Rte, yte):
    """kNN(k<=5) on the exposure buffer, raw reps. Empty buffer -> 0.0."""
    if len(buf_y) == 0:
        return 0.0
    R = torch.stack(buf_R)
    yb = torch.tensor(buf_y)
    k = min(KNN_K, len(buf_y))
    idx = torch.cdist(Rte, R).topk(k, largest=False).indices
    pred = torch.mode(yb[idx], dim=1).values
    return round(float((pred == yte).float().mean()), 6)


def run_one(arm_name, oname, seed, out_path):
    torch.set_num_threads(THREADS)
    set_determinism(seed)
    st = stamp(seed, extra_files=[os.path.abspath(__file__)])
    run_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), st["run_id"])
    snapshot_code(run_dir, extra_files=[os.path.abspath(__file__)])
    log_path = os.path.join(run_dir, "log.jsonl")
    arm = factory_for(arm_name)(seed)
    ptr, pte = probe_sets(seed=0, n_train=400, n_test=400)
    Xtr = torch.stack([x for x, _ in ptr])
    Ytr = torch.stack([y for _, y in ptr])
    Xte = torch.stack([x for x, _ in pte])
    yte_all = torch.stack([y for _, y in pte]).argmax(1)
    task_eval = {}                                    # group name -> (X, y)
    for g in GROUPS:
        name = "mnist-g" + "".join(map(str, g))
        m = torch.isin(yte_all, torch.tensor(g))
        task_eval[name] = (Xte[m], yte_all[m])
    step = 0
    with write_stamped(log_path, st) as f:
        for task, exp in make_schedule(oname, seed):
            Xt, yt = task_eval[task.name]
            buf_R, buf_y = [], []                      # resets every exposure
            for off in range(PHASE + 1):
                if off > 0:
                    s = task.samples[off - 1]
                    arm.step(s)
                    buf_R.append(arm.get_representations())
                    buf_y.append(int(s[1].argmax()))
                    step += 1
                if off in OFFSET_SET:
                    row = {"step": step, "task": task.name, "exp": exp, "off": off,
                           "a1": {"native": native_acc(arm, Xt, yt),
                                  "knn": knn_acc(buf_R, buf_y,
                                                 rep_batch(arm, Xt), yt)}}
                    f.write(json.dumps(row) + "\n")
                if off > 0 and off % RDM_EVERY == 0:
                    row = {"step": step, "task": task.name, "exp": exp, "off": off}
                    row.update(rdm_profiles(rep_batch(arm, Xtr), Ytr))
                    f.write(json.dumps(row) + "\n")
            f.flush()
    ap = os.path.abspath(out_path)
    if os.path.islink(ap):
        os.unlink(ap)
    if not os.path.exists(ap):
        os.symlink(log_path, ap)
    return log_path


def run_all():
    for arm, oname, seed in itertools.product(GATED, ORDERS, SEEDS):
        ap = alias(arm, oname, seed)
        if os.path.islink(ap):
            print(f"skip {ap}", flush=True)
            continue
        t0 = time.time()
        run_one(arm, oname, seed, ap)
        print(f"done {ap} in {time.time()-t0:.0f}s", flush=True)
    rp = alias("sgd", "forward", 0, rerun=True)        # gate (c) triple
    if not os.path.islink(rp):
        run_one("sgd", "forward", 0, rp)
        print(f"done {rp} (determinism rerun)", flush=True)


# ------------------------------------------------------------------- analysis
def curves_of(path):
    """{(task, exp): {off: {'native': v|None, 'knn': v}}} + rdm rows."""
    _, rows = load_stamped(path)
    cur, rdm = {}, {}
    for r in rows:
        key = (r["task"], r["exp"])
        if "a1" in r:
            cur.setdefault(key, {})[r["off"]] = r["a1"]
        elif "rdm_corr" in r:
            rdm.setdefault(key, {})[r["off"]] = r["rdm_corr"]
    return cur, rdm


def sustained_cross(c, bar):
    """R-015: first grid offset o with c[o] >= bar AND c[next(o)] >= bar
    (guards single-probe luck at 1-step resolution). The last grid offset
    counts alone (end of phase, pre-stated)."""
    offs = [o for o in OFFSETS if o in c]
    for i, o in enumerate(offs):
        if c[o] >= bar and (i + 1 == len(offs) or c[offs[i + 1]] >= bar):
            return o
    return None


def cell_times(curve1, curve2, channel):
    """(t1, t2, flags) per registered construct; None if cell excluded."""
    c1 = {o: v[channel] for o, v in curve1.items() if v[channel] is not None}
    c2 = {o: v[channel] for o, v in curve2.items() if v[channel] is not None}
    if not c1 or not c2:
        return None
    plateau = sum(c1[o] for o in PLATEAU_OFFS) / len(PLATEAU_OFFS)
    bar = 0.8 * plateau
    if c1[0] >= bar:                                   # transfer artifact
        return ("excluded", None, None)
    t1 = sustained_cross(c1, bar)
    if t1 is None or t1 == 0:
        return ("excluded", None, None)
    t2 = sustained_cross(c2, bar)
    censored = t2 is None
    if censored:
        t2 = PHASE
    return ("ok", (t1, t2), {"censored": censored, "ceiling": t2 == 0,
                             "onset_ret": round(c2[0] / max(plateau, 1e-9), 4)})


def s_index(t1, t2):
    return (t1 - t2) / (t1 + t2) if (t1 + t2) > 0 else 0.0


def savings(t1, t2):
    return 1 - t2 / t1


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def analyze():
    lines = ["R-015 A1 COMMISSIONING ANALYSIS (attempt #2)", "=" * 50]
    cells = {}          # (arm, order, seed, task) -> per-channel results
    rdms = {}
    for arm, oname, seed in itertools.product(GATED, ORDERS, SEEDS):
        ap = alias(arm, oname, seed)
        if not os.path.exists(ap):
            print(f"incomplete: {ap}")
            return None
        cur, rdm = curves_of(ap)
        rdms[(arm, oname, seed)] = rdm
        for g in GROUPS:
            name = "mnist-g" + "".join(map(str, g))
            for ch in ("native", "knn"):
                if arm == "randproj" and ch == "native":
                    continue
                res = cell_times(cur[(name, 1)], cur[(name, 2)], ch)
                cells[(arm, oname, seed, name, ch)] = res

    def seed_medians(arm, ch):
        out = {}
        for oname in ORDERS:
            per_seed = []
            for seed in SEEDS:
                vals = []
                for g in GROUPS:
                    name = "mnist-g" + "".join(map(str, g))
                    res = cells[(arm, oname, seed, name, ch)]
                    if res and res[0] == "ok":
                        vals.append(s_index(*res[1]))
                per_seed.append(median(vals) if vals else None)
            out[oname] = per_seed
        return out

    def counts(arm, ch):
        exc = cen = ceil = ok = 0
        onset = []
        for k, res in cells.items():
            if k[0] != arm or k[4] != ch or res is None:
                continue
            if res[0] == "excluded":
                exc += 1
            else:
                ok += 1
                cen += res[2]["censored"]
                ceil += res[2]["ceiling"]
                onset.append(res[2]["onset_ret"])
        mo = sum(onset) / len(onset) if onset else float("nan")
        return exc, cen, ceil, ok, mo

    # ---- gate (a): SGD native, all 8 seed-medians > 0 in each order
    sm = seed_medians("sgd", "native")
    gate_a = True
    lines.append("\n--- gate (a): SGD / native, s* seed-medians ---")
    for oname, meds in sm.items():
        ok = all(m is not None and m > 0 for m in meds)
        gate_a = gate_a and ok
        lines.append(f"  {oname:8s} " +
                     " ".join("  na" if m is None else f"{m:+.2f}" for m in meds) +
                     f"  -> {'8/8 OK (exact p=1/256)' if ok else 'FAIL'}")

    # ---- gate (b1): randproj (t1,t2) identical across orders per (seed,task)
    b1 = True
    for seed in SEEDS:
        for g in GROUPS:
            name = "mnist-g" + "".join(map(str, g))
            ts = {cells[("randproj", o, seed, name, "knn")][1] for o in ORDERS}
            if len(ts) != 1:
                b1 = False
                lines.append(f"  b1 VIOLATION seed{seed} {name}: {ts}")
    lines.append(f"\n--- gate (b1): randproj (t1,t2) order-invariance: "
                 f"{'OK (exact)' if b1 else 'FAIL'} ---")

    # ---- gate (b2): randproj knn, no seed-consistent sign AND |mean s*|<=0.10
    rm = seed_medians("randproj", "knn")["forward"]    # order-collapsed via b1
    rv = [m for m in rm if m is not None]
    npos = sum(1 for v in rv if v > 0)
    nneg = sum(1 for v in rv if v < 0)
    mean_s = sum(rv) / len(rv) if rv else float("nan")
    b2 = (min(npos, nneg) >= 1 or len(rv) - npos - nneg > 0) and abs(mean_s) <= 0.10
    lines.append(f"--- gate (b2): randproj/knn seed-medians "
                 f"{[f'{v:+.2f}' for v in rv]} pos={npos} neg={nneg} "
                 f"mean={mean_s:+.3f} -> {'OK' if b2 else 'FAIL'} ---")

    # ---- gate (c): determinism rerun triple
    rp = alias("sgd", "forward", 0, rerun=True)
    if not os.path.exists(rp):
        print("incomplete: rerun triple missing")
        return None
    _, rows_a = load_stamped(alias("sgd", "forward", 0))
    _, rows_b = load_stamped(rp)
    gate_c = rows_a == rows_b
    lines.append(f"--- gate (c): rerun triple row-identical: "
                 f"{'OK' if gate_c else 'FAIL'} ({len(rows_a)} rows) ---")

    # ---- recorded, non-gating
    lines.append("\n--- recorded (non-gating) ---")
    for arm, ch in (("sgd", "native"), ("sgd", "knn"), ("randproj", "knn")):
        allv, sav = [], []
        for oname in ORDERS:
            for seed in SEEDS:
                for g in GROUPS:
                    name = "mnist-g" + "".join(map(str, g))
                    res = cells[(arm, oname, seed, name, ch)]
                    if res and res[0] == "ok":
                        allv.append(s_index(*res[1]))
                        sav.append(savings(*res[1]))
        exc, cen, ceil, ok, mo = counts(arm, ch)
        lines.append(f"  {arm}/{ch}: cells ok={ok} excluded={exc} censored={cen} "
                     f"ceiling(t_re=0)={ceil}")
        if allv:
            lines.append(f"    s* mean={sum(allv)/len(allv):+.3f} "
                         f"median={median(allv):+.3f} | classic savings "
                         f"mean={sum(sav)/len(sav):+.3f} median={median(sav):+.3f} | "
                         f"onset-retention mean={mo:.3f}")

    # A1-geometry secondary: displacement at switches; recovery on re-exposure
    def corrdist(a, b):
        ta, tb = torch.tensor(a), torch.tensor(b)
        ta, tb = ta - ta.mean(), tb - tb.mean()
        return float(1 - (ta @ tb) / (ta.norm() * tb.norm() + 1e-9))
    for arm in GATED:
        disp = []
        for oname, seed in itertools.product(ORDERS, SEEDS):
            rdm = rdms[(arm, oname, seed)]
            seq = []                                    # phase order of rdm keys
            for (name, exp), by_off in rdm.items():
                seq.append(((name, exp), by_off))
            for i in range(1, len(seq)):
                prev = seq[i - 1][1].get(PHASE)
                first = seq[i][1].get(RDM_EVERY)
                if prev and first:
                    disp.append(corrdist(prev, first))
        if disp:
            lines.append(f"  {arm}: rdm_corr switch-displacement "
                         f"mean={sum(disp)/len(disp):.4f} (n={len(disp)})")

    verdict = "PASS" if (gate_a and b1 and b2 and gate_c) else "FAIL"
    lines.append(f"\nGATE (a) SGD known-positive:  {'OK' if gate_a else 'FAIL'}")
    lines.append(f"GATE (b1) pipeline invariance: {'OK' if b1 else 'FAIL'}")
    lines.append(f"GATE (b2) known-zero:          {'OK' if b2 else 'FAIL'}")
    lines.append(f"GATE (c) determinism:          {'OK' if gate_c else 'FAIL'}")
    lines.append(f"\nR-015 VERDICT: {verdict}")
    run_ids = sorted({load_stamped(alias(a, o, s))[0]["run_id"]
                      for a in GATED for o in ORDERS for s in SEEDS})
    lines.append(f"\nEvidence: {len(run_ids)} stamped runs, "
                 f"{run_ids[0]} .. {run_ids[-1]} (+1 determinism rerun)")
    return "\n".join(lines)


def selftest():
    """Plumbing only (synthetic blobs; not the registered arms/data)."""
    from experiments.harness import _blobs
    torch.set_num_threads(THREADS)
    set_determinism(0)
    IN, NC = 20, 4
    stream = _blobs([0, 1], 400, IN, 1, NC)
    t1 = Task("blob01", stream[:100])
    t2 = Task("blob01", stream[100:200])
    arm = SGDArm(IN, NC, hidden=(16,), lr=1e-1, seed=0)
    Xt = torch.stack([x for x, _ in stream[200:280]])
    yt = torch.stack([y for _, y in stream[200:280]]).argmax(1)
    curves = {}
    for exp, task in ((1, t1), (2, t2)):
        buf_R, buf_y = [], []
        cur = {}
        for off in range(101):
            if off > 0:
                s = task.samples[off - 1]
                arm.step(s)
                buf_R.append(arm.get_representations())
                buf_y.append(int(s[1].argmax()))
            if off % 10 == 0:
                cur[off] = {"native": native_acc(arm, Xt, yt),
                            "knn": knn_acc(buf_R, buf_y, rep_batch(arm, Xt), yt)}
        curves[exp] = cur
    global OFFSETS, PLATEAU_OFFS, PHASE
    old = OFFSETS, PLATEAU_OFFS, PHASE
    OFFSETS, PLATEAU_OFFS, PHASE = list(range(0, 101, 10)), [80, 90, 100], 100
    try:
        res = cell_times(curves[1], curves[2], "native")
        assert res is not None and res[0] in ("ok", "excluded"), res
        if res[0] == "ok":
            (a, b), fl = res[1], res[2]
            print(f"blob cell: t1={a} t2={b} s*={s_index(a, b):+.2f} flags={fl}")
    finally:
        OFFSETS, PLATEAU_OFFS, PHASE = old
    print("selftest passed: A1 channels + cell_times plumbing (synthetic only).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        sys.exit(0)
    if not args.analyze_only:
        run_all()
    text = analyze()
    if text is None:
        sys.exit(1)
    text2 = analyze()
    assert text == text2, "analysis rerun not identical"
    text += "\nAnalysis rerun: identical (OK)."
    print(text)
    with open(VERDICT_FILE, "w") as f:
        f.write(text + "\n")
