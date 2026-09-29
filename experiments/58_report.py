"""Writes 58_report.md: experiment 58 (56 rerun batched on the GPU) against 56, cell by
cell, from outputs/ (2026-09-29)."""

import importlib
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reporting import OUT, write  # noqa: E402

m58 = importlib.import_module("58_dieout_gpu")
ORDER = ["alive", "cycle", "fixed", "dead", "blowup"]


def jl(name):
    p = os.path.join(OUT, name)
    return [json.loads(s) for s in open(p)] if os.path.exists(p) else []


def check_section(out):
    p = os.path.join(OUT, "58_dieout_gpu_check.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    vs = list(rows[0]["variants"])
    out.append(f"== check: {len(rows)} cells, each stepped by the stock host and by {len(vs)} batched variants")
    out.append("   stock rerun today vs 56's log, and 56's bookkeeping rewritten for a batch run on the")
    out.append("   stock trajectory vs 56's log: 'same' means every field identical, else the fields that differ")
    for r in rows:
        out.append(f"   {r['tag']:42s} {r['outcome_log']:7s} stock {'same' if not r['stock_now_vs_log'] else r['stock_now_vs_log']}"
                   f"; bookkeeping {'same' if not r['book_on_stock_vs_log'] else r['book_on_stock_vs_log']}")
    out.append("")
    out.append("   per variant: the first step whose state differs from the stock's at all (- = never in 2000),")
    out.append("   the first step it differs by more than 1e-3 of the state's size, the distance at step 2000,")
    out.append("   the outcome (* = not 56's), and how many record fields differ from 56's log")
    for v in vs:
        out.append(f"   -- {v}")
        for r in rows:
            x = r["variants"][v]
            f0, f3 = x["first_step_over"]["0.0"], x["first_step_over"]["0.001"]
            d = x["dist_at"].get("2000")
            out.append(f"      {r['tag']:42s} first {'-' if f0 is None else f0:>5}  >1e-3 {'-' if f3 is None else f3:>5}"
                       f"  at 2000 {d:9.2e}  {x['outcome']:7s}{'*' if x['outcome'] != r['outcome_log'] else ' '}"
                       f"  fields differing {len(x['differs_from_log']):2d}  {x['secs']:7.1f}s")
    out.append("")


def full_section(out, name, log):
    rs = jl(f"58_dieout_gpu_{name}.jsonl")
    if not rs:
        return None
    by = {r["tag"]: r for r in rs}
    out.append(f"== full {name}: {len(rs)} of {len(log)} cells")
    gpu = sum(r["group_secs"] / r["cells_in_group"] for r in rs)
    build = sum(r["build_secs"] / r["cells_in_group"] for r in rs)
    cpu56 = sum(r.get("secs", 0) for r in log.values())
    stamp = os.path.join(OUT, f"58_dieout_gpu_{name}.stamp.json")
    wall = None
    if os.path.exists(stamp):
        ls = [l for l in json.load(open(stamp))["launches"] if l.get("wall_secs")]
        wall = ls[-1]["wall_secs"] if ls else None
    out.append(f"   time: stepping {gpu:.0f} s, building the hosts {build:.0f} s, wall {wall} s "
               f"(the measures of alive and cycle cells on 12 CPU workers, overlapped); 56's cells "
               f"summed {cpu56:.0f} single-thread seconds")
    exact = sum(1 for t, r in by.items() if not m58.same_record(r, log[t]))
    out.append(f"   records identical to 56's in every field: {exact}")
    conf = Counter((log[t]["outcome"], r["outcome"]) for t, r in by.items())
    agree = sum(v for (a, b), v in conf.items() if a == b)
    out.append(f"   outcome agrees with 56: {agree} of {len(by)}")
    out.append("   rows 56's outcome, columns this run's: " + " ".join(f"{o:>7}" for o in ORDER))
    for a in ORDER:
        out.append(f"   {a:>7} " + " ".join(f"{conf.get((a, b), 0):7d}" for b in ORDER))
    for kind, key in (("blowup", "blowup_step"), ("dead", "dead_at"), ("cycle", "period")):
        both = [(log[t], r) for t, r in by.items() if log[t]["outcome"] == kind and r["outcome"] == kind]
        same = sum(1 for a, b in both if a.get(key) == b.get(key))
        out.append(f"   both {kind}: {len(both)}, same {key} in {same}")
    both = [(log[t], r) for t, r in by.items()
            if log[t]["outcome"] in ("alive", "cycle") and r["outcome"] in ("alive", "cycle")]
    for key in ("active_share", "c1", "static", "lock", "pr"):
        d = [abs(a[key] - b[key]) for a, b in both
             if a.get(key) is not None and b.get(key) is not None and a[key] == a[key] and b[key] == b[key]]
        if d:
            out.append(f"   alive or cycle in both ({len(d)}): |{key} difference| median {st.median(d):.2e}, "
                       f"90th percentile {sorted(d)[int(0.9 * len(d))]:.2e}, max {max(d):.2e}")
    per = defaultdict(lambda: [0, 0])
    for t, r in by.items():
        act = t.split("|")[1]
        per[act][0] += log[t]["outcome"] == r["outcome"]
        per[act][1] += 1
    out.append("   outcome agreement by activation: " + ", ".join(f"{a} {v[0]}/{v[1]}" for a, v in per.items()))
    dis = sorted(t for t, r in by.items() if log[t]["outcome"] != r["outcome"])
    if dis:
        out.append(f"   the {len(dis)} cells whose outcome differs (56 -> this run):")
        for t in dis:
            out.append(f"      {t:44s} {log[t]['outcome']:>7} -> {by[t]['outcome']}")
    out.append("")
    return by


def body():
    log = {json.loads(s)["tag"]: json.loads(s) for s in open(os.path.join(OUT, "56_dieout.jsonl"))}
    out = []
    check_section(out)
    f32 = full_section(out, "f32", log)
    f64 = full_section(out, "f64", log)
    full_section(out, "exact", log)
    if f32 and f64:
        both = set(f32) & set(f64)
        same = sum(1 for t in both if f32[t]["outcome"] == f64[t]["outcome"])
        out.append(f"== f32 against f64: outcome agrees in {same} of {len(both)}")
    return "\n".join(out)


if __name__ == "__main__":
    raw = [n for n in ("58_dieout_gpu_check.json", "58_dieout_gpu_f32.jsonl", "58_dieout_gpu_f64.jsonl", "58_dieout_gpu_exact.jsonl",
                       "58_dieout_gpu_check.log", "58_dieout_gpu_f32.log", "58_dieout_gpu_f64.log", "58_dieout_gpu_exact.log")
           if os.path.exists(os.path.join(OUT, n))]
    write("58", "58_dieout_gpu.py", raw, body(),
          notes=["- Compared against `outputs/56_dieout.jsonl` (56's records; 56's own report is 56_report.md)."])
