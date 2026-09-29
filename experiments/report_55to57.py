"""Tables for the report on experiments 55, 55b, 55f (the fill), 56 and 57 (2026-09-29).

Every number the report quotes comes from here. Loads 55 from its three sources (the
first launch's log lines, 55r's and 55f's full records), 55b, 56 and, once present,
57. Usage: venv/bin/python experiments/report_55to57.py [section ...]
sections: 55, 55b, 56, 57, fill (default: all present)
"""

import json
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def jl(name):
    p = os.path.join(HERE, name)
    return [json.loads(s) for s in open(p)] if os.path.exists(p) else []


NUM = r"(-?[\d.]+(?:e-?\d+)?|None|nan)"


def parse_line(s):
    """55's log line back into a record (the first launch has nothing else)."""
    tag, rest = s.split(": ", 1)
    if not rest.startswith("static"):
        return {"tag": tag, "status": rest.split(" at step")[0].strip(), "src": "log"}
    r = {"tag": tag, "status": "ok", "src": "log"}
    for k, v in re.findall(r"(\w+) " + NUM, rest.split(" | ")[0]):
        if k in ("osc",):
            continue
        r[{"sstatic": "s_static"}.get(k, k)] = None if v in ("None", "nan") else float(v)
    m = re.search(r"osc (-?[\d.]+)@(\d+)", rest)
    if m:
        r["osc"], r["osc_lag"] = float(m.group(1)), int(m.group(2))
    return r


def load55():
    recs = {}
    for s in open(os.path.join(HERE, "fatiguelock55.log")):
        s = s.rstrip("\n")
        if ": " in s and ": ERROR" not in s:
            r = parse_line(s)
            recs.setdefault(r["tag"], r)
    for name in ("fatiguelock55_resume.jsonl", "fatiguelock55_fill.jsonl"):
        for r in jl(name):
            r["src"] = name
            recs[r["tag"]] = r  # full records win over log lines
    return recs


def key55(tag):
    net, act, norm, keep, kind, tau, g, seed = tag.split("|")[:8]
    return net, act, int(norm), float(keep), kind, int(tau), float(g), int(seed)


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return st.median(xs) if xs else None


def f(v, spec=".2f"):
    return "   -" if v is None else format(v, spec)


ARMS55 = [("raw", 1), ("relu", 1), ("tri1.0", 1), ("kwta4", 1), ("sign", 1), ("step0", 1),
          ("wta", 1), ("tanh", 1), ("sig4_0", 1), ("step1", 1), ("tri0.3", 1),
          ("raw", 0), ("relu", 0), ("tri1.0", 0), ("kwta4", 0), ("tanh", 0), ("sig4_0", 0),
          ("step1", 0), ("tri0.3", 0)]
FAT_SHOW = [("none", 0, 0.0), ("sub", 30, 1.0), ("sub", 300, 1.0), ("div", 30, 2.0),
            ("div", 300, 2.0), ("dep", 30, 0.5), ("dep", 300, 0.5)]


def section55(recs):
    by = defaultdict(list)
    for tag, r in recs.items():
        by[key55(tag)[:7]].append(r)  # pool the seeds
    print("== 55: cells by source:", Counter(r["src"] for r in recs.values()),
          "status:", Counter(r["status"].split(":")[0] for r in recs.values()))
    for metric, spec in (("static", ".2f"), ("c50", ".2f"), ("lock", ".2f"), ("frozen", ".2f"),
                         ("sens50", ".1f"), ("s_level", ".2f")):
        print(f"\n-- {metric}: median over the 2 seeds; D = both diverged; columns = fatigue "
              + ", ".join(f"{k}{t}" + (f"g{g:g}" if k != 'none' else '') for k, t, g in FAT_SHOW))
        for net in ("ex16", "run39"):
            for keep in (0.0, 0.5, 0.9):
                print(f"   {net} keep {keep}")
                for act, nm in ARMS55:
                    cells = []
                    for kind, tau, g in FAT_SHOW:
                        rs = by.get((net, act, nm, keep, kind, tau, g), [])
                        ok = [r for r in rs if r["status"] == "ok"]
                        if rs and not ok:
                            cells.append("   D")
                        else:
                            cells.append(f(med([r.get(metric) for r in ok]), spec).rjust(5))
                    print(f"     {act:7s} norm {nm}: " + " ".join(cells))


def section55_diverge(recs):
    """Unbounded arms with the norm off: does fatigue keep them bounded?"""
    print("\n== 55: diverged share, unbounded arms, norm off, by fatigue kind x tau (all g, nets, seeds, keeps)")
    cnt = defaultdict(lambda: [0, 0])
    for tag, r in recs.items():
        net, act, nm, keep, kind, tau, g, seed = key55(tag)
        if act in ("raw", "relu", "tri1.0", "kwta4") and nm == 0:
            c = cnt[(act, kind, tau)]
            c[0] += r["status"] == "DIVERGED"
            c[1] += 1
    for act in ("raw", "relu", "tri1.0", "kwta4"):
        row = [f"none {cnt[(act, 'none', 0)][0]}/{cnt[(act, 'none', 0)][1]}"]
        for kind in ("sub", "div", "dep"):
            row.append(kind + " " + " ".join(f"{cnt[(act, kind, t)][0]}/{cnt[(act, kind, t)][1]}"
                                             for t in (3, 10, 30, 100, 300)))
        print(f"   {act:7s} " + " | ".join(row))


def section56():
    rs = jl("dieout56.jsonl")
    if not rs:
        return
    print("\n== 56:", len(rs), "cells", Counter(r["outcome"] for r in rs))
    print("   shape x outcome:", dict(Counter((r.get("shape"), r["outcome"]) for r in rs)))
    L = {"blowup": "B", "dead": "D", "fixed": "F", "cycle": "C", "alive": "A"}
    by = {}
    for r in rs:
        net, act, scale, inp, keep, norm, seed, form = r["tag"].split("|")
        by[(net, act, float(scale), inp, float(keep), norm, int(seed), form)] = r
    SC = [0.1, 0.3, 1.0, 2.0, 4.0, 16.0]
    acts = ["raw", "relu", "rrelu1", "tri0.3", "tri1.0", "tanh", "sig4_0", "sig4_1", "step0",
            "step0.5", "step1", "step2", "cstep", "sign", "wta", "kwta4"]
    KF = [(0.0, "code"), (0.5, "code"), (0.5, "leak"), (0.9, "code"), (0.9, "leak")]

    def cs(net, act, inp, k, norm, form, seeds, field=None):
        out = []
        for s in SC:
            v = [by[(net, act, s, inp, k, norm, sd, form)] for sd in seeds
                 if (net, act, s, inp, k, norm, sd, form) in by]
            if field is None:
                o = [L[x["outcome"]] for x in v]
                out.append(o[0] if len(set(o)) == 1 else o[0].lower())
            else:
                a = [x.get(field) for x in v if x.get(field) is not None]
                out.append(" -" if not a else f"{min(9, int(10 * st.mean(a) + 0.5)):2d}")
        return "".join(out)
    print("   run39: outcome letters (scales 0.1 0.3 1 2 4 16; lowercase = seeds differ); k0 k.5code k.5leak k.9code k.9leak")
    for act in acts:
        for inp in ("none", "kp1"):
            for norm in ("1", "0"):
                print(f"   {act:8s} {inp:4s} norm {norm} | "
                      + " ".join(cs("run39", act, inp, k, norm, fm, (50, 51)) for k, fm in KF)
                      + "   active share x10: "
                      + " ".join(cs("run39", act, inp, k, norm, fm, (50, 51), "active_share") for k, fm in KF))
    print("   ex16, norm on, seed 50: none k0 k.9code k.9leak | mnist the same")
    for act in acts:
        print(f"   {act:8s} " + " ".join(cs("ex16", act, "none", k, "1", fm, (50,))
                                        for k, fm in [(0.0, "code"), (0.9, "code"), (0.9, "leak")])
              + " | " + " ".join(cs("ex16", act, "mnist", k, "1", fm, (50,))
                                 for k, fm in [(0.0, "code"), (0.9, "code"), (0.9, "leak")])
              + "   active x10 mnist k0: " + cs("ex16", act, "mnist", 0.0, "1", "code", (50,), "active_share"))
    pairs = Counter()
    for (net, act, s, inp, k, norm, sd, form), r in by.items():
        if form == "leak":
            o = by.get((net, act, s, inp, k, norm, sd, "code"))
            if o:
                pairs[(act in ("sign", "step0", "cstep", "wta"), r["outcome"] == o["outcome"])] += 1
    print("   code vs leak outcome agreement (order-only?, same?):", dict(pairs))


def section55_sent(recs):
    """What is sent under fatigue (the stored state is what fatigue edits directly)."""
    by = defaultdict(list)
    for tag, r in recs.items():
        by[key55(tag)[:7]].append(r)
    FS = [("none", 0, 0.0), ("sub", 3, 1.0), ("sub", 30, 1.0), ("sub", 300, 1.0), ("div", 30, 2.0),
          ("dep", 30, 0.5), ("dep", 300, 0.5)]
    for metric in ("s_static", "s_level", "osc", "c1", "fac1"):
        print(f"\n== 55 {metric}; columns none, sub3g1, sub30g1, sub300g1, div30g2, dep30, dep300")
        for net in ("run39", "ex16"):
            for keep in (0.0, 0.9):
                print(f"  {net} keep {keep}")
                for act, nm in [("relu", 1), ("tri0.3", 1), ("tri0.3", 0), ("step0", 1), ("sig4_0", 1),
                                ("step1", 1), ("wta", 1), ("tanh", 1)]:
                    cells = []
                    for k in FS:
                        rs = [r for r in by.get((net, act, nm, keep) + k, []) if r["status"] == "ok"]
                        cells.append(f(med([r.get(metric) for r in rs])).rjust(5) if rs else "    D")
                    print(f"    {act:7s} norm {nm}: " + " ".join(cells))


def section55b(recs):
    code, leak = defaultdict(list), defaultdict(list)
    for tag, r in recs.items():
        code[key55(tag)[:7]].append(r)
    for r in jl("fatiguelock55_leak.jsonl"):
        leak[key55(r["tag"].rsplit("|leak", 1)[0])[:7]].append(r)
    print("\n== 55b:", sum(len(v) for v in leak.values()), "records")

    def summ(rs, m):
        ok = [r for r in rs if r["status"] == "ok"]
        if rs and not ok:
            return "    D"
        return f(med([r.get(m) for r in ok])).rjust(5)
    FS = [("none", 0, 0.0), ("sub", 30, 1.0), ("div", 30, 2.0), ("dep", 30, 0.5), ("dep", 300, 0.5)]
    for m in ("static", "s_static", "lock", "sens50", "frozen"):
        print(f"-- {m}: code | leak, norm off, columns none, sub30g1, div30g2, dep30, dep300")
        for net in ("run39", "ex16"):
            for keep in (0.5, 0.9):
                print(f"  {net} keep {keep}")
                for act in ("raw", "relu", "tri1.0", "kwta4", "tanh", "sig4_0", "step1", "tri0.3"):
                    a = " ".join(summ(code.get((net, act, 0, keep) + k, []), m) for k in FS)
                    b = " ".join(summ(leak.get((net, act, 0, keep) + k, []), m) for k in FS)
                    print(f"    {act:7s} off: {a} | {b}")
    dc, dl, nc, nl = Counter(), Counter(), Counter(), Counter()
    for k, rs in code.items():
        net, act, nm, keep, kind, tau, g = k
        if nm == 0 and keep > 0 and act in ("raw", "relu", "tri1.0", "kwta4"):
            for r in rs:
                nc[kind] += 1
                dc[kind] += r["status"] == "DIVERGED"
    for k, rs in leak.items():
        net, act, nm, keep, kind, tau, g = k
        if nm == 0 and act in ("raw", "relu", "tri1.0", "kwta4"):
            for r in rs:
                nl[kind] += 1
                dl[kind] += r["status"] == "DIVERGED"
    print("-- diverged, unbounded arms, norm off, keep 0.5 and 0.9:",
          "; ".join(f"{k} code {dc[k]}/{nc[k]} leak {dl[k]}/{nl[k]}" for k in ("none", "sub", "div", "dep")))
    print("-- norm-on check (seed 50), code/leak: static, lock, s_level, sens50")
    for k, rs in sorted(leak.items()):
        net, act, nm, keep, kind, tau, g = k
        c = [r for r in code.get(k, []) if r["tag"].endswith("|50")]
        if nm != 1 or not c:
            continue
        c, lr = c[0], rs[0]

        def g_(r, m):
            return f(r.get(m)) if r["status"] == "ok" else "D"
        print(f"   {net} {act:6s} {keep} {kind}{tau}: " + " ".join(
            f"{m} {g_(c, m)}/{g_(lr, m)}" for m in ("static", "lock", "s_level", "sens50")))


def section57():
    rs = jl("keepleak57.jsonl")
    if not rs:
        return
    print("\n== 57:", len(rs), "cells", Counter(r["status"].split(":")[0] for r in rs))
    by = defaultdict(list)
    for r in rs:
        net, act, nm, keep, form, seed = r["tag"].split("|")
        by[(net, act, int(nm), float(keep), form)].append(r)
    KF = [(0.0, "code"), (0.25, "code"), (0.25, "leak"), (0.5, "code"), (0.5, "leak"),
          (0.75, "code"), (0.75, "leak"), (0.9, "code"), (0.9, "leak")]
    arms = ([(a, 1) for a in ("tanh", "sig4_0", "sig8_0", "step1", "tri0.3", "tri0.7", "raw", "relu", "tri1.0", "kwta4")]
            + [(a, 0) for a in ("tanh", "sig4_0", "sig8_0", "step1", "tri0.3", "tri0.7", "raw", "relu", "tri1.0", "kwta4",
                                "sign", "step0", "cstep", "wta")])
    for metric, spec in (("static", ".2f"), ("s_static", ".2f"), ("lock", ".2f"), ("rate", "+.2f"),
                         ("rate_fs", "+.2f"), ("s_level", ".2f"), ("sat", ".2f"), ("frozen", ".2f"),
                         ("c50", ".2f"), ("level", ".3g")):
        print(f"\n-- 57 {metric}: median over 4 seeds (D = all diverged); columns k0, then code/leak at 0.25, 0.5, 0.75, 0.9")
        for net in ("run39", "ex16"):
            print(f"  {net}")
            for act, nm in arms:
                cells = []
                for k, fm in KF:
                    rr = by.get((net, act, nm, k, fm), [])
                    ok = [r for r in rr if r["status"] == "ok"]
                    cells.append("    D" if rr and not ok else f(med([r.get(metric) for r in ok]), spec).rjust(5))
                print(f"    {act:7s} norm {nm}: {cells[0]} | " + " | ".join(
                    f"{cells[i]} {cells[i + 1]}" for i in range(1, 9, 2)))


def section_fill():
    p = os.path.join(HERE, "fatiguelock55_fill_check.txt")
    print("\n== 55f:", len(jl("fatiguelock55_fill.jsonl")), "records;",
          open(p).readline().strip() if os.path.exists(p) else "check not written yet")


def main():
    want = set(sys.argv[1:]) or {"55", "55sent", "55b", "56", "57", "fill"}
    recs = load55() if want & {"55", "55sent", "55b"} else None
    if "55" in want:
        section55(recs)
        section55_diverge(recs)
    if "55sent" in want:
        section55_sent(recs)
    if "55b" in want:
        section55b(recs)
    if "fill" in want:
        section_fill()
    if "56" in want:
        section56()
    if "57" in want:
        section57()


if __name__ == "__main__":
    main()
