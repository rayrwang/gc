"""Writes 55_report.md: experiment 55 and its companions 55r (the resume), 55f (the fill)
and 55b (the norm-off arms with the (1 - keep) factor), from outputs/ (2026-09-29).

Split from report_55to57.py (the one generator for 55 to 57 on 09-29, before the report
scheme); the tables are the same. 55's records come from three places: the first
launch's log lines (it was killed before writing its JSON), 55r's full records for the
rest, and 55f's reruns of the first launch's cells with every measure; full records win
over log lines.
"""

import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import OUT, capture, f, jl, med, write  # noqa: E402


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
    for s in open(os.path.join(OUT, "55_fatigue_lock.log")):
        s = s.rstrip("\n")
        if ": " in s and ": ERROR" not in s:
            r = parse_line(s)
            recs.setdefault(r["tag"], r)
    for name in ("55r_fatigue_lock_resume.jsonl", "55f_fatigue_lock_fill.jsonl"):
        for r in jl(name):
            r["src"] = name
            recs[r["tag"]] = r  # full records win over log lines
    return recs


def key55(tag):
    net, act, norm, keep, kind, tau, g, seed = tag.split("|")[:8]
    return net, act, int(norm), float(keep), kind, int(tau), float(g), int(seed)


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
    for r in jl("55b_fatigue_lock_leak.jsonl"):
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


def section_fill():
    p = os.path.join(OUT, "55f_fatigue_lock_fill_check.txt")
    print("\n== 55f:", len(jl("55f_fatigue_lock_fill.jsonl")), "records;",
          open(p).readline().strip() if os.path.exists(p) else "check not written yet")


RAW = ['55_fatigue_lock.log', '55_fatigue_lock.stdout', '55r_fatigue_lock_resume.json', '55r_fatigue_lock_resume.jsonl', '55r_fatigue_lock_resume.stdout', '55f_fatigue_lock_fill.json', '55f_fatigue_lock_fill.jsonl', '55f_fatigue_lock_fill.log', '55f_fatigue_lock_fill.stdout', '55f_fatigue_lock_fill_check.txt', '55b_fatigue_lock_leak.json', '55b_fatigue_lock_leak.jsonl', '55b_fatigue_lock_leak.log', '55b_fatigue_lock_leak.stdout']
HISTORY = {'55_fatigue_lock.log': 'fatiguelock55.log', '55_fatigue_lock.stdout': 'fatiguelock55.stdout', '55r_fatigue_lock_resume.json': 'fatiguelock55_resume.json', '55r_fatigue_lock_resume.stdout': 'fatiguelock55_resume.stdout', '55f_fatigue_lock_fill.json': 'fatiguelock55_fill.json', '55f_fatigue_lock_fill.log': 'fatiguelock55_fill.log', '55f_fatigue_lock_fill.stdout': 'fatiguelock55_fill.stdout', '55f_fatigue_lock_fill_check.txt': 'fatiguelock55_fill_check.txt', '55b_fatigue_lock_leak.json': 'fatiguelock55_leak.json', '55b_fatigue_lock_leak.log': 'fatiguelock55_leak.log', '55b_fatigue_lock_leak.stdout': 'fatiguelock55_leak.stdout'}


def body():
    recs = load55()
    return (capture(section55, recs) + capture(section55_diverge, recs) + capture(section55_sent, recs)
            + capture(section55b, recs) + capture(section_fill))


if __name__ == "__main__":
    write("55", "55_fatigue_lock.py", RAW, body(), HISTORY,
          ["- Companion scripts: `55r_fatigue_lock_resume.py`, `55f_fatigue_lock_fill.py`, "
           "`55b_fatigue_lock_leak.py`. The first launch (7,068 cells, 01:10 EDT) was killed at 3,462; "
           "55r ran the rest with a changed runner and a guarded participation ratio; 55f reran the first "
           "launch's ok cells and reproduced every logged line exactly."])
