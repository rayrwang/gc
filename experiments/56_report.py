"""Writes 56_report.md from outputs/56_dieout.jsonl (2026-09-29; split from report_55to57.py,
same tables)."""

import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import capture, jl, write  # noqa: E402

RAW = ["56_dieout.json", "56_dieout.jsonl", "56_dieout.log", "56_dieout.stdout"]


def section56():
    rs = jl("56_dieout.jsonl")
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
                out.append(" -" if not a else f"{min(9, int(10 * statistics.mean(a) + 0.5)):2d}")
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


if __name__ == "__main__":
    write("56", "56_dieout.py", RAW, capture(section56),
          {"56_dieout.json": "dieout56.json", "56_dieout.log": "dieout56.log", "56_dieout.stdout": "dieout56.stdout"})
