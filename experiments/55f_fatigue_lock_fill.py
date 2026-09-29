"""Fill in the five measures experiment 55's first launch lost (2026-09-29).

RW 2026-09-29 about 02:10 EDT: "ok rerun the exosotng ones iwth the missing bumbers im
going to sleep so 5 vs 9 hrs doebst rrally matter".

55's first launch (all on example 16's network, 3,462 cells before its OOM kill) kept
only the log line per cell; the end-of-run JSON that held the rest was never written.
Missing for those cells: s_level (what columns send on average, the constant level 52
found sets the look of the lock), c10 and fac10 (lag 10), and run 39's ac1_39 and
tau_39. This reruns every cell of that launch that finished ok (the diverged ones have
no measures to fill), with 55r's guarded participation ratio, through pool_runner.

It is also a determinism check: each rerun cell's log line, formatted by 55's own
line(), is compared with the logged one field for field (the time field excepted);
matches, mismatches and every mismatching pair go to outputs/55f_fatigue_lock_fill_check.txt.
Output: outputs/55f_fatigue_lock_fill.jsonl and .json, outputs/55f_fatigue_lock_fill.log. 55's log and
55r's files are not modified.
Usage:
    venv/bin/python experiments/55f_fatigue_lock_fill.py          # all cells
    venv/bin/python experiments/55f_fatigue_lock_fill.py smoke    # two cells, compared
"""

import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

m55r = importlib.import_module("55r_fatigue_lock_resume")
m55 = m55r.m55
OUT = os.path.join(HERE, "outputs", "55f_fatigue_lock_fill.json")
LOG = os.path.join(HERE, "outputs", "55f_fatigue_lock_fill.log")
CHECK = os.path.join(HERE, "outputs", "55f_fatigue_lock_fill_check.txt")


def logged():
    """The ok log lines of cells that have no full record in 55r's results."""
    have = set()
    rp = os.path.join(HERE, "outputs", "55r_fatigue_lock_resume.jsonl")
    if os.path.exists(rp):
        for s in open(rp):
            have.add(json.loads(s)["tag"])
    lines = {}
    for s in open(m55.LOG):
        tag = s.split(":", 1)[0]
        if " static " in s and tag not in have:
            lines[tag] = s.rstrip("\n")
    return lines


def body(line):
    return line.rsplit(" | ", 1)[0]


def main():
    lines = logged()
    js = [j for j in m55.jobs() if m55r.key(j) in lines]
    print(f"{len(lines)} logged ok cells without a full record; {len(js)} jobs", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for j in js[:1] + [j for j in js if j[4] != "none"][:1]:
            r = m55r.run(j)
            a, b = body(m55.line(r)), body(lines[m55r.key(j)])
            print("rerun ", a, "\nlogged", b, "\nsame" if a == b else "\nDIFFERENT",
                  {k: r.get(k) for k in ("s_level", "c10", "fac10", "ac1_39", "tau_39")},
                  flush=True)
        return
    import pool_runner
    jl = OUT.replace(".json", ".jsonl")
    pool_runner.run_jobs(js, m55r.run, jl, LOG, m55.line, 16, 40, OUT)
    same, diff = 0, []
    for s in open(jl):
        r = json.loads(s)
        if r.get("status") != "ok":
            diff.append((r["tag"], r.get("status"), lines.get(r["tag"], "")))
            continue
        a, b = body(m55.line(r)), body(lines[r["tag"]])
        if a == b:
            same += 1
        else:
            diff.append((r["tag"], a, b))
    with open(CHECK, "w") as fh:
        fh.write(f"{same} cells reproduce their logged line exactly, {len(diff)} differ\n\n")
        for t, a, b in diff:
            fh.write(f"{t}\n  rerun  {a}\n  logged {b}\n")
    print(f"check: {same} same, {len(diff)} different ({CHECK})", flush=True)


if __name__ == "__main__":
    main()
