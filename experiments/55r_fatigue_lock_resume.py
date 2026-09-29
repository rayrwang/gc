"""Finish experiment 55 after its OOM kill (2026-09-29).

55 was killed by the kernel at its unit's 60 GB cap at 01:41:43 EDT, 3,462 of 7,068
runs in: 22 workers at 2.5 to 3.5 GB each. The finished runs are the lines of
outputs/55_fatigue_lock.log (its JSON was written only at the end, so none exists); this runs
the rest, and the four runs that failed on the participation ratio's eigenvalue step,
with that step guarded, 16 workers replaced every 40 jobs, and a checkpoint per run
(pool_runner.py). Results: outputs/55r_fatigue_lock_resume.jsonl and .json; log lines appended to
outputs/55_fatigue_lock.log. 55's own file is not modified.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

m55 = importlib.import_module("55_fatigue_lock")
_PART = m55.r39._participation


def _safe_participation(X, d_col):
    try:
        return _PART(X, d_col)
    except Exception:
        return float("nan"), float("nan")


def run(job):
    m55.r39._participation = _safe_participation
    return m55.run(job)


def key(job):
    net, act, norm, keep, kind, tau, g, seed = job
    return f"{net}|{act}|{int(norm)}|{keep}|{kind}|{tau}|{g}|{seed}"


def main():
    done = set()
    for s in open(m55.LOG):
        tag = s.split(":", 1)[0]
        if ": ERROR" in s:
            continue
        done.add(tag)
    js = [j for j in m55.jobs() if key(j) not in done]
    print(f"55: {len(done)} runs in the log, {len(js)} left", flush=True)
    import pool_runner
    pool_runner.run_jobs(js, run, os.path.join(HERE, "outputs", "55r_fatigue_lock_resume.jsonl"), m55.LOG, m55.line,
                   16, 40, os.path.join(HERE, "outputs", "55r_fatigue_lock_resume.json"))


if __name__ == "__main__":
    main()
