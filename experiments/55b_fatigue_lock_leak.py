"""
Amendment to experiment 55: the same cells with a true leaky carry-over (2026-09-29)

RW, 2026-09-29 about 01:00 EDT: "also maybe write ad amendment to 55 with the carry
correcitons? since 56 already has the corrections?"

Why. The code's carry-over (fc.update = keep * x, then the full new input added)
makes the accumulator the leaky average times 1/(1 - keep): 2x at keep 0.5, 10x at
0.9. With the RMS norm on and activity above the norm's 0.9 floor, that one overall
scale divides out exactly; with the norm off, or when quiet activity drops under the
floor, keep also acts as a gain (gc-project 3.10 item 18). 56's smoke showed the size
of it: raw, weight scale 0.3, keep 0.9, norm off, no input, blows up at step 151 as
coded and dies at step 240 with the factor. 55 ran only the coded form, so its
norm-off cells at keep above 0 carry the gain.

What this runs. 55's cell, unchanged, with the (1 - keep) factor on everything that
enters the accumulator: every recurrent contribution (whatever the fatigue does to
the transport, depression included) and the input (kp1 or MNIST). So the carried
state is a true leaky integrator, keep = e^(-dt/tau).
- Main: 55's eight norm-off arms (raw, relu, tri1.0, kwta4, tanh, sig4_0, step1,
  tri0.3) x keep {0.5, 0.9} x all 31 fatigue settings x both anchor networks x seeds
  {50, 51}: 1,984 runs, to set against 55's coded cells.
- Check: four norm-on arms (tri0.3, sig4_0, relu, wta) x keep {0.5, 0.9} x fatigue
  {none, sub tau 30 g 1, div tau 30 g 2, dep tau 30 U 0.5} x both networks x seed 50:
  64 runs. Expected (Claude's): these match 55's coded cells in their statistics
  (not number for number, since the dynamics are chaotic), because the norm divides
  out the scale; if they differ, the floor or the norm's running average matters
  more than the arithmetic says.
Expected for the main block: the scale-dependent activations change most (raw, relu,
tri1.0, kwta4 blow up less, since the hidden gain was 2x or 10x; tanh and sig4_0
saturate less; step1 fires less), tri0.3 moderately (degree 0.3); fatigue's
comparison with the norm as the bound is then made without the extra gain.

Output: fatiguelock55_leak.log, fatiguelock55_leak.json beside this file; tags end
in "|leak". 55's own file is not modified.
Usage:
    venv/bin/python experiments/55b_fatigue_lock_leak.py          # all runs, in parallel
    venv/bin/python experiments/55b_fatigue_lock_leak.py smoke    # two runs, printed
"""

import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

m55 = importlib.import_module("55_fatigue_lock")
m52 = m55.m52
_INSTALL = m55.install
_STREAM = m52.stream
LOG = os.path.join(HERE, "fatiguelock55_leak.log")
OUT = os.path.join(HERE, "fatiguelock55_leak.json")
KEEPS = (0.5, 0.9)
CHECK_ARMS = (("tri0.3", True), ("sig4_0", True), ("relu", True), ("wta", True))
CHECK_FATIGUE = (("none", 0, 0.0), ("sub", 30, 1.0), ("div", 30, 2.0), ("dep", 30, 0.5))


_PART = m55.r39._participation


def _safe_participation(X, d_col):
    try:
        return _PART(X, d_col)
    except Exception:  # eigh failed on near-blown-up states in 4 of 55's cells
        return float("nan"), float("nan")


def run(job):
    keep = job[3]
    m55.r39._participation = _safe_participation

    def install_leak(act, kind, tau, g):
        _INSTALL(act, kind, tau, g)
        from experiments import dire_hosts as dh
        cur = dh.BareAgtX._transport
        dh.BareAgtX._transport = lambda self, x, w, _c=cur, _k=keep: (1.0 - _k) * _c(self, x, w)

    # 55's cell looks these names up at call time, so the wrappers take effect; they
    # are re-applied for every job, since the pool reuses its workers
    m55.install = install_leak
    m52.stream = lambda inp, seed, _k=keep: [(1.0 - _k) * x for x in _STREAM(inp, seed)]
    r = m55.run(job)
    r["tag"] = r["tag"] + "|leak"
    return r


def jobs():
    main = [(net, a, nm, keep, kind, tau, g, s)
            for net in m52.ANCHORS for a, nm in m55.ARMS if not nm for keep in KEEPS
            for kind, tau, g in m55.FATIGUE for s in m55.SEEDS]
    check = [(net, a, nm, keep, kind, tau, g, 50)
             for net in m52.ANCHORS for a, nm in CHECK_ARMS for keep in KEEPS
             for kind, tau, g in CHECK_FATIGUE]
    return main + check


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for job in (("run39", "raw", False, 0.9, "none", 0, 0.0, 50),
                    ("run39", "tri0.3", True, 0.9, "sub", 30, 1.0, 50)):
            print(m55.line(run(job)), flush=True)
        return
    js = jobs()
    # capped at 16 (2026-09-29 01:28 EDT): 55's 22 workers held 53 to 57 GB of the
    # box's 91 GB with swap full; 16 leaves headroom for the desktop
    workers = min(int(os.environ.get("W55B_WORKERS", "22")), 16)
    import pool_runner
    pool_runner.run_jobs(js, run, OUT.replace(".json", ".jsonl"), LOG, m55.line, workers, 40, OUT)


if __name__ == "__main__":
    main()
