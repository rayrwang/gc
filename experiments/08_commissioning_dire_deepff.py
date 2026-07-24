"""
Dir.E deep-feedforward rung: per-column exam machinery commissioned on a
host where every column's answer is known (machinery in dire_exam.py, host =
DeepFFAgt in dire_hosts.py: framework cols, framework conns, the Dir.E delta
that agents.py:1255 leaves TODO).

Era: FREE PLAY. Known answers: on the cycle every column of every learner
should predict near 1; frozen and dire-off arms sit at their baselines; on
noise the input column must be honest while deeper columns may be honestly
predictable (their activity is dominated by its own stationary statistics:
iid at the input is not iid at depth). The per-column persistence baseline
(cosine of a column's activity with its own next activity, derived from the
records) measures exactly that autocorrelation, so the honest-deep-column
reading is prediction-above-persistence, not prediction-above-zero.

BareAgt is deliberately not here: it is expected to behave differently and
gets its own event (09).

Modes:
    08_commissioning_dire_deepff.py             full run
    08_commissioning_dire_deepff.py judge <id>  fresh-process ledger replay
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder
from experiments.dire_hosts import DeepFFHost
from experiments.provenance import stamp

SEEDS = [0, 1, 2]        # free play
DIMS = (64, 64)
N_STEPS = 400
SW_CYCLE, SW_NOISE = 400, 200
TS_CYCLES, TS_STEPS = 6, 200
PROVISIONAL_SETTLE = {"fraction": 0.1, "window": 50, "dwell": 100,
                      "atol": 0.02, "plateau_floor": 0.1}  # unregistered

SUBJECTS = {
    "deep:ff": lambda dim, seed: DeepFFHost(dim, seed, dims=DIMS),
    "deep:dire_off": lambda dim, seed: DeepFFHost(dim, seed, dims=DIMS, dire=False),
    "deep:frozen": lambda dim, seed: DeepFFHost(dim, seed, dims=DIMS, frozen=True),
}

STORE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "d3_store")


def print_report(run_id, manifest, report, integrity):
    cfg = manifest["config"]
    print(f"run {run_id}  era {cfg['era']}  "
          f"({sum(1 for v in integrity.values() if v)}/{len(integrity)} objects verified)")
    print(f"free play, seeds {cfg['seeds']}, dims {cfg['dims']}: known-answer "
          "machinery check, settles nothing\n")
    print("network aggregate (mean over columns), trailing mean per seed:")
    for fixture in ("kp0", "kp1", "kz"):
        for name, per_seed in sorted(report["fixtures"].get(fixture, {}).items()):
            tm = [f"{r['trailing_mean']:+.3f}" if r["trailing_mean"] is not None else "  n/a"
                  for r in per_seed]
            print(f"{fixture:>8} {name:<16} {tm}")
        print()
    print("per-column (mean across seeds): prediction vs persistence baseline")
    print(f"{'fixture':>8} {'subject':<16} {'column':<7} {'predict':>8} {'persist':>8}")
    for (fixture, subject, column), v in sorted(report["columns"].items()):
        if fixture in ("kp1", "kz"):
            p = f"{v['trail']:+.3f}" if v["trail"] is not None else "n/a"
            q = f"{v['persist']:+.3f}" if v["persist"] is not None else "n/a"
            print(f"{fixture:>8} {subject:<16} {column:<7} {p:>8} {q:>8}")
    print("\nsandwich (network aggregate):")
    for name, per_seed in sorted(report["readings"]["sw"].items()):
        r = per_seed[0]
        pre = sum(x["pre_trailing"] for x in per_seed) / len(per_seed)
        post = sum(x["post_immediate"] for x in per_seed) / len(per_seed)
        print(f"  {name:<16} pre {pre:+.3f}  post0 {post:+.3f}  "
              f"settle {r['settle_pre']}/{r['settle_post']} (s0)")


def main():
    store = D3Store(STORE_ROOT)
    if len(sys.argv) > 2 and sys.argv[1] == "judge":
        run_id = sys.argv[2]
        manifest, records = store.load_run(run_id)
        report = de.judge(records, settle=manifest["config"]["provisional_settle"])
        print_report(run_id, manifest, report, store.verify(run_id))
        return

    s = stamp(seed=SEEDS[0], extra_files=[os.path.abspath(__file__)])
    fixtures = de.standard_fixtures(N_STEPS, SW_CYCLE, SW_NOISE, TS_CYCLES, TS_STEPS)
    config = {"era": "free-play", "seeds": SEEDS, "dims": list(DIMS),
              "n_steps": N_STEPS, "sw": [SW_CYCLE, SW_NOISE],
              "ts": [TS_CYCLES, TS_STEPS], "provisional_settle": PROVISIONAL_SETTLE,
              "subjects": sorted(SUBJECTS), "fixtures": [f for f, _ in fixtures]}
    rec = Recorder(store, s, config)
    de.drive_fixtures(SUBJECTS, rec.sink, SEEDS, fixtures)
    run_id, mhash = rec.close()
    print(f"recorded {run_id}  manifest {mhash[:12]}")
    print(f"replay with: 08_commissioning_dire_deepff.py judge {run_id}\n")
    manifest, records = store.load_run(run_id)
    report = de.judge(records, settle=PROVISIONAL_SETTLE)
    print_report(run_id, manifest, report, store.verify(run_id))


if __name__ == "__main__":
    main()
