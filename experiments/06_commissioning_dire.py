"""
Dir.E exam machinery commissioning (DIRE_CONTRACT_DRAFT.md; machinery in
dire_exam.py, recorder in d3.py).

This script commissions the instrument stack, not any Dir.E implementation
(that is 07+, per registered event): it drives the controls (gate nulls,
per-family frozen twins, chance anchor) through every fixture, records
through D3, judges from the store only, and reports the exam verdict: FAIL,
no implementation registered. Its registered moment later is the band-freeze
run, whose bands come from these controls on fresh registered seeds.

Modes:
    06_commissioning_dire.py             full run: drive, record, judge
    06_commissioning_dire.py judge <id>  fresh-process ledger_replay: re-judge
                                         a stored run without driving anything
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder
from experiments.dire_hosts import OneLayerDirEHost
from experiments.dire_refs import PCReference, WakeSleepReference
from experiments.provenance import stamp

# run configuration (this era's choices, frozen with this script's hash)
SEEDS = [101, 211, 307, 401, 503, 601, 701, 811]  # registered, R-020; free play never touches these
N_STEPS = 400            # kp0 / kp1 / kz length
SW_CYCLE, SW_NOISE = 400, 200
TS_CYCLES, TS_STEPS = 6, 200
PROVISIONAL_SETTLE = {"fraction": 0.1, "window": 50, "dwell": 100,
                      "atol": 0.02, "plateau_floor": 0.1}  # unregistered; registered proposal: floor from frozen-twin spread

CONSTANTS = {"ACTIVITY_FLOOR": de.ACTIVITY_FLOOR, "BEAT_NULL_MARGIN": de.BEAT_NULL_MARGIN,
             "KZ_BAND": de.KZ_BAND, "SW_RETENTION_FACTOR": de.SW_RETENTION_FACTOR}

# the band-freeze roster: the gate nulls and every family's same-init frozen
# twin (the contract's N-frozen); the gaussian emitter stays as a non-gating
# chance anchor
SUBJECTS = {
    "null:copy_last": lambda dim, seed: de.CopyLastNull(dim),
    "anchor:chance": lambda dim, seed: de.FrozenRandomNull(dim, seed),
    "frozen:pc": lambda dim, seed: PCReference(dim, seed, lr_w=0.0),
    "frozen:wake_sleep": lambda dim, seed: WakeSleepReference(dim, seed, lr=0.0),
    "frozen:gc_ff": lambda dim, seed: OneLayerDirEHost(dim, seed, frozen=True),
}

STORE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "d3_store")


def judge_run(store, run_id):
    """the back half, from the ledger only: works in a process that never drove."""
    manifest, records = store.load_run(run_id)
    report = de.judge(records, settle=manifest["config"]["provisional_settle"],
                      constants=manifest["config"]["constants"])
    integrity = store.verify(run_id)
    return manifest, report, integrity


def print_report(run_id, manifest, report, integrity):
    cfg = manifest["config"]
    print(f"run {run_id}  ({sum(1 for v in integrity.values() if v)}/{len(integrity)} "
          f"objects verified)")
    print(f"settle parameters PROVISIONAL {cfg['provisional_settle']}, "
          f"seeds PROVISIONAL {cfg['seeds']}\n")
    for fixture, rows in report["fixtures"].items():
        for name, per_seed in rows.items():
            tm = [f"{r['trailing_mean']:+.3f}" if r["trailing_mean"] is not None else "n/a"
                  for r in per_seed]
            print(f"{fixture:>4} {name:<20} trailing mean per seed: {tm}")
    print()
    for name, per_seed in report["readings"]["sw"].items():
        for r in per_seed:
            print(f"  sw {name:<20} s{r['seed']}  pre {r['pre_trailing']:+.3f}  "
                  f"post0 {r['post_immediate']:+.3f}  settle pre/post: "
                  f"{r['settle_pre']}/{r['settle_post']}")
    for name, per_seed in report["readings"]["ts"].items():
        for r in per_seed:
            print(f"  ts {name:<20} s{r['seed']}  t_switch_mean: {r['t_switch_mean']}")
    print("\npromise verdicts:")
    for k, v in report["verdicts"].items():
        print(f"  {k}: {v}")


def main():
    store = D3Store(STORE_ROOT)

    if len(sys.argv) > 2 and sys.argv[1] == "judge":
        print_report(sys.argv[2], *judge_run(store, sys.argv[2]))
        return

    s = stamp(seed=SEEDS[0], extra_files=[os.path.abspath(__file__)])
    config = {"seeds": SEEDS, "n_steps": N_STEPS, "sw": [SW_CYCLE, SW_NOISE],
              "ts": [TS_CYCLES, TS_STEPS], "provisional_settle": PROVISIONAL_SETTLE,
              "constants": CONSTANTS, "subjects": sorted(SUBJECTS)}
    rec = Recorder(store, s, config)
    de.drive(SUBJECTS, rec.sink, SEEDS, N_STEPS, SW_CYCLE, SW_NOISE, TS_CYCLES, TS_STEPS)
    run_id, mhash = rec.close()
    print(f"recorded {run_id}  manifest {mhash[:12]}")
    print(f"replay with: 06_commissioning_dire.py judge {run_id}\n")
    print_report(run_id, *judge_run(store, run_id))


if __name__ == "__main__":
    main()
