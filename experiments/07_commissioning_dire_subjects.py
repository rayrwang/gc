"""
Dir.E subject commissioning, Step 3c feedforward rung (machinery in
dire_exam.py, references in dire_refs.py, hosts in dire_hosts.py).

Era: REGISTERED (R-023, once appended): seeds 101-811, frozen constants
from dire_frozen_constants.json (R-022), verdicts against frozen bars with
roles per the registration: gate subjects (gc host, both refs), drift-gated
controls (bit-identical to the R-020 source run), the dire-off
known-negative, and characterization arms (recorded, never gated). The
free-play era of this script ended with R-020's append; seeds 0/1/2 runs
bind nothing.

Roster: the copy-last gate null and chance anchor, three learners (pc,
wake_sleep, gc feedforward host), three same-init frozen twins (one per
learner family: the contract's N-frozen), the dire-off arm, and
a small forward-rule variant block (characterization: the kp1 pass provably
cannot separate forward rules; sandwich retention and the capacity curves
might).

Fixtures: the standard five, plus characterization: orthogonal period sweep
(8/16/32; 4 is kp1 itself), the rank probe on random non-orthogonal symbols
(16/32/48/64: input dim is 32, gc hidden width 64), and the revisit cycle
(a-b-a-c: memoryless predictors cap at 0.854, stateful ones can reach 1).

Modes:
    07_commissioning_dire_subjects.py             full run
    07_commissioning_dire_subjects.py judge <id>  fresh-process ledger replay
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import dire_exam as de
from experiments.d3 import D3Store, Recorder
from experiments.dire_hosts import OneLayerDirEHost
from experiments.dire_refs import PCReference, WakeSleepReference
from experiments.provenance import stamp

SEEDS = [101, 211, 307, 401, 503, 601, 701, 811]  # registered, R-020
N_STEPS = 400
SW_CYCLE, SW_NOISE = 400, 200
TS_CYCLES, TS_STEPS = 6, 200
PROVISIONAL_SETTLE = {"fraction": 0.1, "window": 50, "dwell": 100,
                      "atol": 0.02, "plateau_floor": 0.1}  # unregistered; registered proposal: floor from frozen-twin spread

CONSTANTS = {"ACTIVITY_FLOOR": de.ACTIVITY_FLOOR, "BEAT_NULL_MARGIN": de.BEAT_NULL_MARGIN,
             "KZ_BAND": de.KZ_BAND, "SW_RETENTION_FACTOR": de.SW_RETENTION_FACTOR}

SUBJECTS = {
    # gate null + non-gating chance anchor
    "null:copy_last": lambda dim, seed: de.CopyLastNull(dim),
    "anchor:chance": lambda dim, seed: de.FrozenRandomNull(dim, seed),
    # learners
    "ref:pc": lambda dim, seed: PCReference(dim, seed),
    "ref:wake_sleep": lambda dim, seed: WakeSleepReference(dim, seed),
    "host:gc_ff": lambda dim, seed: OneLayerDirEHost(dim, seed),
    # same-init frozen twins, one per learner family
    "frozen:pc": lambda dim, seed: PCReference(dim, seed, lr_w=0.0),
    "frozen:wake_sleep": lambda dim, seed: WakeSleepReference(dim, seed, lr=0.0),
    "frozen:gc_ff": lambda dim, seed: OneLayerDirEHost(dim, seed, frozen=True),
    # substrate-only arm
    "host:gc_ff_dire_off": lambda dim, seed: OneLayerDirEHost(dim, seed, dire=False),
    # forward-rule variant block (characterization)
    "var:oja": lambda dim, seed: OneLayerDirEHost(dim, seed, rule="oja"),
    "var:instar": lambda dim, seed: OneLayerDirEHost(dim, seed, rule="instar"),
    "var:bcm_spike": lambda dim, seed: OneLayerDirEHost(dim, seed, rule="bcm", act="spike"),
}

CHARACTERIZATION = (
    [(f"cyc{p}", lambda seed, p=p: de.kp1_cycle(N_STEPS, period=p, seed=seed))
     for p in (8, 16, 32)]
    + [(f"rnd{p}", lambda seed, p=p: de.cycle_random(N_STEPS, p, seed=seed))
       for p in (16, 32, 48, 64)]
    + [("revisit", lambda seed: de.revisit_cycle(N_STEPS, seed=seed))]
)

STORE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "d3_store")


def judge_run(store, run_id):
    manifest, records = store.load_run(run_id)
    report = de.judge(records, settle=manifest["config"]["provisional_settle"],
                      constants=manifest["config"]["constants"])
    return manifest, report, store.verify(run_id)


def print_report(run_id, manifest, report, integrity):
    cfg = manifest["config"]
    print(f"run {run_id}  era {cfg['era']}  "
          f"({sum(1 for v in integrity.values() if v)}/{len(integrity)} objects verified)")
    print("free play: motivates design, settles nothing. constants unfrozen, "
          f"seeds provisional {cfg['seeds']}\n")

    std = [f for f in ("kp0", "kp1", "kz") if f in report["fixtures"]]
    for fixture in std:
        for name, per_seed in sorted(report["fixtures"][fixture].items()):
            tm = [f"{r['trailing_mean']:+.3f}" if r["trailing_mean"] is not None else "  n/a"
                  for r in per_seed]
            print(f"{fixture:>8} {name:<22} {tm}")
        print()

    print("capacity curves (mean trailing score across seeds; kp1 = period 4):")
    curves = defaultdict(dict)
    for fixture, rows in report["fixtures"].items():
        if fixture in ("kp0", "kz"):
            continue
        for name, per_seed in rows.items():
            ms = [r["trailing_mean"] for r in per_seed if r["trailing_mean"] is not None]
            curves[name][fixture] = sum(ms) / len(ms) if ms else None
    order = ["kp1", "cyc8", "cyc16", "cyc32", "rnd16", "rnd32", "rnd48", "rnd64", "revisit"]
    print(f"{'subject':<22} " + " ".join(f"{f:>7}" for f in order))
    for name in sorted(curves):
        row = [curves[name].get(f) for f in order]
        print(f"{name:<22} " + " ".join(f"{v:+.3f}" if v is not None else "    n/a" for v in row))

    print("\nsandwich (retention: pre vs post0; re-acquisition: settle pre/post):")
    for name, per_seed in sorted(report["readings"]["sw"].items()):
        r = per_seed[0]
        pre = [x["pre_trailing"] for x in per_seed]
        post = [x["post_immediate"] for x in per_seed]
        print(f"  {name:<22} pre {sum(pre)/len(pre):+.3f}  post0 {sum(post)/len(post):+.3f}  "
              f"settle {r['settle_pre']}/{r['settle_post']} (s0)")
    print("\ntimescales (t_switch_mean per seed):")
    for name, per_seed in sorted(report["readings"]["ts"].items()):
        ts = [str(r["t_switch_mean"]) for r in per_seed]
        print(f"  {name:<22} {ts}")
    print("\npromise verdicts (expected: refusals until the band-freeze entry):")
    for k, v in report["verdicts"].items():
        print(f"  {k}: {v}")


FAMILY = {"host:gc_ff": "gc_ff", "host:gc_ff_dire_off": "gc_ff",
          "ref:pc": "pc", "ref:wake_sleep": "wake_sleep"}
GATE_SUBJECTS = ("host:gc_ff", "ref:pc", "ref:wake_sleep")
CONTROLS = ("frozen:pc", "frozen:wake_sleep", "frozen:gc_ff",
            "null:copy_last", "anchor:chance")
KNOWN_NEGATIVE = "host:gc_ff_dire_off"
STANDARD = ("kp0", "kp1", "kz", "sw", "ts")


def bar_verdicts(store, records, frozen, settle):
    """R-023 section-2 wiring: roles to primitives. returns (verdicts, lines)."""
    import json as _json
    lines, verdicts = [], {}
    # drift gate first: any control deviation blocks everything
    _, baseline = store.load_run(frozen["source_run"])
    drift_ok = True
    for c in CONTROLS:
        ok, where = de.drift_identical(records, baseline, c, STANDARD)
        drift_ok = drift_ok and ok
        lines.append(f"  drift {c:<22} {'identical' if ok else f'DEVIATES at {where}'}")
    verdicts["drift"] = drift_ok
    for name in GATE_SUBJECTS:
        fam = FAMILY[name]
        gates = {}
        for fixture in ("kp0", "kp1"):
            ok, means = de.above_bar(records, name, fixture,
                                     frozen["bars"][f"{fam}/{fixture}"], settle["window"])
            gates[f"finds-structure/{fixture}"] = ok
            lines.append(f"  {name:<22} {fixture} above {frozen['bars'][f'{fam}/{fixture}']:.3f}: "
                         f"{'pass' if ok else 'FAIL'} {[round(m, 3) for m in means.values()]}")
        ok, _ = de.in_band(records, name, frozen["kz_bands"][fam], settle["window"])
        gates["stays-honest/kz"] = ok
        lines.append(f"  {name:<22} kz in {frozen['kz_bands'][fam]}: {'pass' if ok else 'FAIL'}")
        if name == "host:gc_ff":  # sandwich gates bind the candidate only
            settle_fam = {**settle, "plateau_floor": frozen["plateau_floors"]["gc_ff/cycle"]}
            ok, per_seed = de.sw_gates(records, name, 0.5, 2.0, settle_fam)
            gates["stays-honest/sw"] = ok
            worst = min(per_seed.values(), key=lambda d: d["post0"])
            lines.append(f"  {name:<22} sw retention/reacq: {'pass' if ok else 'FAIL'} "
                         f"(worst seed: {worst})")
        verdicts[name] = all(gates.values())
    ok, means = de.above_bar(records, KNOWN_NEGATIVE, "kp1",
                             frozen["bars"]["gc_ff/kp1"], settle["window"])
    verdicts[KNOWN_NEGATIVE] = not ok  # must fail finds-structure
    lines.append(f"  {KNOWN_NEGATIVE:<22} kp1 must stay under bar: "
                 f"{'pass (fails as expected)' if not ok else 'INCIDENT: passed (leakage-shaped)'}")
    commissioned = (verdicts["drift"] and all(verdicts[n] for n in GATE_SUBJECTS)
                    and verdicts[KNOWN_NEGATIVE])
    verdicts["event"] = ("host:gc_ff COMMISSIONED (feedforward-inductive Dir.E)"
                         if commissioned else "NOT COMMISSIONED: see failing gates")
    return verdicts, lines


def load_frozen():
    import json as _json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dire_frozen_constants.json")) as f:
        return _json.load(f)


def main():
    store = D3Store(STORE_ROOT)
    if len(sys.argv) > 2 and sys.argv[1] == "judge":
        run_id = sys.argv[2]
        manifest, report, integrity = judge_run(store, run_id)
        print_report(run_id, manifest, report, integrity)
        _, records = store.load_run(run_id)
        verdicts, lines = bar_verdicts(store, records, load_frozen(),
                                       manifest["config"]["provisional_settle"])
        print("\nregistered gates (R-023):")
        print("\n".join(lines))
        print(f"\n  event: {verdicts['event']}")
        return

    s = stamp(seed=SEEDS[0], extra_files=[os.path.abspath(__file__)])
    fixtures = de.standard_fixtures(N_STEPS, SW_CYCLE, SW_NOISE, TS_CYCLES, TS_STEPS) \
        + list(CHARACTERIZATION)
    config = {"era": "registered-R023", "seeds": SEEDS, "n_steps": N_STEPS,
              "sw": [SW_CYCLE, SW_NOISE], "ts": [TS_CYCLES, TS_STEPS],
              "provisional_settle": PROVISIONAL_SETTLE, "constants": CONSTANTS,
              "subjects": sorted(SUBJECTS),
              "fixtures": [f for f, _ in fixtures]}
    rec = Recorder(store, s, config)
    de.drive_fixtures(SUBJECTS, rec.sink, SEEDS, fixtures)
    run_id, mhash = rec.close()
    print(f"recorded {run_id}  manifest {mhash[:12]}")
    print(f"replay with: 07_commissioning_dire_subjects.py judge {run_id}\n")
    print_report(run_id, *judge_run(store, run_id))
    _, records = store.load_run(run_id)
    verdicts, lines = bar_verdicts(store, records, load_frozen(), PROVISIONAL_SETTLE)
    print("\nregistered gates (R-023):")
    print("\n".join(lines))
    print(f"\n  event: {verdicts['event']}")


if __name__ == "__main__":
    main()
