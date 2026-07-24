# R-023 | DRAFT: registration of the Dir.E subject event (07 registered run)

**Type:** pre-registration: Step-3c feedforward-rung commissioning verdicts.
The first run where learners are judged against frozen bars (R-022 numbers).
**Status:** DRAFT until RW review, and blocked on one build item: the judge's
bar-application extension (comparing learner readings to frozen constants)
does not exist yet: it is written after this draft, diffed against this
wording (R-017 discipline), and its hash goes into the final entry.
**Blindness note:** free-play runs (seeds 0/1/2) informed design and bind
nothing; every expectation below is grounded in R-022's frozen numbers only.

## 1. Event

07_commissioning_dire_subjects.py, registered era: seeds 101-811 (R-020's),
frozen constants loaded from dire_frozen_constants.json (sha256 afe662b73c6b),
full roster, standard five fixtures plus the characterization block
(recorded, never gated).

## 2. Roster and per-subject registered expectations

Gate subjects (pass/fail against frozen bars):

- `host:gc_ff` (the commissioning candidate): finds-structure = kp0/kp1
  trailing mean above its family bars (0.4154 / 0.1401) on every registered
  seed; stays-honest = kz trailing mean inside [-0.0545, 0.0539] every seed,
  and SW retention post0 >= 0.5 x pre with re-acquisition <= 2.0 x first
  settle. All three hold = Dir.E-feedforward-inductive commissioned;
  any failure routes to the contract's failure branch (repair the
  expectation mechanism; no generation or acceptance logic to compensate).
- `ref:pc`, `ref:wake_sleep` (known-answer arms): must pass finds-structure
  and kz-honesty against their own family bars. A reference failing =
  exam-broken incident (not a mechanism finding): fixture or band redesign.

Control subjects (drift gates, no pass/fail of their own):

- `frozen:pc`, `frozen:wake_sleep`, `frozen:gc_ff`, `null:copy_last`,
  `anchor:chance`: every reading must reproduce the R-020 source run
  bit-for-bit (same seeds, same hashes, CPU-deterministic). Any deviation is
  a charter section-13 reproducibility incident and blocks all verdicts.

Known-negative learner-shaped arm:

- `host:gc_ff_dire_off`: registered expectation = fails finds-structure on
  kp1 (its backward direction is disabled; prediction must not appear).
  Passing = leakage-shaped incident.

Characterization only (recorded, no expectations): `var:oja`, `var:instar`,
`var:bcm_spike`, capacity sweep (cyc8-32, rnd16-64), revisit cycle,
timescale profiles, per-column traces. Non-binding forecast, recorded for
later checking only (RW-ruled 2026-07-20): free play suggests ref:pc fails
the sandwich readings (noise burnout); if that replicates on the registered
seeds it may graduate, at a future event, to a known-positive for the
damage detector. Nothing in this event turns on it.

## 3. Analysis

Judge = dire_exam.judge plus the bar-application extension at the hash cited
in the final entry; verdicts from D3 ledger_replay only; prequential by
construction. SW readings use the frozen settle conventions and the R-021
per-cell plateau floors. T_switch for the gc host is recorded as
characterization only: the Step-4 burn-in prefix derives exclusively from
the BareAgt rung's own export (RW staging ruling 2026-07-20), and no number
from this event may serve as a burn-in source.

## 4. Consequence

PASS for host:gc_ff = the feedforward-inductive Dir.E realization is
commissioned; the mechanism lane advances to the BareAgt rung (08) and the
measurement lane to the detector layer. FAIL = mechanism repair loop, with
the reference arms and dire-off arm localizing whether the exam or the
mechanism broke. Either way the exam machinery's own commissioning (R-022)
stands.
