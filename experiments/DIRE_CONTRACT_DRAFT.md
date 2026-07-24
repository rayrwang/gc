# DIR.E CONTRACT + COMMISSIONING EXAM | DRAFT

**Type:** interface contract for the expectation module (`Dir.E`) and the
known-answer exam that commissions any implementation of it. Plan Step 1b:
written before any implementation exists; the exam fails today by construction.
**Status:** DRAFT until RW line-by-line review. On adoption the exam becomes
the Step-3c commissioning suite (`06_commissioning_dire.py` or similar); the
registered constants in §6 go to REGISTRY.md at freeze time per charter §3.
**Decision provenance:** every design choice below was decided by RW in the
2026-07-19 walkthrough (plan of record, Step-1b block). Nothing here is new.

---

## 1. What Dir.E is contracted to do

Dir.E is the expectation mechanism: it publishes one-step-ahead guesses about
the substrate's next activity and is scored on them. The contract binds the
interface and its promises only. Both realization families must fit behind it
unchanged: foundation-inductive (expectations carried by the learning
foundation itself) and structure-selective (expectations carried by selected
structure on top). The exam never inspects the inside.

Four promises, each carried by a named fixture in §4:

1. **finds-structure**: where learnable structure exists, guesses beat both
   nulls by a registered margin (KP fixtures).
2. **stays-honest**: where no structure exists, guesses claim nothing: score
   stays inside the frozen-control band (KZ fixture), and exposure to
   structureless input does not silently damage the mechanism (SW fixture).
3. **responds**: expectation error drives responsive internal activity change.
   This is the loose-sense generation floor folded in per the F5 ruling: the
   generator is implicit in the network dynamics, so this promise is tested
   here and not on a separate component.
4. **has-a-measurable-timescale**: settling and tracking timescales exist and
   are well-defined (TS fixture). Characterization only: this promise is never
   a latency virtue. A long-open error with responsive activity is inquiry,
   not failure (consistent with the starvation ruling: starvation is a
   persistent error being ignored, never slowness). The single exception: on a
   known-positive fixture, where the pattern is reachable by construction,
   indefinite non-settling is a commissioning failure.

Failure branch (Step-3c gate): an exam failure repairs the expectation
mechanism. No generation or acceptance logic may be added to compensate.

## 2. Interface

Discrete steps. At step t:

1. Dir.E receives history through t-1 and publishes guess `g_t` before
   anything from step t is visible to it.
2. The substrate produces actual activity `a_t`.
3. The harness computes score `s_t = score(g_t, a_t)` and records it.

Guesses live in the same space as the activity they predict, per column:
`g_t` and `a_t` are indexed by a declared readout set of columns fixed in the
fixture manifest. Publication order is enforced by the harness (guess is
content-hashed and logged before `a_t` is computed), so peeking is structurally
impossible rather than merely forbidden.

**Score:** cosine between `g_t` and `a_t`, with an activity floor: steps where
`||a_t||` is below the floor are recorded but excluded from the aggregate (a
cosine against near-silence is noise, and an all-zero guesser must not score).
The floor is measured from controls (§6). The magnitude term is the one open
decision in RW's predicate v2 and stays open here: the exam runs cosine-only,
and closing the magnitude term is a prospective amendment at registration, not
a mid-exam choice.

**Aggregate:** the pass/fail reading is network-wide: per-step mean of
per-column scores over the declared readout set, then a trailing-window mean
(window size in §6). All per-column traces are recorded to D3 regardless (see
D3_RECORDER_SPEC_DRAFT.md §3): verdicts read the aggregate, characterization
reads the columns.

**Exports:** a commissioned run exports its measured timescales (§4, TS). The
Step-4 burn-in prefix is `k x T_switch` with `k = 3` (structural convention,
registered now). This is the only permitted source for that prefix: it closes
the circularity of deriving a burn-in from an unbuilt module.

## 3. Nulls and controls

Run first, across the registered seeds, before any implementation exists;
their spreads freeze the bands in §6.

- **N-frozen**: the same-init, never-learning frozen twin of the
  implementation under test, one per family, publishing whatever its
  untrained dynamics emit. Kills fake structure-finding (the
  random-projection lesson: untrained machinery scores embarrassingly well
  until you check). (Amended in free play 2026-07-20: originally a generic
  frozen-random guesser, a stand-in from before any implementation existed;
  that arm survives as a non-gating chance anchor. The finds-structure
  margin reads: beat N-copy and beat your own frozen twin, both by the
  registered margin.)
- **N-copy**: copy-last-step predictor: `g_t = a_{t-1}`. Kills persistence
  masquerading as prediction. The KP1 fixture is built so this null scores at
  chance by construction (no consecutive repeats).
- **Frozen control** (for KZ and vitals bands): same-init frozen substrate run
  on the same stream; its score spread across seeds defines the two-sided KZ
  band, and its vitals spread defines the characterization bands.

Optional characterization arms, never pass/fail: a Markov chain with a known
entropy floor (calibrates how far short of the ceiling the mechanism lands)
and a shuffled cycle (same symbols as KP1, random order: separates
symbol-familiarity from sequence structure).

## 4. Fixture ladder

**KP0, constant (rung 0, sanity).** One fixed pattern every step. Checks the
mechanics: guesses published in order, scored, error falls. Pass: beat
N-frozen by the registered margin. N-copy is exempt on this rung: on constant
input copy-last is perfect and coincides with the target, so the persistence
null is uninformative here by construction.

**KP1, deterministic cycle (rung 1, the rung that decides).** A
cycle of period 4 over symbols drawn near-orthogonal from a registered seed,
with no consecutive repeats. Pass: after settling, the trailing-window
aggregate beats both N-frozen and N-copy by the registered margin, on every
registered seed.

**KZ, known-zero (iid noise).** A fresh random pattern each step from a
registered generator: nothing is predictable beyond activity statistics.
Reading, in error terms against the frozen-control band:

- error below the band (predicting iid noise better than the frozen control):
  leakage incident. There is no legitimate way to beat chance on iid noise;
  the harness or fixture is leaking the future. Charter §13 incident, fix before
  proceeding.
- error above the band: destabilization finding (noise input degraded the
  mechanism's output beyond control spread). A finding, not a pass.
- error in-band: output-honesty pass. In-band does not certify harmlessness:
  the mechanism could be burning plasticity fitting noise while its error on
  noise looks normal. That harm claim is carried by SW, not KZ.

**SW, sandwich (cycle, noise, cycle).** Run KP1's cycle to settlement, switch
to KZ noise for a registered duration, return to the same cycle. Two
registered readings:

- **retention**: cycle error immediately after the noise block vs immediately
  before it (bounded by a registered factor).
- **re-acquisition**: time to re-settle on the cycle vs the original settling
  time. Slower by more than the registered factor = plasticity burnout, an
  A1-shaped finding (the relearning-savings construct pointed at self-damage).

These two adjudicate the noise-harm claim. Vitals (activity, entropy,
weight-movement) ride alongside as characterization and never adjudicate
(charter §10).

**TS, timescale sweep.** Three settling times: `T_const` (KP0 from scratch),
`T_cycle` (KP1 from scratch), `T_switch` (settled cycle to new cycle). Then
tracking bandwidth: a registered grid of switch rates (new cycle every m
steps); the tracked reading is the fastest rate at which the mechanism still
beats both nulls between switches. "Settled" is a trailing-window criterion:
window mean within a registered fraction of the plateau and staying there for
a registered dwell. Pass is stability, not speed: each T is well-defined,
meaning its spread across seeds and across neighboring grid rates stays within
a registered factor. The magnitudes are characterization. By-product, recorded
only: the per-column settling/lag profile, the first far-from-input look.

## 5. Responds: the activity-response reading

Carried inside KP1 and TS rather than by its own fixture: at each cycle
switch, expectation error opens by construction and is reachable. The reading:
while error is open, the activity-movement statistic (declared in the fixture
manifest, measured against the frozen control's band) departs its frozen band;
error persisting while activity sits inside the frozen band is
starvation-shaped and recorded as such. This is a floor (does activity respond
to error at all), not a speed test; it is the planted-reachable-error test of
the F5 ruling.

## 6. Constants: provenance-tagged, per the Q5 ruling

Every number carries one of three tags. Post-hoc taste is a banned category:
charter §7f extended from metrics to every constant.

| Constant | Tag | Value | Frozen when |
|---|---|---|---|
| activity floor | measured-from-controls | from frozen-control activity spread | after control runs, before implementation |
| beat-null margin (KP0, KP1) | measured-from-controls | from null score spreads across seeds | after null runs, before implementation |
| KZ two-sided band | measured-from-controls | frozen-control error spread across seeds | after control runs, before implementation |
| SW retention factor | measured-from-controls | from frozen-control pre/post spread | after control runs, before implementation |
| SW re-acquisition factor | structural convention | set once at freeze | at registration |
| settled criterion (fraction, dwell) | structural convention | set once at freeze | at registration |
| trailing-window size | structural convention | set once at freeze, one registered power check against control variance | at registration |
| cycle period | structural convention | 4 | now |
| no-consecutive-repeats | structural convention | required | now |
| k (burn-in multiplier) | structural convention | 3 | now |
| switch-rate grid | structural convention | set once at freeze | at registration |
| registered seeds | structural convention | count set once at freeze | at registration |
| burn-in prefix | derived | k x T_switch | after TS runs |

Structural conventions change only by prospective amendment (before the run
they govern), never after seeing results.

**Vacuous-pass rule:** if a band or margin turns out too wide to exclude the
nulls (the exam cannot fail), that is an underpowered fixture: a commissioning
incident. Redesign the fixture; never lower the bar.
**Impossible-pass rule:** if nothing can pass, check the nulls first. Nulls
broken = fixture bug. Nulls fine = the mechanism is what needs repair (§1
failure branch).

## 7. Order of operations

1. Adopt this contract (RW review).
2. Build the exam harness and fixtures against the interface. The exam must
   run and fail: no implementation exists.
3. Run nulls and frozen controls across the registered seeds; freeze the
   measured-from-controls constants; register the constants table.
4. Build the minimal Dir.E implementation (Step 3c) and run the exam.
5. Export `T_switch`; Step 4 derives its burn-in from it.

Gate (Step 1b): this contract and a failing exam exist; nothing implements
them.
