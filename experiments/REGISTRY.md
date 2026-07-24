# REGISTRY — append-only

Pre-registered commitments and their outcomes. Entries are never edited or
deleted; corrections and changes are new entries linking old ones. Governed by
CHARTER.md (whose version+hash each entry cites).

Entry format:
```
## R-NNN | date | type
type: contrast | reading | kill-condition | verdict | revisit | incident
links: R-XXX (superseded/revisits/resolves), charter vN
body: the commitment or outcome, exact.
```
A `contrast` entry must contain: arms, ONE primary outcome, kill/pass thresholds,
analysis plan, data regime (single-pass | repeated+held-out), tuning-budget report
plan. A `verdict` entry must cite its contrast entry and disclose full lineage if
the contrast was ever revisited.

---

## R-000 | 2026-07-07 | conventions
links: charter v1
This registry begins. Append-only from here. Run identity = registered contrast
(charter §1). Nothing below this line is ever rewritten.

## R-001 | 2026-07-07 | kill-condition (standing)
links: charter v1 §14
**Thesis falsifier (registered sentence):** if C1 shows no divergence beyond
seed-noise between history conditions — i.e., knowledge-stock endpoints are
determined by input structure alone, order-variance indistinguishable from
seed-variance after washout — the history-contingency claim (Layer-1 core) is
killed. This is independent of any motivational stopping.

## R-002 | 2026-07-07 | kill-condition (standing)
links: charter v1 §14, D3/D4 specs (to be registered)
**Capture detector (D4, standing):** contradiction-rate falling while downstream
error persists = the cull has been captured (evasion breeding / suppression —
invoices stop while debt grows). Standing kill condition on any critic build it
is measured against; requires D3's ledger fields (per-test exposures, death
causes, downstream-error linkage) to be computable.

## R-003 | 2026-07-07 | correction (scopes R-001)
links: R-001, charter v3 §1 (lineage), §14

R-001 as written conflates three scopes. Corrected reading:

1. A C1 verdict is indexed to (mechanism-implementation, substrate, task-regime).
   A null kills history-contingency FOR THAT TUPLE only. CIFARAgt says nothing
   about BareAgt; this Dir.E says nothing about other Dir.E designs.

2. A null on a MECHANISM-FREE arm (foundation-alone: BareAgt/CIFARAgt without
   Dir.E, MNISTAgt, the nulls) is NOT a wound — it is the control behaving.
   The contrast foundation-alone(null) vs foundation+house(divergence) is the
   POSITIVE result sought. Only a null on the mechanism-CARRYING arm wounds.

3. What R-001's kill actually applies to: the BEST CURRENT mechanism-carrying
   arm, in its registered regime. That kill is a wound to the programme, not
   yet the thesis (L1 is a general claim; no single implementation-null can
   refute it logically).

4. ANTI-DEFLECTION LEDGER (the price of 1-3): every move to a new mechanism
   implementation after a C1 null is a lineage entry — it must link the killed
   predecessor, state the principled reason the new design escapes the failure
   (registered BEFORE the new arm runs), and any claim citing the thesis must
   disclose the implementation-kill count ("survived by revision, N=3").
   Deflection without a pre-stated principled reason carries the
   `verdict-motivated` flag (§1). The ledger is what keeps scoped falsifiers
   from becoming unfalsifiability.

(Degeneration tripwire considered and deliberately left out — the ledger makes
the record honest; stopping remains sovereign per §14.)

## R-004 | 2026-07-07 | commissioning (instrument test: C1)
links: charter v3 §12.1, R-001 (which presumes this passes), R-003 (scoping)

QUESTION: does C1's primary statistic separate a known-history-DEPENDENT arm
from a known-history-INDEPENDENT arm? This gates all downstream C1 verdicts.
This entry tests the INSTRUMENT; no thesis contact (both arms are off-the-shelf).

ARMS (configs frozen as written; zero verdict-guided tuning — any change = revisit):
  - SGDArm(784, 10, hidden=128, lr=1e-2)   — known history-dependent
    (catastrophic forgetting on split-MNIST is textbook)
  - RandomProjArm(784, 128)                — known history-independent
    (never learns; order-effect is zero BY CONSTRUCTION = false-positive control)
  - MNISTAgt arm: optional third point, NON-GATING, 2 seeds only. Expected near
    RandomProj (BCM rescue, not learning — RW). Reported wherever it lands.

DESIGN: schedules AB = [(mnist-A01234, 4000), (mnist-B56789, 4000), (washout, 2000)]
and BA = task order reversed, same washout tail (washout is neutral by
construction: y=None, verified 2026-07-07 after the zeros-label bug catch).
5 seeds x 2 orders per gated arm. Fixed test-split probe sets (400/400), all
three probes recorded every 1000 steps. Single-pass regime throughout.

PRIMARY OUTCOME (computed PER PROBE, all three — ridge, kNN, logistic; a single
probe could be an artifact, per §9 and the MNIST-linear-collapse trap):
  For probe p: competence profile P_p = 10-dim per-class accuracy at END OF
  TRAINING (pre-washout).
  D_order(p) = mean L2 distance between P_p across orders, seed-matched pairs.
  D_seed(p)  = mean L2 distance between P_p across seeds within an order.
  S_p = D_order(p) / D_seed(p)    ("order effect in units of that probe's own
                                    seed noise" — each probe supplies its own
                                    noise floor, so optimizer/tie-break noise in
                                    kNN/logistic self-normalizes)

PASS iff BOTH:
  (a) sensitivity:     S_p(SGD) >= 3.0 on AT LEAST 2 of 3 probes
  (b) false-positive:  S_p(RandomProj) <= 1.5 on ALL 3 probes
      (expected ~0 for ridge (deterministic reps: exact); <=1.5 for the noisy-
      denominator probes means "order effect indistinguishable from that probe's
      own seed noise" — any probe reporting S substantially above its noise
      floor on an arm with provably zero order-effect is hallucinating, and one
      hallucinating probe disqualifies the battery)
FAIL => charter §12.1: instrument redesign preempts everything.

SECONDARY (exploratory, non-gating): the full S_p(t) curve at every probed
washout depth (t = 0..2000 in probe_every steps) — exercises C1's washout leg;
no single-point post-washout number is cited (washout-length dependence is
handled by reporting the curve). Informs R-005 (arms trial), where the
big-world clause / C1xA1 joint reading gets registered under review.

ALSO AT THIS RUN: measure §6 cost cards (FLOPs/step per arm); log
tuning-evaluation count per arm (expected: 0 and 0). Cross-probe agreement on
known arms is itself recorded — commissioning doubles as the probe-battery's
own consistency check and as control data for R-005.

## R-005 | 2026-07-08 | amendment (supersedes R-004's PASS criteria; no runs yet)
links: R-004; the F-ratio/permutation lesson (session transcript)

Design bump: 8 seeds (0..7) x 2 orders per gated arm (was 5); MNISTAgt stays
2 seeds, non-gating. S_p as defined in R-004, reported as EFFECT SIZE.

Gates become exact PAIRED permutation tests (sign-flip, respecting the
seed-matched design): within each seed, swap or keep its AB/BA labels; all
2^8 = 256 relabelings enumerated; recompute S_p under each = the exact null.

PASS iff BOTH:
 (a) sensitivity:    SGD's observed S_p exceeds ALL 256 relabeled values
                     (exact p ~= 0.004) on >= 2 of 3 probes
 (b) false-positive: RandomProj's D_order(p) = 0 within float tolerance on
                     ALL 3 probes. Its reps are bit-identical across orders BY
                     CONSTRUCTION, so this gate tests the PIPELINE, not the arm:
                     any nonzero order-effect here is a plumbing bug (leakage,
                     nondeterminism, probe contamination) = §13 incident.
R-004's fixed thresholds (3.0 / 1.5) demoted to stated expectations, reported
alongside, gating nothing. Correction note: an earlier draft proposed 126
unrestricted relabelings — wrong for a paired statistic; sign-flip is the
scheme that matches D_order's seed-matched pairs.

## R-006 | 2026-07-08 | verdict (commissioning R-004/R-005: FAIL)
links: R-004, R-005; evidence: 32 stamped runs r_20260708_013028_5fd3771_s0 ..
r_20260708_014411_5fd3771_s7; experiments/runs/commissioning_verdict.txt

GATE (a) sensitivity FAIL: SGD S = 1.12/1.85/0.94 (ridge/knn/logistic),
perm-p = 2/256, 2/256, 115/256 — 0 of 3 probes at the 1/256 bar (needed 2).
GATE (b) false-positive PASS: RandomProj D_order = 0.0000 exactly, 3/3 probes,
pipeline clean (no leakage, bit-identical across orders).
VERDICT: FAIL => charter S12.1, instrument redesign preempts everything.

READING (why the known signal was invisible): the probe RETRAINS a fresh
readout on all classes at every checkpoint, repairing at read-time the damage
history does — split-MNIST forgetting lives mostly in the readout, and 128-d
hidden features for the old task remain decodable (random-proj floor: 0.76
with zero learning). Diagnostic in-data: S(t) = 1.3-1.7 while only one task
seen (instrument CAN see representation differences), collapsing to ~1.0 once
both seen. Also learned: washout is a no-op for supervised arms (cannot learn
from unlabeled noise); the washout leg binds only plastic-unsupervised arms.
Stated expectations (S>=3.0 / <=1.5) were wrong about the construct, not the
plumbing. Redesign to be registered as R-007 before any rerun.

## R-007 | 2026-07-08 | registration (C1 redesign after R-006 FAIL; commissioning attempt #2)
links: R-006 (FAIL + diagnosis), R-004/R-005 (retained where unchanged)

REDESIGN, two orthogonal fixes:
 WRITE-SIDE (curriculum): 5 tasks x 2 digits ({01},{23},{45},{67},{89}),
   1600 steps each (8000 total) + washout 2000. Unseen-gaps up to 4 phases.
   K=4 REGISTERED orders over task indices 0..4:
     forward = (0,1,2,3,4)   reverse = (4,3,2,1,0)
     perm3   = (4,1,0,3,2)   perm4   = (3,4,2,1,0)
   perm3/perm4 drawn once from gen(0,"c1_orders") rejecting collisions with
   forward/reverse, frozen HERE before any run.
 READ-SIDE (two constructs per probe p, both from the same runs):
   - drift-S_p (GATES): probes fitted on frozen reps at phase boundaries
     1600/3200/4800/6400 and STORED (weights + standardization stats persisted
     in the log; kNN stores its standardized fit-time train reps). At GATE step
     8000 each stored stale probe is evaluated on current reps; profile =
     concatenated per-class accuracies (4 fit-points x 10 classes = 40-dim).
     Measures representation movement under a once-valid readout — immune to
     the R-006 read-time repair. Stale evals also logged at every probe point
     after fitting (non-gating drift curves; collect early, analyze late).
   - content-S_p (RECORDED, non-gating): R-004's retrained-probe 10-dim
     profile at step 8000. The R-006 dissociation (drift without content
     change) is itself a reading, not a failure.
 ARMS: unchanged (SGD known-positive; RandomProj known-zero; MNISTAgt optional
   non-gating). 8 seeds x 4 orders x 2 gated arms = 64 runs.
 STATISTIC: S = D_order/D_seed generalized to K=4 (D_order = mean pairwise
   distance among the K order-profiles within seed, averaged over seeds;
   D_seed = mean pairwise distance among seeds within order, averaged over
   orders). NULL: Monte-Carlo permutation — 10,000 within-seed relabelings of
   order assignments, drawn from gen(0,"c1_null"); p = (1+#{null>=obs})/10001.
 GATES:
  (a) sensitivity: SGD drift-S_p at p <= 0.001 (top 10 of 10,000) on >= 2 of 3
      probes.
  (b) false-positive: RandomProj D_order = 0 within float tolerance on BOTH
      constructs, all 3 probes (frozen arm: any nonzero = pipeline bug, S13).
 EXPECTATIONS (non-gating): SGD drift-S >= 3; content-S may remain ~1.
 PRE-REGISTERED ESCALATION if (a) fails again: the known-positive is re-cast,
   not the gate loosened — next attempts, each a fresh registration: deeper
   SGD arm (2-3 hidden layers), then CIFAR-10 inputs. A drift-S failure under
   maximum interference with no read-time repair is evidence about the PAIR,
   not the instrument.

## R-008 | 2026-07-08 | verdict (commissioning #2, R-007: FAIL)
links: R-007; evidence: 64 stamped runs r_20260708_120700_s0..124906_s7;
experiments/runs/commissioning2_verdict.txt

GATE (a) FAIL: SGD drift-S = 0.97/1.00/1.00 (ridge/knn/logistic), MC-p =
0.174/0.0026/0.522 — 0/3 at the 0.001 bar (kNN a whisker above it).
GATE (b) PASS: RandomProj D_order = 0 exactly, both constructs, 6/6 readings.

READING — opposite failure to R-006: SATURATION. Stale probes collapse to
near-chance at the gate (fit@1600 -> 0.118 mean acc at 8000; chance = 0.10):
batch-1 lr=1e-2 SGD reorganizes reps so violently that a linear readout dies
within ~1-2 phases REGARDLESS of order — the order-differential signal is
buried under order-independent churn, and the same churn inflates D_seed.
R-006 construct = too forgiving (repair); R-007 construct = too fragile
(saturation). POST-HOC (labeled, S3): short-lag drift (lag 1200-1800, where
stale acc = 0.28-0.59, unsaturated), raw and mean-centered — still 1/3 probes
(kNN p~0.0005; ridge/logistic p 0.12-0.68). The kNN whisker across both
attempts says a weak real signal exists; the shallow arm's rep-level order
effect is simply comparable to its own seed churn. CONCLUSION per R-007's
pre-registered escalation: the KNOWN-POSITIVE was mis-cast — shallow-MLP
batch-1 MNIST reps are order-insensitive relative to their own noise (a real
finding, twice-evidenced by independent constructs). Escalate to the deeper
arm (R-009), gate unchanged.

## R-009 | 2026-07-08 | registration (commissioning #3, per R-007's escalation)
links: R-008 (verdict + mis-cast diagnosis), R-007 (design retained)

CHANGES from R-007 (all else stands — groups, orders, seeds, statistic, MC
null, gate thresholds, zeros checks):
 1. Known-positive re-cast: SGDArm 784-256-256-10 (two hidden layers, read
    the LAST hidden 256). Rationale (commissioning-grade, S1): rep-level
    forgetting grows with depth (task-specialization lives in later layers —
    Ramasesh et al.); the shallow arm's features were generic (randproj floor
    0.76, content-S ~1) => differential drift ~absent, twice-evidenced.
    lr unchanged (1e-2). Depth chosen over lr-reduction because lr cuts
    signal and noise together and reads as tune-to-pass.
 2. Probe grid aligned: probe_every=800 (stale evals exist at exactly
    fit+1600 = one phase of staleness).
 3. GATING construct: short-lag drift-S — each phase-boundary probe
    (fit at 1600/3200/4800/6400) evaluated at fit+1600, per-class profiles
    MEAN-CENTERED per fit-point (removes common-mode decay), concatenated
    (40-dim). At-gate drift-S, raw (uncentered) short-lag drift-S, and
    content-S recorded, non-gating.
 4. RandomProj widened to 256 (match read dimensionality).
GATES unchanged: (a) SGD gating drift-S at MC-p <= 0.001 on >= 2 of 3 probes;
(b) RandomProj D_order = 0 within float tolerance on ALL recorded constructs
and probes (pipeline check).
EXPECTATIONS (non-gating): deeper arm shows the effect on ridge as well as
kNN (kNN near-passes 3x: p .0078/.0026/.0005 — signal exists, weak).
ESCALATION if (a) fails: CIFAR-10 inputs (per R-007); then the construct
itself goes under review (behavior-vs-representation reading of C1).

## R-010 | 2026-07-08 | verdict (commissioning #3, R-009: FAIL)
links: R-009; evidence: 64 stamped runs r_20260708_130501_s0..140235_s7;
experiments/runs/commissioning3_verdict.txt

GATE (a) FAIL — new signature: PERFECT DEGENERACY. All nine drift constructs:
D_order = D_seed to 4 decimals, S = 1.00, MC-p = 1.0000 (all 10,000
relabelings >= observed; relabeling changes nothing).
GATE (b) PASS: 12/12 exact zeros.

DIAGNOSIS (verdict-blind data inspection, logged): stale probes collapse to
CONSTANT PREDICTION (per-class profile [1.0, 0 x 9] — predicts one class for
every input). Vitals: deep-arm participation ratio ~5 of 256 — batch-1 CE
collapsed the rep to ~5 effective dims that rotate wholesale per phase; a
frozen linear map on a rotated low-rank subspace is mean-shift-dominated =>
constant function => profiles carry no order OR seed structure => all
pairwise distances equal => S degenerate at 1. Content acc = 0.80: THE ARM
LEARNS FINE. Root cause across all three attempts: LINEAR PROBES on batch-1
SGD reps cannot hold a stable ruler across time at any staleness (fresh =
repairs damage; long-stale = chance; short-stale + depth = constant). kNN
(neighborhood geometry, not linear readout) near-passed all three attempts —
the surviving signal is RELATIONAL.

R-009's ladder says CIFAR next, then construct review. The diagnosis is
evidence the failure is construct-vs-dynamics, not dataset difficulty —
decision on whether to run the CIFAR rung or amend the escalation to the
construct review (geometry/RSA-style profiles, behavior-level readings)
DEFERRED TO REVIEW (RW). No further runs until registered.

## R-011 | 2026-07-08 | registration (commissioning #4: RSA construct; AMENDED escalation)
links: R-010 (diagnosis), R-009 (design retained), R-003 (deflection pricing:
reason stated before running)

LADDER AMENDMENT, disclosed: R-009's next rung was CIFAR-10. SKIPPED, reason:
R-010's diagnosis (rep collapsed to PR~5 subspace rotating wholesale; stale
probes = constant functions; content healthy at 0.80) shows the failure is
construct-vs-dynamics, not dataset difficulty — CIFAR with the same construct
buys a predicted fourth failure. The construct review (pre-registered in
R-009 as the post-CIFAR step) is pulled forward instead.

CONSTRUCT (charter v4 declaration): GEOMETRY — decoder-free representational
similarity (RSA). At each probe point, per-class centroids of the fixed
probe-set reps -> 10x10 pairwise centroid RDM -> profile = upper triangle
(45 dims). Two variants: rdm_corr (correlation distance on centroid rows —
scale/offset/rotation robust; GATES) and rdm_euclid (Euclidean; confirmatory
second gate). Rationale: all three prior failures were frame-dependence of
linear probes (repair / saturation / rotation-degeneracy); RDMs are invariant
to exactly those nuisances; kNN (the only distance-channel probe) near-passed
all three attempts (p .0078/.0026/.0005) — the surviving signal is relational.
Prior constructs (content, drift, short-lag) remain recorded, non-gating.

DESIGN: identical to R-009 (5x2 curriculum, K=4 frozen orders, 8 seeds,
deep SGDArm 784-256-256-10 + RandomProj-256, probe_every=800, MC null 10k
within-seed relabelings from gen(0,"c1_null")).

GATES:
 (a) sensitivity: SGD rdm_corr-S at MC-p <= 0.001 AND rdm_euclid-S at
     MC-p <= 0.001 (2 of 2 — replaces the extinct 2-of-3 probe redundancy).
 (b) false-positive: RandomProj D_order = 0 within float tolerance on both
     RDM variants + all recorded constructs (pipeline check).

EXPECTATIONS (non-gating): SGD rdm-S >= 2; recency visible as which class-
pairs are separated. PRE-REGISTERED FALLBACK if (a) fails: the finding
"batch-1 SGD endpoint geometry is order-insensitive" is accepted as real
(three constructs deep), and the construct review resolves to BEHAVIOR:
C1 reads the arm's own per-class outputs where a head exists (SGD's head =
textbook forgetting), probe layer retained for headless arms; that reading
gets a fresh registration.

## R-012 | 2026-07-08 | verdict (commissioning #4, R-011: PASS)
links: R-011; evidence: 64 stamped runs r_20260708_141536_s0..153105_s7;
experiments/runs/commissioning4_verdict.txt

GATE (a) PASS, 2/2: SGD rdm_corr S = 2.30, MC-p = 0.0001 (0 of 10,000
relabelings reached the observed value); rdm_euclid S = 2.30, p = 0.0001.
GATE (b) PASS: RandomProj D_order = 0.0000 exactly on all 14 recorded
constructs. (Cosmetic: verdict text prints "14/12" — the label was written
for 12 checks before the two RDM constructs were added; values correct;
noted here per S13, no rerun.)

C1 IS COMMISSIONED with the geometry construct (R-011 spec): order effect =
2.3x seed noise in class-centroid RDMs, exact-null p = 1e-4, known-zero arm
exactly zero throughout, pipeline never implicated in 4 attempts.

THE DISSOCIATION, now one table from the same 64 runs: geometry S = 2.30
(p 1e-4) while EVERY linear-probe construct sits at S = 0.92-1.00 beside it.
Substantive finding produced by commissioning itself: batch-1 SGD order
signatures live in representational GEOMETRY (which class-pairs end up
near/far), and are INVISIBLE to linear decodability — fresh, stale, and
short-lag alike. Anyone probing continual-learning representations with
linear readouts alone would conclude "no order dependence"; the distance
structure says otherwise at 2.3x noise.

Saga summary (R-004..R-012): 4 attempts, 3 diagnosed construct failures
(read-time repair -> saturation -> rotation degeneracy), gates never
loosened, arm re-cast once per pre-registered escalation, ladder amended
once with stated reason. Commissioning controls banked for the arms trial:
SGD and RandomProj endpoint RDM distributions under 4 orders x 8 seeds.
NEXT: R-013 = arms-trial registration (A1 co-registration, C1xA1 joint
reading, big-world clause) — drafted for RW review before anything runs.

## R-013 | 2026-07-09 | registration (commissioning: A1 instrument, known arms; NEW RUNS)
links: R-012 (arms/orders/streams reused; its "banked controls" claim scoped
below), R-007 (frozen orders), R-005 (gate style), charter v4

DRAFT-ERROR DISCLOSURE (charter honesty; error-ledger shape #11): the R-013
DRAFT reviewed 2026-07-09 claimed the banked comm4 runs hold a "5-task x
2-EXPOSURE curriculum" enabling zero-compute retroactive A1 commissioning.
FALSE: R-011's "5x2" = 5 tasks x 2 DIGITS; 04's make_schedule plays each task
ONCE. No re-exposure exists in any banked run, so no t_re is computable and
the retroactive version is unrunnable. R-012's "controls banked for the arms
trial" covers C1 endpoint RDM distributions ONLY, not A1. Also supersedes
R-012's NEXT note: the arms trial is R-014; THIS entry is A1 commissioning
with new known-arm runs. R-014's draft inherits this correction (its §2/§4
premise sentences cite the banked-data version; to be fixed at its review).

QUESTION: unchanged from draft — does the A1 savings construct detect
relearning acceleration where it must exist (SGD) and read zero where it
cannot (frozen random projection)? Instrument test; zero thesis contact.

DESIGN (code = experiments/05_commissioning_a1.py, content sha256[:12] =
7bf70908ffcf, frozen and hash-cited HERE before any registered run; every run
additionally stamps experiments_hash/extra_hash + snapshot_code):
 - Arms as R-011: SGDArm 784-256-256-10 lr=1e-2 (known-positive) and
   RandomProjArm-256 (known-zero). Seeds 0..7. Task streams, groups and probe
   sets identical to R-011 (class_group_tasks, probe_sets(seed=0, 400/400)).
 - Curriculum: each frozen R-007 order played TWICE consecutively (10 phases
   x 1600 = 16000 steps; uniform 4-phase lag between exposures; no washout —
   A1 does not read washout). RE-EXPOSURE STREAMS FRESH SAMPLES (slice
   [1600:3200] of the same single-pass stream): savings must measure the
   task, not recognition of replayed samples. Single-pass regime preserved.
 - A1 channels on offset grid {0,10,..,400} u {450,500,..,1600} per phase
   (dense early: crossings are fast; 0 = onset, pre-learning):
     native: arm's own head, task-restricted accuracy on the task's fixed
       probe-test samples (registered side-channel, SGD only — vitals-style,
       socket untouched).
     a1knn: kNN(k=5, raw reps) fit on reps of samples streamed SO FAR in the
       CURRENT exposure (buffer resets each phase — the anti-leakage rule;
       empty-buffer onset reads 0.0 by convention), eval on task probe-test
       reps. Batched-rep equivalence registered: body(X)/relu(X@W) is
       identical-by-construction to per-sample step() for these two arms.
 - rdm_corr/rdm_euclid every 400 steps (A1-geometry secondary, non-gating).
 - torch threads fixed at 8 (reduction order); set_determinism per run.

CONSTRUCTS:
 - plateau_1 = mean of exposure-1 curve over offsets > 1200; bar = 0.8 x
   plateau_1 (relative threshold, per draft). t_first/t_re = first grid
   offset with curve >= bar (exposure 1 / 2).
 - EXCLUSION (pre-stated): exposure-1 ONSET >= bar -> cell excluded, counted
   (transfer artifact; ratio undefined territory).
 - CENSORING (pre-stated): re-exposure never crosses -> t_re := 1600, cell
   flagged, counted.
 - GATING index s* = (t_first - t_re)/(t_first + t_re). Rationale, found at
   registration: the draft's classic savings 1 - t_re/t_first is Jensen-
   biased NEGATIVE under the null (E[1/t] > 1/E[t]), so "CI contains 0"
   would fail on pure noise; s* is exactly 0-symmetric under exchangeability.
   Classic savings still RECORDED as the interpretable effect size. Signs
   agree cell-wise, so the draft's sign tests are unchanged in meaning.
 - Per-seed statistic: median of s* over the 5 tasks.
 - t_re = 0 (onset above bar) -> s* = 1 CEILING; ceiling fraction reported
   with the pre-stated reading: high fraction = full retention, savings
   LEVEL validated but dynamic range unmeasured on this curriculum ->
   R-014's timing run must check interference before sizing. Onset-retention
   (exposure-2 onset / plateau_1) recorded as the companion number.

GATES (all must pass):
 (a) known-positive: SGD/native s* seed-medians > 0 for ALL 8 seeds in EACH
     of 4 orders (exact one-sided sign test, p = 1/256 per order). A median
     of exactly 0 = instrument-resolution failure (grid quantization),
     diagnosed as such, not passed.
 (b1) pipeline invariance: RandProj a1knn (t_first, t_re) BIT-IDENTICAL
     across the 4 orders per (seed, task). Frozen reps + fixed slices make
     the curve order-independent BY CONSTRUCTION; any deviation is a
     plumbing bug (S13 incident), the R-005-style pipeline gate.
 (b2) known-zero: across the 8 seed-medians (order-collapsed via b1), no
     seed-consistent sign (min(#pos,#neg) >= 1, zeros count as neither) AND
     |mean s*| <= 0.10 (tolerance covers grid quantization).
 (c) determinism: one triple (sgd, forward, s0) rerun row-identical
     (stamp line excluded); analysis pass rerun byte-identical.
FAIL(a) = construct blind -> redesign (geometry-primary fallback) BEFORE
R-014 registers. FAIL(b1|c) = incident, pipeline repair, re-register.
FAIL(b2) = readout leakage -> readout redesign. Per draft §6, R-014 remains
GATED on this verdict and inherits the passing operationalization.

EXPECTATIONS (non-gating): SGD native per-seed median s* in 0.3..0.9, or
ceiling-dominated (either is a pass; they differ in the dynamic-range
caveat). RandProj |mean s*| < 0.05. SGD a1knn (recorded) may dissociate
from native — reportable, the R-012 dissociation lesson applied to A1.

COST: 65 runs (64 + determinism rerun), ~30-60 s each, CPU only; single
analysis pass; collect-early-analyze-late (no savings number computed until
all runs land).

## R-014 | 2026-07-09 | verdict (A1 commissioning attempt #1, R-013: FAIL)
links: R-013; evidence: 64 stamped runs r_20260709_161932_5fd3771_s0 ..
r_20260709_162210_5fd3771_s7 (+1 determinism rerun);
experiments/runs/commissioning_a1_verdict.txt
NUMBERING NOTE: the arms-trial draft file formerly named R-014_DRAFT.md is
renumbered at its registration (registry numbers are strictly sequential;
this entry claims R-014).

GATE (a) known-positive FAIL: SGD/native s* seed-medians all >= 0, none
negative, 28/32 strictly positive — but forward s2,s3, reverse s7, perm4 s4
tie at EXACTLY 0, the pre-registered quantization mode ("median of exactly 0
= instrument-resolution failure"). perm3 passed 8/8 (exact p=1/256).
GATE (b1) PASS: RandProj (t1,t2) exactly order-invariant, 40/40.
GATE (b2) PASS: seed-medians 6 zeros + 2 negative, mean s* = -0.067 (|.|<=0.10).
GATE (c) PASS: rerun triple row-identical (690 rows); analysis rerun identical.

READING (post-verdict inspection, logged): ALL 160 SGD/native cells cross
within the first 50 steps of a 1600-step phase — t1 in {10..50}, t2 in
{10,20}; 46/160 exact ties, 9 at the (10,10) grid floor. The arm learns a
2-class discrimination to 0.8x-plateau in a few dozen batch-1 samples; the
10-step grid gives the ruler 2-5 gradations. The construct is NOT blind:
t2 <= 20 always while t1 ranges to 50 (dominance), failing medians are 0
never negative. Two pre-stated predictions confirmed in-data: (1) Jensen
bias materialized on the known-zero — classic savings mean -0.312 on pure
noise beside s* -0.067 (the s* switch was correct); (2) SGD onset-retention
= 0.000 — forgetting is total at re-exposure onset, ZERO ceiling cells, so
the dynamic-range caveat is NOT triggered and savings comes entirely from
faster relearning. CONSEQUENCE: resolution redesign (R-015), gate unchanged.

## R-015 | 2026-07-09 | registration (A1 commissioning attempt #2: finer ruler)
links: R-014 (diagnosis), R-013 (design retained where unchanged)

CHANGES (code = experiments/05_commissioning_a1.py, content sha256[:12] =
e4b199c4299f, frozen before any registered run; attempt-#1 code preserved in
every R-013 run dir via snapshot_code):
 1. Offset grid refined where the crossings actually live: {0..100 step 1}
    u {102..200 step 2} u {210..400 step 10} u {450..1600 step 50}.
 2. Sustained-crossing rule: t = first grid offset with curve >= bar at that
    offset AND at the next grid offset (the last offset counts alone, end of
    phase). Guards single-probe luck at 1-step resolution. Residual exact
    ties still read as failures per R-013's gate — the gate is NOT loosened.
Everything else identical to R-013: arms, seeds, frozen orders, curriculum,
fresh-sample re-exposure, bar 0.8x plateau, exclusion, censoring, s* gating
index, gates (a)/(b1)/(b2)/(c), aliases comm_a1b_*. Training trajectories
are IDENTICAL to R-013's by construction (probes are read-only, named RNG
streams untouched): attempt #2 samples the same underlying curves finer.

EXPECTATIONS (non-gating): the four tied seed-medians resolve strictly
positive; t1 resolves into ~3..50, t2 into ~2..20; exact ties become rare;
b1/b2/c outcomes unchanged.

## R-016 | 2026-07-09 | verdict (A1 commissioning attempt #2, R-015: PASS)
links: R-015, R-014 (FAIL + diagnosis), R-013 (registration); evidence: 64
stamped runs r_20260709_162553_5fd3771_s0 .. r_20260709_162853_5fd3771_s7
(+1 determinism rerun); experiments/runs/commissioning_a1b_verdict.txt

GATE (a) PASS 4/4 orders: SGD/native s* seed-medians ALL 32 strictly
positive (8/8 per order, exact p = 1/256 each); range +0.08..+0.50.
GATE (b1) PASS: RandProj (t1,t2) exactly order-invariant, 40/40.
GATE (b2) PASS: seed-medians mixed-sign (1 pos, 6 neg, 1 zero), mean s* =
-0.050 (|.| <= 0.10).
GATE (c) PASS: rerun triple row-identical (1990 rows); analysis rerun
identical.

A1 IS COMMISSIONED: behavioral relearning-savings via native-head channel
(arms with heads) / accumulating within-exposure kNN (headless arms), bar =
0.8 x own first-exposure plateau, sustained crossing on the R-015 grid,
gating index s*, exclusion/censoring/ceiling rules as registered. R-014's
diagnosis confirmed in-data: all four previously-tied seed-medians resolved
strictly positive (forward s2 +0.19, s3 +0.21; reverse s7 +0.19; perm4 s4
+0.11) with nothing else changed — the ruler was the problem.

Recorded alongside: (1) the Jensen bias reproduced on the known-zero
(classic savings -0.303 vs s* -0.061 on identical cells) — s* is confirmed
as the only usable known-zero index; (2) rep-level savings dissociation:
SGD a1knn s* mean +0.18 vs native +0.27 — savings visible but smaller in
representation space than in behavior (analogue of the R-012 lesson: channel
choice matters; both carried forward, native gates where a head exists);
(3) onset-retention 0.000 across all SGD cells — forgetting is total at
re-exposure onset, zero ceiling cells, dynamic-range caveat NOT triggered.

A1 saga: 2 attempts, 1 diagnosed resolution failure, gates never loosened.
Both instruments (C1 geometry R-012, A1 savings here) now commissioned on
the same arms, orders, seeds and streams. NEXT: arms-trial registration
(the draft currently in experiments/R-014_DRAFT.md, renumbered at
registration) — its A1 constructs inherit THIS operationalization; its
premise sentences citing "banked 2-exposure data" must be rewritten per
R-013's disclosure; and its curriculum inherits the 5x2-exposure design
with fresh-sample re-exposure slices proven here.

## R-017 | 2026-07-10 | incident (code-vs-registration deviation in R-013 gate b2; verdicts unaffected)
links: R-013 (registration deviated from), R-014/R-016 (verdicts audited),
R-015 (re-registration, same gate wording inherited)

DEVIATION: R-013 registers gate (b2) known-zero as "no seed-consistent sign
(min(#pos,#neg) >= 1, zeros count as neither) AND |mean s*| <= 0.10".
The commissioning code (experiments/05_commissioning_a1.py, line 326, content
hash as registered in R-015) implements:
    b2 = (min(npos, nneg) >= 1 OR count_of_exact_zeros > 0) AND |mean| <= 0.10
The OR-zero branch is MORE PERMISSIVE than the registration: e.g. 7 positive
seed-medians + 1 exact zero would PASS the code but FAIL the registered rule
(zeros count as neither, so 7-pos/0-neg is a consistent sign).

DISCOVERY: sol (GPT-5.6) cross-review of ARMS_TRIAL_DRAFT v5, CHAT.md
2026-07-09 23:35; verified against R-013's text and the code same session.

AUDIT: in R-015's actual data the RandProj/knn seed-medians were 1 pos /
6 neg / 1 exact zero -> min(1,6) = 1 satisfies BOTH the registered rule and
the code; the deviant branch was never the deciding factor. R-014's FAIL was
on gate (a), not (b2). VERDICTS R-014 AND R-016 STAND UNAFFECTED.

DISPOSITION: (1) 05_commissioning_a1.py is a frozen artifact of R-013/R-015
(its content hash is registered) and is left UNTOUCHED — correcting it would
break provenance without changing any verdict. (2) Every PROSPECTIVE
implementation of the b2 rule follows the REGISTERED wording; the arms-trial
draft (v9, gate b-ii) already specifies this and flags this incident inline.
(3) Standing lesson for gate authorship: registration text and gate code
must be diffed as part of commissioning gate (c)-style checks; a
code-vs-registration diff is added to the arms trial's pre-arm checklist via
its gate (a)/(d) review.

## R-018 | 2026-07-11 | correction (scope and prospective conventions)
links: R-000, R-002, R-012, R-013, R-015, R-016, R-017; charter v5

PURPOSE: incorporate the 2026-07-11 full-chain survey/instrument review without
rewriting an issued entry, result, threshold, artifact, or hash. This entry has
no treatment verdict. It corrects prospective data-regime language, records the
scope of the two commissioned operators, and separates commissioned tools from
infrastructure, diagnostics, and sketches.

### Data-regime convention from charter v5

R-000's `single-pass | repeated+held-out` shorthand is prospectively replaced
by an explicit declaration of:

- **prequential (test-then-train):** the locked score is recorded before any
  update from the observation;
- **train-then-score:** the score is a training diagnostic;
- **held-out:** the observation or task family is not used for updating,
  design selection, threshold choice, or tuning before evaluation completes.

Every future registration also states the score/update order and whether its
stream, task family, generator seed, or results were used during development.
A final confirmatory claim uses a preregistered sealed prequential stream or
task family unused during development, or a separate held-out set. This is a
terminology and scope correction. It does not retroactively turn the R-012 or
R-016 commissioning streams into confirmatory held-out evidence.

### C1 scope note

R-012 commissions the registered class-centroid RDM geometry operator on one
known-history-dependent SGD configuration and one known-history-independent
RandomProj configuration. It does not establish a property of `BareAgt`, a
future `Dir.E` mechanism, a critic, or candidate removal. It does not commission
a quantitative between-treatment-arm C1 comparison. Every future treatment use
must pass its registered exact-channel known-positive and known-zero gates.

### A1 scope note

R-016 commissions the registered relearning-savings/readout operator on one
known-positive SGD configuration and one known-zero RandomProj configuration.
The controls shared tasks, streams, orders, and seeds: they were
**experience-matched, not competence-matched**. Their attainable plateaus
differ, and R-013/R-015/R-016 contain no scratch arm. R-016 therefore does not
validate cross-arm comparability under heterogeneous plateaus. Future
heterogeneous-arm use requires a preregistered common-target or competence
check and the A1c novel-task or label-remap companion.

A positive A1 reading means superiority on the registered behavioral or
decodable-representation readout. It does not by itself establish endogenous
use, knowledge, rent, or causal control of action. Those stronger claims require
action-level causal evidence such as the future D1 intervention family. No
R-016 verdict is retracted.

### R-017 disposition reaffirmed

R-017's permissive OR-zero code branch was not decisive in either prior A1
verdict: R-014 failed gate (a), and R-016's known-zero data passed the stricter
registered sign rule. R-014 and R-016 remain unchanged. All prospective gate
implementations follow the registered wording and include a text-versus-code
comparison before collection.

### Instrument-status ledger as of this entry

| Item | Status | Permitted evidential role |
| --- | --- | --- |
| Charter and provenance plumbing | Operational infrastructure | Governs evidence; is not treatment evidence |
| C1 geometry | Commissioned on the R-012 named controls | Future treatment use only after exact-channel gates |
| A1 relearning/readout savings | Commissioned on the R-016 named controls | Future treatment use only after exact-channel and comparability gates |
| Vitals and old-three precondition diagnostics | Diagnostic only | Debugging and failure detection; never an affirmative verdict |
| D3 event ledger | Planned infrastructure; unimplemented | No evidential role until replay and treatment payloads are commissioned |
| D4 static-society detector | Standing kill condition from R-002; uncommissioned measurement | No affirmative verdict; depends on commissioned D3 sentinels |
| A1c, A2, A3, B1, B2, C2, D1, D2 | Uncommissioned plans or sketches | No affirmative verdict until separately registered and commissioned |
| TRANSFER | Parked | Outside the current programme unless prospectively restored |

NUMBERING: this correction consumes R-018. `ARMS_TRIAL_DRAFT.md` remains
unregistered and, if later frozen and accepted, registers as R-019 or later.

## R-019 | 2026-07-19 | conventions
links: R-000, R-018, charter v6
**Writing-style discontinuity note (no experiment, no verdict).** As of
2026-07-18/19 the project's prose rules changed (RW decision, recorded in the
global CLAUDE.md): no all-caps emphasis, no em dashes or prose double hyphens,
no filler-register words ("genuinely", "honestly", "load-bearing", and kin);
Paul Graham's "Write Simply" adopted as the norm: plain words, nothing hidden,
no dumbing down, no omitted detail. Applied to the tracked codebase in commits
5482891 and df45b20, and to CHARTER.md's living half as charter v6 (style only;
no rule change). Existing registry entries R-000 through R-018 and all frozen
or registered experiment documents remain byte-unmodified per the append-only
rule: caps and dash typography in pre-07-19 text is historical formatting, not
semantics, and must not be edited in place. Any future style edit to a
registered document happens as a new entry, never as an in-place change. This
note is not evidence and changes no threshold, wording, or verdict.
NUMBERING: this note consumes R-019; R-018's "ARMS successor registers as
R-019 or later" therefore now reads R-020 or later.

## R-020 | 2026-07-20 | registration (Dir.E exam band-freeze; commissioning of the exam instrument)
links: charter v6 (sections 3, 7f, 13), DIRE_CONTRACT_DRAFT.md,
D3_RECORDER_SPEC_DRAFT.md, R-013/R-015 (pattern), R-017 (gate-diff lesson);
full pre-registration text at experiments/R-020_DRAFT.md (content-frozen by
this entry; the draft file is the registration's long form).

EVENT: 06_commissioning_dire.py at this entry's experiments_hash runs the
control roster (null:copy_last, anchor:chance, frozen:pc, frozen:wake_sleep,
frozen:gc_ff) through the five standard fixtures on the registered seeds
101, 211, 307, 401, 503, 601, 701, 811 (fresh; disjoint from free-play
0/1/2, which bind nothing). One run; verdict entry follows separately.

FREEZES (measured, formulas fixed here, numbers from the run): per-family
kp0/kp1 bars = max(N-copy, own twin) trailing mean + 3 x pooled seed-SD,
pass requires above-bar on every registered seed; per-family KZ band = own
twin's kz trailing mean +/- 3 x seed-SD; global plateau floor = max over
twin x fixture-class cells of (offset + 3 x seed-SD), sanity-bounded at 0.2.
CONVENTIONS (structural, set once): cycle period 4; no consecutive repeats;
burn-in k = 3; symbol dim 32; fixture lengths kp/kz 400, sw 400+200+400,
ts 6x200; settle fraction 0.10, window 50, dwell 100, atol 0.02; SW
retention post0 >= 0.5 x pre; SW re-acquisition <= 2.0 x first settle;
activity floor inactive at this rung (measured version registers at the
BareAgt rung); magnitude term deferred (cosine-only stands, predicate-v2
open decision, prospective amendment only).

EXPECTATIONS (the run's own pass/fail): E1 controls' kz means inside own
bands; E2 copy-last no-plateau on sw/ts; E3 no vacuous band (kp1 bars below
0.9, else underpowered-fixture incident: redesign, never lower the bar); E4
bars exceed own twin means; E5 plateau floor <= 0.2 else twin-pathology
incident. Vacuous-pass and impossible-pass route per contract section 6.

ANALYSIS: judge = dire_exam.judge at this entry's hash, gate wording diffed
against gate code before the run (R-017); stream regime prequential by
construction; verdict derives from D3 ledger_replay only. Determinism gate
forward: the registered subject event's control arms must reproduce this
run bit-for-bit or verdicts block on a section-13 incident.
NUMBERING: this entry consumes R-020; the ARMS successor moves to R-021 or
later. The Dir.E subject event registers separately after this entry's
verdict.

## R-021 | 2026-07-20 | incident (R-020 expectation E5 fired; floor formula amended, RW option A)
links: R-020 (registration), R-017 (kin: formula-vs-intent divergence),
charter sections 12.1, 13.

E5 fired on the first registered run (r_20260720_152001_692f9c9_s101): the
global plateau floor computed to 0.6612, over the 0.2 bound. Investigation:
the twins are not pathological; the registered formula conflated two regimes.
On noise, twin offsets are chance proper (0.05-0.08). On constant input, any
frozen system's output aligns with the only direction in the room (cells
0.42-0.66), and frozen:pc carries real predictive residue on cycles (0.27)
because freezing weights froze learning, not settling inference. A global max
over these regimes is not a chance floor. RW's outlier concern at review
time, and the E5 tripwire added in answer to it, caught the simplification
on the first firing.
DISPOSITION (RW-ruled 2026-07-20, option A): per-family x fixture-class
floors (nine numbers) replace the global max: this restores the original
per-cell design that review had simplified away. E5 recalibrated to the two
regimes: noise cells <= 0.2 (chance); constant/cycle cells < 0.9 (structural
alignment, must leave room for learners). No verdict affected: the run was
controls-only, so no learner-favoring selection is possible. The amended
freeze consumes the same recorded run via ledger_replay (deterministic,
charter 12.1 instrument-redesign path); sequence fully disclosed here.

## R-022 | 2026-07-20 | verdict (band-freeze PASS under R-021-amended formulas; constants frozen)
links: R-020, R-021; source run r_20260720_152001_692f9c9_s101;
frozen constants file experiments/dire_frozen_constants.json (sha256 afe662b73c6b).

All expectations pass: E1 (kz self-consistency, 3/3 families), E2 (copy-last
no-plateau), E3 (no vacuous band: kp1 bars 0.078-0.185, all far below 0.9),
E4 (bars exceed own twins), E5 amended (noise cells 0.055-0.077 under 0.2;
constant/cycle cells 0.12-0.66 under 0.9). Frozen numbers: kp1 bars
gc_ff 0.1401, pc 0.1845, wake_sleep 0.0776; kp0 bars gc_ff 0.4154,
pc 0.6612, wake_sleep 0.5958; kz bands gc_ff [-0.0545, 0.0539],
pc [-0.0768, 0.0770], wake_sleep [-0.0604, 0.0627]; plateau floors = the
nine cells in the json. These are numbers now, not formulas; changing any is
a new entry. The Dir.E exam instrument is commissioned for control
separation; subject verdicts await the registered subject event (draft at
experiments/R-023_DRAFT.md; registers after the judge's bar-application
extension is built and hashed).

## R-023 | 2026-07-20 | registration (Dir.E subject event: Step-3c feedforward-rung verdicts)
links: R-020/R-021/R-022 (bands), DIRE_CONTRACT_DRAFT.md, charter sections
3, 7f, 11, 13; full pre-registration text at experiments/R-023_DRAFT.md
(content-frozen by this entry after RW point-by-point review, five rulings
2026-07-20: burn-in wording corrected to BareAgt-rung-only; sandwich gates
bind the candidate only, refs owe prediction gates, pc-burnout forecast
recorded non-binding; reference failure blocks all verdicts (conservative);
control drift blocks all verdicts; dire-off must-fail kept with every-seed
protection).

EVENT: 07_commissioning_dire_subjects.py, registered era, seeds
101/211/307/401/503/601/701/811, frozen constants sha256 afe662b73c6b,
judge = dire_exam.judge + bar-application extension, all at
experiments_hash 199c89f82830 (gate wording diffed against gate code pre-append,
ten checks clean). Verdicts from D3 ledger_replay only.

GATES: host:gc_ff (candidate): kp0/kp1 above family bars 0.4154/0.1401
every seed; kz inside [-0.0545, 0.0539] every seed; SW retention >= 0.5 x
pre and re-acquisition <= 2.0 x first settle per seed (frozen gc/cycle
plateau floor 0.2117). ref:pc, ref:wake_sleep: prediction gates against
own family bars; any reference failure = exam-broken incident, no verdicts
issue. Controls (3 twins, copy-last, chance anchor): bit-identical scores
and guess-hashes to R-022's source run over the standard five; any
deviation = section-13 incident, no verdicts. host:gc_ff_dire_off: must
fail kp1 finds-structure; clearing the bar on all seeds = leakage-shaped
incident. Characterization (variants, capacity, revisit, timescales):
recorded, never gated; T_switch characterization only, burn-in derives
exclusively from the BareAgt rung (RW staging ruling).

CONSEQUENCE: all gates green = feedforward-inductive Dir.E commissioned;
mechanism lane advances to the BareAgt rung, measurement lane to the
detector layer. Any failure routes per the contract failure branch.
Verdict entry follows separately.

R-023 CORRECTION (2026-07-20, pre-run): the registered experiments_hash
covered a 07 script with a syntax error (unterminated string, line 204: the
script could not parse, so no run ever executed under that hash). One-line
repair applied; no logic changed; gate-diff rerun clean. Corrected
experiments_hash for the event: c41a42be13e2. The registered run executes under
this hash.

## R-024 | 2026-07-20 | verdict (R-023 subject event: PASS; feedforward-inductive Dir.E commissioned)
links: R-023 (registration, incl. hash correction), R-022 (bands); source
run r_20260720_154625_692f9c9_s101 (1249/1249 objects verified; verdicts
from ledger_replay).

DRIFT GATE: all five controls bit-identical to R-022's source run (scores
and guess-hashes, standard five fixtures, all eight seeds). GATES:
host:gc_ff kp0 1.000 (bar 0.4154) and kp1 0.999-1.000 (bar 0.1401) on
every seed; kz inside band every seed (max |0.032|); SW retention worst
0.754 (>= 0.5) and re-acquisition 66-67 vs first-settle 71 (<= 2.0x, and
faster in absolute terms: savings-shaped). ref:pc and ref:wake_sleep pass
all prediction gates every seed. host:gc_ff_dire_off stays under the kp1
bar on every seed (max 0.122 vs 0.140): fails as registered.
EVENT VERDICT: host:gc_ff COMMISSIONED (feedforward-inductive, predict-only
rung).

RECORDED (characterization, non-gating): the non-binding pc-burnout
forecast replicated on fresh seeds (retention 0.222, re-settle 275 vs 57):
eligible to graduate to a damage-detector known-positive at a future event.
Capacity and revisit patterns replicated from free play (softhebb-family
features degrade gracefully on the rank probe where oja/instar collapse;
every memoryless subject sits at the 0.854 revisit ceiling). T_switch for
the host ~84-88: characterization only; burn-in derives exclusively from
the BareAgt rung.

SCOPE: commissioned means the mechanism keeps the contract's promises on
these fixtures against these controls. It is Bucket-A prep by the plan's
own accounting: no evidence about synthesis-first vs kill-first, no
candidate, no thesis claim. Two cosmetic defects noted for pre-08 repair
(post-run edits would change the registered hash, so they stand as-is in
this event's artifact): the report header prints the free-play banner in
registered era, and the legacy promise-verdict block still shows module
refusals; the registered-gates block supersedes both.

## R-025 | 2026-07-21 (22:51 EDT / 2026-07-22 02:51 UTC) | registration (BareAgt-rung band-freeze)
links: R-020 (pattern), R-024 (rung handoff), R-025_DRAFT.md (freeze page,
hash 6bbc043c48d2). Registered script: 14_registered_bare_freeze.py, hash
245a213103a6, smoke-executed end to end pre-registration (R-023 lesson).

RULINGS: RW accepted the freeze page's five decision points as proposed
("yeah good enough, run it", 2026-07-21 ~22:46 EDT, mid-flight AC854):
horizon 12000 (measured, hz12 curve); differential bar = beat frozen twin
by 3 sigma of twin seed spread AND per-seed differential > 0 on >= 7 of 8
seeds; activity floor = 1st percentile of twin norms, global; responds
statistic as trialed (13_responds_trial, 18x separation); fixture-length
conventions sw 12000+6000+12000, ts 6x2000.

PRE-REGISTRATION AMENDMENT (smoke-caught, before this entry): activity
floor construction takes the 1st percentile of NONZERO twin norms, with
the zero-norm fraction recorded alongside: structural zeros (unreachable
columns) form a point mass that pins any low percentile to exactly 0, and
zeros are already excluded from scoring by the cosine zero-norm guard.
Freeze-page row 7 carries the amendment.

SEEDS: 3252, 4635, 5149, 5344, 6134, 7540, 7776, 9490. Drawn at t0 via
secrets, disjoint from free-play 0-9 and feedforward 101-811; free play
banned from touching them.

ARMS: x frozen twin + x dire-off (faithful knobs softhebb/triangle/
ss=1e-2/lr_e=0.1, n_cols=49, CRN graphs). Fixtures kp1(12000), kz(12000),
ts(6x2000), stream namespace "r025". E-checks E1-E5 per script docstring;
any failure = incident, no constants issue. Constants freeze to
bare_frozen_constants.json (sha recorded in the R-026 outcome entry).
Recording: per-step network aggregates through the chunked recorder;
per-column detail regenerable deterministically from (config, seed,
stream namespace), CPU numerics pinned (CUDA_VISIBLE_DEVICES empty).

SCOPED OUT (carried from the freeze page): 24000-horizon imprint drift
(open question, candidate future fixture); instar+relu family (excluded
until stabilized); 400-step screening's horizon blind spot (covered by
the boundedness gate). Outcome entry follows as R-026.

## R-026 | 2026-07-21 (22:59 EDT) | outcome (R-025 band-freeze: PASS, constants frozen)
links: R-025. E-checks E1-E5 all green, incl. E5 drift (twin seed 3252 kp1
rerun bit-identical by aggregate-trace hash). Constants frozen to
bare_frozen_constants.json, sha 7d1fc274c7a6, written once.

MEASURED: twin differential mean -0.0415, sigma 0.0415 -> gap bar 0.1245;
activity floor (p1 of nonzero twin norms) = 1.000000; twin responds mean
0.0021, sigma 0.0014 -> responds bar 0.0062. dire_off differentials
(recorded): -0.1792, -0.1059, +0.1119, -0.1367, -0.1093, +0.3030, -0.0860,
0.0000: wide spread as expected at this horizon with frozen E readouts.

OBSERVED CONSTRUCTION DETAILS (recorded, not amended: constants are
frozen): (1) the floor's exact value 1.0 sits on the input column's
unit-norm symbol mass inside the pooled distribution; scoring applies to
internal columns only, so the floor's bite is on genuinely quiet internal
steps; the R-027 floor-delta reading will show its effect. (2) The fresh
seeds drew a wide twin spread (rep11-like, not hz12-like), so the gap bar
(0.1245) sits at free play's measured faithful gap (0.124): the subject
event is severe, and a faithful failure is a live outcome.

## R-027 | 2026-07-21 (23:00 EDT) | registration (BareAgt-rung subject event)
links: R-025/R-026 (constants, sha 7d1fc274c7a6). Registered script:
15_registered_bare_subjects.py, hash ed4b70dc0766, smoke-executed end to
end. Stream namespace "r025" (identical streams to the band-freeze: the
drift gate depends on it).

ROSTER: bare:x_faithful (subject, gated), bare:stock (BareAgtDirE,
recorded reading, non-gating; FORECAST: uncertain, never probed at this
horizon), bare:x_twin + bare:x_dire_off (controls, drift-gated against
R-025's stored aggregate hashes).

GATES: faithful differential: mean(kp1-kz) - twin mean >= 0.1245 AND
per-seed diff > 0 on >= 7 of 8 seeds; faithful responds >= 0.0062;
dire_off must fail the full differential conjunction; controls bounded and
bit-identical to R-025 traces; twin floored-vs-frozen delta recorded,
incident threshold 0.01 absolute.

DECLARED SEMANTICS: a faithful gate failure is a registered negative
verdict for this recipe at this rung (bars stand, recipe retriable at a
future event); only control misbehavior (dire_off clearing the bar, drift,
control NaN) is incident-shaped. Non-binding forecast, recorded for
honesty: joint faithful pass probability ~40-60% given the measured bar
severity (sign clause ~0.8 x gap clause ~coin-flip).

PRE-REGISTRATION AMENDMENTS (smoke-caught in 15 before this entry):
must-fail tests the bar's full conjunction (dire_off's gap is inflated by
dir.a learning erasing the twin's persistence asymmetry: rep11: so the
sign clause discriminates); floor-delta demoted from sigma/2 gate to
recorded reading with 0.01 absolute incident threshold; T_switch plateau
floor built from seed-level twin ts means (per-step pooling inflates
sigma). T_switch export: settle fraction 0.1, window 500, dwell 500, atol
0.02, plateau floor = twin ts seed-mean + 3 sigma; median over switches
and seeds becomes Step 4's burn-in prefix (closes 3c).

SCOPE: this event reads differential, responds, and T_switch only; no
sandwich (retention/re-acquisition) gate exists at the bare rung yet: the
sw conventions in R-025 row 4 are declared but dormant. Floor policy: the
frozen floor applies to per-column scoring identically for every arm;
recorded aggregate traces stay unfloored so drift compares machinery.
Verdict entry follows as R-028.

## R-028 | 2026-07-21 (23:3x EDT) | verdict (R-027 subject event: FAITHFUL FAILS differential bar; controls clean; honest negative)
links: R-027 (registration), R-026 (constants sha 7d1fc274c7a6). Verdicts
from the registered run (experiments/runs/r027.log; aggregate traces in
D3 as r027_* runs):

GATES: drift PASS (twin and dire_off bit-identical to R-025 traces, all
seeds, all fixtures); control boundedness PASS (no NaN anywhere);
dire_off must-fail PASS (fails the conjunction as registered); floor
delta PASS (twin floored-vs-frozen 0.0033, threshold 0.01). faithful
responds PASS: 0.0482 vs bar 0.0062 (~8x, replicating the trial's
separation on fresh seeds). **faithful differential FAIL: mean -0.0191
(gap +0.0223 vs bar 0.1245), per-seed signs 1/8 (min 7). The registered
verdict stands: the faithful recipe is NOT commissioned at the bare rung.**
Stock (non-gating reading): -0.0547, signs 1/8: no world imprint, forecast
resolved. T_switch export: median 500 with 20 of 33 instances exactly at
the 500-step resolution limit (settle faster than resolvable at
window=dwell=500): exported as "<= ~500, resolution-limited", and 3c does
NOT close: the export's subject failed its lead gate, so Step 4's burn-in
may use the number only provisionally, flagged.

POST-EVENT DIAGNOSTICS (from the event's own stored unfloored aggregates,
recorded 23:4x): recomputing differentials unfloored and
aggregate-ordered on the same faithful runs reads 5/8 positive (mean
-0.012, values -0.111..+0.029) vs the floored gate statistic's 1/8. Two
findings: (1) CONSTRUCTION FLAW, mine: the frozen floor (1.0, calibrated
on twin norms) bites learned dynamics differentially: the floor-delta
guard watched the twin, the one arm whose dynamics the floor was
calibrated on, and never audited the floor's bite on subjects. A future
event needs a bite-audited floor construction. The gate verdict stands as
registered regardless: bars are bars. (2) GENERALIZATION QUESTION, real:
even unfloored, fresh-graph differentials (mean -0.012, weak 5/8) are far
below free play's development-graph readings (+0.083, 9/10 at the same
horizon): the free-play arc developed recipe and horizon on graphs 0-9
and may have overfit the draw family, which is exactly the failure mode
registration exists to expose. Disambiguation probe (16, free play): the
2x2 of graphs {0-9, registered} x stream namespace {rep11, r025}.

READING: the instruments all behaved (drift bit-identical, controls
clean, responds replicated); the finding did not. This is the discipline
working as designed: R-021's ancestor caught a broken bar, R-028 catches
either a floor artifact, a non-generalizing effect, or both. Faithful
retriable at a future event under a repaired floor and a pre-stated
fresh-graph sample size.

R-028 ADDENDUM (2026-07-21 23:5x EDT): ROOT CAUSE FOUND: the free-play
differential was a fixture-ORDER artifact, and the registered null is the
true reading. Chain of evidence: (1) RW asked whether the graph sets
differed in input fanout; the topology audit found them statistically
alike BUT exposed dev seed 7: input fanout 0, reachability 0, a closed
world whose matched differential must be exactly 0: hz12 had read it as
+0.074. (2) Design audit: the free-play runners (10/11/12/16) drive ONE
host through kp1 then kz sequentially, so kp1 is measured on a fresh
substrate (steps 0-12000) and kz on an aged one (12000-24000): the
"differential" contains early-vs-late margin drift that grows with
horizon: the entire "world imprint arrives at 6000" curve is this
artifact's growth curve. The registered scripts (14/15) build a fresh
host per fixture: the matched design. (3) Pre-registered prediction:
if artifact, BOTH missing 2x2 cells read positive regardless of graphs
or streams. CONFIRMED by 16: dev graphs x r025 streams +0.144 (10/10);
registered graphs x rep11 streams +0.084 (8/8): the same registered
graphs that read -0.019 (1/8) under matched design read +0.084 (8/8)
under the ordered design. The design, not the world, produced the
finding.

STANDING AFTER THE ADDENDUM: contaminated free-play readings: all
sweep-confirm differentials (10), rep11 differentials incl. the instar
family's, the hz12 horizon curve, gen16 cells. Surviving: the responds
reading (within-stream, order-free; replicated registered at 8x), the
frozen twin's offset (stationary dynamics, no aging), boundedness and
collapse-plateau findings, all feedforward-rung events (fresh host per
fixture throughout via drive_fixtures), and the registered pair's
machinery. The world-imprint-at-depth question is REOPENED as unanswered,
not answered negative: matched-design evidence to date is ~zero signal
(R-028 gate 1/8; unfloored 5/8 weak), measured correctly for the first
time. Next free play: the horizon question rerun under matched design
(fresh host per fixture), then any retry registers with a bite-audited
floor and pre-stated sample.

## R-029 | 2026-07-22 (04:18 EDT) | registration (fan3 transport band-freeze)
links: R-028 + addendum (matched-design mandate, retry prescription),
rep18 (free-play basis: fan3 gap +0.070 vs twin, n=30 matched fresh
graphs, per-seed sd ~0.14, paired-positive ~0.70), vis21 (conflict
visibility 3.3%: transport has a marginal consumer; detector events ride
the subject event non-gating). Registered script:
22_registered_fan3_freeze.py, hash 40591ab348ff, smoke-executed end to
end. RW rulings: the event certifies a truth, not a magnitude ("the
registered event certifies a truth... ok do it?", 2026-07-22 ~04:0x EDT);
bars powered for the measured small effect.

SEEDS (32, drawn at t0 via secrets, disjoint from all free-play 0-39,
feedforward 101-811, R-025's eight): 1190 1337 1666 1722 2309 2597 2699
2779 3474 3485 3650 3833 3888 3994 4315 4522 4634 4862 4972 5293 5311
6942 7045 7210 7314 7533 8516 8691 9377 9457 9465 9664. Free play banned
from touching them.

ARMS: fan3_twin + fan3_direoff (fan3 knobs: p_a=.3, p_e=.6, dist_exp=1,
input_fanout=3, softhebb/triangle/ss=1e-2/lr_e=.1, n_cols=49). Fixtures
kp1+kz at 12000, namespace "r029", fresh host per fixture (matched by
construction, ledger #40). BARS PRE-STATED: gap = fan3 mean differential
minus twin mean >= 3 sigma of twin seed spread; signs >= 20 of 32
paired-positive (null probability 2.5%; power ~0.79 at the measured
rate). FLOOR: p1 of nonzero twin norms, WITH bite audit on free-play
fan3 runs frozen as an envelope (caps: exclusion < 20%, margin delta <
0.02); registered-run bite outside the envelope = incident (the R-028
floor lesson made construction). E-checks E1-E5 per script docstring.
Constants freeze to fan3_frozen_constants.json; outcome entry R-030;
subject event registers as R-031 against the frozen numbers with
fan3 gated, dire_off must-fail (full conjunction), twins drift-gated,
and the vis21-scoped ts arm recorded non-gating (first registered
conflict-detector events). Declared semantics unchanged from R-027:
subject failure = negative verdict, control misbehavior = incident.

## R-030 | 2026-07-22 (10:03 BST; entry appended 19:10 BST after a watcher gap) | outcome (R-029 band-freeze: PASS, constants frozen)
links: R-029. E-checks E1-E5 all green, including E4, the first bite
audit: floor 1.0 (p1 of nonzero twin norms, zero fraction 0.006, the
input-symbol mass detail recurring from R-026), exclusion 3.5% of scored
steps, subject margin delta 0.0114, both inside the declared caps
(20%, 0.02). Constants frozen to fan3_frozen_constants.json, sha
6333ac07e5b0. MEASURED: twin differential mean -0.0537, sigma 0.0117 ->
gap bar 0.0350 (free-play prediction from rep18 held: sigma estimate
0.012); sign bar stands at 20/32 as registered. dire_off differentials
recorded in the constants. E5 drift bit-identical. The nine-hour gap
between completion and this entry was watcher mortality (session-side
only); the run, constants, and ledger were unaffected, which is the
detachment design doing its job. Subject event registers next as R-031.

## R-031 | 2026-07-22 (19:12 BST) | registration (fan3 transport subject event)
links: R-029 (bars pre-stated), R-030 (constants sha 6333ac07e5b0).
Registered script: 23_registered_fan3_subjects.py, hash 2691d2798ea9,
smoke-executed end to end. Stream namespace "r029" (identical streams to
the band-freeze; drift gate depends on it).

ROSTER: fan3 (subject, gated: gap >= 0.0350 with floored scoring AND
paired signs >= 20/32, paired meaning each seed's differential beats the
SAME seed's frozen twin differential); fan3_twin + fan3_direoff
(controls: drift-gated bit-identical to R-029 traces; dire_off must fail
the full conjunction); fan3 ts arm (recorded, non-gating: per-column
conflict events at the cal20 cell = the first registered detector
readings, the responds value, and a provisional T_switch).

BITE ENVELOPE: subject exclusion and floored-vs-unfloored margin delta
must sit inside R-030's audited caps (20%, 0.02); outside = incident
flag, never a verdict flip. Declared semantics unchanged: subject failure
= registered negative verdict; control misbehavior = incident.
Non-binding forecast, recorded: joint pass probability ~75-80% at the
rep18-measured effect (+0.070, paired rate ~0.70); the likeliest failure
shape is a clean gap with 18-19/32 signs. Verdict entry follows as R-032.

R-031 CORRECTION (2026-07-22 20:12 BST, pre-verdict): the registered hash
2691d2798ea9 covered a ts-arm defect: settled_at received dwell twice
(explicit argument + the SETTLE dict), TypeError at the first ts future,
no verdicts produced, nothing recorded. The smoke missed it because the
settle call was guarded behind "if not SMOKE": the branch never executed
under smoke. One-line repair (dwell removed from SETTLE), no logic
change; the repaired branch exercised directly at real call shape.
Corrected hash: 51b9739f11c3. Lesson appended to the R-023 family: smoke
coverage must include every branch the registered run will take;
SMOKE-guarded code is uncovered code. The registered run executes under
the corrected hash.

## R-032 | 2026-07-22 (20:5x BST) | verdict (R-031: INCIDENT, no certificate; must-fail arm cleared the bar)
links: R-031 (corrected hash 51b9739f11c3), R-030 (constants). Verdicts
from experiments/runs/r031b.log; aggregate traces in D3 as r031_* runs.

OUTCOME: fan3 cleared both gates (gap +0.0679 vs bar 0.0350; paired
signs 23/32 vs min 20; bite inside envelope: exclusion 1.5%, delta
0.0012). BUT fan3_direoff ALSO cleared the full conjunction (gap +0.0461,
paired 20/32), violating the must-fail gate. By the declared semantics,
control misbehavior = incident: NO CERTIFICATE ISSUES. Drift and
boundedness clean.

MECHANISM READING (what the incident teaches): on the fan3 topology,
Dir.A learning alone, with E conns frozen at random init, produces a
kp1-kz differential of +0.046 over the twin. World-coupled dynamics
(which the responds statistic already certified) are legible through ANY
fixed readout at this connectivity: the differential is therefore not an
E-specific transport reading on this topology. The E-learning increment,
fan3 minus dire_off, is +0.022 (recorded, uncertified). Ordering: twin
-0.054 < dire_off -0.008 < fan3 +0.014.

AUDIT FAILURE, MINE (filed as ledger #41): the frozen constants have
carried dire_off's diffs since R-030 (10:03 BST); computing the must-fail
against them takes four lines and was never done; the R-031 registration
then FORECAST "dire_off must fail" against data on disk that said
otherwise. Compounding design gap: rep18 gave the raw family a dire_off
arm but never fan3, so free play promoted a candidate without its full
control suite; the confound was catchable a day earlier.

RECORDED (non-gating): registered visibility 1.0% (76/7840 col-switch
pairs, falses 0.025/1k: lower than free play's 3.3%, graph variance);
responds mean 0.0373; T_switch median 6822 (fan3 settles slowly: two
orders above faithful's resolution-limited ~500; provisional, and moot
while no certificate stands).

CONSEQUENCE: the transport-via-E claim is re-specified, not retried
as-was: the honest baseline for an E-specific claim is dire_off, not the
twin, and the E-increment (+0.022, spread unmeasured) needs free-play
characterization (including the fan3_direoff arm rep18 lacked) before
any powered event. The twin remains the null for dynamics-level claims;
dire_off becomes the null for E-level claims: a two-null structure the
R-026 zero-point lesson implied and this incident makes mandatory.
Sparsity design work proceeds regardless; its visibility target stands.

R-032 ADDENDUM (2026-07-22 21:2x BST, RW-prompted): the E-increment given
above as "+0.022 (recorded, uncertified)" overstated its standing. The
paired per-seed computation (fan3_i minus dire_off_i, same seeds, both
arms drift-verified) gives mean +0.0220, sd 0.3143, SE 0.0556, t = 0.40,
signs 19/32: indistinguishable from zero at n=32. Corrected standing:
there is currently NO evidence of an E-specific transport contribution at
the bare rung, on any tested topology: faithful read null (R-028) and
fan3's twin-relative gap decomposes entirely into Dir.A-side dynamics
within measurement error (R-032). Dir.E's demonstrated functions to date
are the feedforward rung (R-024) and self-model prediction of endogenous
dynamics. The design work (sparsity et al.) now carries the full burden
of making E-learning matter measurably, with the two-null structure and
this paired statistic as the baseline to beat.
