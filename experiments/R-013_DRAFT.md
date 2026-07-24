# R-013 | DRAFT — SUPERSEDED by REGISTRY.md R-013 (registered 2026-07-09)
#
# This draft's premise was FALSE: it claims the banked comm4 runs hold a
# 2-exposure curriculum ("all checkpoints and per-task trajectories" — also
# overstated). R-011's "5x2" = 5 tasks x 2 DIGITS, each task seen ONCE; no
# t_re exists in banked data. The registered R-013 keeps the constructs and
# known-answer logic but runs NEW known-arm runs (re-exposure curriculum,
# fresh-sample slices) and gates on the symmetric index s* (the draft's
# ratio is Jensen-biased under the null). Kept for the record.

**Type:** pre-registration — commissioning of the A1 (relearning-savings) instrument
on BANKED data from the C1 commissioning saga. No new training runs.
**Status:** DRAFT until RW review; then appended to REGISTRY.md. Analysis runs
only after registration (blindness: savings numbers have never been computed —
asserted here; the banked evidence dirs are append-only and untouched).

---

## 1. Question

Does the A1 savings construct detect relearning acceleration where it must exist
(SGD) and read zero where it cannot (frozen random projection)? C1 required four
attempts to find a construct that sees; A1 gets its known-answer test BEFORE the
arms trial (R-014) makes it a co-primary.

## 2. Data (banked, read-only)

The R-011/R-012 commissioning runs: arms SGD and RandProj, 4 orders × 8 seeds,
5-task × 2-exposure curriculum, all checkpoints and per-task loss/accuracy
trajectories logged at probe points. The R-010 deep-arm runs are included as
RECORDED-ONLY (no registered expectation; its PR-collapse makes it a stress
case, not a known answer).

## 3. Constructs (declared before analysis, charter v4 §3)

**A1-behavioral (candidate-primary):** for each (arm, order, seed, task):
t_first = steps from task onset to crossing the registered threshold on first
exposure; t_re = same on re-exposure. Savings = 1 − t_re/t_first.
Threshold: RELATIVE — 80% of that task's own first-exposure plateau accuracy
(plateau = mean over final quarter of the exposure) — relative, to avoid
arm-difficulty confounds. Readout per arm: native task loss where the arm has
one (SGD); registered kNN readout on representations otherwise (RandProj).
Aggregate: median savings across tasks; uncertainty across seeds.

**A1-geometry (secondary):** RDM displacement at task switch (1 − corr of
class-centroid RDM before/after) and its recovery rate on re-exposure.
Recorded alongside; dissociation from behavioral is reportable, not gating.

**Limit stated now:** the banked curriculum has TWO exposures per task, so this
commissioning validates the savings LEVEL only. The savings SLOPE (across ≥3
cycles) cannot be validated retroactively; R-014's curriculum adds a third
exposure and the slope reading carries "level-validated, slope-first-use" status.

## 4. Known-answer expectations (registered)

| Arm | Expectation | Test |
|---|---|---|
| SGD | savings > 0 | exact sign test across seeds per order; all 4 orders |
| RandProj | savings ≈ 0 | CI contains 0; no seed-consistent sign |
| Deep (R-010) | none registered | recorded |

PASS = both registered expectations met AND analysis rerun is bit-identical
(determinism gate on the analysis code, seeded).
FAIL on SGD = the behavioral construct is blind → redesign (geometry-primary is
the pre-stated fallback candidate) BEFORE R-014 registration.
FAIL on RandProj (nonzero savings) = readout leakage (the kNN readout itself
learns) → readout redesign; same consequence.

## 5. Provenance & procedure

Analysis code committed (SHA in the verdict entry) before first execution on
banked dirs; banked dirs opened read-only; one analysis pass; verdict appended
with the saga-style honesty (a FAIL is a diagnosed finding, not a retry-until-
green). Estimated cost: analysis code + one afternoon; zero training compute.

## 6. Consequence for R-014

R-014 (arms trial) registration is GATED on R-013's verdict: its A1 construct
declarations inherit whatever operationalization passes here. A FAIL here delays
R-014 by construct-redesign time and is cheaper than any alternative discovery
path (mid-trial construct failure = amendment-in-flight, the worst case).
