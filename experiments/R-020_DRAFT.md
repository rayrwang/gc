# R-020 | DRAFT: registration of the Dir.E exam band-freeze (06 registered run)

**Type:** pre-registration: commissioning of the Dir.E exam instrument. Freezes
every constant the exam needs, from control runs on fresh seeds, before any
implementation is judged. The companion subject event (07 registered, R-021)
gets its own draft after this entry's verdict.
**Status:** DRAFT until RW line-item review (yes/no/amend per row); then
appended to REGISTRY.md and run once.
**Blindness:** no learner runs in this event. All free-play results to date
(runs of 2026-07-20, seeds 0/1/2) are development evidence and bind nothing.

## 1. Event

The 06 script at its then-current hash runs the control roster through the
five standard fixtures on the registered seeds: gate null `null:copy_last`,
chance anchor `anchor:chance`, and the three per-family frozen twins
(`frozen:pc`, `frozen:wake_sleep`, `frozen:gc_ff`). Its outputs freeze the
measured constants below; the entry's verdict is pass/fail on the sanity
expectations in section 5.

## 2. Seed policy

Eight fresh seeds, declared here, disjoint from the free-play seeds 0/1/2:

    101, 211, 307, 401, 503, 601, 701, 811

Rule: registered runs use these and only these; free play never touches them
again; any change is a new entry. (Eight = the R-013/R-015 precedent.)

## 3. Constants table (the rulings requested)

| # | constant | tag | proposed | notes |
|---|----------|-----|----------|-------|
| 1 | cycle period | structural | 4 | set in walkthrough |
| 2 | no consecutive repeats | structural | required | set in walkthrough |
| 3 | burn-in k | structural | 3 | set in walkthrough |
| 4 | symbol dim | structural | 32 | free-play value carried |
| 5 | fixture lengths | structural | kp/kz 400; sw 400+200+400; ts 6x200 | free-play values carried |
| 6 | settle fraction | structural | 0.10 | |
| 7 | settle window | structural | 50 | power check in section 4 |
| 8 | settle dwell | structural | 100 | |
| 9 | settle atol | structural | 0.02 | absolute tolerance floor |
| 10 | plateau floor | measured | one global number: max over all twin x fixture-class cells of (trailing-mean offset + 3 x seed-SD) | retires the free-play 0.1; suppress-not-fabricate asymmetry accepted (RW 07-20); sanity bound in E5 |
| 11 | activity floor | structural, v0 | 0 (inactive) | input symbols are never silent at this rung; the measured version binds at the BareAgt rung (08) and is registered there |
| 12 | beat-null margin | measured | bar = max(N-copy, own twin) trailing mean + 3 x pooled seed-SD; pass = above bar on every registered seed | finds-structure gate, kp0/kp1 |
| 13 | KZ band | measured | own twin's kz trailing mean, two-sided, +/- 3 x its seed-SD | leakage below, destabilization above |
| 14 | SW retention factor | structural | post0 >= 0.5 x pre | harm bound; twins exempt (no-plateau) |
| 15 | SW re-acquisition factor | structural | re-settle <= 2.0 x first settle | slower = burnout finding |
| 16 | magnitude term | deferred | cosine-only stands | predicate v2's open decision; closing it is a future prospective amendment |

Structural rows change only by prospective amendment before the run they
govern. Measured rows freeze from this event's outputs and are thereafter
numbers, not formulas.

## 4. Analysis plan

Judge = `dire_exam.judge` at the `experiments_hash` of this entry's stamp;
gate wording above is to be diffed against the gate code before the run
(R-017 lesson). Stream regime: prequential, test-then-train at every step, by
construction of the interface. Records through D3; the verdict derives from
`ledger_replay` only. Window power check, registered here: per-step scores
are cosines with observed per-step SD near 0.18, so a 50-sample trailing mean
has SE near 0.026 and the 3-sigma measured quantities in rows 10/12/13 sit
near 0.08: small against the unit score scale; window 50 is powered for
bands, and any band that fails to exclude its null is handled by row V below,
not by widening.

## 5. Registered expectations (the entry's own pass/fail)

- E1: every control's kz trailing mean lies inside its own family band
  (self-consistency of row 13's construction).
- E2: `null:copy_last` reads no-plateau on sw and ts (it never goes anywhere).
- E3: no measured band is vacuous: each family's kp1 bar (row 12) must sit
  below 0.9, else the exam cannot fail a competent learner and the fixture is
  underpowered: commissioning incident, redesign, never lower the bar.
- E4: no impossible pass: each family's kp1 bar must sit above the family
  twin's own trailing mean (the bar excludes its own null by construction).
- E5: the computed global plateau floor must not exceed 0.2; a floor above
  that means some twin's chance behavior is pathological: incident,
  investigate the twin, never inherit the number.
- V (standing): vacuous-pass = underpowered fixture = incident; impossible-
  pass = check nulls first (nulls broken = fixture bug; nulls fine = repair
  the mechanism). Charter section 7f extends to every constant here.

## 6. Determinism gate for R-021

CPU-deterministic stack. In the registered 07 event, every control arm must
reproduce this event's numbers bit-for-bit (same seeds, same code hashes);
any deviation is a reproducibility incident (charter section 13) and blocks
verdicts until diagnosed.

## 7. Cost

Both registered events are minutes of CPU. Zero training compute at stake;
the expensive thing being spent here is the right to call the next 07 run a
verdict.
