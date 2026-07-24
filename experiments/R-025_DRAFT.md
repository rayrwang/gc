# R-025 DRAFT: BareAgt-rung band-freeze registration (freeze page)

Status: DRAFT, every row awaits RW yes/no/amend. Style per R-020: one row per
constant, provenance tag on each. Registers the bare-rung pair's bands and
constructions; the subject event (R-026) runs against these frozen numbers.

Provenance corpus: free play 09 (first contact, convergence probe), 10 (216-
config sweep, ledger kinds sweep-screen/confirm), 11 (instar replication +
controls, kind rep11), 12 (horizon curve, kind hz12). All free-play seeds
0-9; all CPU numerics.

## Rows

1. Subjects. bare:stock (BareAgtDirE, bit-faithful wrapper) and bare:x
   faithful (BareAgtXC softhebb+triangle ss=1e-2 lr_e=0.1), with per-seed
   controls on identical CRN graphs: x frozen twin, x dire-off.
   [carried from the 09 roster; stock keeps the fidelity-anchor role]

2. n_cols = 49, d_col = 32, symbol dim = 32. [carried-forward free-play
   values, no walkthrough pedigree; same honesty tag as R-020 rows 4-5]

3. Registered horizon = 12000 steps for kp1/kz. [measured: hz12 gap curve
   +0.021 @3000, +0.098 @6000, +0.124 @12000, +0.106 @24000; smallest
   horizon after which the gap stops growing = 12000]

4. Fixture lengths beyond kp1/kz: sw = 12000 + 6000 noise + 12000 re-cycle;
   ts = 6 x 2000. [convention, scaled from the feedforward shapes; set once]

5. Lead differential bar (the finds-structure analogue): mean kp1-kz
   differential minus frozen-twin differential >= 3 sigma of the twin's
   per-seed differential spread, sigma measured in this registered run, and
   sign-positive on >= 7 of 8 registered seeds. [construction registered
   now, numbers frozen from the registered run's own controls, R-022
   pattern; free-play reference: gap +0.124, twin sigma ~0.027]

6. Frozen-twin offset is the zero point everywhere: no raw-zero readings.
   [measured: twin offset -0.041 to -0.042, horizon-flat, hz12]

7. Activity floor (deferred at R-020 row 11, binds here): scoring exclusion;
   a column-step with actual norm below floor records but leaves the
   aggregate. Construction: floor = 1st percentile of NONZERO per-step
   actual norms pooled over the frozen-twin registered runs, one global
   number; the zero-norm fraction is recorded alongside. [construction
   registered now, number frozen at t0. AMENDED pre-registration: smoke
   caught that structural zeros (unreachable columns) form a point mass
   that pins any low percentile to exactly 0; zeros are already excluded
   from scoring by the cosine zero-norm guard, so the floor calibrates on
   the living distribution]

8. Responds statistic (RW declaration pending): proposal: at each ts
   switch, mean absolute change in per-column trailing activity norm
   (window 500, all columns, host-read), averaged over switches, exceeds
   the frozen twin's same statistic by 3 sigma of the twin's seed spread.
   [needs RW ruling: declared activity-movement reading, not derivable.
   TRIALED in free play (13_responds_trial, kind resp13, seeds 0-9):
   faithful 0.0430 +/- 0.0099 vs twin 0.0023 +/- 0.0020; bar at twin +
   3 sigma = 0.0083; every faithful seed clears it individually, worst
   seed 0.0289; ~18x mean separation. the construction behaves.]

9. Seed block: 8 fresh seeds drawn at t0, disjoint from free-play 0-9 and
   from feedforward's 101-811; free play banned from touching them.
   [anti-ceremony guarantee, R-020 pattern]

10. Boundedness gate: any NaN or activity explosion in a registered run =
    that subject fails, no reruns. [standard]

11. T_switch export: settled-at on the registered ts subject, exported as
    Step 4's burn-in prefix (closes 3c). [carried from the staging ruling]

12. Magnitude term: stays deferred pending predicate v2 adoption; cosine
    only. [unchanged]

## Scoped observations carried, not bars

- Long-horizon drift: at 24000 the faithful sign count degrades to 6/10
  with high per-seed variance (some graphs +0.18..+0.23, some -0.06..-0.09)
  while the twin stays flat. Recorded as an open question (imprint
  stability), candidate future fixture; deliberately outside this event's
  bars.
- instar+relu family: positive differential where it survives but NaNs on
  half of graphs (Dir.A-side, dire-off shares the explosions). Excluded
  from subjects; revisit only behind a stabilization fix.
- Screen blind spot: 400-step viability does not predict 3000+ step
  boundedness (lr_e=0.3 relu/raw NaNs). Reflected in row 10.
