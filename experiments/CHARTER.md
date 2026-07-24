# CHARTER: experimental discipline for the measurement battery

This file is the **living half**: freely revisable, current text is normative.
Immutability lives elsewhere: the CHANGELOG at the bottom of this file (append-only),
REGISTRY.md (append-only), and content-hash stamps on artifacts. Since this file is
untracked (journal-status), the charter version = the version number + content hash
recorded in the CHANGELOG; provenance stamps carry that hash, so every verdict
permanently points at the exact rule-text it ran under even as this text moves.

Scope (§15): this charter constrains how things **count**, not what gets built.
Instrument specs and instrument-specific registered readings live in REGISTRY.md.
No dates appear in this document; the build order bends to correctness, not calendar.

## §1 Two halves and a log
- **Living half**: this file + instrument spec notes. Edit freely.
- **Append-only**: the CHANGELOG below (every edit: date, delta, why, and a
  flag: did any live or adverse verdict motivate this change?) and REGISTRY.md.
- **Run identity = the registered contrast.** A run addressing an already-registered
  contrast/primary outcome is a **revisit**, regardless of implementation changes.
  Revisits link the prior entry and state the delta and reason. Prior verdicts are
  never deleted. Any claim citing a revisited contrast discloses the full lineage
  ("run 3x: loss, loss, win"). Revising an instrument after it ruled against an arm
  requires commissioning-grade justification (shown broken on known controls), else
  the entry carries a permanent `verdict-motivated` flag.

## §2 Provenance or it isn't evidence
Every artifact (plot, JSONL, table) carries: seed, git SHA of the code, run ID,
step counter, charter version+hash — stamped mechanically (experiments/provenance.py).
Unstamped numbers may not be cited in any writeup, including informal ones.

## §3 Register before running
Before any run that will be cited: the contrast (which arms), exactly **one primary
outcome per instrument**, kill/pass thresholds, and the analysis plan go to
REGISTRY.md. Competence-instrument registrations must state **which construct**
they read (**content**: fresh-probe decodable; **drift**: movement under a
frozen probe; **behavior**: the arm's own outputs) and why; the three
dissociate (established by R-006/R-008), and an undeclared construct is how an
instrument silently measures the wrong thing. Registered runs run on
**committed substrate** where practical
(commit first; it is free); when not, the dirty diff is captured (hash in the
stamp, patch in the run dir) and the run is reconstructible as SHA + patch. No
re-running for stamp cosmetics: if the same code is later committed, equivalence
(old SHA + patch == new commit) is checkable and recordable; a rerun adds nothing
under determinism, and a rerun that differs is a reproducibility incident (§13). Post-hoc analyses are permitted, permanently labeled `post-hoc`;
they may motivate a claim, never settle one.

## §4 Hands-off after t0
Once a run starts: no tweaks, no vibes-based early stops, no peek-then-adjust.
**Looking is allowed, touching voids**: read-only observation (debugger, vitals)
is encouraged; any write (parameter poke, restart, schedule nudge) voids the run
(new run ID). Infra crashes void the run: logged, never repaired in place.
Enforcement is architectural: run processes expose read-only telemetry and no
intervention write-path exists.

## §5 Debugging vs tuning; parity
- **Debugging-to-spec** (mechanism does what its spec says: cull fires on
  contradiction, no NaNs, ledger records) is unlimited and always free, but
  **verdict-blind**: fixes are judged against unit tests/spec behavior, never
  against the primary outcome. The discriminator is *what number you consulted*
  to decide the change worked.
- **Tuning** (config search judged by the primary outcome) is budgeted: parity
  binds **after registration**: verdict-guided configuration evaluations are
  counted per arm and reported with results. Before registration: free play.
- **Maturity disclosure**: reports disclose each arm's maturity
  ("Dir.E v0, ~2 weeks; SGD baseline, standard recipe"). Parity governs what we
  did; disclosure covers what history did.

## §6 Cost reporting
Per-step / per-sample is the default normalization. Each arm gets a **one-time
cost card** (measured FLOPs-per-step constant) so any reader can convert;
no per-run FLOP accounting. Wall-clock may be logged as an informal vital.

## §7 Cheating taxonomy: any of these invalidates a verdict
a. **Task-boundary leakage**: task IDs as input; optimizer/plasticity resets or any
   schedule keyed to task switches. (The mechanism *detecting* change itself is the
   phenomenon under study, not cheating — the rule bars *experimenter-supplied*
   boundary information.)
b. **Privileged internal state**: reading anything outside the declared input
   channel (true env state instead of observations; probe labels or probe accuracy;
   the curriculum's future).
c. **Protocol-informed configuration**: arm settings justified by the *test's*
   structure rather than the task (memory decay set to outlast a known washout
   length; representation sized to what the probe rewards; matched delay queues).
d. **Probe leakage**: probes are read-only linear readouts on frozen
   representations; probe gradients never reach the arm; the probe recipe is fixed
   globally, never tuned per-arm.
e. **Data-dependent analysis windows**: washout tails and run lengths are set at
   registration; no ending a window early after peeking.
f. **Metric swaps after first data**: primary outcome locked at registration;
   friendlier metrics discovered later are `post-hoc`.

## §8 Stream and evaluation regime
Single-pass does not by itself make a stream held-out. Every registry entry
states the order in which each scored observation is handled:

- **Prequential (test-then-train):** compute and record the locked score before
  any update from that observation, then optionally learn from it. The
  observation is unseen when scored but becomes training data afterward; it is
  not permanently held-out.
- **Train-then-score:** any score recorded after learning from the observation is
  a training diagnostic, not evidence of generalization to unseen data.
- **Held-out:** the observation or task family is never used for parameter
  updates, design selection, threshold choice, or tuning before the registered
  evaluation is complete.

Replayed or recycled observations count as fresh prequential events only on
their first encounter. Later scores on them are training diagnostics. A final
confirmatory claim must use a preregistered sealed prequential stream or task
family not used during development, or a separate held-out evaluation set.
Reusing a stream, task family, generator seed, or its results across design
iterations makes it development evidence even when each run scores
test-then-train. The registry records the regime, score/update order, and prior
development use.

## §9 Comparisons only — the three nulls
No absolute claims. Every reported result is arm-vs-arm or arm-vs-null:
- **zero-retention null** (backprop, no replay): kills fake savings;
- **no-history null** (scratch learner): kills fake rent;
- **raw-input / random-projection null**: kills fake representation learning
  (88% on MNIST sounds impressive until raw pixels probe the same).

## §10 Weights are for watching, behavior is for judging
Verdict instruments (Tier-2 A/B/C/D families) read competence profiles only.
Diagnostics/vitals (Tier-1: conn-norm curves, distribution shape, plasticity/
movement) read weights freely but **inform, never adjudicate**. Corollary (the
Othello caveat): probe-decodability of structure is not evidence the system
*uses* it.

## §11 Verdict language
Nulls are findings, reported at equal prominence to wins. Reports count
**contradiction events** (binary, per-test); a sentence ranking conjectures on a
continuous goodness scale is invalid. Instrument-specific registered readings
(e.g. C1's) live in their registry entries, reviewed at registration time.

## §12 Schedule rules
Tactical (hour-to-hour) order changes happen only by the three pre-registered
reorders:
1. **Commissioning failure** (instrument cannot separate known-different arms) =>
   instrument redesign preempts everything.
2. **Preconditions-gate failure** => substrate-dynamics work is legitimately pulled
   forward, the one pre-registered case where substrate work jumps the queue
   (distinguishing mandated substrate work from avoidance).
3. **A candidate stabilizes** => D2 jumps the queue.
The **big-picture plan** (adding / deleting / merging instruments and build items)
is revisable at will, with a CHANGELOG entry stating what and why.

## §13 Truth-channel incidents
Instruments (probes, ledger, vitals) are engineering with their own failure log.
A misbehaving instrument is an incident report, never a silent re-run.

## §14 Two stopping registers
Motivational stopping (bored, done, life) is sovereign and owed to no one.
The thesis falsifier is a separate registered sentence (REGISTRY.md, standing
entries). Quitting and being-refuted are different events; records never blur them.

---

## CHANGELOG (append-only)
- **v6** (2026-07-19): style-only pass on the living half, applying the new
  project-wide writing rules (RW, 07-18/19): no all-caps emphasis, no em dashes
  or prose double hyphens, no filler-register words; Paul Graham "Write Simply"
  adopted as the prose norm. Concretely: title and section dashes became colons;
  "WHICH CONSTRUCT" became bold lowercase; "an honest flag" became "a flag"
  (the flag's question is unchanged and carries the content); "state each arm's
  maturity honestly" became "disclose each arm's maturity". No rule semantics,
  thresholds, procedures, or section numbering changed; no verdict is affected.
  Prior CHANGELOG entries left byte-identical per append-only. The codebase-wide
  application is in git (5482891, df45b20); REGISTRY.md R-019 records the
  discontinuity so pre-07-19 typography in frozen documents reads as historical.
  Motivating verdicts: none (style, not evidence).
- **v5** (2026-07-11) — §8 replaces the false equation of single-pass online
  data with held-out validation. It distinguishes prequential test-then-train,
  train-then-score diagnostics, and genuinely held-out evaluation; requires
  score/update order and prior development use in registrations; and requires
  sealed evaluation for confirmatory claims. The correction arose from the
  full-chain survey/instrument review, not from an adverse treatment verdict.
  R-012 and R-016 remain commissioned on their named controls; no issued result
  or threshold changed.
- **v4** (2026-07-08) — §3: construct declaration (content/drift/behavior)
  required in competence-instrument registrations; motivated by the R-006/R-008
  commissioning failures (construct errors, both directions; adverse verdicts
  against the INSTRUMENT motivated this, none existed against arms).
  Taxonomy revision (big-picture, §12): M-family dissolved (M1=D2-v0, M2=rider);
  B3 folded into B1 as a stressor type; C3 delisted (absorbed into this charter);
  washout tails moved from C1's definition to the shared curriculum toolkit;
  TRANSFER instrument named and PARKED (capability-flank, out of scope per the
  all-in-on-goal-genesis decision); C1 gains a registered-future reading:
  convergence RATE of history-contingency (the point where the battery touches
  Deutsch's stated crux from below); D3 schema must distinguish death-by-
  contradiction from death-by-displacement (the synthesis-vs-cull fork).
  Tier structure restated: Charter -> vitals + PRECONDITIONS GATE (old-3
  detector, S12.2 reorder) -> A/B/C/D.
- **v3** (2026-07-07) — housekeeping rename: battery/ -> experiments/; CHARTER.md
  and REGISTRY.md moved inside experiments/; experiment scripts numbered like
  examples/ (00_smoke.py); stamp key battery_hash -> experiments_hash.
  No rule changes; no verdicts existed.
- **v2** (2026-07-07) — §3: commit-before-registered-runs preference; dirty runs
  reconstructible via captured diff (provenance diff_hash + substrate.diff);
  no-rerun-for-stamp-cosmetics rule. Motivated by Raymond's question about
  uncommitted experiment code; no verdicts existed.
- **v1** (2026-07-07) — initial charter, adopted after point-by-point review.
  Raymond's edits incorporated: two-halves architecture (§1); looking-vs-touching
  (§4); debugging/tuning split + post-registration parity + maturity disclosure
  (§5); FLOP demoted to one-time cost card (§6); single-pass caveat (§8);
  raw-input third null (§9); weights-for-watching carve-out (§10); C1 reading
  deferred to registry (§11); big-picture plan revisability (§12).
  Motivating verdicts: none (no runs exist yet).
