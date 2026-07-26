# R-033 DRAFT: Dir.E rule-level control at the bare rung (hebb vs delta)

Status: DRAFT, every row awaits RW yes/no/amend. Nothing implemented; the
patch in §6 is specified, not applied. Registry stands at R-032.

## 1. What this tests, and why it is not already answered

AOFF26 (07-23) established that on the recurrent substrate the delta rule
converges to a **near-perfect, exactly fixture-blind** predictor: the failure
is at the optimum, not short of it, because the endogenous map dominates the
predictable variance. "Predicting activity is not extracting world."
COMP25 generalised R-032 from "E adds nothing" to "E-learning actively
obscures" (vs-doff negative across the whole signed family, to -0.223 at
beta 2.0, where frozen-E readouts show +0.24).

Both findings are properties of a **least-squares** objective. A Hebbian
update has no least-squares objective and therefore no reason to land on that
fixed point: its attractor is correlational structure, not the minimum-error
self-model. The exact property that makes hebb a worse predictor, that it
never subtracts what it already predicts, is what would stop it collapsing
onto the self-model. Crosstalk is a defect against a clean target and
possibly a feature against a self-similar one.

**The control exists and was run in the wrong regime.** Free play, 07-20,
one-layer feedforward host: `dire_rule="hebb"` 0.80 vs `"delta"` 1.00 on the
cycle, cause named as crosstalk unlearning. That regime has no degeneracy to
avoid, so being a worse predictor is simply worse there and the result does
not transfer. At the bare rung the knob has never existed: `BareAgtX`
hardcodes the delta form, confirmed in code (`dire_hosts.py`, E update inside
the `BareAgtX` step; no `dire_rule` parameter on its `__init__`). The
registry contains no hebb-vs-delta comparison at any rung; every "hebb" hit
in REGISTRY.md is `softhebb`, which is the Dir.A rule.

Framing note: the AOFF26 design axes were (1) change the TARGET, (2) change
the DYNAMICS. This is neither. It keeps the target and drops the
least-squares criterion, so it is a third axis and should be named as one.

## 2. Claim under test

Against the registered fan3 constants, an error-free Hebbian Dir.E update
produces a kp1-kz differential that is less negative than learned-delta's,
i.e. it recovers some of the world-legibility that E-learning destroys.
Directional and modest by construction: the interesting outcome is the sign
of (hebb - delta), not whether hebb beats the frozen twin.

## 3. Constants: carried unchanged from the frozen fan3 page

No new constants. `fan3_frozen_constants.json` is reused as-is, so this arm
is scored against numbers frozen before it was conceived.

  family: p_a=0.3, p_e=0.6, dist_exp=1.0, input_fanout=3, rule=softhebb,
          transport=triangle, ss=0.01, lr_e=0.1
  horizon 12000 | window 500 | activity_floor 1.0 | n_seeds 32
  twin_diff_mean -0.053691 | twin_diff_sigma 0.011679
  diff_bar_gap 0.035036 | sign_min 20 | zero_norm_fraction 0.006248
  stream_ns r029, same 32 registered seeds

`lr_e=0.1` is carried unchanged deliberately. It was tuned for delta and is
almost certainly wrong for hebb, which has no error term to shrink its own
updates. Carrying it keeps this a rule swap rather than a rule-plus-tuning
comparison; a swept lr_e is a separate follow-up, not this run.

## 4. Arms

  fan3          delta E (the incumbent)              existing
  fan3_hebb     hebb E, all else identical           NEW, the subject
  fan3_twin     frozen twin, dynamics-null           existing, zero point
  fan3_direoff  E frozen at init, mechanism-null     existing

Matched design per ledger #40: fresh host per fixture, identical CRN graphs
per seed, the two nulls already carry per-seed diffs in the frozen page.
The only thing that varies between fan3 and fan3_hebb is the E update.

## 5. Reading, decided before the run

Primary: mean(hebb diff) - mean(delta diff) across the 32 registered seeds,
with the paired sign gate (each seed's hebb diff vs the SAME seed's delta
diff), sign_min 20 as registered.

  hebb - delta > 0, sign gate passes  -> the least-squares objective is
      implicated in the degeneracy. Promotes the objective axis alongside
      target and dynamics. Does NOT show hebb is a good rule.
  hebb - delta ~ 0                    -> degeneracy is not objective-specific;
      it is the substrate, and the design burden stays on target/dynamics.
      This is the outcome that most cleanly narrows the search.
  hebb - delta < 0                    -> the one-layer result transfers;
      error-free updating is worse everywhere; close the axis.

All three are informative. There is no result here that wastes the run,
which is the property to want given how cheap it is.

## 6. Patch, specified not applied

`BareAgtX.__init__` gains `dire_rule="delta"` stored as `self.dire_rule`,
mirroring `OneLayerDirEHost`. In its E update, branch:

    target = lo.nr_1.actual
    pre    = self.prev_actual[hi.loc]
    if self.dire_rule == "hebb":
        upd = torch.outer(pre, target)
    else:
        upd = torch.outer(pre, target - prev_expect[lo.loc])
    hi.conns[(lo.loc, Dir.E)] = w + self.lr_e * upd / (1 + pre @ pre)

Note the (pre, post) ordering is BareAgtX's own (i,o) convention and differs
from `OneLayerDirEHost`'s (o,i); the hebb translation is "replace the error
with the raw actual", not a transpose. Default `"delta"` leaves every
existing registered run bit-identical, which the drift gate must confirm
before the subject arm runs.

## 7. Pre-registered failure modes

**E-norm divergence is the live risk and it is not a differential failure.**
Hebb has no error term, so nothing bounds weight growth except the NLMS
denominator. Over 12000 recurrent steps this may run away where it did not
on a short feedforward fixture. Register now: if fan3_hebb's E-conn norms
grow monotonically past the sorted-conn-norm vital's runaway signature, the
arm is recorded FAILED ON VITALS and the differential is not read at all.
That is a finding about the rule, not a null result, and it must not be
salvaged by adding normalisation mid-run.

**Saturation.** If majority-sign rises toward the comp25 unsigned-gate
pathology (100% one-vs-rest, responds ~0.001), same disposition: vitals
failure, differential unread.

**Do not repair and re-register.** Per the R-023-family lesson, any fix
found after the subject arm starts makes this run an INCIDENT and the fixed
version is a new number. Smoke-guarded code is uncovered code: the smoke
path must exercise `dire_rule="hebb"` itself, not just the default.

## 8. Cost

One new arm over 32 seeds at horizon 12000, against constants and nulls that
already exist. No new fixtures, no new constants, no re-registration of the
frozen page. This is the cheapest open question on the board and it sits on
an axis the AOFF26 note did not name.
