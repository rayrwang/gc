# D3 v0 RECORDER | DRAFT (candidate-free per the F2 ruling)

**Type:** spec for the D3 trace, detector, and event infrastructure, restricted
to what can exist before the existence observation. Plan Step 1b (document) and
Step 2 (build).
**Status:** DRAFT until RW line-by-line review.
**Scope rule (F2, RW 07-18):** don't build the container without first knowing
if there's anything to hold. v0 carries continuous traces and
candidate-agnostic detector events only. Everything candidate-shaped
(candidate envelope, candidate event vocabulary, lifecycle operators, the
physical-realization + playback decision) waits for the existence observation:
commissioned Dir.E running on the real substrate, traces showing individuable,
re-identifiable, behavior-bearing structure. The container is then fitted to
what was observed.

---

## 1. Three components, separably versioned

1. **Trace/artifact store**: immutable and content-addressed. Holds continuous
   neural state summaries, input slices, context prefixes, and later
   checkpoints. Artifacts are written once and referenced by hash; nothing is
   edited in place.
2. **Detectors**: map continuous traces to discrete events. Each detector is
   separately versioned and configuration-hashed; its windows, statistics,
   boundaries, and hysteresis live in its configuration, never in the neural
   substrate and never in an unversioned payload (Step-3a rule). A standing
   threshold-drift sentinel watches detector configurations.
3. **Event envelope**: append-only and versioned. Fields: schema version,
   run ID, event ID, detector identity + configuration hash, window and
   evidence, provenance stamp (charter §2), typed payload references (into the
   store, by hash), and lifecycle transitions (v0: incident state only, see
   §4).

Transport and reference rules are mechanism-independent and freeze against
synthetic fixtures (§6). Semantic payloads and detector configurations stay
provisional at `v0.x` until real Dir.E traces exist; then a prospective refit
and complete recommissioning are required before any trial freeze.

## 2. Identity and manifests

Every role or party handle is opaque and namespace-scoped to its fixture or
run. An execution manifest establishes artifact identity only, never
causal-role identity across runs. Manifest schema: checkpoint or parameter
hash, architecture and code hash, configuration, hierarchical RNG derivation,
numeric mode/backend, initialization mode, context references. Validation
rejects any initialization mode that has not passed its fixtures.

## 3. Trace set v0

The minimum is what the Dir.E exam requires recorded
(DIRE_CONTRACT_DRAFT.md §2): per step and per column of the declared readout
set: guess, actual, score, below-floor flag; the network aggregate; the
declared activity-movement statistic and the standard vitals (activity,
entropy, weight-movement summaries); fixture stream identity (which fixture,
which position in it); and the guess content-hash written before the actual
(the publication-order enforcement artifact). Plus the manifest (§2) per run.

Traces are continuous records. Nothing in the trace schema names candidates,
incumbents, successors, or lineage.

## 4. Event vocabulary v0 (candidate-agnostic)

Detector events that presuppose no candidates:

- **error-band exits**: score/error crossing a registered band, low side and
  high side separately (the KZ leakage and destabilization readings ride on
  these).
- **settled** and **settlement-lost**: the trailing-window criterion met or
  un-met (the TS readings ride on these).
- **tracking-lost**: nulls no longer beaten between switches at a given
  switch rate.
- **response events**: error-open with activity responsive vs error-open with
  activity inside the frozen band (the starvation-shaped reading; descriptive,
  per the F5 starvation ruling).
- **conflict opened**: parties are opaque externally supplied handles (fixture
  arms), with source recorded as `external` or `internal`, both first-class
  and unranked. Different activations alone are not a conflict; the conflict
  predicate direction is RW's cloud-separability predicate v2, formally
  adopted at refit on RW's review surface. Arity above two and mixed-source
  records are future schema extensions; a binary record is a v0 encoding
  convenience only.
- **resource deletion**: capacity-driven garbage collection events (the
  deleter can never read conflict state; that rule is commissioned in
  Step 3b, but the event type exists in v0).
- **incidents**: leakage, underpowered-fixture, reproducibility (charter §13
  class), each carrying evidence references.

Gated until the existence observation (do not add fields, types, or stubs for
these in v0): candidate generated, candidate rejected, incumbent flagged,
successor accepted, incumbent displaced, lineage archived.

## 5. Replay

`ledger_replay` (re-deriving verdict statistics from recorded events and
traces) and `execution_replay` (re-running the substrate from a manifest) are
distinct operations and never conflated in records. v0 must support
ledger_replay fully; execution_replay support is bounded by the manifest
schema (§2) and the determinism regime already in experiments/determinism.py:
CPU paths bit-for-bit, GPU paths under a commissioned numerical tolerance that
no verdict-relevant statistic may cross.

## 6. Fixtures (Step-2 freeze, all synthetic)

Serialization and round-trip; append-only behavior (rejected in-place edits);
referential-integrity sweeps (every reference resolves); content-address
verification; planted detector crossings; near-boundary jitter; hysteresis;
detector-version replay (old events re-derivable under their recorded
configuration); malformed events rejected; uncommissioned initialization
modes rejected; ledger_replay vs execution_replay distinguished in output.

## 7. Gate

Step-2 gate: transport fixtures and synthetic detector known answers pass;
all referenced artifacts resolve; no semantic payload or detector is
represented as frozen or commissioned against the live substrate. Nothing in
v0 is evidence about the thesis; this is plumbing, and reports must say so
(plan Step-5 gate language).
