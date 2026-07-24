"""
Machinery for the Dir.E commissioning exam (DIRE_CONTRACT_DRAFT.md).

Fixture stream generators, the two nulls, the scorer, and the interface any
Dir.E implementation must fit behind. No implementation lives here; the exam
runner (06_commissioning_dire.py) fails until one is registered in Step 3c.

Constants carry provenance tags per the contract's section 6: structural
conventions are set here; measured-from-controls values are None until the
null/control runs freeze them (a None band makes every pass verdict
impossible, so the exam cannot silently pass while unfrozen).
"""

import hashlib

import torch

# structural conventions (contract section 6, set now)
CYCLE_PERIOD = 4
BURN_IN_K = 3
SYMBOL_DIM = 32          # fixture symbol dimensionality, set once

# structural conventions not yet registered: None here; runs that need them
# must pass explicit values labeled provisional in their reports
SETTLE_FRACTION = None   # trailing-window mean within this fraction of plateau
SETTLE_WINDOW = None     # trailing-window length, steps
SETTLE_DWELL = None      # must stay settled this long, steps

# measured-from-controls (frozen only by registered null/control runs)
ACTIVITY_FLOOR = None
BEAT_NULL_MARGIN = None
KZ_BAND = None           # (low, high) on error, two-sided, from frozen-control spread
SW_RETENTION_FACTOR = None


class DirEInterface:
    """The contract interface. Both realization families implement exactly this.

    At step t the harness calls guess() before the actual is visible, then
    observe(actual). guess() returns a vector in the readout space.
    """

    def guess(self):
        raise NotImplementedError

    def observe(self, actual):
        raise NotImplementedError


class FrozenRandomNull(DirEInterface):
    """N-frozen: same-init, never-learning guesser (untrained dynamics)."""

    def __init__(self, dim=SYMBOL_DIM, seed=0):
        self.gen = torch.Generator().manual_seed(seed)
        self.dim = dim

    def guess(self):
        return torch.randn(self.dim, generator=self.gen)

    def observe(self, actual):
        pass


class CopyLastNull(DirEInterface):
    """N-copy: g_t = a_{t-1}. Kills persistence masquerading as prediction."""

    def __init__(self, dim=SYMBOL_DIM):
        self.last = torch.zeros(dim)

    def guess(self):
        return self.last.clone()

    def observe(self, actual):
        self.last = actual.clone()


def derive_seed(*parts):
    """hierarchical rng derivation (manifest schema): every role gets its own
    stream, so a fixture and a subject can never share one (the kz leakage
    class this exam exists to catch, first caught in its own harness)."""
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


def orthogonalish_symbols(n, dim=SYMBOL_DIM, seed=0):
    """n near-orthogonal unit symbols from a registered seed (gram-schmidt on gaussians)."""
    assert n <= dim, f"only {dim} orthogonal symbols exist in {dim} dims; use random_symbols"
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, dim, generator=g)
    q, _ = torch.linalg.qr(m.T)
    return q.T[:n].contiguous()


def random_symbols(n, dim=SYMBOL_DIM, seed=0):
    """n random unit symbols, no orthogonalization: linearly dependent past
    n = dim, which is the point (the rank probe)."""
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, dim, generator=g)
    return m / m.norm(dim=1, keepdim=True)


def kp0_constant(steps, dim=SYMBOL_DIM, seed=0):
    """rung 0: one fixed pattern every step."""
    sym = orthogonalish_symbols(1, dim, seed)[0]
    for _ in range(steps):
        yield sym


def kp1_cycle(steps, period=CYCLE_PERIOD, dim=SYMBOL_DIM, seed=0):
    """rung 1: deterministic cycle, no consecutive repeats by construction."""
    syms = orthogonalish_symbols(period, dim, seed)
    for t in range(steps):
        yield syms[t % period]


def kz_noise(steps, dim=SYMBOL_DIM, seed=0):
    """known-zero: a fresh iid pattern each step, nothing predictable."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        yield torch.randn(dim, generator=g)


def sw_sandwich(cycle_steps, noise_steps, dim=SYMBOL_DIM, seed=0):
    """cycle -> noise -> same cycle, with phase labels for the three readings."""
    for s in kp1_cycle(cycle_steps, dim=dim, seed=seed):
        yield "pre", s
    for s in kz_noise(noise_steps, dim=dim, seed=derive_seed("sw-noise", seed)):
        yield "noise", s
    for s in kp1_cycle(cycle_steps, dim=dim, seed=seed):
        yield "post", s


def cycle_random(steps, period, dim=SYMBOL_DIM, seed=0):
    """deterministic cycle over random non-orthogonal unit symbols (rank probe:
    memorization ceilings bite past the feature rank)."""
    syms = random_symbols(period, dim, seed)
    for t in range(steps):
        yield syms[t % period]


def revisit_cycle(steps, dim=SYMBOL_DIM, seed=0):
    """a -> b -> a -> c: what follows `a` depends on phase. two of the four
    transitions (b->a, c->a) are deterministic; the two after-a steps are
    ambiguous for a memoryless predictor (best guess spans b and c, cosine
    0.707 each), so the memoryless ceiling is (1+1+0.707+0.707)/4 = 0.854 on
    orthogonal symbols, while stateful predictors can reach 1. a
    family-membership test, characterization only."""
    a, b, c = orthogonalish_symbols(3, dim, seed)
    pattern = [a, b, a, c]
    for t in range(steps):
        yield pattern[t % 4]


def ts_switch(n_cycles, steps_per_cycle, dim=SYMBOL_DIM, seed=0):
    """cycle -> new cycle -> ... for settling-time and tracking-bandwidth reads."""
    for c in range(n_cycles):
        for s in kp1_cycle(steps_per_cycle, dim=dim, seed=derive_seed("ts-cycle", seed, c)):
            yield c, s


def score(guess, actual, floor=None):
    """cosine with activity floor; below-floor steps return None (recorded, excluded)."""
    a = actual.flatten().float()
    if floor is not None and a.norm() < floor:
        return None
    g = guess.flatten().float()
    denom = g.norm() * a.norm()
    if denom == 0:
        return 0.0
    return float((g @ a) / denom)


def publication_hash(guess):
    """content-hash a guess before the actual is computed (publication-order artifact)."""
    return hashlib.sha256(guess.numpy().tobytes()).hexdigest()


def run_stream(impl, stream, sink, floor=None, fixture="", subject="", seed=0):
    """drive one implementation through one labeled stream, emitting the full
    per-step record (the contract's trace set; also D3 v0's schema) into sink."""
    for step, item in enumerate(stream):
        label, actual = item if isinstance(item, tuple) else (None, item)
        g = impl.guess()
        h = publication_hash(g)
        s = score(g, actual, floor)
        sink({
            "fixture": fixture, "label": label, "step": step,
            "subject": subject, "seed": seed,
            "guess": g.tolist(), "actual": actual.tolist(),
            "guess_hash": h, "score": s, "below_floor": s is None,
        })
        impl.observe(actual)


def run_stream_multi(impl, stream, sink, floor=None, fixture="", subject="", seed=0,
                     tap_sink=None):
    """drive a multi-column implementation (guesses()/actuals() dicts keyed by
    col loc) through one labeled stream. tick alignment: guesses emitted after
    step t target the activity that materializes at step t+1; hashes are taken
    at emission (publication order). tap_sink, when given and the impl exposes
    taps(), receives one record per per-sender pre-sum contribution: raw
    material, unhashed, kept out of the score stream (route it to its own
    chunk namespace, e.g. fixture + ".tap", so judge replay never sees it);
    the taps emitted at step t assemble the guesses emitted at step t."""
    g_prev, h_prev = None, None
    for step, item in enumerate(stream):
        label, a = item if isinstance(item, tuple) else (None, item)
        impl.observe(a)
        if tap_sink is not None and getattr(impl, "taps", None) is not None:
            for tp in impl.taps() or []:
                tap_sink({"kind": "tap", "fixture": fixture + ".tap",
                          "subject": subject, "seed": seed, "step": step, **tp})
        acts = impl.actuals()
        if g_prev is not None:
            for col, g in g_prev.items():
                sc = score(g, acts[col], floor)
                sink({
                    "fixture": fixture, "label": label, "step": step,
                    "subject": subject, "seed": seed,
                    "column": f"{col[0]},{col[1]}",
                    "guess": g.tolist(), "actual": acts[col].tolist(),
                    "guess_hash": h_prev[col], "score": sc,
                    "below_floor": sc is None,
                })
        g_prev = impl.guesses()
        h_prev = {col: publication_hash(g) for col, g in g_prev.items()}


def _netify(records):
    """split columnar records out and synthesize per-step network-aggregate
    records (mean score over columns: the contract's network-wide reading),
    so every existing judge section reads multi-column subjects unchanged."""
    records = [r for r in records if r.get("kind") != "tap"]  # raw material,
    # never score-shaped: a replay that loads a run holding tap chunks judges
    # the same stream it would have without them
    plain = [r for r in records if "column" not in r]
    colrecs = [r for r in records if "column" in r]
    if not colrecs:
        return records, []
    acc = {}
    for r in colrecs:
        if r["score"] is not None:
            key = (r["fixture"], r["label"], r["subject"], r["seed"], r["step"])
            acc.setdefault(key, []).append(r["score"])
    net = [{"fixture": f, "label": lb, "subject": s, "seed": sd, "step": st,
            "score": sum(v) / len(v), "below_floor": False, "guess_hash": None}
           for (f, lb, s, sd, st), v in sorted(acc.items(), key=lambda kv: kv[0][4])]
    return plain + net, colrecs


def column_tables(colrecs, window):
    """per-column trailing means plus the persistence baseline (cosine of a
    column's activity with its own next activity, from the recorded actuals):
    the derived copy-last null, and the autocorrelation reading that decides
    which columns iid input is even iid at."""
    out = {}
    groups = {}
    for r in colrecs:
        groups.setdefault((r["fixture"], r["subject"], r["column"], r["seed"]), []).append(r)
    for (fixture, subject, column, seed), rs in groups.items():
        rs.sort(key=lambda r: r["step"])
        xs = [r["score"] for r in rs if r["score"] is not None]
        acts = [torch.tensor(r["actual"]) for r in rs]
        persist = []
        for a, b in zip(acts[:-1], acts[1:]):
            d = a.norm() * b.norm()
            persist.append(float(a @ b / d) if d > 0 else 0.0)
        cell = out.setdefault((fixture, subject, column), {"trail": [], "persist": []})
        cell["trail"].append(sum(xs[-window:]) / window if len(xs) >= window else None)
        cell["persist"].append(sum(persist[-window:]) / window if len(persist) >= window else None)
    return {k: {"trail": _mean(v["trail"]), "persist": _mean(v["persist"])}
            for k, v in out.items()}


def settled_at(scores, fraction, window, dwell, atol, plateau_floor):
    """first step where the trailing-window mean is within tolerance of the
    final plateau and stays there for dwell steps; None if never or if no
    plateau exists (|plateau| < plateau_floor: settling is undefined for a
    subject that never went anywhere). tolerance = max(fraction * |plateau|,
    atol): the absolute floor keeps a near-zero plateau from demanding
    impossible closeness. parameters must be passed explicitly: they are
    unregistered structural conventions until the freeze entry exists."""
    xs = [s for s in scores if s is not None]
    if len(xs) < window + dwell:
        return None
    plateau = sum(xs[-window:]) / window
    if abs(plateau) < plateau_floor:
        return None
    tol = max(abs(plateau) * fraction, atol)
    run = 0
    for i in range(window, len(xs)):
        mean = sum(xs[i - window:i]) / window
        if abs(mean - plateau) <= tol:
            run += 1
            if run >= dwell:
                return i - dwell + 1
        else:
            run = 0
    return None


def drive_fixtures(subjects, sink, seeds, fixtures, floor=None, seed_of=None):
    """the exam's front half: every subject through every fixture at every seed.
    subjects: {name: factory(dim, seed) -> DirEInterface};
    fixtures: [(name, factory(seed) -> stream)]. fresh instance per run.
    seed_of(name, seed) overrides the per-subject seed derivation (default =
    namespaced derive_seed): events that need two subjects to share draws,
    like a bitwise shadow pair, declare it explicitly there."""
    if seed_of is None:
        seed_of = lambda name, seed: derive_seed("subject", name, seed)
    for seed in seeds:
        for name, make in subjects.items():
            for fixture, stream_of in fixtures:
                impl = make(SYMBOL_DIM, seed_of(name, seed))
                runner = run_stream_multi if hasattr(impl, "guesses") else run_stream
                runner(impl, stream_of(seed), sink,
                       floor=floor, fixture=fixture, subject=name, seed=seed)


def standard_fixtures(n_steps, sw_cycle, sw_noise, ts_cycles, ts_steps):
    """the five fixtures of the registered exam, in ladder order."""
    return [
        ("kp0", lambda seed: kp0_constant(n_steps, seed=seed)),
        ("kp1", lambda seed: kp1_cycle(n_steps, seed=seed)),
        ("kz", lambda seed: kz_noise(n_steps, seed=seed)),
        ("sw", lambda seed: sw_sandwich(sw_cycle, sw_noise, seed=seed)),
        ("ts", lambda seed: ts_switch(ts_cycles, ts_steps, seed=seed)),
    ]


def drive(subjects, sink, seeds, n_steps, sw_cycle, sw_noise, ts_cycles, ts_steps, floor=None):
    """standard-exam entry point, byte-compatible with the 06 era."""
    drive_fixtures(subjects, sink, seeds,
                   standard_fixtures(n_steps, sw_cycle, sw_noise, ts_cycles, ts_steps),
                   floor=floor)


def _scores(records, fixture, subject, seed, label=None):
    return [r["score"] for r in records
            if r["fixture"] == fixture and r["subject"] == subject
            and r["seed"] == seed and (label is None or r["label"] == label)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _settle_label(scores, settle):
    """settled step, or why not: 'no-plateau' (subject never went anywhere)
    vs 'never' (went somewhere, never met the criterion)."""
    t = settled_at(scores, **settle)
    if t is not None:
        return t
    xs = [x for x in scores if x is not None]
    w = settle["window"]
    if len(xs) >= w:
        plateau = sum(xs[-w:]) / w
        if abs(plateau) < settle["plateau_floor"]:
            return "no-plateau"
    return "never"


def judge(records, settle, constants=None):
    """the exam's back half: verdicts from the record only. settle = dict of
    provisional settled-criterion parameters (labeled provisional until
    registered). returns a report dict; never touches a live run. multi-column
    records are folded to their network aggregate for every standard section;
    per-column tables ride in report["columns"]."""
    constants = constants or {}
    records, colrecs = _netify(records)
    subjects = sorted({r["subject"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    w = settle["window"]
    report = {"subjects": subjects, "seeds": seeds, "fixtures": {}, "readings": {}, "verdicts": {}}
    if colrecs:
        report["columns"] = column_tables(colrecs, w)

    plain = sorted({r["fixture"] for r in records} - {"sw", "ts"})
    for fixture in plain:
        rows = {}
        for name in subjects:
            per_seed = []
            for seed in seeds:
                xs = [x for x in _scores(records, fixture, name, seed) if x is not None]
                per_seed.append({"seed": seed, "n": len(xs), "trailing_mean": _mean(xs[-w:])})
            rows[name] = per_seed
        report["fixtures"][fixture] = rows

    # sw readings: retention (pre trailing vs post immediate) + re-acquisition times
    sw = {}
    for name in subjects:
        per_seed = []
        for seed in seeds:
            pre = [x for x in _scores(records, "sw", name, seed, "pre") if x is not None]
            post = [x for x in _scores(records, "sw", name, seed, "post") if x is not None]
            per_seed.append({
                "seed": seed,
                "pre_trailing": _mean(pre[-w:]),
                "post_immediate": _mean(post[:w]),
                "settle_pre": _settle_label(pre, settle),
                "settle_post": _settle_label(post, settle),
            })
        sw[name] = per_seed
    report["readings"]["sw"] = sw

    # ts readings: settling time per cycle segment; T_switch = mean over post-switch segments
    ts = {}
    for name in subjects:
        per_seed = []
        for seed in seeds:
            segs = {}
            for r in records:
                if r["fixture"] == "ts" and r["subject"] == name and r["seed"] == seed:
                    segs.setdefault(r["label"], []).append(r["score"])
            times = {c: _settle_label(xs, settle) for c, xs in sorted(segs.items())}
            switch_times = [t for c, t in times.items() if c > 0 and isinstance(t, int)]
            per_seed.append({"seed": seed, "settle_per_cycle": times,
                             "t_switch_mean": _mean(switch_times)})
        ts[name] = per_seed
    report["readings"]["ts"] = ts

    # verdicts: every promise refuses until its constants freeze and an
    # implementation exists; the refusal wording is the point
    has_impl = any(not n.startswith("null:") for n in subjects)
    unfrozen = [k for k in ("ACTIVITY_FLOOR", "BEAT_NULL_MARGIN", "KZ_BAND", "SW_RETENTION_FACTOR")
                if constants.get(k) is None]
    report["verdicts"] = {
        "finds-structure": f"no verdict: unfrozen {unfrozen}" if unfrozen else "ready",
        "stays-honest": f"no verdict: unfrozen {unfrozen}" if unfrozen else "ready",
        "responds": "no verdict: activity-movement statistic undeclared, no implementation",
        "has-a-measurable-timescale": "characterization only: settle parameters provisional",
        "overall": "PASS candidate" if has_impl and not unfrozen
                   else "FAIL: no implementation registered" + ("" if not unfrozen else f"; constants unfrozen {unfrozen}"),
    }
    return report


# bar application (R-023 judge extension): primitives comparing recorded
# readings to frozen constants. event scripts wire these to their registered
# roster roles; nothing here decides which subject faces which gate.

def _trailing_by_seed(records, fixture, subject, window, label=None):
    out = {}
    for seed in sorted({r["seed"] for r in records}):
        xs = [r["score"] for r in records
              if r["fixture"] == fixture and r["subject"] == subject
              and r["seed"] == seed and (label is None or r["label"] == label)
              and r["score"] is not None]
        out[seed] = sum(xs[-window:]) / window if len(xs) >= window else None
    return out

def above_bar(records, subject, fixture, bar, window):
    """finds-structure gate: trailing mean above bar on every seed."""
    means = _trailing_by_seed(records, fixture, subject, window)
    ok = all(m is not None and m > bar for m in means.values())
    return ok, means

def in_band(records, subject, band, window):
    """kz honesty gate: trailing mean inside [lo, hi] on every seed."""
    lo, hi = band
    means = _trailing_by_seed(records, "kz", subject, window)
    ok = all(m is not None and lo <= m <= hi for m in means.values())
    return ok, means

def sw_gates(records, subject, retention_factor, reacq_factor, settle):
    """sandwich gates, per seed: retention post0 >= factor x pre trailing;
    re-acquisition settle_post <= factor x settle_pre. non-numeric settling
    on the known-positive phases fails the gate (contract: indefinite
    non-settling on a known-positive is a commissioning failure)."""
    w = settle["window"]
    per_seed, ok = {}, True
    for seed in sorted({r["seed"] for r in records}):
        pre = [r["score"] for r in records if r["fixture"] == "sw" and r["subject"] == subject
               and r["seed"] == seed and r["label"] == "pre" and r["score"] is not None]
        post = [r["score"] for r in records if r["fixture"] == "sw" and r["subject"] == subject
                and r["seed"] == seed and r["label"] == "post" and r["score"] is not None]
        pre_trail = sum(pre[-w:]) / w
        post0 = sum(post[:w]) / w
        s_pre, s_post = _settle_label(pre, settle), _settle_label(post, settle)
        retention = post0 >= retention_factor * pre_trail
        reacq = isinstance(s_pre, int) and isinstance(s_post, int) and s_post <= reacq_factor * s_pre
        per_seed[seed] = {"pre": round(pre_trail, 4), "post0": round(post0, 4),
                          "settle_pre": s_pre, "settle_post": s_post,
                          "retention": retention, "reacq": reacq}
        ok = ok and retention and reacq
    return ok, per_seed

def drift_identical(records_a, records_b, subject, fixtures):
    """reproducibility gate: score and guess-hash sequences bit-identical
    between two runs for one subject over the named fixtures."""
    for fixture in fixtures:
        for seed in sorted({r["seed"] for r in records_a if r["subject"] == subject}):
            def seq(rs):
                return [(r["score"], r["guess_hash"]) for r in rs
                        if r["fixture"] == fixture and r["subject"] == subject
                        and r["seed"] == seed]
            if seq(records_a) != seq(records_b):
                return False, (fixture, seed)
    return True, None
