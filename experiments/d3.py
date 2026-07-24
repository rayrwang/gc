"""
D3 v0 first slice (D3_RECORDER_SPEC_DRAFT.md): the recorder.

Content-addressed write-once store + run manifests + append-only ledger +
read/verify. Sized to what the Dir.E exam emits; detectors, richer event
vocabulary, execution_replay, and everything candidate-shaped are deferred
(spec sections 4-5; F2 gate).

Layout under a root directory:
    objects/ab/abcdef....json.gz   write-once objects, keyed by sha256 of
                                   canonical json bytes (hashed uncompressed)
    ledger.jsonl                   append-only, ordered; entries point into
                                   objects/ by hash
"""

import gzip
import hashlib
import json
import os

SCHEMA_VERSION = "d3-0.1"


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class D3Store:
    def __init__(self, root):
        self.root = root
        self.objects = os.path.join(root, "objects")
        self.ledger_path = os.path.join(root, "ledger.jsonl")
        os.makedirs(self.objects, exist_ok=True)

    def _path(self, h):
        return os.path.join(self.objects, h[:2], h + ".json.gz")

    def put(self, obj):
        """store an object, return its hash. write-once: an existing hash is
        left untouched (same hash = same canonical bytes, by keying)."""
        b = canonical_bytes(obj)
        h = hashlib.sha256(b).hexdigest()
        path = self._path(h)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with gzip.open(tmp, "wb") as f:
                f.write(b)
            os.rename(tmp, path)  # atomic: no partially-written object is ever visible
        return h

    def get(self, h):
        with gzip.open(self._path(h), "rb") as f:
            return json.loads(f.read())

    def check(self, h):
        """re-hash stored bytes against the key (tamper/corruption detection)."""
        try:
            with gzip.open(self._path(h), "rb") as f:
                return hashlib.sha256(f.read()).hexdigest() == h
        except OSError:
            return False

    def append_ledger(self, entry):
        """append one envelope entry. no rewrite api exists anywhere in this
        class: the ledger only grows."""
        entry = dict(entry, schema_version=SCHEMA_VERSION)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def ledger(self):
        if not os.path.exists(self.ledger_path):
            return []
        with open(self.ledger_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def find_run(self, run_id):
        for e in self.ledger():
            if e.get("kind") == "run-recorded" and e.get("run_id") == run_id:
                return e
        raise KeyError(f"run {run_id} not in ledger")

    def load_run(self, run_id):
        """ledger_replay read side: reconstruct the exact record stream the
        recorder was handed, in original order."""
        manifest = self.get(self.find_run(run_id)["manifest"])
        records = []
        for _key, h in manifest["chunks"]:
            records.extend(self.get(h))
        return manifest, records

    def verify(self, run_id):
        """referential integrity + content-address verification for one run."""
        entry = self.find_run(run_id)
        ok = {"manifest": self.check(entry["manifest"])}
        if ok["manifest"]:
            for key, h in self.get(entry["manifest"])["chunks"]:
                ok[key] = self.check(h)
        return ok


class Recorder:
    """streaming sink for one run: buffers records by (fixture, subject, seed),
    flushing any buffer that reaches chunk_size to the store at once, so memory
    stays bounded on long runs; close() flushes remainders + writes manifest +
    one run-recorded ledger entry. per-key record order is preserved; chunk
    keys carry a sequence suffix (key#n). a crashed run leaves orphan objects
    but no ledger entry, so no partial run is ever visible."""

    def __init__(self, store, stamp, config, chunk_size=100_000):
        self.store, self.stamp, self.config = store, stamp, config
        self.chunk_size = chunk_size
        self.buffers = {}  # insertion-ordered: preserves drive order
        self.refs = []     # (key#seq, hash) in flush order
        self.seqs = {}
        self.n_records = 0

    def _flush(self, key):
        recs = self.buffers[key]
        if not recs:
            return
        seq = self.seqs.get(key, 0)
        self.refs.append((f"{key}#{seq}", self.store.put(recs)))
        self.seqs[key] = seq + 1
        self.buffers[key] = []

    def sink(self, record):
        key = f"{record['fixture']}/{record['subject']}/s{record['seed']}"
        self.buffers.setdefault(key, []).append(record)
        self.n_records += 1
        if len(self.buffers[key]) >= self.chunk_size:
            self._flush(key)

    def close(self):
        for key in self.buffers:
            self._flush(key)
        manifest = {"schema_version": SCHEMA_VERSION, "stamp": self.stamp,
                    "config": self.config, "chunk_size": self.chunk_size,
                    "chunks": self.refs}
        mhash = self.store.put(manifest)
        self.store.append_ledger({"kind": "run-recorded",
                                  "run_id": self.stamp["run_id"],
                                  "event_id": "rec:" + self.stamp["run_id"],
                                  "manifest": mhash,
                                  "n_records": self.n_records})
        return self.stamp["run_id"], mhash


def selftest():
    import shutil
    import tempfile
    root = tempfile.mkdtemp()
    try:
        store = D3Store(root)
        rec = Recorder(store, {"run_id": "r_test_s0"}, {"seeds": [0]})
        rows = [{"fixture": "kp0", "subject": "null:x", "seed": 0, "step": i, "score": 0.5}
                for i in range(10)]
        for r in rows:
            rec.sink(r)
        run_id, mhash = rec.close()
        manifest, back = store.load_run(run_id)
        assert back == rows, "round-trip failed"
        assert store.put(rows) == manifest["chunks"][0][1], "write-once keying broke"
        assert all(store.verify(run_id).values()), "verify failed on clean store"
        rec2 = Recorder(store, {"run_id": "r_test_s1"}, {"seeds": [0]}, chunk_size=4)
        for r in rows:
            rec2.sink(r)
        assert max(len(b) for b in rec2.buffers.values()) <= 4, "buffer unbounded"
        run_id2, _ = rec2.close()
        manifest2, back2 = store.load_run(run_id2)
        assert back2 == rows, "chunked round-trip failed"
        assert len(manifest2["chunks"]) == 3, "expected 4+4+2 chunking"
        assert all(store.verify(run_id2).values()), "verify failed on chunked run"
        chunk_hash = manifest["chunks"][0][1]
        with gzip.open(store._path(chunk_hash), "wb") as f:  # tamper
            f.write(b'[{"forged": true}]')
        assert not store.verify(run_id)[manifest["chunks"][0][0]], "tamper undetected"
        assert not hasattr(store, "rewrite"), "no rewrite api may exist"
        print(f"selftest passed: {run_id} manifest={mhash[:12]} tamper detected, "
              f"chunked flush ok")
    finally:
        shutil.rmtree(root)




# detector layer (spec section 1 component 2, first two detectors): versioned,
# config-hashed instruments that turn score streams into ledger events. the
# arithmetic is the exam judge's; the machinery (identity, hysteresis via
# dwell counts, incremental state, events) is what makes readings testify.
# the settled detector is the ONLINE variant of the judge's retrospective
# criterion (plateau = current window mean, stability = spread of recent
# window means), documented as such: the two agree on clean signals and are
# commissioned separately.

class DetectorBase:
    kind = "detector"

    def __init__(self, store, config, run_id, stream):
        self.store, self.config = store, dict(config)
        self.run_id, self.stream = run_id, stream
        self.config_hash = hashlib.sha256(canonical_bytes(self.config)).hexdigest()[:12]
        self.t = 0
        self.events = []

    def emit(self, kind, evidence):
        entry = self.store.append_ledger({
            "kind": kind, "run_id": self.run_id, "stream": self.stream,
            "event_id": f"{self.kind}:{self.stream}:{self.t}",
            "detector": self.kind, "config_hash": self.config_hash,
            "t": self.t, "evidence": evidence,
        })
        self.events.append(entry)
        return entry


class BandExitDetector(DetectorBase):
    """two-sided band on a score stream. config: lo, hi, enter_dwell (steps
    out of band before an exit opens: hysteresis against boundary jitter),
    exit_dwell (steps back in band before it closes)."""

    kind = "band-exit"

    def __init__(self, store, config, run_id, stream):
        super().__init__(store, config, run_id, stream)
        self.state = None          # None | "low" | "high"
        self.run_out = self.run_in = 0

    def feed(self, x):
        self.t += 1
        c = self.config
        side = "low" if x < c["lo"] else ("high" if x > c["hi"] else None)
        if self.state is None:
            if side is None:
                self.run_out = 0
            else:
                self.run_out = self.run_out + 1 if side == getattr(self, "_side", side) else 1
                self._side = side
                if self.run_out >= c["enter_dwell"]:
                    self.state = side
                    self.emit("band-exit-open", {"side": side, "value": round(x, 4),
                                                 "dwell": self.run_out})
                    self.run_in = 0
        else:
            if side == self.state:
                self.run_in = 0
            else:
                self.run_in += 1
                if self.run_in >= c["exit_dwell"]:
                    self.emit("band-exit-close", {"side": self.state, "value": round(x, 4)})
                    self.state, self.run_out = None, 0


class SettledDetector(DetectorBase):
    """online settling: window means computed incrementally; settled when the
    last `dwell` window-means span at most tol = max(fraction * |mean|, atol)
    and the mean clears plateau_floor; settlement-lost when the window mean
    departs the settled value by more than tol."""

    kind = "settled"

    def __init__(self, store, config, run_id, stream):
        super().__init__(store, config, run_id, stream)
        self.buf, self.means, self.settled_at_val = [], [], None

    def feed(self, x):
        self.t += 1
        c = self.config
        self.buf.append(x)
        if len(self.buf) < c["window"]:
            return
        if len(self.buf) > c["window"]:
            self.buf.pop(0)
        m = sum(self.buf) / c["window"]
        self.means.append(m)
        if len(self.means) > c["dwell"]:
            self.means.pop(0)
        tol = max(c["fraction"] * abs(m), c["atol"])
        if self.settled_at_val is None:
            if (len(self.means) == c["dwell"] and abs(m) >= c["plateau_floor"]
                    and max(self.means) - min(self.means) <= tol):
                self.settled_at_val = m
                self.emit("settled", {"plateau": round(m, 4)})
        elif abs(m - self.settled_at_val) > tol:
            self.emit("settlement-lost", {"plateau": round(self.settled_at_val, 4),
                                          "value": round(m, 4)})
            self.settled_at_val, self.means = None, []


class ConflictOpenedDetector(DetectorBase):
    """channel-3 conflict detector under predicate v2, provisionally
    accepted (RW 2026-07-21, tuning expected: retunes are recalibration
    events with new config hashes, never silent adjustment). the claim
    width is the expecter's own measured jitter: the first cal_window
    scores calibrate mean and std, then the threshold mean - k*std is
    frozen for the stream's lifetime. below-threshold persisting past
    enter_dwell opens a conflict; back above past exit_dwell closes it.
    config: k, cal_window, enter_dwell, exit_dwell, source (the causal
    source value per the taxonomy ruling: organic events at depth carry
    "unattributed" until tap-based attribution exists)."""

    kind = "conflict"

    def __init__(self, store, config, run_id, stream):
        super().__init__(store, config, run_id, stream)
        self.cal = []
        self.threshold = None
        self.state = None
        self.run_out = self.run_in = 0

    def feed(self, x):
        self.t += 1
        c = self.config
        if self.threshold is None:
            self.cal.append(x)
            if len(self.cal) == c["cal_window"]:
                m = sum(self.cal) / len(self.cal)
                sd = (sum((v - m) ** 2 for v in self.cal)
                      / max(1, len(self.cal) - 1)) ** 0.5
                self.threshold = m - c["k"] * sd
            return
        if self.state is None:
            if x < self.threshold:
                self.run_out += 1
                if self.run_out >= c["enter_dwell"]:
                    self.state = "open"
                    self.emit("conflict-opened",
                              {"value": round(x, 4),
                               "threshold": round(self.threshold, 4),
                               "dwell": self.run_out,
                               "source": c.get("source", "unattributed")})
                    self.run_in = 0
            else:
                self.run_out = 0
        else:
            if x < self.threshold:
                self.run_in = 0
            else:
                self.run_in += 1
                if self.run_in >= c["exit_dwell"]:
                    self.emit("conflict-closed",
                              {"value": round(x, 4),
                               "threshold": round(self.threshold, 4),
                               "source": c.get("source", "unattributed")})
                    self.state, self.run_out = None, 0


def detector_selftest():
    """planted known answers per the step-2 fixture list: clean crossing,
    boundary jitter under hysteresis, ramp-to-plateau, plateau break."""
    import shutil
    import tempfile
    root = tempfile.mkdtemp()
    try:
        store = D3Store(root)
        cfg = {"lo": -0.1, "hi": 0.1, "enter_dwell": 3, "exit_dwell": 3}
        d = BandExitDetector(store, cfg, "r_test", "s")
        for x in [0.0] * 10 + [0.5] * 10 + [0.0] * 10:  # clean crossing + return
            d.feed(x)
        kinds = [e["kind"] for e in d.events]
        assert kinds == ["band-exit-open", "band-exit-close"], kinds
        d2 = BandExitDetector(store, cfg, "r_test", "j")
        for i in range(60):  # boundary jitter: alternates in/out, never 3 in a row
            d2.feed(0.12 if i % 2 else 0.08)
        assert d2.events == [], "hysteresis failed on jitter"
        scfg = {"fraction": 0.1, "window": 5, "dwell": 5, "atol": 0.02,
                "plateau_floor": 0.1}
        sd = SettledDetector(store, scfg, "r_test", "p")
        ramp = [i / 20 for i in range(20)] + [1.0] * 20 + [0.2] * 10
        for x in ramp:
            sd.feed(x)
        kinds = [e["kind"] for e in sd.events]
        # the trailing constant 0.2 is long enough to legitimately settle at
        # the new plateau: re-settling is the intended lifecycle
        assert kinds == ["settled", "settlement-lost", "settled"], kinds
        n = len(store.ledger())
        assert n == 5, f"ledger should hold the 5 events, has {n}"
        # conflict detector, planted answers: calibrate on jitter around 0.8,
        # sustained collapse opens, recovery closes, jitter-scale wobble and
        # sub-dwell spikes stay silent
        ccfg = {"k": 3.0, "cal_window": 20, "enter_dwell": 3, "exit_dwell": 3,
                "source": "planted"}
        cd = ConflictOpenedDetector(store, ccfg, "r_test", "c")
        cal = [0.8 + 0.02 * (-1) ** i for i in range(20)]
        for x in cal + [0.81] * 10 + [0.3] * 10 + [0.82] * 10:
            cd.feed(x)
        kinds = [e["kind"] for e in cd.events]
        assert kinds == ["conflict-opened", "conflict-closed"], kinds
        assert cd.events[0]["evidence"]["source"] == "planted"
        cd2 = ConflictOpenedDetector(store, ccfg, "r_test", "c2")
        for x in cal + [0.8 + 0.03 * (-1) ** i for i in range(40)]:  # wobble
            cd2.feed(x)
        assert cd2.events == [], "jitter-scale wobble must not open"
        cd3 = ConflictOpenedDetector(store, ccfg, "r_test", "c3")
        for x in cal + [0.81] * 10 + [0.3] * 2 + [0.81] * 20:  # sub-dwell spike
            cd3.feed(x)
        assert cd3.events == [], "sub-dwell spike must not open"
        n = len(store.ledger())
        assert n == 7, f"ledger should hold 7 events, has {n}"
        print(f"detector selftest passed: {n} events, "
              f"band cfg {BandExitDetector(store, cfg, 'x', 'y').config_hash} "
              f"settle cfg {SettledDetector(store, scfg, 'x', 'y').config_hash} "
              f"conflict cfg {ConflictOpenedDetector(store, ccfg, 'x', 'y').config_hash}")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    selftest()
    detector_selftest()
