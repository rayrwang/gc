"""The commit check and launch record for every experiment run (2026-09-29).

Launches run only from committed code (RW 2026-09-29, the experiments plan: "source
commited then when the run finished the rperot is written"). start() checks that every
gc file the run has imported so far is committed and unmodified, and refuses the launch
with exit code 4 and the list of files if one is not, so the commit hash alone
identifies the code that ran and no copies of the code are kept. Smoke and test
launches pass allow_dirty=True (or set GC_ALLOW_DIRTY=1) and are recorded as dirty.
When a run resumes into a results file an earlier launch already wrote to, start()
also refuses (exit code 5; GC_ALLOW_MIXED=1 overrides) if the imported files differ
from that launch's, so one results file never mixes two versions of the code, as
experiment 55 did when 55r resumed it with a changed runner.

Each launch appends a record to <base>.stamp.json (base: the results path without its
extension, in experiments/outputs/): the commit, the imported gc files with content
hashes and states, argv, host, the Python, torch, numpy and CUDA versions, the GPU and
driver, `pip freeze` and its hash, whatever counts the caller passes, start and end,
and at the end whether any imported file changed during the run. The report generator
copies it into the committed report (reporting.py). A failure while writing the record
prints a warning and never stops the run; the commit check is not covered by that.

Use, in a script that does not go through pool_runner (pool_runner.run_jobs calls it
itself), after its imports and before the work:

    import launch
    rec = launch.start(os.path.join(HERE, "outputs", "58_name"), {"cells": len(jobs)},
                       allow_dirty=smoke)
    ...
    rec.finish("done", finished=n_done, failed=n_failed)

Split out of pool_runner.py the same day (RW: "ok do that"), so the rule holds for
every experiment, not only the pooled ones.
"""

import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return ""


def _git(*args):
    return _run(["git", "-C", GC_ROOT, *args])


def gc_files():
    """gc source files imported by this process, relative to the repo root."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        # absolute paths to real files only: torch.ops and torch.classes report fake
        # relative names ("_ops.py", "_classes.py") that would resolve against the cwd
        if not f or not f.endswith(".py") or not os.path.isabs(f):
            continue
        f = os.path.realpath(f)
        if (f.startswith(GC_ROOT + os.sep) and os.sep + "venv" + os.sep not in f
                and os.path.isfile(f)):
            out.add(os.path.relpath(f, GC_ROOT))
    return out


def _sha(rel):
    try:
        return hashlib.sha256(open(os.path.join(GC_ROOT, rel), "rb").read()).hexdigest()[:16]
    except OSError:
        return None


def _describe(files):
    files = sorted(files)
    tracked = set(_git("ls-files", "--", *files).splitlines()) if files else set()
    modified = set(_git("diff", "--name-only", "HEAD", "--", *files).splitlines()) if files else set()
    return {f: {"sha256": _sha(f),
                "state": "modified" if f in modified else "committed" if f in tracked else "untracked"}
            for f in files}


def _load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {"launches": []}


def _gate(base, allow_dirty, resuming):
    files = _describe(gc_files())
    dirty = sorted(f for f, d in files.items() if d["state"] != "committed")
    if dirty and not allow_dirty:
        print("LAUNCH REFUSED: these gc files are modified or untracked; commit them first "
              "(smoke and test launches: allow_dirty=True or GC_ALLOW_DIRTY=1):", flush=True)
        for f in dirty:
            print(f"  {files[f]['state']:10} {f}", flush=True)
        sys.exit(4)
    prev = [l for l in _load(base + ".stamp.json")["launches"] if l.get("to_run") != 0]
    if resuming and prev and os.environ.get("GC_ALLOW_MIXED") != "1":
        old = {**prev[-1].get("worker_files", {}), **prev[-1].get("files", {})}
        changed = sorted(f for f in files if f in old and old[f]["sha256"] != files[f]["sha256"])
        added = sorted(f for f in files if f not in old)
        if changed or added:
            print(f"RESUME REFUSED: the code differs from the launch of {prev[-1]['start']} "
                  f"(commit {prev[-1].get('commit')}), whose results are already in this file; "
                  "write to a new file, or set GC_ALLOW_MIXED=1:", flush=True)
            for f in changed:
                print(f"  changed {f}", flush=True)
            for f in added:
                print(f"  new     {f}", flush=True)
            sys.exit(5)
    return files, dirty


class _Record:
    def __init__(self, base, info, files, dirty):
        self.path = base + ".stamp.json"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.doc = _load(self.path)
        try:
            import torch
            tv, cv = torch.__version__, torch.version.cuda
        except Exception:
            tv = cv = None
        try:
            import numpy
            nv = numpy.__version__
        except Exception:
            nv = None
        freeze = _run([sys.executable, "-m", "pip", "freeze"]).splitlines()
        diff = _git("diff", "HEAD")
        self.files = files
        rec = {"start": _now(), "end": None, "status": "running", "argv": sys.argv,
               "host": socket.gethostname(), "commit": _git("rev-parse", "HEAD") or None,
               "dirty_files": dirty, "reproducible_from_commit": not dirty,
               "dirty_tracked": bool(_git("status", "--porcelain", "--untracked-files=no")),
               "diff_hash": hashlib.sha256(diff.encode()).hexdigest()[:16] if diff else None,
               "python": sys.version.split()[0], "torch": tv, "cuda": cv, "numpy": nv,
               "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"]) or None,
               "pip_freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest()[:16],
               "pip_freeze": freeze, **(info or {}),
               "files": files, "worker_files": {}, "dirty_worker_files": [],
               "changed_during_run": None}
        self.doc["launches"].append(rec)
        self.rec = rec
        self.worker_seen = set()
        self.write()

    def worker(self, files):
        new = set(files) - set(self.files) - self.worker_seen
        if new:
            self.worker_seen |= new
            d = _describe(new)
            self.rec["worker_files"].update(d)
            bad = sorted(f for f, v in d.items() if v["state"] != "committed")
            if bad:
                self.rec["dirty_worker_files"] += bad
                self.rec["reproducible_from_commit"] = False
                print(f"WARNING: workers imported uncommitted files: {bad}", flush=True)
            self.write()

    def finish(self, status, **extra):
        allf = {**self.files, **self.rec["worker_files"]}
        self.rec["changed_during_run"] = sorted(f for f, d in allf.items() if _sha(f) != d["sha256"])
        self.rec.update(end=_now(), status=status, **extra)
        self.write()

    def write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.doc, fh, indent=1)
        os.replace(tmp, self.path)


class Launch:
    """What start() returns: the record, fail-soft. .ok says whether it was written;
    .dirty lists the uncommitted files of a launch allowed to run dirty."""

    def __init__(self, base, info, files, dirty):
        self.dirty = dirty
        self.path = base + ".stamp.json"
        try:
            self.r = _Record(base, info, files, dirty)
        except Exception as e:
            self.r = None
            print(f"LAUNCH RECORD WARNING: {type(e).__name__}: {e}", flush=True)

    @property
    def ok(self):
        return self.r is not None

    def _do(self, name, *a, **kw):
        if self.r is None:
            return None
        try:
            return getattr(self.r, name)(*a, **kw)
        except Exception as e:
            print(f"LAUNCH RECORD WARNING in {name}: {type(e).__name__}: {e}", flush=True)
            return None

    def worker(self, files):
        self._do("worker", files)

    def finish(self, status, **extra):
        self._do("finish", status, **extra)

    def changed(self):
        return self.r.rec.get("changed_during_run") if self.r is not None else None


def start(base, info=None, allow_dirty=False, resuming=False, check=True):
    """Checks the code and opens the launch record (see the module docstring). base: the
    results path without extension; resuming: the run continues a results file an
    earlier launch wrote to; check=False (a launch that will run nothing) records
    without refusing."""
    allow_dirty = allow_dirty or os.environ.get("GC_ALLOW_DIRTY") == "1"
    if check:
        files, dirty = _gate(base, allow_dirty, resuming)
    else:
        files = _describe(gc_files())
        dirty = sorted(f for f, d in files.items() if d["state"] != "committed")
    rec = Launch(base, info, files, dirty)
    print(f"launch record {rec.path}" + ("" if rec.ok else " NOT WRITTEN (see warning)")
          + (f"; DIRTY launch, not reproducible from the commit: {dirty}" if dirty else ""),
          flush=True)
    return rec
