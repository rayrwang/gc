"""Run a job list in a process pool with a checkpoint and a launch record (2026-09-29).

Every finished job is appended at once to a JSON-lines file under its job key, so a
run that dies resumes by skipping the keys already there (experiment 55 was
OOM-killed at 3,462 of 7,068 on 09-29, its JSON written only at the end).

Workers are replaced by running the jobs in chunks of workers * max_tasks, each chunk
in a fresh pool, so per-worker memory cannot creep (55's reached 2.5 to 3.5 GB each).
The first version used ProcessPoolExecutor(max_tasks_per_child=...) instead, and on
CPython 3.14.4 with forkserver it stalled: 55r stopped at exactly 640 = 16 workers x 40
jobs, every worker retired and none was started in its place. A chunk costs a short
tail at its end, while the last jobs finish.

If nothing finishes for `stall` seconds the run exits with code 3, so a hang ends the
unit instead of sitting idle. Jobs whose worker died are not recorded, so a rerun
redoes them.

Launches run only from committed code (RW 2026-09-29, the experiments plan: "source
commited then when the run finished the rperot is written"). Before any job starts,
every gc file the run has imported must be committed and unmodified; if one is not,
the launch is refused with exit code 4 and the list of files, so the commit hash alone
identifies the code that ran and no copies of the code are kept. Smoke and test
launches pass allow_dirty=True (or GC_ALLOW_DIRTY=1) and are recorded as dirty. A
resume whose imported files differ from those of an earlier launch that ran jobs is
refused with exit code 5 (GC_ALLOW_MIXED=1 overrides), so one result file never mixes
two versions of the code, as experiment 55 did when 55r resumed it with a changed
runner.

Every launch appends a record to <base>.stamp.json beside the results (base = the
JSON-lines path without .jsonl): the commit, the imported gc files with content hashes
and states (in the main process and in the workers), argv, host, the Python, torch,
numpy and CUDA versions, the GPU and driver, `pip freeze` and its hash, the job counts,
start and end, and at the end whether any imported file changed while the run was
going. The report generator copies the provenance from there into the committed report
(reporting.py). The first version of the stamp (earlier on 09-29) also copied
uncommitted files beside the results; the commit requirement replaced that.
"""

import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

GC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPORTED = False


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception:
        return ""


def _git(*args):
    return _run(["git", "-C", GC_ROOT, *args])


def _gc_files():
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


def _call(run, job):
    """Runs one job in a worker; the first result from each worker also carries the gc
    files that worker imported (removed again before the result is written)."""
    global _REPORTED
    r = run(job)
    if not _REPORTED and isinstance(r, dict):
        _REPORTED = True
        r["_gc_files"] = sorted(_gc_files())
    return r


def _load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {"launches": []}


def _gate(base, allow_dirty, allow_mixed):
    """Refuses a launch from uncommitted code (exit 4) or a resume on changed code (exit 5)."""
    files = _describe(_gc_files())
    dirty = sorted(f for f, d in files.items() if d["state"] != "committed")
    if dirty and not allow_dirty:
        print("LAUNCH REFUSED: these gc files are modified or untracked; commit them first "
              "(smoke and test launches: allow_dirty=True or GC_ALLOW_DIRTY=1):", flush=True)
        for f in dirty:
            print(f"  {files[f]['state']:10} {f}", flush=True)
        sys.exit(4)
    prev = [l for l in _load(base + ".stamp.json")["launches"] if (l.get("finished") or 0) > 0]
    if prev and not allow_mixed:
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


class _Stamp:
    def __init__(self, base, info, files, dirty):
        self.path = base + ".stamp.json"
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
               "pip_freeze": freeze, **info,
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


class _Safe:
    """The stamp is a record, never a reason to stop a run: any failure in it prints a
    warning and the run goes on without it. (The commit gate is not inside it.)"""

    def __init__(self, make):
        try:
            self.s = make()
        except Exception as e:
            self.s = None
            print(f"STAMP WARNING: {type(e).__name__}: {e}", flush=True)

    @property
    def ok(self):
        return self.s is not None

    def _do(self, name, *a, **kw):
        if self.s is None:
            return None
        try:
            return getattr(self.s, name)(*a, **kw)
        except Exception as e:
            print(f"STAMP WARNING in {name}: {type(e).__name__}: {e}", flush=True)
            return None

    def worker(self, files):
        self._do("worker", files)

    def finish(self, status, **extra):
        self._do("finish", status, **extra)

    def changed(self):
        return self.s.rec.get("changed_during_run") if self.s is not None else None


def run_jobs(jobs, run, jsonl, log=None, line=None, workers=16, max_tasks=40, out_json=None,
             stall=900, allow_dirty=False):
    done = set()
    if os.path.exists(jsonl):
        for s in open(jsonl):
            try:
                done.add(json.loads(s)["job_key"])
            except Exception:
                pass
    todo = [j for j in jobs if repr(j) not in done]
    chunk = workers * max_tasks
    base = jsonl[:-6] if jsonl.endswith(".jsonl") else jsonl
    os.makedirs(os.path.dirname(os.path.abspath(jsonl)), exist_ok=True)
    allow_dirty = allow_dirty or os.environ.get("GC_ALLOW_DIRTY") == "1"
    allow_mixed = os.environ.get("GC_ALLOW_MIXED") == "1"
    if todo:
        files, dirty = _gate(base, allow_dirty, allow_mixed)
    else:  # nothing will run, so nothing to refuse; the record still says what the code was
        files = _describe(_gc_files())
        dirty = sorted(f for f, d in files.items() if d["state"] != "committed")
    stamp = _Safe(lambda: _Stamp(base, {"jobs": len(jobs), "already_done": len(jobs) - len(todo),
                                        "to_run": len(todo), "workers": workers, "chunk": chunk},
                                 files, dirty))
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"on {workers} workers, chunks of {chunk}; stamp {base + '.stamp.json'}"
          + ("" if stamp.ok else " NOT WRITTEN (see warning)")
          + (f"; DIRTY launch, not reproducible from the commit: {dirty}" if dirty else ""),
          flush=True)
    fl = open(log, "a") if log else None
    k = failed = 0
    with open(jsonl, "a") as fj:
        for c0 in range(0, len(todo), chunk):
            with ProcessPoolExecutor(workers) as ex:
                futs = {ex.submit(_call, run, j): j for j in todo[c0:c0 + chunk]}
                pending = set(futs)
                while pending:
                    fin, pending = wait(pending, timeout=stall, return_when=FIRST_COMPLETED)
                    if not fin:
                        print(f"POOL STALL: nothing finished in {stall}s with {len(pending)} "
                              f"pending; exiting", flush=True)
                        fj.flush()
                        if fl:
                            fl.flush()
                        stamp.finish("stalled", finished=k - failed, failed=failed)
                        os._exit(3)  # systemd then stops the workers left in the unit
                    for fut in fin:
                        j = futs[fut]
                        k += 1
                        try:
                            r = fut.result()
                        except Exception as e:  # the worker died: left unrecorded for the rerun
                            failed += 1
                            print(f"POOL {type(e).__name__} on {j!r}: {e}", flush=True)
                            continue
                        wf = r.pop("_gc_files", None)
                        if wf:
                            stamp.worker(wf)
                        r["job_key"] = repr(j)
                        fj.write(json.dumps(r) + "\n")
                        fj.flush()
                        if fl and line:
                            fl.write(line(r) + "\n")
                            fl.flush()
                        if k % 200 == 0:
                            print(f"{k}/{len(todo)}", flush=True)
    if fl:
        fl.close()
    if out_json:
        rs = [json.loads(s) for s in open(jsonl)]
        with open(out_json, "w") as fh:
            json.dump(rs, fh)
    stamp.finish("done", finished=k - failed, failed=failed)
    changed = stamp.changed()
    print("done" + (f"; WARNING, changed during the run: {changed}" if changed else ""), flush=True)
