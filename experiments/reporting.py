"""Shared pieces of the experiment reports (2026-09-29).

RW's plan for experiments, 2026-09-29: raw results stay on disk in outputs/, which git
does not track; what is committed for an experiment is its source (NN_name.py), its
report generator (NN_report.py) and the report (NN_report.md) beside them. The source
is committed before the run (launch.py refuses launches from uncommitted code),
the report after it, and the report carries what makes the run reproducible.

write() puts the header on a report: the source script; the launch records copied from
outputs/ (launch.py: commit, versions, pip freeze hash, job counts), or, for runs from
before the records, the script's first commit and the raw files' last-write
times, so a reader can compare them; each raw file's size, line count and sha256, and
the commit where a file committed before the move to outputs/ can still be found; and
the commit and state of the generator itself. Tables go in a fenced block so they keep
their columns on GitHub. A generated report is never edited by hand: rerun its
generator. Readings of the results belong in gc-project, not here.
"""

import ast
import contextlib
import datetime
import hashlib
import io
import json
import os
import statistics as st
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
GC_ROOT = os.path.dirname(HERE)


def jl(name):
    """Records of a JSON-lines file in outputs/ ([] if absent)."""
    p = os.path.join(OUT, name)
    return [json.loads(s) for s in open(p)] if os.path.exists(p) else []


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return st.median(xs) if xs else None


def f(v, spec=".2f"):
    return "   -" if v is None else format(v, spec)


def capture(fn, *a, **kw):
    """What a print-based table function prints, as a string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", GC_ROOT, *args], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def _state(rel):
    if _git("ls-files", "--", rel) == "":
        return "untracked"
    return "modified" if _git("diff", "--name-only", "HEAD", "--", rel) else "committed"


def _title(script):
    """The docstring's first paragraph, joined into one line."""
    try:
        doc = ast.get_docstring(ast.parse(open(os.path.join(HERE, script)).read())) or ""
    except (OSError, SyntaxError):
        doc = ""
    return " ".join(doc.strip().split("\n\n")[0].split()) or script


def _committed(old):
    """(commit, identical?) for the last commit that added or changed experiments/<old>."""
    h = _git("log", "-1", "--diff-filter=AM", "--format=%h", "--", f"experiments/{old}")
    if not h:
        return None, None
    try:
        blob = subprocess.run(["git", "-C", GC_ROOT, "show", f"{h}:experiments/{old}"],
                              capture_output=True, timeout=120).stdout
    except Exception:
        return h, None
    return h, blob


def _stamp_lines(path):
    doc = json.load(open(path))
    rel = os.path.relpath(path, HERE)
    ran = [l for l in doc.get("launches", []) if l.get("to_run") != 0]  # to_run 0: nothing left to do
    idle = len(doc.get("launches", [])) - len(ran)
    out = []
    for l in ran:
        repro = l.get("reproducible_from_commit")
        out.append(
            f"- Launch {l['start']} to {l.get('end')}, {l.get('status')}: commit "
            f"`{(l.get('commit') or '?')[:12]}`"
            + ("" if repro is None else ", reproducible from the commit" if repro
               else f", NOT reproducible from the commit (uncommitted: {l.get('dirty_files')})")
            + "".join(f"; {l[k]} {w}" for k, w in (("finished", "finished"), ("failed", "failed"),
                                                   ("already_done", "already done")) if l.get(k) is not None)
            + f"; `{' '.join(l.get('argv', []))}` on {l.get('host')}; "
            f"Python {l.get('python')}, torch {l.get('torch')}, numpy {l.get('numpy')}"
            + (f", CUDA {l['cuda']}" if l.get("cuda") else "")
            + (f"; GPU {l['gpu']}" if l.get("gpu") else "")
            + (f"; pip freeze sha256 {l['pip_freeze_sha256']}" if l.get("pip_freeze_sha256") else "")
            + (f"; files changed during the run: {l['changed_during_run']}" if l.get("changed_during_run") else "")
            + f" (`{rel}`)")
    if idle:
        out.append(f"- {idle} launch{'es' if idle > 1 else ''} in `{rel}` ran no jobs "
                   "(everything was already done).")
    return out, bool(ran)


def write(num, script, raw, body, history=None, notes=()):
    """Writes NN_report.md. raw: file names in outputs/; history: {raw name: the path it was
    committed under before the move to outputs/}; notes: extra lines for the header."""
    history = history or {}
    lines = [f"# Experiment {num}", "", _title(script), "",
             f"Generated by `experiments/{num}_report.py`; do not edit by hand, rerun the generator.", "",
             f"- Source: `experiments/{script}` (its docstring gives the premises and the design).", ""]
    stamped = False
    run_lines = []
    stems = []
    for name in raw:
        stem = name.split(".")[0]
        if stem not in stems:
            stems.append(stem)
    for stem in stems:
        p = os.path.join(OUT, stem + ".stamp.json")
        if os.path.exists(p):
            ls, ran = _stamp_lines(p)
            run_lines += ls
            stamped = stamped or ran
    if not stamped:
        first = _git("log", "--diff-filter=A", "--format=%h %ci", "--", f"experiments/{script}").splitlines()
        first = first[-1] if first else "not committed yet"
        run_lines.insert(0, "- Run: no launch record (the run predates the runner's launch records of "
                            f"2026-09-29). The script was first committed in {first}; the raw files' "
                            "modification times below are the nearest thing to the run's end (a copy "
                            "or checkout can change them); `git log -- experiments/" + script
                         + "` shows any later change to the script.")
    lines += run_lines + list(notes) + ["", "Raw data, in `experiments/outputs/` (not tracked by git):", "",
                                        "| file | bytes | lines | sha256 (16) | modified | committed earlier as |",
                                        "|---|---:|---:|---|---|---|"]
    for name in raw:
        p = os.path.join(OUT, name)
        b = open(p, "rb").read()
        n = b.count(b"\n") + (1 if b and not b.endswith(b"\n") else 0)
        when = datetime.datetime.fromtimestamp(os.path.getmtime(p)).astimezone().isoformat(timespec="seconds")
        old = history.get(name)
        h, blob = _committed(old) if old else (None, None)
        was = "-" if not h else (f"`experiments/{old}` at {h}, "
                                 + ("byte-identical" if blob == b else "DIFFERS from this file"
                                    if blob is not None else "not compared"))
        lines.append(f"| {name} | {len(b):,} | {n:,} | {hashlib.sha256(b).hexdigest()[:16]} | "
                     f"{when} | {was} |")
    gen = [f"experiments/{num}_report.py", "experiments/reporting.py"]
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    lines += ["", f"- Report written {now} at commit `{_git('rev-parse', '--short=12', 'HEAD')}`; generator "
              + ", ".join(f"`{g.split('/')[-1]}` {_state(g)}" for g in gen)
              + " (a report is written before its own commit, so it cannot name that commit).", "",
              "```text", body.rstrip("\n"), "```", ""]
    path = os.path.join(HERE, f"{num}_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def log_report(num, script, logs, history, notes=()):
    """For experiments whose log is already the table: the logs, whole, under the header."""
    body = "\n\n".join((f"== {name}\n" if len(logs) > 1 else "") + open(os.path.join(OUT, name)).read()
                       for name in logs)
    write(num, script, logs, body, history,
          ["- The log is the result: one line per cell, as the script printed it."] + list(notes))
