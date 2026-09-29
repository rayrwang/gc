"""Run a job list in a process pool with a checkpoint (2026-09-29).

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
"""

import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait


def run_jobs(jobs, run, jsonl, log=None, line=None, workers=16, max_tasks=40, out_json=None,
             stall=900):
    done = set()
    if os.path.exists(jsonl):
        for s in open(jsonl):
            try:
                done.add(json.loads(s)["job_key"])
            except Exception:
                pass
    todo = [j for j in jobs if repr(j) not in done]
    chunk = workers * max_tasks
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"on {workers} workers, chunks of {chunk}", flush=True)
    fl = open(log, "a") if log else None
    k = 0
    with open(jsonl, "a") as fj:
        for c0 in range(0, len(todo), chunk):
            with ProcessPoolExecutor(workers) as ex:
                futs = {ex.submit(run, j): j for j in todo[c0:c0 + chunk]}
                pending = set(futs)
                while pending:
                    fin, pending = wait(pending, timeout=stall, return_when=FIRST_COMPLETED)
                    if not fin:
                        print(f"POOL STALL: nothing finished in {stall}s with {len(pending)} "
                              f"pending; exiting", flush=True)
                        fj.flush()
                        if fl:
                            fl.flush()
                        os._exit(3)  # systemd then stops the workers left in the unit
                    for fut in fin:
                        j = futs[fut]
                        k += 1
                        try:
                            r = fut.result()
                        except Exception as e:  # the worker died: left unrecorded for the rerun
                            print(f"POOL {type(e).__name__} on {j!r}: {e}", flush=True)
                            continue
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
    print("done", flush=True)
