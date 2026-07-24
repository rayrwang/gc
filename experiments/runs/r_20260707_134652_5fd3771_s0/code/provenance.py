"""Provenance stamping (CHARTER.md §2): unstamped numbers are not evidence.

Usage:
    from battery.provenance import stamp, run_id, write_stamped, load_stamped

    s = stamp(seed=42)               # dict, one per run, cite everywhere
    with write_stamped(path, s) as f:  # stamp = first line of every output file
        f.write(json.dumps(row) + "\\n")
    rows = load_stamped(path)        # raises on missing/malformed stamp
"""
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTER = os.path.join(ROOT, "CHARTER.md")


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def charter_version():
    """(version, content_hash) of the charter this run counts under."""
    try:
        with open(CHARTER, "rb") as f:
            content = f.read()
    except FileNotFoundError:
        return "none", "0" * 12
    h = hashlib.sha256(content).hexdigest()[:12]
    version = "v?"
    for line in content.decode(errors="ignore").splitlines():
        if line.startswith("- **v"):            # changelog is newest-first
            version = line.split("**")[1]        # e.g. "v2"
            break
    return version, h


def run_id(seed):
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    return f"r_{time.strftime('%Y%m%d_%H%M%S')}_{sha}_s{seed}"


def _hash_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode())
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:12]


def battery_files():
    d = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]


def stamp(seed, extra_files=()):
    """The identity record for one run. Generate once at t0, cite everywhere.

    git_sha/git_dirty cover TRACKED code (the substrate) only; battery/ and
    experiment scripts are git-excluded, so they are covered separately by
    content hash (battery_hash, extra_hash) and by snapshot_code().
    """
    version, chash = charter_version()
    dirty = bool(_git("status", "--porcelain"))
    return {
        "run_id": run_id(seed),
        "seed": seed,
        "git_sha": _git("rev-parse", "HEAD") or "nogit",
        "git_dirty": dirty,                      # tracked substrate code only
        "diff_hash": hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest()[:12] if dirty else None,
        "battery_hash": _hash_files(battery_files()),
        "extra_hash": _hash_files(list(extra_files)) if extra_files else None,
        "charter_version": version,
        "charter_hash": chash,
        "env": _env_fields(),
        "t0": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _env_fields():
    """Bit-identity holds only on same hardware + library versions; record them."""
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "default_dtype": str(torch.get_default_dtype()),
        }
    except Exception:
        return {"torch": None}


def snapshot_code(run_dir, extra_files=()):
    """Copy the code that actually ran into the run's output dir (true reproducibility
    for untracked code: the run dir carries its own source)."""
    import shutil
    dst = os.path.join(run_dir, "code")
    os.makedirs(dst, exist_ok=True)
    for p in list(battery_files()) + list(extra_files):
        shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
    diff = _git("diff", "HEAD")
    if diff:                                     # dirty tracked code: keep the patch
        with open(os.path.join(dst, "substrate.diff"), "w") as f:
            f.write(diff + "\n")
    return dst


@contextmanager
def write_stamped(path, stamp_dict, step=None):
    """Open an output file whose first line is the stamp (plus step if given)."""
    rec = dict(stamp_dict)
    if step is not None:
        rec["step"] = step
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_stamp": rec}) + "\n")
        yield f


def load_stamped(path):
    """Load a stamped JSONL; raise if the stamp is missing. The enforcement."""
    with open(path) as f:
        first = f.readline()
        try:
            head = json.loads(first)
            stamp_rec = head["_stamp"]
            assert "run_id" in stamp_rec and "charter_hash" in stamp_rec
        except Exception:
            raise ValueError(f"{path}: no provenance stamp — not evidence (CHARTER §2)")
        rows = [json.loads(line) for line in f if line.strip()]
    return stamp_rec, rows


def selftest():
    s = stamp(seed=7, extra_files=[__file__])
    assert s["run_id"].endswith("_s7") and len(s["charter_hash"]) == 12
    assert len(s["battery_hash"]) == 12 and len(s["extra_hash"]) == 12
    import tempfile
    rd = tempfile.mkdtemp()
    dst = snapshot_code(rd, extra_files=[__file__])
    assert os.path.exists(os.path.join(dst, "provenance.py"))
    p = "/tmp/prov_selftest.jsonl"
    with write_stamped(p, s, step=123) as f:
        f.write(json.dumps({"x": 1}) + "\n")
    rec, rows = load_stamped(p)
    assert rec["seed"] == 7 and rec["step"] == 123 and rows == [{"x": 1}]
    import io
    try:
        with open(p, "w") as f:
            f.write('{"x": 1}\n')
        load_stamped(p)
        raise AssertionError("unstamped file was accepted")
    except ValueError:
        pass
    os.remove(p)
    print(f"selftest passed: stamp={rec['run_id']} charter={rec['charter_version']}/{rec['charter_hash']}")


if __name__ == "__main__":
    selftest()
