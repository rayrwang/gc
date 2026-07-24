"""The curriculum harness (CHARTER §-wide): the arm-agnostic socket everything
plugs into. C1/A1/vitals are run-configs + analyses over its stamped logs, not
separate programs.

An Arm exposes exactly two methods (§10 behavior-is-for-judging):
    step(x)                 one online sample in; learn from it (or not)
    get_representations()   frozen readout of internal state for the last input
The harness never sees weights, never calls arm-specific APIs. Vitals read
weights via a separate registered side-channel, not through this socket.

Probes (§7d) are a fixed battery run globally-identically on FROZEN reps,
read-only: ridge + kNN + logistic. Reporting all three IS the across-probe
robustness check (example 04 finding #4).

Nulls (§9): SGDArm (zero-retention w/o replay), RandomProjArm (raw/random-proj).
gc arms: MNISTHebbArm now; BareAgt/CIFARAgt adapters follow the same pattern
(inject a rep_fn); BareAgt carries the input-coupling caveat, wrap post-commission.
"""
import json
import os
from abc import ABC, abstractmethod

import torch

from experiments.determinism import gen, set_determinism
from experiments.provenance import snapshot_code, stamp, write_stamped


# ---------------------------------------------------------------- the socket
class Arm(ABC):
    name = "arm"
    @abstractmethod
    def step(self, x, learn=True): ...
    @abstractmethod
    def get_representations(self): ...          # -> 1-D float tensor
    def cost_per_step(self):                     # §6 cost card (override; optional)
        return None
    def weight_matrices(self):                   # Tier-1 vitals side-channel (§10)
        return None


class SGDArm(Arm):
    """Small MLP, online backprop, NO replay = the zero-retention null (§9).
    hidden may be an int (one hidden layer) or a tuple (deeper, R-009);
    representations = the LAST post-ReLU hidden layer."""
    name = "sgd-mlp"
    def __init__(self, in_dim, n_classes, hidden=128, lr=1e-2, seed=0):
        g = gen(seed, "init")
        dims = (hidden,) if isinstance(hidden, int) else tuple(hidden)
        layers, d = [], in_dim
        for hd in dims:
            layers += [torch.nn.Linear(d, hd), torch.nn.ReLU()]
            d = hd
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, n_classes)
        for p in list(self.body.parameters()) + list(self.head.parameters()):
            if p.dim() > 1:
                torch.nn.init.kaiming_uniform_(p, generator=g)
        self.opt = torch.optim.SGD(
            list(self.body.parameters()) + list(self.head.parameters()), lr=lr)
        self._h = None

    def step(self, x, learn=True):
        xf, y = x
        if learn and y is not None:
            self.body.train()
            logits = self.head(self.body(xf.unsqueeze(0)))
            loss = torch.nn.functional.cross_entropy(logits, y.argmax().unsqueeze(0))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            self._h = self.body(xf)                      # LAST post-ReLU hidden

    def get_representations(self):
        return self._h.detach().clone()

    def weight_matrices(self):
        out = {f"fc{i//2+1}": m.weight for i, m in enumerate(self.body) 
               if isinstance(m, torch.nn.Linear)}
        out["head"] = self.head.weight
        return out


class RandomProjArm(Arm):
    """Fixed random projection + ReLU, no learning = the raw/random-proj null (§9)."""
    name = "random-proj"
    def __init__(self, in_dim, hidden=128, seed=0):
        g = gen(seed, "init")
        self.W = torch.randn(in_dim, hidden, generator=g) / (in_dim ** 0.5)
        self._h = None
    def step(self, x, learn=True):
        xf, _ = x
        self._h = torch.relu(xf @ self.W)
    def get_representations(self):
        return self._h.clone()
    def weight_matrices(self):
        return {"Wproj": self.W.T}


class GCArm(Arm):
    """Adapter for gc agents. rep_fn(agt) -> 1-D tensor knows the agent's readout:
        MNISTAgt : lambda a: a.cols[1, 0].nr_1.actual
        CIFARAgt : lambda a: a.get_representations()
        BareAgt  : lambda a: torch.cat([c.nr_1.actual for c in a.internal_cols()])
                   (input-coupling caveat — wrap post-commissioning)
    """
    def __init__(self, agt, rep_fn, name="gc", use_lrn=True, weights_fn=None):
        self.agt, self.rep_fn, self.name, self.use_lrn = agt, rep_fn, name, use_lrn
        self._weights_fn = weights_fn
    def weight_matrices(self):
        return self._weights_fn(self.agt) if self._weights_fn else None
    def step(self, x, learn=True):
        xf, _ = x
        lrn = self.use_lrn and learn
        try:
            self.agt.step([xf], use_lrn=lrn, disable_print=True)
        except TypeError:
            self.agt.step([xf], disable_print=True)      # BareAgt: no use_lrn kw
    def get_representations(self):
        return self.rep_fn(self.agt).detach().clone().flatten().float()


def mnist_hebb_arm(ispec, ospec, seed=0, use_lrn=True):
    """The day-one Hebbian arm (fast, single-layer, the controlled learn-vs-random rig)."""
    from src.agents import MNISTAgt, MNISTCfg
    agt = MNISTAgt(MNISTCfg(ispec, ospec), f"/tmp/harness_mnistagt_{seed}")
    # no debug_init(): that opens the pygame debugger GUI — examples-only,
    # battery runs are headless (looking is allowed, but not by accident)
    return GCArm(agt, lambda a: a.cols[1, 0].nr_1.actual, name="mnist-hebb", use_lrn=use_lrn)


# ---------------------------------------------------------------- probe battery
def _accuracy(pred, y):
    return (pred == y.argmax(1)).float().mean().item()

def probe_ridge(Rtr, Ytr, Rte, Yte, lam=1e-2):
    n, d = Rtr.shape
    A = Rtr.T @ Rtr + lam * torch.eye(d)
    W = torch.linalg.solve(A, Rtr.T @ Ytr)
    return (Rte @ W).argmax(1)

def probe_knn(Rtr, Ytr, Rte, Yte, k=5):
    d2 = torch.cdist(Rte, Rtr)                    # (te, tr) euclidean
    idx = d2.topk(k, largest=False).indices       # k nearest
    votes = Ytr.argmax(1)[idx]                     # (te, k)
    return torch.mode(votes, dim=1).values

def probe_logistic(Rtr, Ytr, Rte, Yte, steps=200, lr=0.1, seed=0):
    d, c = Rtr.shape[1], Ytr.shape[1]
    W = torch.zeros(d, c, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=lr, max_iter=steps)
    yt = Ytr.argmax(1)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Rtr @ W, yt)
        loss.backward(); return loss
    opt.step(closure)
    return (Rte @ W).argmax(1)

PROBES = {"ridge": probe_ridge, "knn": probe_knn, "logistic": probe_logistic}

def _collect_reps(arm, samples):
    out = []
    for s in samples:
        arm.step((s[0], None), learn=False)              # read-only (S7d)
        out.append(arm.get_representations())
    return torch.stack(out)


def fit_probes_store(arm, probe_train):
    """Fit all probes on frozen reps NOW; return a serializable snapshot
    (weights + standardization stats; kNN keeps its standardized train reps).
    Stored in the log so any future analysis can re-apply them (R-007)."""
    Rtr_raw = _collect_reps(arm, probe_train)
    Ytr = torch.stack([y for _, y in probe_train])
    mu, sd = Rtr_raw.mean(0), Rtr_raw.std(0) + 1e-6
    Rtr = (Rtr_raw - mu) / sd
    d, c = Rtr.shape[1], Ytr.shape[1]
    A = Rtr.T @ Rtr + 1e-2 * torch.eye(d)
    W_ridge = torch.linalg.solve(A, Rtr.T @ Ytr)
    W_log = torch.zeros(d, c, requires_grad=True)
    opt = torch.optim.LBFGS([W_log], lr=0.1, max_iter=200)
    yt = Ytr.argmax(1)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Rtr @ W_log, yt)
        loss.backward(); return loss
    opt.step(closure)
    r4 = lambda T: [[round(float(v), 4) for v in row] for row in T]
    return {"mu": [round(float(v), 4) for v in mu], "sd": [round(float(v), 4) for v in sd],
            "ridge_W": r4(W_ridge), "logistic_W": r4(W_log.detach()),
            "knn_Rtr": r4(Rtr), "knn_y": [int(v) for v in yt]}


def eval_stored_probes(arm, stored, probe_test):
    """Per-class accuracy of a STORED (stale) probe snapshot on CURRENT reps."""
    Rte_raw = _collect_reps(arm, probe_test)
    Yte = torch.stack([y for _, y in probe_test])
    yte = Yte.argmax(1)
    mu = torch.tensor(stored["mu"]); sd = torch.tensor(stored["sd"])
    Rte = (Rte_raw - mu) / sd
    preds = {"ridge": (Rte @ torch.tensor(stored["ridge_W"])).argmax(1),
             "logistic": (Rte @ torch.tensor(stored["logistic_W"])).argmax(1)}
    Rtr = torch.tensor(stored["knn_Rtr"]); ytr = torch.tensor(stored["knn_y"])
    idx = torch.cdist(Rte, Rtr).topk(5, largest=False).indices
    preds["knn"] = torch.mode(ytr[idx], dim=1).values
    out = {}
    for name, pred in preds.items():
        pc = []
        for c in range(Yte.shape[1]):
            m = yte == c
            pc.append(round(float((pred[m] == c).float().mean()), 4) if bool(m.any()) else None)
        out[name] = pc
    return out


def probe_battery(arm, probe_train, probe_test):
    """Collect frozen reps for the probe sets (read-only, NO learning) and run all
    three probes. Returns {probe_name: accuracy}. §7d: never touches the arm."""
    def reps(samples):
        out = []
        for s in samples:
            arm.step((s[0], None), learn=False)          # read-only, no learning (§7d)
            out.append(arm.get_representations())
        return torch.stack(out)
    Rtr_raw = reps(probe_train); Ytr = torch.stack([y for _, y in probe_train])
    Rte = reps(probe_test);  Yte = torch.stack([y for _, y in probe_test])
    mu, sd = Rtr_raw.mean(0), Rtr_raw.std(0) + 1e-6   # standardize on train stats only
    Rtr, Rte = (Rtr_raw - mu) / sd, (Rte - mu) / sd
    yte = Yte.argmax(1)
    accs, per_class = {}, {}
    for name, fn in PROBES.items():
        pred = fn(Rtr, Ytr, Rte, Yte)
        accs[name] = float((pred == yte).float().mean())
        pc = []
        for c in range(Ytr.shape[1]):                # the 10-dim competence profile (R-004)
            m = yte == c
            pc.append(round(float((pred[m] == c).float().mean()), 4) if bool(m.any()) else None)
        per_class[name] = pc
    return accs, per_class, Rtr_raw


# ---------------------------------------------------------------- tasks + driver
class Task:
    """A named stream of (x, one_hot_y). The arm is NEVER told task identity (§7a)."""
    def __init__(self, name, samples):
        self.name, self.samples = name, list(samples)

def run_curriculum(arm_factory, schedule, seed, out_path,
                   probe_every=2000, probe_n=200, extra_files=(),
                   probe_train=None, probe_test=None, probe_fit_steps=()):
    """schedule = [(Task, n_steps), ...]. Feeds samples one at a time; at
    probe_every steps runs the frozen probe battery on FIXED probe sets
    (pass tasks.probe_sets() — same sets every checkpoint, held out from the
    stream, §7d). Fallback when none given (synthetic selftests only):
    resample from seen task samples. Everything born stamped + snapshotted (§2)."""
    set_determinism(seed)
    st = stamp(seed, extra_files=extra_files)
    run_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), st["run_id"])
    snapshot_code(run_dir, extra_files=extra_files)
    log_path = os.path.join(run_dir, "log.jsonl")   # collision-proof: run_id is unique
    arm = arm_factory(seed)
    seen, step = [], 0
    g = gen(seed, "probe_sample")
    fixed_probes = probe_train is not None
    stored_probes = {}                               # fit_step -> snapshot (R-007)
    with write_stamped(log_path, st) as f:
        for task, n in schedule:
            for i in range(n):
                s = task.samples[i % len(task.samples)]
                arm.step(s)
                seen.append(task)
                step += 1
                if step in probe_fit_steps:
                    snap = fit_probes_store(arm, probe_train)
                    stored_probes[step] = snap
                    f.write(json.dumps({"step": step, "stale_fit": snap}) + "\n")
                    f.flush()
                if step % probe_every == 0:
                    if fixed_probes:
                        ptr, pte = probe_train, probe_test
                    else:
                        pool = [x for t in set(id(t) for t in seen) for x in
                                next(tt for tt in seen if id(tt) == t).samples]
                        perm = torch.randperm(len(pool), generator=g)[:2 * probe_n]
                        pts = [pool[k] for k in perm.tolist()]
                        ptr, pte = pts[:probe_n], pts[probe_n:2 * probe_n]
                    if len(pte) >= 10:
                        accs, per_class, Rraw = probe_battery(arm, ptr, pte)
                        from experiments import vitals as _vitals
                        row = {"step": step, "task": task.name,
                               "probes": accs, "per_class": per_class}
                        if stored_probes:
                            row["stale_eval"] = {fs: eval_stored_probes(arm, sp, pte)
                                                 for fs, sp in stored_probes.items()}
                        f.write(json.dumps(row) + "\n")
                        f.write(json.dumps({"step": step,
                                            "vitals": _vitals.collect(arm, Rraw)}) + "\n")
                        f.flush()
    ap = os.path.abspath(out_path)                   # friendly name -> latest run
    if os.path.islink(ap):
        os.unlink(ap)
    if not os.path.exists(ap):                       # never clobber a real file
        os.symlink(log_path, ap)
    return log_path


# ---------------------------------------------------------------- selftest
def _blobs(classes, n, in_dim, seed, n_classes):
    g = gen(seed, f"blob{classes}")
    centers = torch.randn(n_classes, in_dim, generator=g) * 4
    xs = []
    for _ in range(n):
        c = classes[torch.randint(len(classes), (1,), generator=g).item()]
        x = centers[c] + torch.randn(in_dim, generator=g)
        y = torch.zeros(n_classes); y[c] = 1            # global one-hot width
        xs.append((x, y))
    return xs

def selftest():
    set_determinism(0)
    IN, NC = 20, 6
    taskA = Task("A(0-2)", _blobs([0, 1, 2], 400, IN, 1, NC))
    taskB = Task("B(3-5)", _blobs([3, 4, 5], 400, IN, 2, NC))
    sched = [(taskA, 2000), (taskB, 2000), (taskA, 2000)]
    for factory, tag in [
        (lambda s: SGDArm(IN, NC, seed=s), "sgd"),
        (lambda s: RandomProjArm(IN, seed=s), "randproj"),
    ]:
        out = run_curriculum(factory, sched, seed=0,
                             out_path=f"/tmp/harness_selftest_{tag}.jsonl",
                             probe_every=2000, probe_n=120)
        from experiments.provenance import load_stamped
        _, rows = load_stamped(out)
        assert rows, f"{tag}: no probe rows"
        last = [r for r in rows if "probes" in r][-1]["probes"]
        assert any("vitals" in r for r in rows), f"{tag}: no vitals rows"
        assert set(last) == {"ridge", "knn", "logistic"}, last
        print(f"{tag:9s} final probes: " +
              "  ".join(f"{k}={v:.2f}" for k, v in last.items()))
    print("selftest passed: socket + 3-probe battery + curriculum driver + stamping.")

if __name__ == "__main__":
    selftest()
