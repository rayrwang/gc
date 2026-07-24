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

from battery.determinism import gen, set_determinism
from battery.provenance import snapshot_code, stamp, write_stamped


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
    """Small MLP, online backprop, NO replay = the zero-retention null (§9)."""
    name = "sgd-mlp"
    def __init__(self, in_dim, n_classes, hidden=128, lr=1e-2, seed=0):
        g = gen(seed, "init")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_classes))
        for p in self.net.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_uniform_(p, generator=g)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr)
        self._h = None

    def step(self, x, learn=True):
        xf, y = x
        if learn and y is not None:
            self.net.train()
            logits = self.net(xf.unsqueeze(0))
            loss = torch.nn.functional.cross_entropy(logits, y.argmax().unsqueeze(0))
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        with torch.no_grad():
            self._h = self.net[1](self.net[0](xf))       # post-ReLU hidden

    def get_representations(self):
        return self._h.detach().clone()

    def weight_matrices(self):
        return {"fc1": self.net[0].weight, "fc2": self.net[2].weight}


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
    agt.debug_init()
    return GCArm(agt, lambda a: a.cols[1, 0].nr_1.actual, name="mnist-hebb", use_lrn=use_lrn)


# ---------------------------------------------------------------- probe battery
def _accuracy(pred, y):
    return (pred == y.argmax(1)).float().mean().item()

def probe_ridge(Rtr, Ytr, Rte, Yte, lam=1e-2):
    n, d = Rtr.shape
    A = Rtr.T @ Rtr + lam * torch.eye(d)
    W = torch.linalg.solve(A, Rtr.T @ Ytr)
    return _accuracy((Rte @ W).argmax(1), Yte)

def probe_knn(Rtr, Ytr, Rte, Yte, k=5):
    d2 = torch.cdist(Rte, Rtr)                    # (te, tr) euclidean
    idx = d2.topk(k, largest=False).indices       # k nearest
    votes = Ytr.argmax(1)[idx]                     # (te, k)
    pred = torch.mode(votes, dim=1).values
    return _accuracy(pred, Yte)

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
    return _accuracy((Rte @ W).argmax(1), Yte)

PROBES = {"ridge": probe_ridge, "knn": probe_knn, "logistic": probe_logistic}

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
    accs = {name: fn(Rtr, Ytr, Rte, Yte) for name, fn in PROBES.items()}
    return accs, Rtr_raw


# ---------------------------------------------------------------- tasks + driver
class Task:
    """A named stream of (x, one_hot_y). The arm is NEVER told task identity (§7a)."""
    def __init__(self, name, samples):
        self.name, self.samples = name, list(samples)

def run_curriculum(arm_factory, schedule, seed, out_path,
                   probe_every=2000, probe_n=200, extra_files=(),
                   probe_train=None, probe_test=None):
    """schedule = [(Task, n_steps), ...]. Feeds samples one at a time; at
    probe_every steps runs the frozen probe battery on FIXED probe sets
    (pass tasks.probe_sets() — same sets every checkpoint, held out from the
    stream, §7d). Fallback when none given (synthetic selftests only):
    resample from seen task samples. Everything born stamped + snapshotted (§2)."""
    set_determinism(seed)
    st = stamp(seed, extra_files=extra_files)
    run_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), st["run_id"])
    snapshot_code(run_dir, extra_files=extra_files)
    arm = arm_factory(seed)
    seen, step = [], 0
    g = gen(seed, "probe_sample")
    fixed_probes = probe_train is not None
    with write_stamped(out_path, st) as f:
        for task, n in schedule:
            for i in range(n):
                s = task.samples[i % len(task.samples)]
                arm.step(s)
                seen.append(task)
                step += 1
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
                        accs, Rraw = probe_battery(arm, ptr, pte)
                        from battery import vitals as _vitals
                        f.write(json.dumps({"step": step, "task": task.name,
                                            "probes": accs}) + "\n")
                        f.write(json.dumps({"step": step,
                                            "vitals": _vitals.collect(arm, Rraw)}) + "\n")
                        f.flush()
    return out_path


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
        from battery.provenance import load_stamped
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
