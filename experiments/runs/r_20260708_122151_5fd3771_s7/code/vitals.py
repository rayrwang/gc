"""Tier-1 vitals (CHARTER §10: weights are for watching, behavior is for judging).

Diagnostics that INFORM, never adjudicate. Logged at every probe point, riding
the representations already computed for probing (no extra passes).

- weight vitals: sorted column-norm curves per weight matrix (sorting removes
  unit identity -> curves overlayable across checkpoints; steepening head =
  rich-get-richer runaway, collapsing tail = dead capacity).
- rep vitals (v0 proxies for conjecture-entropy, upgraded when expect-patterns
  exist): unit-usage entropy (dead-capacity early warning) + participation ratio
  (effective dimensionality of the rep cloud; collapse detector, cf. Karpathy
  "silently collapsed onto a tiny manifold").
"""
import torch


def _sorted_norms(W, points=64):
    norms = W.reshape(W.shape[0], -1).norm(dim=1) if W.dim() > 1 else W.abs()
    s = norms.sort().values
    if len(s) > points:                       # downsample for log size
        idx = torch.linspace(0, len(s) - 1, points).long()
        s = s[idx]
    return [round(float(v), 5) for v in s]


def weight_vitals(arm, points=64):
    fn = getattr(arm, "weight_matrices", None)
    if fn is None:
        return None
    mats = fn()
    if not mats:
        return None
    return {name: _sorted_norms(W.detach().float(), points) for name, W in mats.items()}


def rep_vitals(R):
    """R: (n, d) raw (unstandardized) representations from the probe pass."""
    R = R.detach().float()
    usage = R.abs().mean(0)                               # per-unit mean activation
    p = usage / (usage.sum() + 1e-12)
    unit_entropy = float(-(p * (p + 1e-12).log()).sum() / torch.log(torch.tensor(float(len(p)))))
    Rc = R - R.mean(0)
    cov = (Rc.T @ Rc) / max(1, len(R) - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    pr = float((eig.sum() ** 2) / ((eig ** 2).sum() + 1e-12))
    return {
        "unit_entropy": round(unit_entropy, 4),           # 1.0 = perfectly even usage
        "participation_ratio": round(pr, 2),              # effective dimensionality
        "dead_frac": round(float((usage < 1e-6).float().mean()), 4),
        "mean_norm": round(float(R.norm(dim=1).mean()), 4),
    }


def collect(arm, R):
    out = {"rep": rep_vitals(R)}
    w = weight_vitals(arm)
    if w is not None:
        out["weights"] = w
    return out


def selftest():
    R = torch.randn(200, 64)
    v = rep_vitals(R)
    assert 0.95 < v["unit_entropy"] <= 1.0                # iid gaussian: even usage
    assert v["participation_ratio"] > 40                  # near-full rank
    Rc = torch.randn(200, 1) @ torch.randn(1, 64)         # rank-1 collapse
    vc = rep_vitals(Rc)
    assert vc["participation_ratio"] < 2.0, vc            # collapse detected
    class FakeArm:
        def weight_matrices(self):
            return {"W": torch.randn(32, 16)}
    w = weight_vitals(FakeArm())
    assert "W" in w and len(w["W"]) == 32
    assert w["W"] == sorted(w["W"])                        # sorted curve
    print(f"selftest passed: gaussian PR={v['participation_ratio']}, "
          f"collapsed PR={vc['participation_ratio']}, sorted-norm curve ok.")


if __name__ == "__main__":
    selftest()
