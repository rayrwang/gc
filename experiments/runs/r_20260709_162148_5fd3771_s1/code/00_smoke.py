"""00: Smoke run (pre-registration free play, §5): all three arms on real MNIST
digit-splits. Question: does the plumbing work on real data, do learner curves
move, does random-proj stay flat? NO verdicts here.
Expectation (not a registered claim): SGD probes climb on A then shift on B;
random-proj flat; MNISTAgt ~ flat-ish (BCM rescue, not real learning — RW)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from experiments.harness import SGDArm, RandomProjArm, mnist_hebb_arm, run_curriculum
from experiments.tasks import digit_split_tasks, probe_sets

def main():
    torch.set_default_dtype(torch.float32)
    A, B = digit_split_tasks(seed=0)
    ptr, pte = probe_sets(seed=0, n_train=400, n_test=400)
    sched = [(A, 4000), (B, 4000)]
    arms = {
        "sgd":      lambda s: SGDArm(784, 10, seed=s),
        "randproj": lambda s: RandomProjArm(784, seed=s),
        "mnisthebb": None,  # built below (needs specs)
    }
    from src import iotypes as T
    ispec, ospec = [T.I_Vector(784)], [T.O_Vector(10)]
    arms["mnisthebb"] = lambda s: mnist_hebb_arm(ispec, ospec, seed=s)
    os.makedirs("experiments/runs", exist_ok=True)
    for tag, factory in arms.items():
        out = run_curriculum(factory, sched, seed=0,
                             out_path=f"experiments/runs/smoke_{tag}.jsonl",
                             probe_every=1000, extra_files=[__file__],
                             probe_train=ptr, probe_test=pte)
        from experiments.provenance import load_stamped
        _, rows = load_stamped(out)
        pr = [r for r in rows if "probes" in r]
        print(f"{tag:10s} " + " ".join(f"{r['probes']['ridge']:.2f}" for r in pr), flush=True)

if __name__ == "__main__":
    main()
