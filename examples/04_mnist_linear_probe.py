
"""
Learning representations of MNIST digits with a local (Hebbian) rule.

Architecture: one 128-unit post-ReLU hidden layer fed the raw image, learned online
by BCM, read directly as the rep. Show digits one at a time, train a linear probe on
the hidden activations, and compare learning ON vs the SAME net frozen at init,
against two pixel baselines: a linear probe (floor) and a ReLU MLP (ceiling).

Findings (controlled same-init A/B, raw input, multiple seeds; sweep in 06_mnist_sweep.py):

1. A local rule needs stabilization or it collapses (every unit drifts to one
   feature). basic/instar/Oja all collapse without external winner-take-all; BCM
   does NOT, because its sliding threshold theta=<y^2> is per-neuron homeostasis.
   CAVEAT (likely MNIST-specific): theta stabilizes each unit but doesn't stop two
   units learning the same feature -- the BCM literature adds lateral inhibition on
   natural images. "BCM needs no competition" is an easy-MNIST result; expect
   competition to become necessary on harder data (now confirmed on CIFAR, see 08).

2. Learning beats random only a little and only in the right regime: BCM beats its
   frozen init +2.1% (ridge) at width 128, ~+3.5% at width 64, ~0 by 256-512 as
   random ReLU features saturate. Width drives absolute accuracy; the gap is a
   rescue that only shows where random is weak.

3. Read the post-ReLU hidden DIRECTLY. A second linear readout hop both bottlenecks
   and collapses the rep under learning (-12% vs frozen) -- a readout with no
   nonlinearity after it is the one place local learning reliably hurts.

4. The gap is robust across probes (kNN/ridge/logistic all positive), not a linear-
   probe artifact -- nearest-neighbour cluster structure improves too. See 05.

5. SOBERING: the learned rep does NOT beat a linear classifier on the pixels. Ladder
   (width 128, logistic, 3 seeds): linear-on-pixels 85.0, frozen 82.3, learned 84.1,
   ReLU MLP 93.0. Learning closes the gap TOWARD the linear floor but stays below it
   -- helps vs random, not useful in absolute terms. Without the floor the +2% reads
   as a win.

Caveat that bit us: learn-vs-frozen MUST be controlled (same init, fixed seed,
seed-averaged) or the gap is pure init noise that flips sign. This script's online-
SGD probe is NOT controlled (noisy live numbers); trust the 06_mnist_sweep.py same-init gap.
"""

import os
import sys

sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import argparse
import datetime
import itertools
import multiprocessing
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# isort: off
from src.agents import MNISTCfg, MNISTAgt
from src.envs import run_env
from src.envs import MNISTDataset
from src.envs import MNISTEnvCfg, MNISTEnv


def get_representations(agt: MNISTAgt):
    return agt.cols[1, 0].nr_1.actual.clone()  # the single post-ReLU hidden layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str,
        help="Directory to load saved network from.")
    parser.add_argument("--size", type=int,
        help="The number of modules to initialize in the network.")
    args = parser.parse_args()
    assert not (args.load and args.size), \
        "Size argument is only valid when initializing."

    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    cfg = MNISTEnvCfg("passive")
    ispec, ospec = MNISTEnv.get_specs(cfg)
    ctx = multiprocessing.get_context("spawn")  # Avoid duplicating memory
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    env_process = ctx.Process(
        target=run_env,
        args=(cfg, MNISTEnv, input_queue, output_queue, True,),
        daemon=True)
    env_process.start()

    # Create agent
    if args.load:
        agt = MNISTAgt.load(args.load)
    else:
        if args.size:
            N_COLS = args.size
        else:
            N_COLS = 20
        AGT_PATH = f"{project_root_path}/saves/mnist_repr_agt-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"
        agt = MNISTAgt(MNISTCfg(ispec, ospec), AGT_PATH)
    assert isinstance(agt, MNISTAgt)  # load() returns base AgtBase; narrow for use_lrn
    agt.debug_init()

    # Actual (experimental): Takes in agent's internal representations
    classifier = nn.Linear(get_representations(agt).shape[0], 10)
    optim = torch.optim.SGD(classifier.parameters(), lr=1e-2)

    # No learning control: Above but with static weights
    no_lrn_agt = MNISTAgt(MNISTCfg(ispec, ospec),
        f"{project_root_path}/saves/mnist_repr_no_lrn_agt-{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}")
    no_lrn_classifier = nn.Linear(get_representations(no_lrn_agt).shape[0], 10)
    no_lrn_optim = torch.optim.SGD(no_lrn_classifier.parameters(), lr=1e-2)

    # Regular control: ReLU MLP on the image (the strong nonlinear ceiling)
    control_classifier = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    control_optim = torch.optim.SGD(control_classifier.parameters(), lr=1e-2)

    # Linear baseline: a linear probe straight on the image (the floor -- is the
    # learned rep even worth more than a linear classifier on the raw pixels?)
    linear_classifier = nn.Linear(784, 10)
    linear_optim = torch.optim.SGD(linear_classifier.parameters(), lr=1e-2)

    def training_step(x, classifier, label, optimizer):
        x = x.to(torch.get_default_device()).to(torch.get_default_dtype())
        label = label.to(torch.get_default_device()).to(torch.get_default_dtype())
        pred = classifier(x)
        loss = F.cross_entropy(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mnist_test = MNISTDataset(train=False)

    REPORT_INTERVAL = 10
    TEST_INTERVAL = 50
    TEST_SIZE = 500
    for step in itertools.count():
        if step % REPORT_INTERVAL == 0 and step % TEST_INTERVAL != 0:
            print(f"Step {step}")

        # Receive percept from env
        i = None
        label = None
        while True:
            while not input_queue.empty():
                i, label = input_queue.get()
            if i is not None:
                break

        # Train classifiers ###################################################
        # Actual
        agt.step(i, disable_print=True)
        training_step(get_representations(agt), classifier, label, optim)

        # No learning control
        no_lrn_agt.step(i, use_lrn=False, disable_print=True)
        training_step(get_representations(no_lrn_agt), no_lrn_classifier, label, no_lrn_optim)

        # Regular control + linear baseline (both on the image)
        img, = i
        training_step(img, control_classifier, label, control_optim)
        training_step(img, linear_classifier, label, linear_optim)

        # Test classifiers ####################################################
        if step % TEST_INTERVAL == 0:
            print(f"\nTesting on step {step}...")
            correct = 0
            no_lrn_correct = 0
            control_correct = 0
            linear_correct = 0
            total = 0
            with torch.no_grad():
                for i in random.sample(range(len(mnist_test)), k=TEST_SIZE):
                    img, label = mnist_test[i]
                    img, label = img.to(torch.get_default_device()), label.to(torch.get_default_device())

                    # Actual
                    agt.step([img], use_lrn=False, disable_print=True)  # don't learn on test
                    representations = get_representations(agt)
                    pred = classifier(representations)
                    if torch.argmax(pred) == torch.argmax(label):
                        correct += 1

                    # No lrn control
                    no_lrn_agt.step([img], use_lrn=False, disable_print=True)
                    representations = get_representations(no_lrn_agt)
                    pred = no_lrn_classifier(representations)
                    if torch.argmax(pred) == torch.argmax(label):
                        no_lrn_correct += 1

                    # Regular control (ReLU MLP on image)
                    control_pred = control_classifier(img)
                    if torch.argmax(control_pred) == torch.argmax(label):
                        control_correct += 1

                    # Linear baseline (linear probe on image)
                    if torch.argmax(linear_classifier(img)) == torch.argmax(label):
                        linear_correct += 1
                        
                    total += 1

            accuracy = 100*correct/total
            no_lrn_accuracy = 100*no_lrn_correct/total
            control_accuracy = 100*control_correct/total
            linear_accuracy = 100*linear_correct/total
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"No lrn control accuracy: {no_lrn_accuracy:.2f}%")
            print(f"Linear-on-image baseline: {linear_accuracy:.2f}%")
            print(f"Control (ReLU MLP) accuracy: {control_accuracy:.2f}%\n")
            with SummaryWriter("runs/mnist_representations") as writer:
                writer.add_scalars("Accuracy", {
                    "accuracy": accuracy,
                    "no lrn control accuracy": no_lrn_accuracy,
                    "linear-on-image baseline": linear_accuracy,
                    "control accuracy": control_accuracy
                }, global_step=step)
