
"""
Learning representations of MNIST digits with a local (Hebbian) rule.

Architecture: one 128-unit post-ReLU hidden layer, fed the raw grayscale image,
learned online by BCM, read directly as the representation. (Width 128: the
learn-vs-random gap is largest at narrow widths and washes out by ~256 as random
ReLU features saturate the task; 128 trades some gap for ~85% absolute. Input is
raw -- BCM is continuous, so it does not need the discretized input the old
discrete rule did.) Show digits one at a time, train a linear probe on the hidden
activations, and compare learning ON vs the SAME net frozen at its random init,
against two image baselines: a linear probe on the raw pixels (the floor) and a
ReLU MLP on the pixels (the strong nonlinear ceiling).

Findings (controlled same-init A/B, raw input, multiple seeds; rule and machinery
search in the gitignored _sweep.py, width/probe sweep on this agent):

1. A local rule needs some stabilizing mechanism or it collapses: without one
   every unit drifts to the same feature and the rep degenerates to chance.
   basic / instar / Oja all collapse here without an external winner-take-all and
   are otherwise interchangeable. BCM did NOT collapse without one: its sliding
   threshold theta = <y^2> is computed from each neuron's own activity (per-neuron
   homeostasis -- a unit potentiates only above its own running average), and on
   this task that was enough on its own -- adding explicit competition to BCM did
   not help and slightly hurt. Hence this agent uses BCM (fc.lrn_adaptive).
   CAVEAT (likely MNIST-specific): theta is per-neuron homeostasis, not a direct
   between-neuron competition signal -- it stabilizes each unit but does not
   reliably stop two units learning the same feature. The BCM literature reports
   exactly that redundancy on natural images and routinely ADDS lateral inhibition
   / DoG / contrast normalization to fix it. So "BCM needs no competition" is an
   easy-MNIST result, not a property of BCM; expect competition to become
   necessary on harder / natural data -- an untested prediction to check before
   relying on it.

2. Learning then beats random, but only a little and only in the right regime.
   Reading the post-ReLU hidden directly, BCM beats its own frozen init by a
   controlled +2.1% (ridge) at width 128, rising to ~+3.5% at width 64 and
   shrinking to ~0 by 256-512 as random ReLU features saturate (~90% at 512). So
   width drives absolute accuracy and the learning gap is a rescue that only shows
   where random is weak: narrow width here, the harsh spike activation (+12% when
   random spike features are weak ~45%), harder datasets.

3. Read the post-ReLU hidden DIRECTLY. Stacking a second linear readout hop and
   reading THAT instead both bottlenecks the rep and collapses it under learning
   (BCM on the final linear projection: -12% vs frozen). A readout layer with no
   nonlinearity after it is the one place local learning reliably hurts.

4. The gap is robust across probe types, not a linear-probe artifact: kNN, ridge,
   and logistic regression are all positive. At width 128 ridge/logistic lead
   (+2.1 / +2.6) and the parameter-free kNN is smaller (+1.6); at width 64, where
   the effect is strongest, all three agree (~+3.3 to +3.8), so the nearest-
   neighbour cluster structure improves too, not just linear decodability. See
   05_mnist_all_probes.py.

5. SOBERING -- the learned rep does NOT beat a linear classifier on the raw
   pixels. Full ladder (width 128, raw, logistic, 3 seeds): linear-on-pixels
   85.0%, frozen rep 82.3%, learned rep 84.1% (+1.8 over frozen), ReLU MLP on
   pixels 93.0%. So learning closes the gap TOWARD the linear floor but stays
   BELOW it -- a 128-unit ReLU projection is worth slightly less than logistic
   regression on the 784 pixels (likely more below with full data; here only an
   8k subset). "Learning helps" is true vs random, but the representation is not
   useful in absolute terms on MNIST. That is what the linear floor reveals;
   without it the +2% reads as a win.

Caveat that bit us repeatedly: the learn-vs-frozen comparison MUST be controlled
(same init weights, fixed seed, averaged over seeds) or the gap is pure init
noise -- on a single uncontrolled run it flips sign between runs. The probe in
this script (online SGD, independent inits) is NOT controlled, so its live
numbers are noisy; the trustworthy gap is the _sweep.py / same-init measurement.
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
    return agt.cols[1, 0].nr_1[0].clone()  # the single post-ReLU hidden layer


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
    no_lrn_agt = MNISTAgt(MNISTCfg(ispec, ospec), f"{project_root_path}/saves/mnist_repr_no_lrn_agt")
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
