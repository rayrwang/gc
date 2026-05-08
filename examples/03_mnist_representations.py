
"""
(WIP) Learning representations of random MNIST digits
"""

import os
import sys
sys.path.insert(0, (project_root_path := os.path.dirname(os.path.dirname(__file__))))

import multiprocessing
import argparse
import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.agents import MNISTCfg, MNISTAgt
from src.envs import run_env
from src.envs import MNISTDataset
from src.envs import MNISTEnvCfg, MNISTEnv


def get_representations(agt: MNISTAgt):
    return torch.cat([
        col.nr_1[0].clone() 
        for col in agt.cols.values()
        if not agt.is_i(col.loc)])  # Don't look at inputs (cheating)


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
        AGT_PATH = f"{project_root_path}/saves/mnist_repr_agt"
        agt = MNISTAgt(MNISTCfg(ispec, ospec), AGT_PATH)
    agt.debug_init()

    # Actual (experimental): Takes in agent's internal representations
    classifier = nn.Linear(get_representations(agt).shape[0], 10)
    optim = torch.optim.SGD(classifier.parameters(), lr=1e-2)

    # No learning control: Above but with static weights
    no_lrn_agt = MNISTAgt(MNISTCfg(ispec, ospec), f"{project_root_path}/saves/mnist_repr_no_lrn_agt")
    no_lrn_classifier = nn.Linear(get_representations(no_lrn_agt).shape[0], 10)
    no_lrn_optim = torch.optim.SGD(no_lrn_classifier.parameters(), lr=1e-2)

    # Regular control: Takes in the image
    control_classifier = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    control_optim = torch.optim.SGD(control_classifier.parameters(), lr=1e-2)

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
        representations = get_representations(agt)
        representations = representations.to(torch.get_default_device()).to(torch.get_default_dtype())
        label = label.to(torch.get_default_device()).to(torch.get_default_dtype())
        pred = classifier(representations)
        loss = F.cross_entropy(pred, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # No learning control
        no_lrn_agt.step(i, use_lrn=False, disable_print=True)
        representations = get_representations(no_lrn_agt)
        representations = representations.to(torch.get_default_device()).to(torch.get_default_dtype())
        label = label.to(torch.get_default_device()).to(torch.get_default_dtype())
        pred = no_lrn_classifier(representations)
        loss = F.cross_entropy(pred, label)
        no_lrn_optim.zero_grad()
        loss.backward()
        no_lrn_optim.step()

        # Regular control
        img, = i
        img = img.to(torch.get_default_device()).to(torch.get_default_dtype())
        label = label.to(torch.get_default_device()).to(torch.get_default_dtype())
        pred = control_classifier(img)
        loss = F.cross_entropy(pred, label)
        control_optim.zero_grad()
        loss.backward()
        control_optim.step()

        # Test classifiers ####################################################
        if step % TEST_INTERVAL == 0:
            print(f"\nTesting on step {step}...")
            correct = 0
            no_lrn_correct = 0
            control_correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(TEST_SIZE):
                    img, label = random.choice(mnist_test)
                    img, label = img.to(torch.get_default_device()), label.to(torch.get_default_device())

                    # Actual
                    agt.step([img], disable_print=True)
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

                    # Regular control
                    control_pred = control_classifier(img)
                    if torch.argmax(control_pred) == torch.argmax(label):
                        control_correct += 1
                    total += 1

            accuracy = 100*correct/total
            no_lrn_accuracy = 100*no_lrn_correct/total
            control_accuracy = 100*control_correct/total
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"No lrn control accuracy: {no_lrn_accuracy:.2f}%")
            print(f"Control accuracy: {control_accuracy:.2f}%\n")
            with SummaryWriter("runs/mnist_representations") as writer:
                writer.add_scalars("Accuracy", {
                    "accuracy": accuracy,
                    "no lrn control accuracy": no_lrn_accuracy,
                    "control accuracy": control_accuracy
                }, global_step=step)
