
import multiprocessing
import argparse
import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents import MNISTCfg, MNISTAgt
from src.envs import get_default, run_env
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
        AGT_PATH = "./saves/agt0"
        agt = MNISTAgt(MNISTCfg(ispec, ospec), AGT_PATH)
    agt.debug_init()

    # Classifier: Takes in agent's internal representations
    classifier = nn.Linear(get_representations(agt).shape[0], 10)
    optim = torch.optim.SGD(classifier.parameters(), lr=1e-1)

    # Control classifier: Takes in the image
    control_classifier = nn.Linear(784, 10)
    control_optim = torch.optim.SGD(control_classifier.parameters(), lr=1e-1)

    mnist_test = MNISTDataset(train=False)
    wait_propagate = 5  # Number of steps to wait for the image to propagate through the agent
    label = None
    for step in itertools.count():
        if step % 10 == 0 and step % 100 != 0:
            print(f"Step {step}")

        # Receive percept from env
        i = None
        while not input_queue.empty():
            i, label = input_queue.get()
        i = i or get_default(ispec)

        # Train classifiers
        if label is not None:
            # Actual classifier
            for _ in range(wait_propagate):
                agt.step(i, True)
            representations = get_representations(agt)
            representations = representations.to(torch.get_default_device()).to(torch.get_default_dtype())
            label = label.to(torch.get_default_device()).to(torch.get_default_dtype())
            pred = F.softmax(classifier(representations), dim=0)
            loss = F.cross_entropy(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Control classifier
            img, = i
            img = img.to(torch.get_default_device()).to(torch.get_default_dtype())
            label = label.to(torch.get_default_device()).to(torch.get_default_dtype())
            pred = F.softmax(control_classifier(img), dim=0)
            loss = F.cross_entropy(pred, label)
            control_optim.zero_grad()
            loss.backward()
            control_optim.step()

        # Test classifiers
        if step % 50 == 0:
            print(f"\nTesting on step {step}...")
            correct = 0
            control_correct = 0
            total = 0
            with torch.no_grad():
                for _ in range(40):
                    img, label = random.choice(mnist_test)
                    img, label = img.to(torch.get_default_device()), label.to(torch.get_default_device())

                    # Actual classifier
                    for _ in range(wait_propagate):
                        agt.step([img], True)
                    representations = get_representations(agt)
                    pred = classifier(representations)
                    if torch.topk(pred, 1).indices[0] == torch.topk(label, 1).indices[0]:
                        correct += 1

                    # Control classifier
                    control_pred = control_classifier(img)
                    if torch.topk(control_pred, 1).indices[0] == torch.topk(label, 1).indices[0]:
                        control_correct += 1
                    total += 1
            print(f"Accuracy: {100*correct/total:.2f}%")
            print(f"Control accuracy: {100*control_correct/total:.2f}%\n")
