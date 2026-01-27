
import datetime
import multiprocessing
import argparse

import torch

from src.agents import MNISTCfg, MNISTAgt
from src.envs import get_default, run_env
from src.envs import MNISTEnvCfg, MNISTEnv

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
    cfg = MNISTEnvCfg("active")
    ispec, ospec = MNISTEnv.get_specs(cfg)
    ctx = multiprocessing.get_context("spawn")  # Avoid duplicating memory
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    env_process = ctx.Process(target=run_env,
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
            N_COLS = 200 if torch.cuda.is_available() else 50
        AGT_PATH = "./saves/agt0"
        agt = MNISTAgt(MNISTCfg(N_COLS, ispec, ospec), AGT_PATH)
    agt.debug_init()

    t_start = datetime.datetime.now()
    while True:
        t_now = datetime.datetime.now()
        print(f"\n{t_now} (Elapsed: {t_now-t_start})")

        # Receive percept from env
        i = None
        while not input_queue.empty():
            i, _ = input_queue.get()
        i = i or get_default(ispec)
        print(f"Input: {i}")

        o = agt.step(i)

        # Send action to env
        print(f"Output: {o}")
        output_queue.put(o)

    # TODO Test quality of representations

