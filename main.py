
import datetime
import multiprocessing
import sys

import torch

from src.agents import Cfg, Agt
from src.envs import get_default, run_env
from src.envs import GridEnvCfg, GridEnv

if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    specs = GridEnv.get_specs(GridEnvCfg(size=4))
    ispec, ospec = specs
    ctx = multiprocessing.get_context("spawn")  # Avoid duplicating memory
    agt_pipe, env_pipe = ctx.Pipe()
    env_process = ctx.Process(target=run_env,
                              args=(GridEnvCfg(size=4), GridEnv, env_pipe, True,),
                              daemon=True)
    env_process.start()

    # Create agent
    if len(sys.argv) > 1:
        N_COLS = int(sys.argv[1])
    else:
        N_COLS = 200 if torch.cuda.is_available() else 50
    AGT_PATH = "./saves/agt0"
    agt = Agt(Cfg(N_COLS, ispec, ospec), AGT_PATH)
    agt.debug_init()

    t_start = datetime.datetime.now()
    while True:
        t_now = datetime.datetime.now()
        print(f"\n{t_now} (Elapsed: {t_now-t_start})")

        # Receive percept from env
        i = None
        while agt_pipe.poll():
            i = agt_pipe.recv()
        print(f"Input: {i}")
        i = i or get_default(ispec)

        o = agt.step(i)

        # Send action to env
        print(f"Actions: {o}")
        agt_pipe.send(o)
