
from dataclasses import dataclass

import cv2
import torch

from . import iotypes as T


def get_default(iospec: list[T.I | T.O]) -> list[torch.Tensor]:
    """Get all zeros (default) inputs or outputs"""
    default = []
    for spec in iospec:
        if type(spec) in [T.I_Vector, T.O_Vector]:
            default.append(torch.zeros(spec.d))
        else:
            raise NotImplementedError
    return default


def run_env(cfg, env: type, pipe, show: bool):
    """
    cfg should be instance

    env should be the class itself
    """
    specs = env.get_specs(cfg)
    if cfg is None:
        env_instance = env()
    else:
        env_instance = env(cfg)
    while True:
        # Receive action from agt
        o = None
        while pipe.poll():
            o = pipe.recv()
        o = o or get_default(specs[1])
        i = env_instance._step(o)

        # Send percept to agt
        pipe.send(i)

        if show:
            env_instance._show()


# Virtual environments ########################################################
@dataclass
class GridEnvCfg:
    size: int = 4
class GridEnv:
    @staticmethod
    def get_specs(cfg: GridEnvCfg) -> tuple[list[T.I], list[T.O]]:
        ispec = [T.I_Vector(d=cfg.size**2)]

        # Horizontal and vertical movement,
            # place and break square
        n_actions = 4
        ospec = [T.O_Vector(d=n_actions)]
        return ispec, ospec

    def __init__(self, cfg: GridEnvCfg):
        self.size = size = cfg.size

        self.ispec, self.ospec = self.get_specs(cfg)

        self.grid = torch.zeros(size, size, dtype=torch.int32)
        self.pos = (0, 0)

    def _step(self, a: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(a) == len(self.ospec)

        right, left, up, down = a[0]
        pos_x, pos_y = self.pos
        threshold = 0.2
        if right > threshold and (pos_x+1) < self.size:
            if self.grid[pos_x+1, pos_y] == 0:
                self.pos = (pos_x+1, pos_y)
        elif left > threshold and (pos_x-1) >= 0:
            if self.grid[pos_x-1, pos_y] == 0:
                self.pos = (pos_x-1, pos_y)
        if up > threshold and (pos_y+1) < self.size:
            if self.grid[pos_x, pos_y+1] == 0:
                self.pos = (pos_x, pos_y+1)
        elif down > threshold and (pos_y-1) >= 0:
            if self.grid[pos_x, pos_y-1] == 0:
                self.pos = (pos_x, pos_y-1)

        grid = self.grid.clone()
        grid[*self.pos] = 1
        p = [grid.reshape(self.size**2)]
        return p

    def _show(self) -> None:
        img = self.grid.clone()

        img[*self.pos] = 255  # Draw agt body
        img = img.to(torch.uint8).cpu().numpy()
        # convert from grid to image space
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow("env", img)
        cv2.waitKey(1)


# Real world environments #####################################################
