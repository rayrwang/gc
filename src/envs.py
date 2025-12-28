
from dataclasses import dataclass
from abc import ABC, abstractmethod

import cv2
import torch

from . import iotypes as T


class EnvCfgBase(ABC):
    pass
class EnvBase(ABC):
    @staticmethod
    @abstractmethod
    def get_specs(cfg: EnvCfgBase) -> tuple[list[T.I_Base], list[T.O_Base]]:
        ...

    @abstractmethod
    def _step(self, a: list[torch.Tensor]) -> list[torch.Tensor]:
        ...

    @abstractmethod
    def _show(self) -> None:
        ...


def get_default(iospec: list[T.I_Base | T.O_Base]) -> list[torch.Tensor]:
    """Get all zeros inputs or outputs"""
    default = []
    for spec in iospec:
        if type(spec) in [T.I_Vector, T.O_Vector]:
            default.append(torch.zeros(spec.d))
        else:
            raise NotImplementedError
    return default


def run_env(
        cfg: EnvCfgBase,
        env: type[EnvBase],
        percept_queue,
        action_queue,
        show: bool):
    env_instance = env(cfg)
    _, ospec = env.get_specs(cfg)
    while True:
        # Receive action from agt
        o = None
        while not action_queue.empty():
            o = action_queue.get()
        o = o or get_default(ospec)
        i = env_instance._step(o)

        # Send percept to agt
        percept_queue.put(i)

        if show:
            env_instance._show()


# Virtual environments ########################################################
@dataclass
class GridEnvCfg(EnvCfgBase):
    width: int = 4
class GridEnv(EnvBase):
    def __init__(self, cfg: GridEnvCfg):
        self.cfg = cfg
        self.width = width = cfg.width

        self.ispec, self.ospec = self.get_specs(cfg)

        self.grid = torch.zeros(width, width, dtype=torch.int32, device="cpu")
        self.pos = (0, 0)

        self.opencv_init = False

    @staticmethod
    def get_specs(cfg: GridEnvCfg) -> tuple[list[T.I_Base], list[T.O_Base]]:
        ispec = [T.I_Vector(d=cfg.width**2)]

        # Horizontal and vertical movement,
            # place and break square
        n_actions = 4
        ospec = [T.O_Vector(d=n_actions)]
        return ispec, ospec

    def _step(self, a: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(a) == len(self.ospec)

        # React to actions
        right, left, up, down = a[0]
        pos_x, pos_y = self.pos
        threshold = 0.2
        d_x, d_y = 0, 0

        if right > threshold:
            d_x += 1
        if left > threshold:
            d_x -= 1
        new_x = min(max(pos_x+d_x, 0), self.width-1)
        if self.grid[new_x, pos_y] != 0:  # Occupied
            new_x = pos_x

        if up > threshold:
            d_y += 1
        if down > threshold:
            d_y -= 1
        new_y = min(max(pos_y+d_y, 0), self.width-1)
        if self.grid[pos_x, new_y] != 0:  # Occupied
            new_y = pos_y

        self.pos = new_x, new_y

        # Return percepts
        grid = self.grid.clone()
        grid[*self.pos] = 1
        p = [grid.reshape(self.width**2)]
        return p

    def _show(self) -> None:
        img = self.grid.clone()

        img[*self.pos] = 255  # Draw agt body
        img = img.to(torch.uint8).cpu().numpy()
        # Convert from grid to image space
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if not self.opencv_init:   
            self.opencv_init = True 
            cv2.namedWindow("grid env", cv2.WINDOW_NORMAL)
            WINDOW_H = 200  # TODO calculate using monitor resolution
            cv2.resizeWindow("grid env", (int(WINDOW_H*img.shape[1]/img.shape[0]),
                                          WINDOW_H))
        cv2.imshow("grid env", img)
        cv2.waitKey(1)


# Real world environments #####################################################
