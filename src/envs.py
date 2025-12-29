
from dataclasses import dataclass
from abc import ABC, abstractmethod

import cv2
import torch
import mss
import numpy as np
# import pyautogui

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
        elif type(spec) is T.I_Video:
            default.append(torch.zeros(spec.h, spec.w, spec.c))
        elif type(spec) is T.O_Keyboard:
            default.append(torch.zeros(len(spec.keys)))
        elif type(spec) is T.O_MouseMovement:
            default.append(torch.zeros(2))
        elif type(spec) is T.O_MouseButtons:
            default.append(torch.zeros(len(spec.buttons)))
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

        # Perform actions
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


@dataclass
class ComputerEnvCfg(EnvCfgBase):
    image_w: int
    image_h: int
    keys: list[str]
class ComputerEnv(EnvBase):
    @staticmethod
    def get_specs(cfg: ComputerEnvCfg):
        screen = T.I_Video(w=cfg.image_w, h=cfg.image_h)

        keyboard = T.O_Keyboard(keys=cfg.keys)
        mouse_movement = T.O_MouseMovement()
        mouse_buttons = T.O_MouseButtons()

        ispec = [screen]
        ospec = [keyboard, mouse_movement, mouse_buttons]
        return ispec, ospec

    def __init__(self, cfg: ComputerEnvCfg):
        self.cfg = cfg
        self.image_w = cfg.image_w
        self.image_h = cfg.image_h

        self.ispec, self.ospec = self.get_specs(cfg)

        self.screen = None  # In BGR

        self.opencv_init: bool = False

    def _step(self, a: list[torch.Tensor]) ->  list[torch.Tensor]:
        # TODO perform keyboard and mouse actions

        # Take screenshot
        with mss.mss(with_cursor=True) as sct:
            monitor = sct.monitors[0]
            img = np.array(sct.grab(monitor))
            img = img[:, :, 0:3]
            img = cv2.resize(img, (self.image_w, self.image_h))
            self.screen = img

        p = cv2.cvtColor(self.screen.copy(), cv2.COLOR_BGR2RGB)
        p = torch.from_numpy(p)
        return [p]

    def _show(self):
        img = self.screen.astype(np.uint8)
        if not self.opencv_init:   
            self.opencv_init = True 
            cv2.namedWindow("computer env", cv2.WINDOW_NORMAL)
            WINDOW_H = 200  # TODO calculate using monitor resolution
            cv2.resizeWindow("computer env", (int(WINDOW_H*img.shape[1]/img.shape[0]),
                                              WINDOW_H))
        cv2.imshow("computer env", img)
        cv2.waitKey(1)


# Real world environments #####################################################
# TODO
