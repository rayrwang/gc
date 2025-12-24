
import random
from dataclasses import dataclass
from abc import ABC

import mss
import numpy as np
import cv2
import torch

from . import iotypes as T


def get_default(iospec: list[T.I | T.O]) -> list[torch.Tensor]:
    """ Get all zeros (default) inputs or outputs """
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
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # convert from grid to image space
        cv2.imshow("env", img)
        cv2.waitKey(1)


class ComputerEnv:
    @staticmethod
    def get_specs():
        screen = T.I_Video(h=int(2560/2), w=int(1440/2))

        keyboard = T.O_Keyboard(keys=['w', 'a', 's', 'd'])
        mouse = T.O_Mouse()

        ispec = [screen]
        ospec = [keyboard, mouse]
        return ispec, ospec

    def __init__(self):
        ...

    def _step(self, a: list[torch.Tensor]) ->  list[torch.Tensor]:
        # action (agent's output) -> percept (agent's input)

        if a:
            keys, (mouse_movement, mouse_buttons) = a
            assert len(keys) == len(self.keyboard.keys)
            assert len(mouse_movement) == 2
            assert len(mouse_buttons) == len(self.mouse.buttons)

            # Press keys
            for i, key in enumerate(keys):
                if key == 0:
                    pyautogui.keyUp(self.keyboard.keys[i])
                elif key == 1:
                    pyautogui.keyDown(self.keyboard.keys[i])

            # Mouse
            pyautogui.move(mouse_movement)

            for i, button in enumerate(mouse_buttons):
                if button == 0:
                    pyautogui.mouseUp(self.mouse.buttons[i])
                elif button == 1:
                    pyautogui.mouseDown(self.mouse.buttons[i])

        # Take screenshot
        with mss.mss(with_cursor=True) as sct:
            monitor = sct.monitors[0]
            img = np.array(sct.grab(monitor))
            img = cv2.resize(img, (self.screen.h, self.screen.w))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return [img]


# Real world environments #####################################################
@dataclass
class RobotEnvCfg:
    n_cameras: int
    n_motors: int


class RobotEnv:
    @staticmethod
    def get_specs(cfg: RobotEnvCfg):
        h, w = 960, 1280

        cameras = []
        motors = []

        for i in range(cfg.n_cameras):
            cameras.append(T.I_Video(name=f"camera{i}", h=h, w=w))

        for i in range(cfg.n_motors):
            motors.append(T.O_Scalar(name=f"motor{i}"))

        ispec = cameras
        ospec = motors

        return ispec. ospec

    def __init__(self, cfg: RobotEnvCfg):
        # Video captures
        self.caps = []
        for i in range(cfg.n_cameras):
            new_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.caps.append(new_cap)

    def _step(self, a):

        # Take pictures
        for i in range(self.n_cameras):
            self.caps[i].grab()
        images = []
        for i in range(self.n_cameras):
            _, img = self.caps[i].retrieve()
            self.images.append(img)

        return images
