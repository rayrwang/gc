
import random
import signal
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import torch
import torchvision
import torchvision.transforms.functional
from torch.utils.data import Dataset

from . import iotypes as T

Specs = tuple[list[T.I_Base], list[T.O_Base]]
Aux = Any  # Additional info e.g. labels
Percepts = list[torch.Tensor]
Actions = list[torch.Tensor]


class EnvCfgBase(ABC):
    pass
class EnvBase[CfgT: EnvCfgBase](ABC):
    def __init__(self, cfg: CfgT):
        ...

    @staticmethod
    @abstractmethod
    def get_specs(cfg: CfgT) -> Specs:
        ...

    @abstractmethod
    def _step(self, a: Actions) -> Percepts | tuple[Percepts, Aux]:
        ...

    @abstractmethod
    def _show(self) -> None:
        ...


def get_default(iospec: Sequence[T.I_Base | T.O_Base]) -> Percepts | Actions:
    """Get all zeros inputs or outputs"""
    default = []
    for spec in iospec:
        if isinstance(spec, (T.I_Vector, T.O_Vector)):
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
    # Match the main process's dtype in this spawned subprocess. (Was a module-level import
    # side-effect that silently flipped torch's global default to fp16 for ANY importer of
    # envs; set here instead so importing a Dataset no longer mutates global state.)
    torch.set_default_dtype(torch.float16)
    # Suppress keyboard interrupt traceback
    signal.signal(signal.SIGINT, lambda _, __: sys.exit(0))

    env_instance = env(cfg)
    _, ospec = env.get_specs(cfg)

    COOLDOWN_SEND = 0.01
    t_prev_send = time.perf_counter()
    while True:
        # Receive action from agt
        o = None
        while not action_queue.empty():
            o = action_queue.get()
        o = o or get_default(ospec)
        i = env_instance._step(o)

        # Send percept to agt
        if (time.perf_counter() - t_prev_send) > COOLDOWN_SEND:
            t_prev_send = time.perf_counter()
            percept_queue.put(i)

        if show:
            env_instance._show()


# Virtual environments ########################################################
@dataclass
class GridEnvCfg(EnvCfgBase):
    width: int = 4
class GridEnv(EnvBase[GridEnvCfg]):
    def __init__(self, cfg: GridEnvCfg):
        self.cfg = cfg
        self.width = width = cfg.width

        self.ispec, self.ospec = self.get_specs(cfg)

        self.grid = torch.zeros(width, width, dtype=torch.int32, device="cpu")
        self.pos = (0, 0)

        self.opencv_init = False

    @staticmethod
    def get_specs(cfg: GridEnvCfg) -> Specs:
        ispec: list[T.I_Base] = [T.I_Vector(d=cfg.width**2)]

        # Horizontal and vertical movement,
        # TODO place and break square
        n_actions = 4
        ospec: list[T.O_Base] = [T.O_Vector(d=n_actions)]
        return ispec, ospec

    def _step(self, a: Actions) -> Percepts:
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


class MNISTDataset(Dataset):
    def __init__(self, train=True):
        self.mnist = torchvision.datasets.MNIST("data", train=train, download=True)
        self.len = len(self.mnist)

    def __getitem__(self, i):  # ty: ignore[invalid-method-override]
        (image_raw, label_raw) = self.mnist[i]
        image = torchvision.transforms.functional.to_tensor(image_raw)
        image = image.reshape(-1)
        label = torch.zeros(10)  # TODO just use raw label?
        label[label_raw] = 1
        return image, label

    def __len__(self):
        return self.len


class FashionMNISTDataset(Dataset):
    # Drop-in for MNISTDataset: same 28x28 grayscale, 10 classes, so it reuses
    # MNISTAgt / MNISTEnv specs unchanged. Kept attribute name `mnist` so the
    # offline probe example can read .mnist.data / .mnist.targets like 05 does.
    def __init__(self, train=True):
        self.mnist = torchvision.datasets.FashionMNIST("data", train=train, download=True)
        self.len = len(self.mnist)

    def __getitem__(self, i):  # ty: ignore[invalid-method-override]
        (image_raw, label_raw) = self.mnist[i]
        image = torchvision.transforms.functional.to_tensor(image_raw)
        image = image.reshape(-1)
        label = torch.zeros(10)
        label[label_raw] = 1
        return image, label

    def __len__(self):
        return self.len
@dataclass
class MNISTEnvCfg(EnvCfgBase):
    # active  : choose image based on agent's action
    # passive : randomly show images
    mode: Literal["active", "passive"]
class MNISTEnv(EnvBase[MNISTEnvCfg]):
    def __init__(self, cfg: MNISTEnvCfg):
        self.cfg = cfg
        self.mode = cfg.mode

        self.mnist = MNISTDataset()

        (self.image, self.label) = random.choice(self.mnist)

        self.opencv_init = False

    @staticmethod
    def get_specs(cfg: MNISTEnvCfg) -> Specs:
        ispec: list[T.I_Base] = [T.I_Vector(d=28*28)]
        ospec: list[T.O_Base] = [T.O_Vector(d=10)]
        return ispec, ospec

    def _step(self, a: Actions) -> tuple[Percepts, Aux]:
        if self.mode == "active":
            # Get image of digit corresponding to largest activation in action
            digits, = a
            assert digits.shape == (10,)
            # TODO? use None i/o when unavailable rather than all zeros (get_default)
            if not torch.allclose(digits, torch.zeros(10, dtype=torch.get_default_dtype(), device="cpu")):
                # Get new digit
                digit = torch.argmax(digits)
                while True:
                    (image, label) = random.choice(self.mnist)
                    label_int = torch.argmax(label)
                    if label_int == digit:
                        self.image, self.label = image, label
                        break
        elif self.mode == "passive":
            (self.image, self.label) = random.choice(self.mnist)
        return [self.image.clone()], self.label

    def _show(self) -> None:
        img = self.image.clone() * 255

        img = img.to(torch.uint8).cpu().numpy()
        img = img.reshape(28, 28, 1)
        if not self.opencv_init:
            self.opencv_init = True 
            cv2.namedWindow("mnist env", cv2.WINDOW_NORMAL)
            WINDOW_H = 200  # TODO calculate using monitor resolution
            cv2.resizeWindow("mnist env", (int(WINDOW_H*img.shape[1]/img.shape[0]),
                                          WINDOW_H))
        cv2.imshow("mnist env", img)
        cv2.waitKey(1)


class CIFARDataset(Dataset):
    def __init__(self, train=True):
        self.cifar = torchvision.datasets.CIFAR10("data", train=train, download=True)
        self.len = len(self.cifar)

    def __getitem__(self, i):  # ty: ignore[invalid-method-override]
        (image_raw, label_raw) = self.cifar[i]
        image = torchvision.transforms.functional.to_tensor(image_raw)  # (3, 32, 32) in [0, 1]
        label = torch.zeros(10)
        label[label_raw] = 1
        return image, label

    def __len__(self):
        return self.len
@dataclass
class CIFAREnvCfg(EnvCfgBase):
    # active  : choose image based on agent's action
    # passive : randomly show images
    mode: Literal["active", "passive"]
class CIFAREnv(EnvBase[CIFAREnvCfg]):
    def __init__(self, cfg: CIFAREnvCfg):
        self.cfg = cfg
        self.mode = cfg.mode

        self.cifar = CIFARDataset()

        (self.image, self.label) = random.choice(self.cifar)

        self.opencv_init = False

    @staticmethod
    def get_specs(cfg: CIFAREnvCfg) -> Specs:
        ispec: list[T.I_Base] = [T.I_Video(w=32, h=32, c=3)]
        ospec: list[T.O_Base] = [T.O_Vector(d=10)]
        return ispec, ospec

    def _step(self, a: Actions) -> tuple[Percepts, Aux]:
        if self.mode == "active":
            classes, = a
            assert classes.shape == (10,)
            if not torch.allclose(classes, torch.zeros(10, dtype=torch.get_default_dtype(), device="cpu")):
                target = torch.argmax(classes)
                while True:
                    (image, label) = random.choice(self.cifar)
                    if torch.argmax(label) == target:
                        self.image, self.label = image, label
                        break
        elif self.mode == "passive":
            (self.image, self.label) = random.choice(self.cifar)
        return [self.image.clone()], self.label

    def _show(self) -> None:
        img = (self.image.clone() * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        img = img[:, :, ::-1].copy()  # CHW float -> HWC uint8, RGB -> BGR for cv2
        if not self.opencv_init:
            self.opencv_init = True
            cv2.namedWindow("cifar env", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("cifar env", 200, 200)
        cv2.imshow("cifar env", img)
        cv2.waitKey(1)


class CIFAR100Dataset(Dataset):
    def __init__(self, train=True):
        self.cifar = torchvision.datasets.CIFAR100("data", train=train, download=True)
        self.len = len(self.cifar)

    def __getitem__(self, i):  # ty: ignore[invalid-method-override]
        (image_raw, label_raw) = self.cifar[i]
        image = torchvision.transforms.functional.to_tensor(image_raw)  # (3, 32, 32) in [0, 1]
        label = torch.zeros(100)
        label[label_raw] = 1
        return image, label

    def __len__(self):
        return self.len


# Real world environments #####################################################
# TODO
