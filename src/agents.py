
"""
NOTE outline:

init:
    env, debugger
    agt: initial architecture
        cols: activations, weights
            io cols
            internal cols
        conns
for each timestep:
    step env:
        receive actions
        if needed: change state
        provide observations
    step agt:
        receive inputs (propagate to input cols)
        for each col:
            apply self inhibition to activations
                by using actual and expected
            global (within whole agent):
                learning rules for conns
                pruning and growing conns
                output to other cols using conns
            local (within col):
                apply learning rules
                apply activity rules by adding the propagations
                    to the updated values
            
            send debug info
        provide outputs
"""

from __future__ import annotations

import pickle
import os
from typing import Annotated
from dataclasses import dataclass
import time
import random
import math
from enum import Enum
from abc import ABC, abstractmethod
import multiprocessing
import shutil

import torch
import numpy as np
from tqdm import tqdm

from . import iotypes as T
from . import funcs as fc

class Dir(Enum):  # Direction of connection
    A = 0  # Actual
    E = 1  # Expectations


# Type hints ##################################################################
Loc = tuple[int, int]
Neur = Annotated[list[torch.Tensor], 2]  # 2 copies of activations


# Activations and weights inits ###############################################
def neur(d: int) -> Neur:  # Activations
    """
    nr_<name>[0] : actual activations
    nr_<name>[1] : expectations
    """
    return [torch.randn(d), torch.zeros(d)]

def syn(d_x: int, d_y: int) -> torch.Tensor:  # Internal weights
    return 2 * torch.randn(d_x, d_y) / (d_x**0.5) + (0*d_x**0.5)

def conn(c1: ColBase, c2: ColBase, direction: Dir) -> torch.Tensor:  # External weights
    if direction == Dir.A:
        d1 = c1.a_pre.shape[0]
        d2 = c2.a_post.shape[0]
    else:
        d1 = c1.e_pre.shape[0]
        d2 = c2.e_post.shape[0]
    if direction == Dir.E:
        return 2 * torch.randn(d1, d2) / (d1**0.5) + 0.0
    else:
        return 2 * torch.randn(d1, d2) / (d1**0.5) + (0*d1**0.5)
    

# Constants ###################################################################
D_DEFAULT = 1024


# Classes #####################################################################
class ColCfgBase(ABC):
    ...
class ColBase(ABC):
    def __init__(self):  # Type hints
        self.loc: Loc
        self.cfg: ColCfgBase

        self.conns: dict[tuple[Loc, Dir], torch.Tensor] | None

    @abstractmethod
    def step(self) -> None:
        ...

    # For input to and from other cols using conns
    @property
    @abstractmethod
    def a_pre(self): ...
    @a_pre.setter
    @abstractmethod
    def a_pre(self, value): ...

    @property
    @abstractmethod
    def a_post(self): ...
    @a_post.setter
    @abstractmethod
    def a_post(self, value): ...

    @property
    @abstractmethod
    def e_pre(self): ...
    @e_pre.setter
    @abstractmethod
    def e_pre(self, value): ...

    @property
    @abstractmethod
    def e_post(self): ...
    @e_post.setter
    @abstractmethod
    def e_post(self, value): ...

    @property
    def count(self) -> tuple[int, int, int, int, int]:  # Count number of elements of each type of data
        NR = 0  # activations
        IS = 0  # internal (inside this col) weights
        ES = 0  # external (to other cols) weights
        IH = 0  # hyperparameters for internal weights
        EH = 0  # hyperparameters for external weights
        for name in vars(self):
            if name.startswith("nr_"):
                NR += getattr(self, name)[0].numel()
            elif name.startswith("is_"):
                IS += getattr(self, name).numel()
            elif name.startswith("ih_"):
                IH += getattr(self, name).numel()

        for weight in self.conns.values():
            ES += weight.numel()

        return NR, IS, ES, IH, EH

    def to(self, *args, **kwargs) -> None:
        assert self.weights_loaded

        for name, value in vars(self).items():
            if name.startswith("is_"):
                setattr(self, name, value.to(*args, **kwargs))

        for (loc, d), w in self.conns.items():
            self.conns[(loc, d)] = w.to(*args, **kwargs)

    def save(self, path: str, keep_weights: bool) -> None:
        # Create save directory
        if not os.path.exists(f"{path}/{self.loc}"):
            os.mkdir(f"{path}/{self.loc}")

        # Save cfg
        with open(f"{path}/{self.loc}/cfg", "wb") as f:
            pickle.dump(self.cfg, f)

        # Save activations
        for name in vars(self):
            if name.startswith("nr_"):
                if not os.path.exists(f"{path}/{self.loc}"):
                    os.mkdir(f"{path}/{self.loc}")
                torch.save(getattr(self, name), f"{path}/{self.loc}/{name}")

        # Only save weights if they're loaded, otherwise would save None's
        if self.weights_loaded:

            if not keep_weights:
                self.weights_loaded = False

            # Save internal weights
            for name in vars(self):
                if name.startswith("is_"):
                    torch.save(getattr(self, name), f"{path}/{self.loc}/{name}")
                    if not keep_weights:
                        setattr(self, name, None)

            # Save conns
            with open(f"{path}/{self.loc}/conns", "wb") as f:
                pickle.dump(self.conns, f)
            if not keep_weights:
                self.conns = None

    @staticmethod
    def init_and_load(path: str, name: str, load_weights: bool) -> ColBase:
        loc = eval(name)
        with open(f"{path}/{name}/cfg", "rb") as f:
            cfg = pickle.load(f)
        col = Col(loc, cfg, skip_init=True)
        col.load(path=path, load_weights=load_weights)
        return col

    def load(self, path: str, load_weights: bool) -> None:
        # Load cfg
        with open(f"{path}/{self.loc}/cfg", "rb") as f:
            self.cfg = pickle.load(f)

        # Load activations
        for name in vars(self):
            if name.startswith("nr_"):
                setattr(self, name, torch.load(f"{path}/{self.loc}/{name}"))

        # Load weights
        if load_weights:
            for name in vars(self):
                if name.startswith("is_"):
                    setattr(self, name, torch.load(f"{path}/{self.loc}/{name}"))

            with open(f"{path}/{self.loc}/conns", "rb") as f:
                self.conns = pickle.load(f)

            self.weights_loaded = True


class I_ColBase(ColBase):
    @abstractmethod
    def ipt(self, x: list[torch.Tensor]) -> None:
        ...
class O_ColBase(ColBase):
    @abstractmethod
    def out(self) -> torch.Tensor:
        ...


@dataclass
class BareColCfg(ColCfgBase):
    d: int = D_DEFAULT
class BareCol(ColBase):  # 1 layer, no internals
    def __init__(self, loc, cfg, skip_init):
        self.loc = loc

        self.cfg = cfg
        self.d = cfg.d

        self.conns: dict[tuple[Loc, Dir], torch.Tensor] = {}

        self.nr_1: Neur | None

        if skip_init:
            self.nr_1 = None
            self.weights_loaded = False
        else:
            self.nr_1 = neur(cfg.d)
            self.weights_loaded = True

    # For input to and from other cols using conns
    @property
    def a_pre(self): return self.nr_1[0]
    @a_pre.setter
    def a_pre(self, value): self.nr_1[0] = value

    @property
    def a_post(self): return self.nr_1[0]
    @a_post.setter
    def a_post(self, value): self.nr_1[0] = value

    @property
    def e_pre(self): return self.nr_1[0]
    @e_pre.setter
    def e_pre(self, value): self.nr_1[0] = value

    @property
    def e_post(self): return self.nr_1[1]
    @e_post.setter
    def e_post(self, value): self.nr_1[1] = value

    def step(self):
        self.nr_1[0] = fc.update(self.nr_1[0])
        self.nr_1[1] = fc.update(self.nr_1[1])


I_VectorColCfg = BareColCfg
class I_VectorCol(BareCol, I_ColBase):
    def step(self):
        pass

    def ipt(self, x) -> None:
        self.nr_1[0] = x


O_VectorColCfg = BareColCfg
class O_VectorCol(BareCol, O_ColBase):
    def out(self) -> torch.Tensor:
        return self.nr_1[0].clone()


@dataclass
class ColCfg(ColCfgBase):
    ...
class Col(ColBase):
    """A module within the whole network"""
    def __init__(self, loc: Loc, cfg: ColCfg, skip_init: bool = False):
        self.loc = loc
        self.cfg = cfg

        self.conns: dict[tuple[Loc, Dir], torch.Tensor] = {}

        if skip_init:
            ...
        else:
            # Activations
            self.nr_1 = neur(1024)
            self.nr_2 = neur(1024)
            self.nr_3 = neur(128)
            self.nr_4 = neur(1024)
            self.nr_5 = neur(1024)

            # Weights (within col)
            self.is_1_2 = syn(1024, 1024)
            self.is_2_3_f = syn(1024, 128)
            self.is_2_3_b = syn(128, 1024)
            self.is_2_4 = syn(1024, 1024)
            self.is_4_5 = syn(1024, 1024)

            # Conns (weights between cols)
            # nr_4 -> nr_1 (actual)
            # nr_5 -> nr_2 (expectation)

            self.weights_loaded = True

    def inhibit(self):
        # Lateral inhibition for winner take all, by combining actual and expected
        self.nr_1 = fc.inhibit(self.nr_1)
        self.nr_2 = fc.inhibit(self.nr_2)
        self.nr_3 = fc.inhibit(self.nr_3)
        self.nr_4 = fc.inhibit(self.nr_4)
        self.nr_5 = fc.inhibit(self.nr_5)

    @property
    def a_pre(self): return self.nr_4[0]
    @a_pre.setter
    def a_pre(self, value): self.nr_4[0] = value

    @property
    def a_post(self): return self.nr_1[0]
    @a_post.setter
    def a_post(self, value): self.nr_1[0] = value

    @property
    def e_pre(self): return self.nr_5[0]
    @e_pre.setter
    def e_pre(self, value): self.nr_5[0] = value

    @property
    def e_post(self): return self.nr_2[1]
    @e_post.setter
    def e_post(self, value): self.nr_2[1] = value

    def step(self):
        atv = fc.atv
        lrn = fc.lrn
        update = fc.update
        update_e = fc.update_e

        # Apply learning rules
        self.is_1_2   = lrn(self.nr_1[0], self.is_1_2, self.nr_2[0])
        self.is_2_3_f = lrn(self.nr_2[0], self.is_2_3_f, self.nr_3[0])
        self.is_2_3_b = lrn(self.nr_3[0], self.is_2_3_b, self.nr_2[0])
        self.is_2_4   = lrn(self.nr_2[0], self.is_2_4, self.nr_4[0])
        self.is_4_5   = lrn(self.nr_4[0], self.is_4_5, self.nr_5[0])

        # Apply activity rules (propagate activations)
        nr_1_ = update(self.nr_1[0])
        nr_2_ = update(self.nr_2[0]) \
            + atv(self.nr_1[0], self.is_1_2, self.nr_2[0]) \
            + atv(self.nr_3[0], self.is_2_3_b, self.nr_2[0])
        nr_3_ = update(self.nr_3[0]) \
            + atv(self.nr_2[0], self.is_2_3_f, self.nr_3[0])
        nr_4_ = update(self.nr_4[0]) + atv(self.nr_2[0], self.is_2_4, self.nr_4[0])
        nr_5_ = update(self.nr_5[0]) + atv(self.nr_4[0], self.is_4_5, self.nr_5[0])

        # Update actual
        self.nr_1[0] = nr_1_
        self.nr_2[0] = nr_2_
        self.nr_3[0] = nr_3_
        self.nr_4[0] = nr_4_
        self.nr_5[0] = nr_5_

        # Update expectations
        self.nr_1[1] = update_e(self.nr_1[1])
        self.nr_2[1] = update_e(self.nr_2[1])
        self.nr_3[1] = update_e(self.nr_3[1])
        self.nr_4[1] = update_e(self.nr_4[1])
        self.nr_5[1] = update_e(self.nr_5[1])


@dataclass
class Cfg:  # Agent configuration
    n_cols: int       # Number of columns
    ispec: list[T.I]  # Input specification
    ospec: list[T.O]  # Output specification
class Agt:  # Agent
    def __init__(self,
        cfg: Cfg,
        path: str,
        skip_init: bool=False  # For loading from save
    ):

        self.cfg = cfg
        self.path = path

        self.n_cols = cfg.n_cols  # Number of columns
        self.ispec  = cfg.ispec  # Input specification
        self.ospec  = cfg.ospec  # Output specification

        # mode:
        #     single : only run on either gpu or cpu
        #     double : transfer between both
        self.mode = "single"
        GPU_COL_CAPACITY = 200
        if torch.cuda.is_available() and self.n_cols > GPU_COL_CAPACITY:  
            self.mode = "double"

        self.I_cols: list[I_ColBase] = []
        self.O_cols: list[O_ColBase] = []
        self.cols: dict[Loc, ColBase] = {}  # location : col

        # Which cols to keep on gpu when running in double mode
        self.cache: list[Loc] = []  

        if not skip_init:
            # Reset or create save directory
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            width = math.ceil(self.n_cols**(1/2))  # Side length

            # Initialize io columns
            for i, spec in tqdm(enumerate(self.ispec),
                                desc="Initializing input cols",
                                total=len(self.ispec)):
                assert isinstance(spec, T.I)
                if type(spec) is T.I_Vector:
                    loc = (i+1, 0)
                    col = I_VectorCol(loc, I_VectorColCfg(spec.d), False)
                    self.I_cols.append(col)
                    self.cols[loc] = col
                    self.free_col(col)
                else:
                    raise NotImplementedError

            for i, spec in tqdm(enumerate(self.ospec),
                                desc="Initializing output cols",
                                total=len(self.ospec)):
                assert isinstance(spec, T.O)
                if type(spec) is T.O_Vector:
                    loc = (0, i+1)
                    col = O_VectorCol(loc, O_VectorColCfg(spec.d), False)
                    self.O_cols.append(col)
                    self.cols[loc] = col
                    self.free_col(col)
                else:
                    raise NotImplementedError

            # Initialize bulk of (internal, non io) columns
            for i, (y, x) in tqdm(enumerate(np.ndindex((width,width))),
                                  desc="Initializing bulk of cols",
                                  total=self.n_cols):
                if i < self.n_cols:
                    loc = (x+1, y+1)  # Offset +1 since 0 is for input/output
                    col = Col(loc, ColCfg())
                    self.cols[loc] = col
                    self.free_col(col)
                else:
                    break

            # Set cache
            if self.mode == "double":
                self.cache = random.sample(list(self.cols.keys()), int(0.8*GPU_COL_CAPACITY))

            # Initialize connections
            for loc, col in tqdm(self.cols.items(), desc="Initializing conns"):
                self.load_col(col)
                if col in self.O_cols:  # no conns for output cols
                    continue
                # Independent probability for each possible conn
                for other_loc in self.cols.keys():
                    if other_loc != loc:
                        distance = fc.dist(loc, other_loc)
                        if self.is_io(loc):
                            p = 0.6 / (distance + 1e-3)**2  # Compensate for fewer possible directions
                        else:
                            p = 0.3 / (distance + 1e-3)**2

                        if (self.is_io(col.loc) and self.is_io(other_loc)) \
                            or self.is_i(other_loc):
                            # Don't directly connect io cols
                            # and don't connect to input cols
                            allowed = []
                        elif self.is_io(col.loc):
                            # An io col can't expect, can only provide actual
                            allowed = [Dir.A]
                        else:
                            allowed = list(Dir)
                        for direction in allowed:
                            if random.random() < p:
                                col.conns[(other_loc, direction)] = conn(col, self.cols[other_loc], direction)
                                break  # Only at most one conn per target?
                self.free_col(col)

            # Create directories for all cols
            for col in self.cols.values():
                os.mkdir(f"{path}/{col.loc}")

            self.use_debug = False

    def step(self, ipt: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(ipt) == len(self.ispec), f"Expected input of length {len(self.ispec)} but got length {len(ipt)}"

        # Receive inputs
        for col, x in zip(self.I_cols, ipt):
            x = x.to(torch.get_default_device()).to(torch.get_default_dtype())
            col.ipt(x)

        for col in tqdm(self.cols.values(), desc="Stepping cols..."):
            # Step col
            self.load_col(col)

            # Use lateral inhibition to combine actual and expected activations
            if hasattr(col, "inhibit"):
                col.inhibit()

            # Apply learning rule to connections
            lrn = fc.lrn
            for (loc, direction), weight in col.conns.items():
                if direction == Dir.A:
                    weight = lrn(col.a_pre, weight, self.cols[loc].a_post)
                elif direction == Dir.E:
                    ...
                col.conns[(loc, direction)] = weight

            # Do output to other cols
            for (loc, direction), weight in col.conns.items():
                if direction == Dir.A:
                    self.cols[loc].a_post += fc.atv(col.a_pre, weight, self.cols[loc].a_post)
                elif direction == Dir.E:
                    self.cols[loc].e_post += fc.atv(col.e_pre, weight, self.cols[loc].e_post)

            # Internal lrn, update, atv
            col.step()

            self.free_col(col)

            if self.use_debug:
                self.debug_update()

        # Return outputs
        out = []
        for col in self.O_cols:
            out.append(col.out())
        return out

    def is_i(self, loc):
        return any([loc == col.loc for col in self.I_cols])
    def is_o(self, loc):
        return any([loc == col.loc for col in self.O_cols])
    def is_io(self, loc):
        return any([loc == col.loc for col in self.I_cols+self.O_cols])

    def debug_init(self):
        from .debugger import debugger  # To avoid circular import

        self.use_debug = True
        self.pipes = {}

        ctx = multiprocessing.get_context("spawn")  # Avoid duplicating memory

        # Overview of activity for all cols
        self.pipes["overview"] = ctx.Pipe()

        # Detailed information about a single specific col
        self.pipes["col"] = ctx.Pipe()

        # Information about a conn
        self.pipes["conn"] = ctx.Pipe()

        # activation tensor
        self.pipes["atv"] = ctx.Pipe()

        # Start debug process
        self.debug_process = ctx.Process(target=debugger, args=(self.path, self.pipes))
        self.debug_process.start()

        # Timestamps of most recent updates, to calculate cooldowns
        self.t_prevs = {}
        self.t_prevs["overview"] = time.time()
        self.t_prevs["col"] = time.time()
        self.t_prevs["conn"] = time.time()
        self.t_prevs["atv"] = time.time()

    def debug_update(self):
        def stats(x, is_weight=False):
            shape = tuple(x.shape)
            threshold = 1.0  # temp
            d = (torch.sum(torch.where(x < threshold, 0.0, 1.0)) / x.numel()).item()
            n = torch.linalg.vector_norm(x).item()
            m = torch.mean(x).item()
            s = torch.std(x).item()
            h = None
            # Histogram
            if is_weight:
                # 43 bins: (-inf, -0.1025), [-0.1025, -0.0975), ..., [-0.0025, 0.0025), ..., [0.0975, 0.1025), [0.1025, inf)
                bins = torch.tensor([float("-inf")] + [0.005*i - 0.1025 for i in range(42)] + [float("inf")], device="cpu", dtype=torch.float64)
            else:
                # 43 bins: (-inf, -2.05), [-2.05, -1.95), ..., [-0.05, 0.05), ..., [1.95, 2.05), [2.05, inf)
                bins = torch.tensor([float("-inf")] + [0.1*i - 2.05 for i in range(42)] + [float("inf")], device="cpu", dtype=torch.float64)
            h, _ = torch.histogram(x.cpu().to(torch.float64), bins)  # NOTE issue if use lower float precision
            h = h.tolist()
            assert x.numel() == sum(h), "Failed to calculate histogram, probably because tensor is NaN"
            return (shape, d, n, m, s, h)

        COOLDOWN_OVERVIEW = 0.2
        COOLDOWN_COL = 0.5
        COOLDOWN_CONN = 0.1
        COOLDOWN_ATV = 0.1

        # Send overview information #####################################################################
        pipe, _ = self.pipes["overview"]
        if time.time()-self.t_prevs["overview"] > COOLDOWN_OVERVIEW:
            self.t_prevs["overview"] = time.time()

            info = {}
            info["timestamp"] = time.time()
            nrns = 0
            copies = 0
            isyns = 0
            esyns = 0

            sum_density = 0
            for loc, col in self.cols.items():
                for name, x in vars(col).items():
                    if name.startswith("nr_"):
                        copies = len(x)  # Assume is same for all activations
                        x = x[0]
                        nrns += x.numel()
                        threshold = 1.0
                        sum_density += torch.sum(torch.where(x < threshold, 0.0, 1.0)).item()
                    elif name.startswith("is_"):
                        isyns += x.numel()
                for _, weight in col.conns.items():
                    esyns += weight.numel()
            info["nrns"] = nrns
            info["copies"] = copies
            info["isyns"] = isyns
            info["esyns"] = esyns
            info["syns"] = isyns + esyns

            info["density"] = sum_density / nrns
            pipe.send(info)

        # Send information for single col #####################################################################
        request = None
        pipe, _ = self.pipes["col"]
        while pipe.poll():  # Get most recent request
            request = pipe.recv()
        if request is not None and (time.time()-self.t_prevs["col"]) > COOLDOWN_COL:
            self.t_prevs["col"] = time.time()

            col = self.cols[request]
            info = {}
            info["timestamp"] = time.time()
            info["loc"] = col.loc  # for debugger to verify info is up to date
            info["nrns"], info["isyns"], info["esyns"], _, _ = col.count
            info["syns"] = info["isyns"] + info["esyns"]
            # Values of activations
            for name, x in vars(col).items():
                if name.startswith("nr_"):
                    info[name] = [stats(x_i) for x_i in x]
                elif name.startswith("is_"):
                    info[name] = stats(x, True)

            conns_skeleton = {}
            for (loc, direction), _ in col.conns.items():
                conns_skeleton[(loc, direction)] = None
            info["conns"] = conns_skeleton

            pipe.send(info)

        # Send information for specific conn #####################################################################
        request = None
        pipe, _ = self.pipes["conn"]
        while pipe.poll():  # Get most recent request
            request = pipe.recv()
        if request is not None and (time.time()-self.t_prevs["conn"]) > COOLDOWN_CONN:
            self.t_prevs["conn"] = time.time()

            loc, conn_loc, conn_dir = request
            conn = self.cols[loc].conns.get((conn_loc, conn_dir))
            info = {}
            info["timestamp"] = time.time()
            info["request"] = request  # for debugger to verify info is up to date
            if conn is not None:
                info["valid"] = True
                info["stats"] = stats(conn, True)
            else:
                info["valid"] = False
            pipe.send(info)

        # Send information for single activation tensor #####################################################################
        request = None
        pipe, _ = self.pipes["atv"]
        while pipe.poll():  # Get most recent request
            request = pipe.recv()
        if request is not None and (time.time()-self.t_prevs["atv"]) > COOLDOWN_ATV:
            self.t_prevs["atv"] = time.time()

            loc, i = request
            if hasattr(self.cols[loc], f"nr_{i}"):
                x = getattr(self.cols[loc], f"nr_{i}")[0]
                info = {}
                info["timestamp"] = time.time()
                info["request"] = request  # for debugger to verify info is up to date
                info["x"] = x.detach().cpu().numpy()
                pipe.send(info)

    def load_col(self, c: ColBase) -> None:
        if self.mode == "single":
            pass
        elif self.mode == "double":
            c.to("cuda")
        else:
            raise Exception("mode not implemented")

    def free_col(self, c: ColBase) -> None:
        if self.mode == "single":
            pass
        elif self.mode == "double":
            if c.loc not in self.cache:
                c.to("cpu")
        else:
            raise Exception("mode not implemented")

    def save(self) -> None:
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        # Save cfg
        print("Saving cfg...")
        with open(f"{self.path}/cfg", "wb") as f:
            pickle.dump(self.cfg, f)

        # Save cols
        for loc, col in tqdm(self.cols.items(), desc="Saving cols..."):
            col.save(self.path, keep_weights=True)

    @staticmethod
    def load(path: str, load_weights: bool) -> Agt:
        # Load cfg
        print("Loading cfg...")
        with open(f"{path}/cfg", "rb") as f:
            cfg = pickle.load(f)

        agt = Agt(cfg, path, skip_init=True)

        # Load cols
        for name in tqdm(os.listdir(path), desc="Loading cols..."):
            if name != "cfg":
                col = Col.init_and_load(path, name, load_weights)

                loc = eval(name)
                agt.cols[loc] = col
        return agt

    def verify(self):
        print("Checking agent has right number of columns...", end="")
        assert (self.n_cols + len(self.ispec) + len(self.ospec)) == len(list(self.cols))
        print("✔️")

        print("Checking location in agent's dictionary key matches column location...", end="")
        for (loc,col) in self.cols.items():
            assert loc == col.loc, f"Key {loc} in dictionary has col with loc {col.loc}"
        print("✔️")

        print("Checking for uniform column location dimensionality...", end="")
        dim = len(list(self.cols.keys())[0])  # loc dim of first col
        for loc in self.cols.keys():
            assert len(loc) == dim, f"Different location dimensionality: {dim} and {len(loc)}"
        print("✔️")

        print("Checking that there are no location collisions...", end="")  # (inefficient but readable)
        for i, loc1 in enumerate(self.cols.keys()):
            for j, loc2 in enumerate(self.cols.keys()):
                if i != j:
                    assert loc1 != loc2, f"Location collision: {loc1} and {loc2}"
        print("✔️")

        print("Checking that targets of conns exist...", end="")
        for col in self.cols.values():
            self.load_col(col)
            for (loc, _) in col.conns.keys():
                assert loc in self.cols
            self.free_col(col)
        print("✔️")

        print("Checking that there are a correct number of IO cols...", end="")
        assert len(self.I_cols) == len(self.ispec)
        assert len(self.O_cols) == len(self.ospec)
        print("✔️")

        print("Passed all checks!")

