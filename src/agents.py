
"""
init or load from disk:
    env
    agt:
        cols: activations, internal weights
            io cols
            internal cols
        conns (external weights)
    debugger
for each timestep, in parallel processes:
    step env:
        respond to actions
        provide percepts
    step agt:
        receive inputs (propagate to input cols)
        for each col: (first pass)
            apply learning rule both locally (within a col)
            and globally (across the whole agent)

            apply activity rule (propagate activations) locally and globally
                the new activations will only be used after all of them
                have been computed, which happens in the second pass
            
            send debug info

        then again for each col: (second pass)
            apply self inhibition to activations
                by using actual and expected values
                
            set current activations equal to new
            reset new activations

            send debug info

        provide outputs (read from output cols)
save to disk
"""

from __future__ import annotations

import pickle
import os
from typing import Annotated, Literal
from dataclasses import dataclass
import time
import random
import math
from enum import Enum
from abc import ABC, abstractmethod
import multiprocessing
import shutil
import sys

import torch
import numpy as np
from tqdm import tqdm

from . import iotypes as T
from . import funcs as fc

class Dir(Enum):  # Direction of connection
    A = 0  # Actual
    E = 1  # Expectations


# Type hints ##################################################################
Loc = tuple[int, int]  # Location of col (module)
Activs = Annotated[list[torch.Tensor], 2]  # 2 copies of activations (actual and expectations)
Weights = torch.Tensor

Input = torch.Tensor
Output = torch.Tensor
Inputs = list[Input]
Outputs = list[Output]


# Activations and weights inits ###############################################
# Activations
def activs(d: int) -> Activs:
    """
    nr_<name>[0] : actual activations
    nr_<name>[1] : expectations
    """
    return [torch.randn(d), torch.zeros(d)]

# Internal weights
def weights(d_x: int, d_y: int, scale: float = 2.0) -> Weights:
    return scale * torch.randn(d_x, d_y) / (d_x**0.5) + (0*d_x**0.5)

# External weights
def conn(c1: ColBase, c2: ColBase, direction: Dir, scale: float = 2.0) -> Weights:
    if direction == Dir.A:
        d1 = c1.a_pre.shape[0]
        d2 = c2.a_post.shape[0]
    elif direction == Dir.E:
        d1 = c1.e_pre.shape[0]
        d2 = c2.e_post.shape[0]
    return scale * torch.randn(d1, d2) / (d1**0.5)
    

# Constants ###################################################################
D_DEFAULT = 1024


# Classes #####################################################################
class ColCfgBase(ABC):
    ...
class ColBase(ABC):
    def __init__(self):  # Type hints
        self.loc: Loc
        self.cfg: ColCfgBase

        # Activations:
            # current: self.nr_<name>
            # new: self.nr_<name>_
        # Weights: self.is_<name>

        self.conns: dict[tuple[Loc, Dir], Weights] | None

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def update_activations(self) -> None:
        ...

    # Maps
        # conn source or target: {a, e}_{pre, post}{ , _}
        #                        ^ actual or expectation conn
        #                               ^ source or target
        #                                          ^ current or new
        #                 v
    conn_layer_dict: dict[str, tuple[str, Literal[0, 1]]]
        # to                         ^ name of activation layer and 
        #                                 ^ kind (actual or expectations)

    def __getattr__(self, name):
        if name in self.conn_layer_dict:
            layer_name, i = self.conn_layer_dict[name]
            return getattr(self, layer_name)[i]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name in self.conn_layer_dict:
            layer_name, i = self.conn_layer_dict[name]
            getattr(self, layer_name)[i] = value
        else:
            super().__setattr__(name, value)  # Default behavior

    @property
    def count(self) -> tuple[int, int, int, int, int]:
        """Count number of elements of each type of data"""
        nrns = 0  # Activations
        isyns = 0  # Internal (inside this col) weights
        esyns = 0  # External (to other cols) weights
        ihyps = 0  # Hyperparameters for internal weights
        ehyps = 0  # Hyperparameters for external weights
        for name in vars(self):
            if name.startswith("nr_"):
                nrns += getattr(self, name)[0].numel()
            elif name.startswith("is_"):
                isyns += getattr(self, name).numel()
            elif name.startswith("ih_"):
                ihyps += getattr(self, name).numel()

        for weight in self.conns.values():
            esyns += weight.numel()

        return nrns, isyns, esyns, ihyps, ehyps

    def to(self, *args, **kwargs) -> None:
        """Move all tensors between cpu and gpu"""
        assert self.weights_loaded

        for name, value in vars(self).items():
            if name.startswith("is_"):
                setattr(self, name, value.to(*args, **kwargs))

        for (loc, d), w in self.conns.items():
            self.conns[(loc, d)] = w.to(*args, **kwargs)

    def save(self, agt_path: str, keep_weights: bool) -> None:
        # Create save directory
        if not os.path.exists(f"{agt_path}/{self.loc}"):
            os.makedirs(f"{agt_path}/{self.loc}")

        # Save type of col
        with open(f"{agt_path}/{self.loc}/type", "wb") as f:
            pickle.dump(type(self), f)

        # Save cfg
        with open(f"{agt_path}/{self.loc}/cfg", "wb") as f:
            pickle.dump(self.cfg, f)

        # Save activations
        for name in vars(self):
            if name.startswith("nr_"):
                torch.save(getattr(self, name), f"{agt_path}/{self.loc}/{name}")

        # Only save weights if they're loaded, otherwise would save None's
        if self.weights_loaded:
            self.to("cpu")  # To avoid issues when loading
            if not keep_weights:
                self.weights_loaded = False

            # Save internal weights
            for name in vars(self):
                if name.startswith("is_"):
                    torch.save(getattr(self, name), f"{agt_path}/{self.loc}/{name}")
                    if not keep_weights:
                        setattr(self, name, None)

            # Save conns
            with open(f"{agt_path}/{self.loc}/conns", "wb") as f:
                pickle.dump(self.conns, f)
            if not keep_weights:
                self.conns = None

    @staticmethod
    def init_and_load(
            agt_path: str, 
            name: str, 
            load_activations: bool, 
            load_weights: bool) -> ColBase:
        loc = eval(name)
        with open(f"{agt_path}/{name}/type", "rb") as f:
            col_type = pickle.load(f)
        with open(f"{agt_path}/{name}/cfg", "rb") as f:
            cfg = pickle.load(f)
        col = col_type(loc, cfg, skip_init=True)
        col.load(agt_path=agt_path, load_activations=load_activations, load_weights=load_weights)
        return col

    def load(self,
            agt_path: str,
            load_activations: bool,
            load_weights: bool) -> None:
        # Load cfg
        with open(f"{agt_path}/{self.loc}/cfg", "rb") as f:
            self.cfg = pickle.load(f)

        # Load activations
        if load_activations:
            for name in vars(self):
                if name.startswith("nr_"):
                    setattr(self, name, torch.load(f"{agt_path}/{self.loc}/{name}"))

        # Load weights
        if load_weights:
            for name in vars(self):
                if name.startswith("is_"):
                    setattr(self, name, torch.load(f"{agt_path}/{self.loc}/{name}"))

            with open(f"{agt_path}/{self.loc}/conns", "rb") as f:
                self.conns = pickle.load(f)

            self.weights_loaded = True


class I_ColBase(ColBase):
    @abstractmethod
    def ipt(self, x: Input) -> None:
        ...
class O_ColBase(ColBase):
    @abstractmethod
    def out(self) -> Output:
        ...


@dataclass
class BareColCfg(ColCfgBase):
    d: int = D_DEFAULT
class BareCol(ColBase):  # 1 layer, no internal weights
    def __init__(self, loc, cfg, skip_init=False):
        self.loc = loc

        self.cfg = cfg
        self.d = cfg.d

        self.conns: dict[tuple[Loc, Dir], Weights] = {}

        self.nr_1: Activs | None

        if skip_init:
            self.nr_1, self.nr_1_ = None, None
            self.weights_loaded = False
        else:
            self.nr_1, self.nr_1_ = activs(cfg.d), activs(cfg.d)
            self.weights_loaded = True

    conn_layer_dict = {
        "a_pre": ("nr_1", 0),
        "a_pre_": ("nr_1_", 0),
        "a_post": ("nr_1", 0),
        "a_post_": ("nr_1_", 0),
        "e_pre": ("nr_1", 0),
        "e_pre_": ("nr_1_", 0),
        "e_post": ("nr_1", 1),
        "e_post_": ("nr_1_", 1),
    }

    def step(self):
        pass

    def update_activations(self):
        self.nr_1 = self.nr_1_.copy()  # Intentional shallow copy

        self.nr_1_[0] = fc.update(self.nr_1_[0])
        self.nr_1_[1] = fc.update_e(self.nr_1_[1])


I_VectorColCfg = BareColCfg
class I_VectorCol(BareCol, I_ColBase):
    def update_activations(self):
        self.nr_1 = self.nr_1_.copy()  # Intentional shallow copy

        # Receives perceptual input, don't reset

    def ipt(self, x: Input) -> None:
        self.nr_1_[0] = x


O_VectorColCfg = BareColCfg
class O_VectorCol(BareCol, O_ColBase):
    def out(self) -> Output:
        return self.nr_1[0].clone().cpu()


@dataclass
class ColCfg(ColCfgBase):
    ...
class Col(ColBase):  # Column (module) within the agent (whole network)
    def __init__(self, loc: Loc, cfg: ColCfg, skip_init: bool = False):
        self.loc = loc
        self.cfg = cfg

        # Conns (weights between cols)
            # nr_4 -> nr_1 (actual)
            # nr_5 -> nr_2 (expectation)
        self.conns: dict[tuple[Loc, Dir], Weights] = {}

        if skip_init:
            # Activations
            self.nr_1, self.nr_1_ = None, None
            self.nr_2, self.nr_2_ = None, None
            self.nr_3, self.nr_3_ = None, None
            self.nr_4, self.nr_4_ = None, None
            self.nr_5, self.nr_5_ = None, None

            # Weights (within col)
            self.is_1_2 = None
            self.is_2_3_f = None
            self.is_2_3_b = None
            self.is_2_4 = None
            self.is_4_5 = None
        else:
            # Activations: current and new versions
            self.nr_1, self.nr_1_ = activs(1024), activs(1024)
            self.nr_2, self.nr_2_ = activs(1024), activs(1024)
            self.nr_3, self.nr_3_ = activs(128), activs(128)
            self.nr_4, self.nr_4_ = activs(1024), activs(1024)
            self.nr_5, self.nr_5_ = activs(1024), activs(1024)

            # Weights (within col)
            self.is_1_2 = weights(1024, 1024)
            self.is_2_3_f = weights(1024, 128)
            self.is_2_3_b = weights(128, 1024)
            self.is_2_4 = weights(1024, 1024)
            self.is_4_5 = weights(1024, 1024)

            self.weights_loaded = True

    def inhibit(self):
        # Lateral inhibition for winner take all, by combining actual and expected
        self.nr_1_ = fc.inhibit(self.nr_1_)
        self.nr_2_ = fc.inhibit(self.nr_2_)
        self.nr_3_ = fc.inhibit(self.nr_3_)
        self.nr_4_ = fc.inhibit(self.nr_4_)
        self.nr_5_ = fc.inhibit(self.nr_5_)

    conn_layer_dict = {
        "a_pre": ("nr_4", 0),
        "a_pre_": ("nr_4_", 0),
        "a_post": ("nr_1", 0),
        "a_post_": ("nr_1_", 0),
        "e_pre": ("nr_5", 0),
        "e_pre_": ("nr_5_", 0),
        "e_post": ("nr_2", 1),
        "e_post_": ("nr_2_", 1),
    }

    def step(self):
        # Apply learning rules
        self.is_1_2   = fc.lrn(self.nr_1[0], self.is_1_2, self.nr_2[0])
        self.is_2_3_f = fc.lrn(self.nr_2[0], self.is_2_3_f, self.nr_3[0])
        self.is_2_3_b = fc.lrn(self.nr_3[0], self.is_2_3_b, self.nr_2[0])
        self.is_2_4   = fc.lrn(self.nr_2[0], self.is_2_4, self.nr_4[0])
        self.is_4_5   = fc.lrn(self.nr_4[0], self.is_4_5, self.nr_5[0])

        # Apply activity rule (propagate activations)
        self.nr_1_[0]
        self.nr_2_[0] += fc.atv(self.nr_1[0], self.is_1_2, self.nr_2_[0]) \
            + fc.atv(self.nr_3[0], self.is_2_3_b, self.nr_2_[0])
        self.nr_3_[0] += fc.atv(self.nr_2[0], self.is_2_3_f, self.nr_3_[0])
        self.nr_4_[0] += fc.atv(self.nr_2[0], self.is_2_4, self.nr_4_[0])
        self.nr_5_[0] += fc.atv(self.nr_4[0], self.is_4_5, self.nr_5_[0])

    def update_activations(self):
        # Set current activations equal to new activations
        self.nr_1 = self.nr_1_.copy()  # Intentional shallow copy
        self.nr_2 = self.nr_2_.copy()
        self.nr_3 = self.nr_3_.copy()
        self.nr_4 = self.nr_4_.copy()
        self.nr_5 = self.nr_5_.copy()

        # Reset new activations
        self.nr_1_[0] = fc.update(self.nr_1_[0])
        self.nr_2_[0] = fc.update(self.nr_2_[0])
        self.nr_3_[0] = fc.update(self.nr_3_[0])
        self.nr_4_[0] = fc.update(self.nr_4_[0])
        self.nr_5_[0] = fc.update(self.nr_5_[0])

        self.nr_1_[1] = fc.update_e(self.nr_1_[1])
        self.nr_2_[1] = fc.update_e(self.nr_2_[1])
        self.nr_3_[1] = fc.update_e(self.nr_3_[1])
        self.nr_4_[1] = fc.update_e(self.nr_4_[1])
        self.nr_5_[1] = fc.update_e(self.nr_5_[1])


class AgtBase(ABC):
    def step(self, ipt: Inputs, disable_print: bool = False) -> Outputs:
        assert len(ipt) == len(self.ispec), f"Expected input of length {len(self.ispec)} but got length {len(ipt)}"

        # Receive inputs
        for col, x in zip(self.I_cols, ipt):
            x = x.to(torch.get_default_device()).to(torch.get_default_dtype())
            col.ipt(x)

        # First pass: compute new weights and activations
        for col in (bar := tqdm(self.cols.values(), desc="Computing new weights and activations...", disable=disable_print)):
            # Used to communicate debugger exited
            if self.pipes["overview"][0].poll():
                bar.close()
                self.save()
                sys.exit()

            self.load_col(col)

            # Apply learning and activity rules internally
            col.step()

            # Apply learning rule to connections
            lrn = fc.lrn
            for (loc, direction), weight in col.conns.items():
                if direction == Dir.A:
                    weight = lrn(col.a_pre, weight, self.cols[loc].a_post)
                elif direction == Dir.E:
                    ...  # TODO
                col.conns[(loc, direction)] = weight

            # Do output to other cols
            for (loc, direction), weight in col.conns.items():
                if self.is_i(col.loc):  # Don't discretize inputs
                    if direction == Dir.A:
                        self.cols[loc].a_post_ += col.a_pre @ weight
                    elif direction == Dir.E:
                        self.cols[loc].e_post_ += col.e_pre @ weight
                else:
                    if direction == Dir.A:
                        self.cols[loc].a_post_ += fc.atv(col.a_pre, weight, self.cols[loc].a_post_)
                    elif direction == Dir.E:
                        self.cols[loc].e_post_ += fc.atv(col.e_pre, weight, self.cols[loc].e_post_)

            self.free_col(col)

            if self.use_debug:
                self.debug_update()

        # Second pass: set current activations equal to new, and reset new
        for col in (bar := tqdm(self.cols.values(), desc="Updating and resetting activations...", disable=disable_print)):
            # Used to communicate debugger exited
            if self.pipes["overview"][0].poll():  # TODO not here?
                bar.close()
                self.save()
                sys.exit()

            # Use lateral inhibition to combine actual and expected activations
            if hasattr(col, "inhibit"):
                col.inhibit()

            col.update_activations()

            if self.use_debug:  # TODO not here?
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

        # Activation tensor
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
        def stats(x, is_weight):
            shape = tuple(x.shape)
            threshold = 1.0  # temp
            d = (torch.sum(torch.where(x < threshold, 0.0, 1.0)) / x.numel()).item()
            n = torch.linalg.vector_norm(x).item()
            m = torch.mean(x).item()
            s = torch.std(x).item()
            # Histogram: The histogram displays from -2 to +2 standard deviations,
            # with 43 bins in total, where each bin is 1/10 of a std wide.
            #
            # For example if the std is 1 (and so each bin is 0.1 wide), the bins are:
            # (-inf, -2.05), [-2.05, -1.95), ..., [-0.05, 0.05), ..., [1.95, 2.05), [2.05, inf)
            # 
            # For activation tensors, the histogram std is fixed at 1, but for weights
            # it is computed by rounding down the actual std to 1eX, 2eX, or 5eX,
            # giving bin widths of 1e(X-1), 2e(X-1), and 5e(X-1).
            h = None
            if is_weight:
                # Compute bin size
                bin_width = s / 10
                # Repetitive but avoids bugs
                if bin_width < 0.0015:
                    bin_width = 0.001
                elif 0.0015 <= bin_width < 0.003:
                    bin_width = 0.002
                elif 0.003 <= bin_width < 0.006:
                    bin_width = 0.005
                elif 0.006 <= bin_width < 0.015:
                    bin_width = 0.01
                elif 0.015 <= bin_width < 0.03:
                    bin_width = 0.02
                elif 0.03 <= bin_width < 0.6:
                    bin_width = 0.05
                else:
                    bin_width = 0.1
            else:  # Actuvations
                bin_width = 0.1
            bins = torch.tensor([float("-inf")] + [bin_width*i - 20.5*bin_width for i in range(42)] + [float("inf")], device="cpu", dtype=torch.float64)
            h, _ = torch.histogram(x.cpu().to(torch.float64), bins)  # NOTE issue if use lower float precision
            h = h.tolist()
            assert x.numel() == sum(h), "Failed to calculate histogram, probably because tensor is NaN"
            return (shape, d, n, m, s, (h, bin_width))

        COOLDOWN_OVERVIEW = 0.2
        COOLDOWN_COL = 0.5
        COOLDOWN_CONN = 0.1
        COOLDOWN_ATV = 0.1

        # Send overview information ###########################################
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
                    # Only count current activations, not new
                    if name.startswith("nr_") and name[-1] != "_":
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

        # Send information for single col #####################################
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
                # Only send current activations, not new
                if name.startswith("nr_") and name[-1] != "_":
                    info[name] = [stats(x_i, False) for x_i in x]
                elif name.startswith("is_"):
                    info[name] = stats(x, True)

            conns_skeleton = {}
            for (loc, direction), _ in col.conns.items():
                conns_skeleton[(loc, direction)] = None
            info["conns"] = conns_skeleton

            pipe.send(info)

        # Send information for specific conn ##################################
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

        # Send information for single activation tensor #######################
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
        if torch.cuda.is_available():
            # While available memory is less than 5% of total
            while torch.cuda.memory.mem_get_info()[0] \
                    < 0.05*torch.cuda.memory.mem_get_info()[1]:
                to_evict = random.choice(list(self.cols.values()))
                to_evict.to("cpu")
                torch.cuda.memory.empty_cache()
            c.to("cuda")

    def free_col(self, c: ColBase) -> None:
        if torch.cuda.is_available():
            # Should only be necessary during large inits,
            # since loading while running already frees memory
            available, total = torch.cuda.memory.mem_get_info()
            # Use stronger threahold so that newly loaded cols
            # are not immediately freed
            if available < 0.04*total:
                c.to("cpu")

    def save(self, keep_weights: bool = True) -> None:
        print(f"\nSaving agent to \"{self.path}\":")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Save type of agt
        with open(f"{self.path}/type", "wb") as f:
            pickle.dump(type(self), f)

        # Save cfg
        print("Saving cfg...")
        with open(f"{self.path}/cfg", "wb") as f:
            pickle.dump(self.cfg, f)

        # Save cols
        for loc, col in tqdm(self.cols.items(), desc="Saving cols"):
            col.save(self.path, keep_weights=keep_weights)
        print("done saving.")

    @staticmethod
    def load(path: str,
            load_activations: bool = True,
            load_weights: bool = True) -> AgtBase:
        print(f"\nLoading agent from \"{path}\":")

        # Load type of agt
        with open(f"{path}/type", "rb") as f:
            agt_type = pickle.load(f)

        # Load cfg
        print("Loading cfg...")
        with open(f"{path}/cfg", "rb") as f:
            cfg = pickle.load(f)

        agt = agt_type(cfg, path, skip_init=True)

        # Load cols
        for name in tqdm(os.listdir(path), desc="Loading cols"):
            if name not in ["type", "cfg"]:
                col = Col.init_and_load(path, name, load_activations, load_weights)

                loc = eval(name)
                agt.cols[loc] = col
                if isinstance(col, I_ColBase):
                    agt.I_cols.append(col)
                elif isinstance(col, O_ColBase):
                    agt.O_cols.append(col)
        print("done loading.")
        return agt

    def verify(self):
        print("\nChecking agent has right number of columns...", end="")
        assert (self.n_cols + len(self.ispec) + len(self.ospec)) \
            == len(list(self.cols))
        assert len(self.ispec) == len(self.I_cols)
        assert len(self.ospec) == len(self.O_cols)
        print("✔️")

        print("Checking location key in Agt.cols matches column location...", end="")
        for (loc, col) in self.cols.items():
            assert loc == col.loc, f"Key {loc} in dictionary has col with loc {col.loc}!"
        print("✔️")

        print("Checking for uniform column location dimensionality...", end="")
        loc0 = list(self.cols.keys())[0]
        dim0 = len(loc0)
        for loc in self.cols.keys():
            assert len(loc) == dim0, \
                f"Different location dims at {loc0} and {loc}!"
        print("✔️")

        print("Checking that there are no location collisions...", end="")  # (inefficient but readable)
        for i, loc1 in enumerate(self.cols.keys()):
            for j, loc2 in enumerate(self.cols.keys()):
                if i != j:
                    assert loc1 != loc2, f"Location collision at {loc1}!"
        print("✔️")

        print("Checking that targets of conns exist...", end="")
        for col in self.cols.values():
            self.load_col(col)
            for (loc, _) in col.conns.keys():
                assert loc in self.cols
            self.free_col(col)
        print("✔️")

        print("Passed all checks!")


@dataclass
class Cfg:
    n_cols: int            # Number of columns (modules)
    ispec: list[T.I_Base]  # Input specification
    ospec: list[T.O_Base]  # Output specification
class Agt(AgtBase):  # Agent
    def __init__(self,
            cfg: Cfg,
            path: str,
            skip_init: bool=False):  # For loading from save

        self.cfg = cfg
        self.path = path

        self.n_cols = cfg.n_cols  # Number of columns
        self.ispec  = cfg.ispec  # Input specification
        self.ospec  = cfg.ospec  # Output specification

        self.I_cols: list[I_ColBase] = []
        self.O_cols: list[O_ColBase] = []
        self.cols: dict[Loc, ColBase] = {}  # location : col

        if not skip_init:
            print("\nInitializing new agent...")

            # Reset or create save directory
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            width = math.ceil(self.n_cols**(1/2))  # Side length

            # Initialize io columns
            for i, spec in tqdm(enumerate(self.ispec),
                                desc="Initializing input cols",
                                total=len(self.ispec)):
                assert isinstance(spec, T.I_Base)
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
                assert isinstance(spec, T.O_Base)
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

            print("done init.")


@dataclass
class BareCfg:
    n_cols: int
    ispec: list[T.I_Base]
    ospec: list[T.O_Base]
class BareAgt(AgtBase):
    def __init__(self,
            cfg: Cfg,
            path: str,
            skip_init: bool=False):  # For loading from save

        self.cfg = cfg
        self.path = path

        self.n_cols = cfg.n_cols  # Number of columns
        self.ispec  = cfg.ispec  # Input specification
        self.ospec  = cfg.ospec  # Output specification

        self.I_cols: list[I_ColBase] = []
        self.O_cols: list[O_ColBase] = []
        self.cols: dict[Loc, ColBase] = {}  # location : col

        if not skip_init:
            print("\nInitializing new agent...")

            # Reset or create save directory
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            width = math.ceil(self.n_cols**(1/2))  # Side length

            # Initialize io columns
            for i, spec in tqdm(enumerate(self.ispec),
                                desc="Initializing input cols",
                                total=len(self.ispec)):
                assert isinstance(spec, T.I_Base)
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
                assert isinstance(spec, T.O_Base)
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
                    col = BareCol(loc, BareColCfg(32))
                    self.cols[loc] = col
                    self.free_col(col)
                else:
                    break

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

            print("done init.")


@dataclass
class MNISTCfg:
    ispec: list[T.I_Base]
    ospec: list[T.O_Base]
class MNISTAgt(AgtBase):
    def __init__(self,
            cfg: Cfg,
            path: str,
            skip_init: bool=False):  # For loading from save

        self.cfg = cfg
        self.path = path

        self.ispec = ispec = cfg.ispec  # Input specification
        self.ospec = ospec = cfg.ospec  # Output specification

        assert len(ispec) == 1
        assert type(ispec[0]) is T.I_Vector
        assert ispec[0].d == 784

        assert len(ospec) == 1
        assert type(ospec[0]) is T.O_Vector
        assert ospec[0].d == 10

        self.I_cols: list[I_ColBase] = []
        self.O_cols: list[O_ColBase] = []
        self.cols: dict[Loc, ColBase] = {}  # location : col

        if not skip_init:
            print("\nInitializing new agent...")

            # Reset or create save directory
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            col1 = I_VectorCol((1, 0), I_VectorColCfg(784))
            self.cols[1, 0] = col1
            self.I_cols.append(col1)
            col2 = BareCol((1, 1), BareColCfg(32))
            self.cols[1, 1] = col2
            col3 = BareCol((1, 2), BareColCfg(32))
            self.cols[1, 2] = col3
            col_out = O_VectorCol((0, 1), O_VectorColCfg(10))
            self.cols[0, 1] = col_out
            self.O_cols.append(col_out)

            for loc, other_loc in [((1, 0), (1, 1)), ((1, 1), (1, 2))]:
                self.cols[loc].conns[other_loc, Dir.A] = conn(self.cols[loc], self.cols[other_loc], Dir.A)

            # Create directories for all cols
            for col in self.cols.values():
                os.mkdir(f"{path}/{col.loc}")

            self.use_debug = False

            print("done init.")
