
"""
Naming conventions:

nr_<name>  : Current activations
nr_<name>_ : New activations
is_<name>  : Internal weights (within each col (module))
conns      : External weights (connections between cols)

TODO change?

-------------------------------------------------------------------------------

Order of operations:

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

import ast
import atexit
import json
import math
import multiprocessing
import os
import random
import shutil
import signal
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Literal

import dacite
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# isort: off
from . import iotypes as T
from .iotypes import spec2dict, dict2spec
from . import funcs as fc


class Dir(Enum):  # Direction (kind) of connection
    A = 0  # Actual / "prediction errors" : connects actual to actual activations
    E = 1  # Expectations / "predictions" : connects actual to expectations activations


# Type hints ##################################################################
Loc = tuple[int, int]  # Location of col (module)

Input = torch.Tensor
Output = torch.Tensor
Inputs = list[Input]
Outputs = list[Output]


# Activations and weights classes #############################################
@dataclass(slots=True)
class Activs:
    actual: torch.Tensor  # actual activations
    expect: torch.Tensor  # expectations
    avg: torch.Tensor     # time average of activations
    avg_sq: torch.Tensor  # time average of squares of activations
    rms_avg: float = 1.0  # time average of layer's root mean square
    # TODO super cheap compared to weights, add as many as you want:
        # cache of softmaxed actual activations used as learning gate
        # difference through time (current - previous)
        # per activation plasticity/step size
        # batchnorm statistics
torch.serialization.add_safe_globals([Dir, Activs])


Weights = torch.Tensor  # TODO use dataclass? if need to attach more info


# Activations and weights inits ###############################################
# Activations
def activs(d: int) -> Activs:
    activations = torch.randn(d)
    return Activs(
        actual=activations,
        expect=torch.zeros(d),
        avg=activations.clone(),
        avg_sq=activations**2)

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
    # TODO take into account number of incoming conns
    return scale * torch.randn(d1, d2) / (d1**0.5)
    

# Constants ###################################################################
D_DEFAULT = 1024  # Layer dimensionality
ALPHA = 0.2  # Decay factor for activations EMA
ALPHA_RMS = 0.2  # Decay factor for activations whole layer RMS


# Classes #####################################################################
COLCFG_REGISTRY = {}
class ColCfgBase(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            COLCFG_REGISTRY[cls.__name__] = cls
COL_REGISTRY = {}
class ColBase(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            COL_REGISTRY[cls.__name__] = cls

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
        # Slight variations between subclasses
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
            if i == 0:  # TODO temporary hack
                return getattr(self, layer_name).actual
            elif i == 1:
                return getattr(self, layer_name).expect
            else:
                raise NotImplementedError
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.conn_layer_dict:
            layer_name, i = self.conn_layer_dict[name]
            if i == 0:  # TODO temporary hack
                getattr(self, layer_name).actual = value
            elif i == 1:
                getattr(self, layer_name).expect = value
            else:
                raise NotImplementedError
        else:
            super().__setattr__(name, value)  # Default behavior

    @property
    def count(self) -> tuple[int, int, int]:
        """Count number of elements of each type of data"""
        nrns = 0  # Activations
        isyns = 0  # Internal (within this col) weights
        esyns = 0  # External (to other cols) weights
        for name, value in vars(self).items():
            if name.startswith("nr_") and name[-1] != "_":
                nrns += value.actual.numel()
            elif name.startswith("is_"):
                isyns += value.numel()

        for weight in self.conns.values():
            esyns += weight.numel()

        return nrns, isyns, esyns

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
        os.makedirs(f"{agt_path}/{self.loc}", exist_ok=True)

        # Save type of col
        with open(f"{agt_path}/{self.loc}/type", "w") as f:
            f.write(type(self).__name__)

        # Save type of cfg
        with open(f"{agt_path}/{self.loc}/cfg_type", "w") as f:
            f.write(type(self.cfg).__name__)

        # Save cfg
        with open(f"{agt_path}/{self.loc}/cfg", "w") as f:
            json.dump(asdict(self.cfg), f)

        # Save activations
        for name, activations in vars(self).items():
            if name.startswith("nr_"):
                torch.save(activations, f"{agt_path}/{self.loc}/{name}")

        # Only save weights if they're loaded, otherwise would save None's
        if self.weights_loaded:
            if not keep_weights:
                self.weights_loaded = False

            # Save internal weights
            for name, weights in vars(self).items():
                if name.startswith("is_"):
                    torch.save(weights, f"{agt_path}/{self.loc}/{name}")
                    if not keep_weights:
                        setattr(self, name, None)

            # Save conns
            torch.save(self.conns, f"{agt_path}/{self.loc}/conns")
            
            if not keep_weights:
                self.conns = None

    @staticmethod
    def init_and_load(
            agt_path: str, 
            name: str, 
            load_activations: bool, 
            load_weights: bool) -> ColBase:
        loc = ast.literal_eval(name)
        with open(f"{agt_path}/{name}/type") as f:
            col_type = COL_REGISTRY[f.read().strip()]
            
        with open(f"{agt_path}/{name}/cfg_type") as f:
            cfg_type = COLCFG_REGISTRY[f.read().strip()]
        with open(f"{agt_path}/{name}/cfg") as f:
            cfg = dacite.from_dict(cfg_type, json.load(f))
        col = col_type(loc, cfg, skip_init=True)
        col.load(agt_path=agt_path, load_activations=load_activations, load_weights=load_weights)
        return col

    def load(self,
            agt_path: str,
            load_activations: bool,
            load_weights: bool) -> None:
        # Load activations
        if load_activations:
            for name in vars(self):
                if name.startswith("nr_"):
                    setattr(self, name,
                        torch.load(f"{agt_path}/{self.loc}/{name}",
                            map_location=torch.get_default_device()))

        # Load weights
        if load_weights:
            for name in vars(self):
                if name.startswith("is_"):
                    setattr(self, name,
                        torch.load(f"{agt_path}/{self.loc}/{name}",
                            map_location=torch.get_default_device()))

            self.conns = torch.load(f"{agt_path}/{self.loc}/conns",
                map_location=torch.get_default_device())

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
    def __init__(self, loc: Loc, cfg: BareColCfg, skip_init: bool = False):
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

    def update_activations(self, use_norm: bool = True):
        # Compute new activations averages
        # TODO compute averages of normalized activations not raw?
        new_avg_1 = ALPHA*self.nr_1_.actual + (1-ALPHA)*self.nr_1.avg
        new_avg_sq_1 = ALPHA*self.nr_1_.actual**2 + (1-ALPHA)*self.nr_1.avg_sq
        if use_norm:
            new_rms_avg = ALPHA_RMS*torch.sqrt(torch.mean(self.nr_1_.actual**2)).item() \
                + (1-ALPHA_RMS)*self.nr_1.rms_avg
            new_rms_avg = max(0.9, new_rms_avg)  # Don't amplify low activity
        else:
            new_rms_avg = 1.0

        # Move new activations to current
            # Intentional shallow copy
        self.nr_1.actual = self.nr_1_.actual / new_rms_avg
        self.nr_1.expect = self.nr_1_.expect
        self.nr_1.avg = new_avg_1
        self.nr_1.avg_sq = new_avg_sq_1
        self.nr_1.rms_avg = new_rms_avg

        # Reset new activations
        self.nr_1_.actual = fc.update(self.nr_1_.actual)
        self.nr_1_.expect = fc.update_e(self.nr_1_.expect)


I_VectorColCfg = BareColCfg
class I_VectorCol(BareCol, I_ColBase):
    def update_activations(self):
        # Compute new activations averages
        new_avg_1 = ALPHA*self.nr_1_.actual + (1-ALPHA)*self.nr_1.avg
        new_avg_sq_1 = ALPHA*self.nr_1_.actual**2 + (1-ALPHA)*self.nr_1.avg_sq

        # Move new activations to current
            # Intentional shallow copy
        self.nr_1.actual = self.nr_1_.actual
        self.nr_1.expect = self.nr_1_.expect
        self.nr_1.avg = new_avg_1
        self.nr_1.avg_sq = new_avg_sq_1

        # Receives perceptual input, don't reset

    def ipt(self, x: Input) -> None:
        x = x.to(torch.get_default_device()).to(torch.get_default_dtype())
        self.nr_1_.actual = x


O_VectorColCfg = BareColCfg
class O_VectorCol(BareCol, O_ColBase):
    def out(self) -> Output:
        return self.nr_1.actual.clone().cpu()


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

            self.weights_loaded = False
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
        for new in (self.nr_1_, self.nr_2_, self.nr_3_, self.nr_4_, self.nr_5_):
            after_inhibit = fc.inhibit(new)
            for f in fields(new):
                setattr(new, f.name, getattr(after_inhibit, f.name))

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
        self.is_1_2   = fc.lrn(self.nr_1.actual, self.is_1_2, self.nr_2.actual)
        self.is_2_3_f = fc.lrn(self.nr_2.actual, self.is_2_3_f, self.nr_3.actual)
        self.is_2_3_b = fc.lrn(self.nr_3.actual, self.is_2_3_b, self.nr_2.actual)
        self.is_2_4   = fc.lrn(self.nr_2.actual, self.is_2_4, self.nr_4.actual)
        self.is_4_5   = fc.lrn(self.nr_4.actual, self.is_4_5, self.nr_5.actual)

        # Apply activity rule (propagate activations)
        # self.nr_1_.actual  # No local inputs
        self.nr_2_.actual += fc.atv(self.nr_1.actual, self.is_1_2, self.nr_2_.actual) \
            + fc.atv(self.nr_3.actual, self.is_2_3_b, self.nr_2_.actual)
        self.nr_3_.actual += fc.atv(self.nr_2.actual, self.is_2_3_f, self.nr_3_.actual)
        self.nr_4_.actual += fc.atv(self.nr_2.actual, self.is_2_4, self.nr_4_.actual)
        self.nr_5_.actual += fc.atv(self.nr_4.actual, self.is_4_5, self.nr_5_.actual)

    def update_activations(self):
        for curr, new in zip(
                (self.nr_1,  self.nr_2,  self.nr_3,  self.nr_4,  self.nr_5),
                (self.nr_1_, self.nr_2_, self.nr_3_, self.nr_4_, self.nr_5_),
                strict=True):
            # Compute new activations averages
            new_avg = ALPHA*new.actual + (1-ALPHA)*curr.avg
            new_avg_sq = ALPHA*new.actual**2 + (1-ALPHA)*curr.avg_sq

            # Move new activations to current
                # Intentional shallow copy
            curr.actual = new.actual
            curr.expect = new.expect
            curr.avg = new_avg
            curr.avg_sq = new_avg_sq

            # Reset new activations
            new.actual = fc.update(new.actual)
            new.expect = fc.update_e(new.expect)


CFG_REGISTRY = {}
class CfgBase(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            CFG_REGISTRY[cls.__name__] = cls
AGT_REGISTRY = {}
class AgtBase(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            AGT_REGISTRY[cls.__name__] = cls

    def __init__(self, cfg, path: str, skip_init: bool = False):
        self.cfg = cfg
        self.path = path

        self.age = None  # Lifetime total number of ticks (steps)

        self.I_cols: list[I_ColBase] = []
        self.O_cols: list[O_ColBase] = []
        self.cols: dict[Loc, ColBase] = {}

        self.use_debug = False
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda _, __: sys.exit(0))

    def create_directory(self):
        # Reset or create save directory
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)

        # Create directories for all cols
        for col in self.cols.values():
            os.mkdir(f"{self.path}/{col.loc}")

    def cleanup(self):
        if hasattr(self, "bar"):  # tqdm progress bar
            self.bar.close()
        self.save()

    @abstractmethod
    def step(self, ipt: Inputs, disable_print: bool = False) -> Outputs:
        # Step all cols.
        # Lives in each subclass since the loops diverge.
        # Subclasses may add kwargs.
        ...

    def is_i(self, loc):
        return any(loc == col.loc for col in self.I_cols)
    def is_o(self, loc):
        return any(loc == col.loc for col in self.O_cols)
    def is_io(self, loc):
        return any(loc == col.loc for col in self.I_cols + self.O_cols)

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
        self.debug_process = ctx.Process(
            target=debugger,
            args=(self.path, self.pipes),
            daemon=True)
        self.debug_process.start()

        # Timestamps of most recent updates, to calculate cooldowns
        self.t_prevs = {}
        self.t_prevs["overview"] = time.time()
        self.t_prevs["col"] = time.time()
        self.t_prevs["conn"] = time.time()
        self.t_prevs["atv"] = time.time()
        self.t_prevs["norm_curves"] = 0.0  # 0 -> first overview carries curves

    def debug_update(self):
        def stats(x, is_weight):
            shape = tuple(x.shape)
            d = fc.density(x)
            n = torch.linalg.vector_norm(x).item()
            m = torch.mean(x).item()
            s = torch.std(x).item()
            
            # Histogram: The histogram displays approximately from -2 to +2 standard deviations,
            # with 43 bins in total, where each bin is 1/10 of a std wide.
            #
            # For example if the std is 1 (and so each bin is 0.1 wide), the bins are:
            # (-inf, -2.05), [-2.05, -1.95), ..., [-0.05, 0.05), ..., [1.95, 2.05), [2.05, inf)
            # 
            # For activation tensors the bins are fixed at 0.1 wide,
            # but for weights it is computed by rounding std/10 to 1eX, 2eX, or 5eX.
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
                elif 0.03 <= bin_width < 0.06:
                    bin_width = 0.05
                else:
                    bin_width = 0.1
            else:  # Activations
                bin_width = 0.1
            bins = torch.tensor([float("-inf")] + [bin_width*i - 20.5*bin_width for i in range(42)] + [float("inf")], device="cpu", dtype=torch.float64)
            h, _ = torch.histogram(x.cpu().to(torch.float64), bins)  # NOTE issue if use lower float precision
            h = h.tolist()
            has_nan = x.numel() != sum(h)
            all_nan = sum(h) == 0
            return (shape, d, n, m, s, (h, bin_width), has_nan, all_nan)

        # TODO make adaptive
        COOLDOWN_OVERVIEW = 0.2
        COOLDOWN_COL = 0.5
        COOLDOWN_CONN = 0.1
        COOLDOWN_ATV = 0.1
        COOLDOWN_NORM_CURVES = 2.0  # reads every weight value, so much slower

        # Send overview information ###########################################
        pipe, _ = self.pipes["overview"]
        if time.time()-self.t_prevs["overview"] > COOLDOWN_OVERVIEW:
            self.t_prevs["overview"] = time.time()

            info = {}
            info["timestamp"] = time.time()
            info["age"] = self.age
            nrns = 0  # Activations
            copies = 0  # Copies of activations (for additional info)
            isyns = 0  # Internal weights (within each col)
            esyns = 0  # External weights (between cols)

            actuals = []  # Accumulate actual activations for computing global density
            for col in self.cols.values():
                # This doesn't use col.count since also need to calculate density
                for name, x in vars(col).items():
                    # Only count current activations, not new
                    if name.startswith("nr_") and name[-1] != "_":
                        copies = len(fields(x))  # Assume is same for all activations
                        x = x.actual
                        nrns += x.numel()
                        actuals.append(x)
                    elif name.startswith("is_"):
                        isyns += x.numel()
                for weight in col.conns.values():
                    esyns += weight.numel()
            info["nrns"] = nrns
            info["copies"] = copies
            info["isyns"] = isyns
            info["esyns"] = esyns
            info["syns"] = isyns + esyns

            info["density"] = fc.density(torch.cat(actuals))

            # Sorted per-unit in-norm curves for the stats page. Unlike the
            # counting walk above this reads every weight value, so it rides
            # only the overview messages that clear its own slower cooldown.
            if time.time()-self.t_prevs["norm_curves"] > COOLDOWN_NORM_CURVES:
                self.t_prevs["norm_curves"] = time.time()
                info["norm_curves"] = self.norm_curves()

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
            info["nrns"], info["isyns"], info["esyns"] = col.count
            info["syns"] = info["isyns"] + info["esyns"]
            # Values of activations
            for name, x in vars(col).items():
                # Only send current activations, not new
                if name.startswith("nr_") and name[-1] != "_":
                    # NOTE Depends on Activs dataclass preserving order
                    info[name] = [stats(getattr(x, x_i.name), False)
                        for x_i in fields(x)
                        if isinstance(getattr(x, x_i.name), torch.Tensor)]
                elif name.startswith("is_"):
                    info[name] = stats(x, True)

            conns_skeleton = {}
            for (loc, direction) in col.conns:
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
                full_activs = getattr(self.cols[loc], f"nr_{i}")
                x = full_activs.actual
                x_avg = full_activs.avg
                info = {}
                info["timestamp"] = time.time()
                info["request"] = request  # for debugger to verify info is up to date
                info["x"] = x.cpu().numpy()
                info["x_avg"] = x_avg.cpu().numpy()
                pipe.send(info)

    def norm_curves(self, points: int = 64) -> dict[str, list[float]]:
        """Sorted per-unit incoming-norm curves for the debugger's stats page.

        "conns": each unit's L2 norm over its total incoming external weights,
        pooled across every conn targeting its col, divided by sqrt(in-degree).
        conn() already divides by sqrt(sender dim), so a healthy unit sits near
        the init scale (2.0) regardless of in-degree, sender size, or col.
        "internal": per-unit incoming norms of each is_* matrix (one matrix
        per target layer, so no cross-matrix pooling; weights() puts the init
        at the same 2.0). Head lifting above 2.0 = rich-get-richer runaway;
        flat near-zero tail = dead capacity (its length = the dead fraction).
        """
        sumsq = {}  # (receiving loc, width) -> per-unit sum of squares
        n_in = {}   # (receiving loc, width) -> number of incoming conns
        internal = []
        for col in self.cols.values():
            for (other_loc, _), w in col.conns.items():
                if w is None:
                    continue
                part = w.detach().float().pow(2).sum(dim=0).cpu()
                key = (other_loc, part.shape[0])
                if key in sumsq:
                    sumsq[key] += part
                    n_in[key] += 1
                else:
                    sumsq[key] = part
                    n_in[key] = 1
            for name, w in vars(col).items():
                if name.startswith("is_") and w is not None:
                    internal.append(
                        w.detach().float().pow(2).sum(dim=0).sqrt().cpu())
        curves = {}
        if sumsq:
            conns = torch.cat(
                [(ss / n_in[key]).sqrt() for key, ss in sumsq.items()])
            curves["conns"] = fc.sorted_curve(conns, points)
        if internal:
            curves["internal"] = fc.sorted_curve(torch.cat(internal), points)
        return curves

    def load_col(self, c: ColBase) -> None:
        if torch.get_default_device().type == "cuda":
            # While available memory is less than 5% of total
            while True:
                avail, tot = torch.cuda.memory.mem_get_info()
                if avail >= 0.05*tot:
                    break
                to_evict = random.choice(list(self.cols.values()))
                to_evict.to("cpu", non_blocking=True)
                torch.cuda.memory.empty_cache()
            c.to("cuda", non_blocking=True)

    def free_col(self, c: ColBase) -> None:
        if torch.get_default_device().type == "cuda":
            # Should only be necessary during large inits,
            # since loading while running already frees memory
            available, total = torch.cuda.memory.mem_get_info()
            # Use stronger threshold so that newly loaded cols
            # are not immediately freed
            if available < 0.04*total:
                c.to("cpu", non_blocking=True)

    def save(self, keep_weights: bool = True) -> None:
        print(f"\nSaving agent to \"{self.path}\":")
        os.makedirs(self.path, exist_ok=True)

        # Save age
        with open(f"{self.path}/age", "w") as f:
            f.write(str(self.age))

        # Save type of agt
        with open(f"{self.path}/type", "w") as f:
            f.write(type(self).__name__)

        # Save type of cfg
        with open(f"{self.path}/cfg_type", "w") as f:
            f.write(type(self.cfg).__name__)

        # Save cfg
        print("Saving cfg...")
        cfg_data = {}
        for field in fields(self.cfg):
            val = getattr(self.cfg, field.name)
            cfg_data[field.name] = ([spec2dict(s) for s in val]
                if field.name in ("ispec", "ospec") else val)
        with open(f"{self.path}/cfg", "w") as f:
            json.dump(cfg_data, f)

        # Save cols
        for col in tqdm(self.cols.values(), desc="Saving cols"):
            col.save(self.path, keep_weights=keep_weights)
        print("done saving.")

    @staticmethod
    def load(path: str,
            load_activations: bool = True,
            load_weights: bool = True) -> AgtBase:
        print(f"\nLoading agent from \"{path}\":")

        # Load type of agt
        with open(f"{path}/type") as f:
            agt_type = AGT_REGISTRY[f.read().strip()]

        # Load type of cfg
        with open(f"{path}/cfg_type") as f:
            cfg_type = CFG_REGISTRY[f.read().strip()]

        # Load cfg
        print("Loading cfg...")
        with open(f"{path}/cfg") as f:
            cfg_data = json.load(f)
            kwargs = {}
            for field in fields(cfg_type):
                if field.name not in cfg_data:
                    raise ValueError(f"Saved cfg is missing field {field.name!r}")
                kwargs[field.name] = ([dict2spec(s) for s in cfg_data[field.name]]
                    if field.name in ("ispec", "ospec") else cfg_data[field.name])
            cfg = cfg_type(**kwargs)

        agt = agt_type(cfg, path, skip_init=True)

        # Load age
        with open(f"{path}/age") as f:
            agt.age = int(f.read())

        # Load cols
        for name in tqdm(os.listdir(path), desc="Loading cols"):
            # Skip agt metadata and any stray files (.DS_Store, editor temps);
            # col saves are directories named by their loc tuple
            if name in ("type", "cfg", "cfg_type") or not os.path.isdir(f"{path}/{name}"):
                continue
            col = Col.init_and_load(path, name, load_activations, load_weights)

            loc = ast.literal_eval(name)
            agt.cols[loc] = col
            if isinstance(col, I_ColBase):
                agt.I_cols.append(col)
            elif isinstance(col, O_ColBase):
                agt.O_cols.append(col)
        print("done loading.")
        return agt


@dataclass
class Cfg(CfgBase):
    n_cols: int            # Number of columns (modules)
    ispec: list[T.I_Base]  # Input specification
    ospec: list[T.O_Base]  # Output specification
class Agt(AgtBase):  # Agent
    def __init__(self,
            cfg: Cfg,
            path: str,
            skip_init: bool = False):  # For loading from save

        super().__init__(cfg, path, skip_init)

        self.n_cols = cfg.n_cols  # Number of columns
        self.ispec  = cfg.ispec  # Input specification
        self.ospec  = cfg.ospec  # Output specification

        if not skip_init:
            print("\nInitializing new agent...")

            self.age: int = 0

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
                # TODO randomly sample the required number of conns (more efficient)
                for other_loc in self.cols:
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

            self.create_directory()

            print("done init.")

    def step(self, ipt: Inputs, disable_print: bool = False) -> Outputs:
        assert len(ipt) == len(self.ispec), f"Expected input of length {len(self.ispec)} but got length {len(ipt)}"

        # Receive inputs
        for col, x in zip(self.I_cols, ipt, strict=True):
            col.ipt(x)

        # First pass: compute new weights and activations
        self.bar = tqdm(self.cols.values(), desc="Computing new weights and activations...", disable=disable_print)
        for col in self.bar:
            # Used to communicate debugger exited
            if self.use_debug and self.pipes["overview"][0].poll():
                sys.exit(0)

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
            # Activations only, don't need to load weights
        self.bar = tqdm(self.cols.values(), desc="Updating and resetting activations...", disable=disable_print)
        for col in self.bar:
            # Used to communicate debugger exited
            if self.use_debug and self.pipes["overview"][0].poll():  # TODO not here?
                sys.exit(0)

            # Use lateral inhibition to combine actual and expected activations
            if hasattr(col, "inhibit"):
                col.inhibit()

            col.update_activations()

            if self.use_debug:  # TODO not here?
                self.debug_update()

        self.age += 1

        # Return outputs
        return [col.out() for col in self.O_cols]

    def verify(self) -> None:
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
        for loc in self.cols:
            assert len(loc) == dim0, \
                f"Different location dims at {loc0} and {loc}!"
        print("✔️")

        print("Checking that targets of conns exist...", end="")
        for col in self.cols.values():
            self.load_col(col)
            for (loc, _) in col.conns:
                assert loc in self.cols
            self.free_col(col)
        print("✔️")

        print("Passed all checks!")


@dataclass
class BareCfg(CfgBase):
    n_cols: int
    ispec: list[T.I_Base]
    ospec: list[T.O_Base]
class BareAgt(AgtBase):
    def __init__(self,
            cfg: BareCfg,
            path: str,
            skip_init: bool = False):  # For loading from save

        super().__init__(cfg, path, skip_init)

        self.n_cols = cfg.n_cols  # Number of columns
        self.ispec  = cfg.ispec  # Input specification
        self.ospec  = cfg.ospec  # Output specification

        if not skip_init:
            print("\nInitializing new agent...")

            self.age = 0

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
                for other_loc in self.cols:
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
                                col.conns[(other_loc, direction)] = \
                                    conn(col, self.cols[other_loc], direction, 2.0)
                                break  # Only at most one conn per target?
                self.free_col(col)

            self.create_directory()

            print("done init.")

    def step(self, ipt: Inputs, disable_print: bool = False) -> Outputs:
        assert len(ipt) == len(self.ispec), f"Expected input of length {len(self.ispec)} but got length {len(ipt)}"

        # Receive inputs
        for col, x in zip(self.I_cols, ipt, strict=True):
            col.ipt(x)

        # First pass: compute new weights and activations
        # TODO cache triangle & softmax activations rather than recomputing on the fly?
            # triangle cache can be locally in the loop since it's sender,
            # but softmaxed cache has to be attached to Activs since it's on the receiver end
                # (would also require multiple passes)
        self.bar = tqdm(self.cols.values(), desc="Computing new weights and activations...", disable=disable_print)
        for col in self.bar:
            # Used to communicate debugger exited
            if self.use_debug and self.pipes["overview"][0].poll():
                sys.exit(0)

            self.load_col(col)

            # Apply learning and activity rules internally
            col.step()

            # Apply learning rule to connections
            for (loc, direction), weight in col.conns.items():
                if direction == Dir.A:
                    weight = fc.lrn_oja_gated(
                        col.nr_1.actual,
                        weight,
                        fc.softmax_wta(self.cols[loc].nr_1.actual, beta=0.5, signed=True),
                        self.cols[loc].nr_1.actual,
                        ss=1e-2)
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
                        self.cols[loc].a_post_ += fc.atv_triangle(col.a_pre, weight, power=0.3)
                    elif direction == Dir.E:
                        self.cols[loc].e_post_ += fc.atv_triangle(col.e_pre, weight, power=0.3)

            self.free_col(col)

            if self.use_debug:
                self.debug_update()

        # Second pass: set current activations equal to new, and reset new
            # Activations only, don't need to load weights
        self.bar = tqdm(self.cols.values(), desc="Updating and resetting activations...", disable=disable_print)
        for col in self.bar:
            # Used to communicate debugger exited
            if self.use_debug and self.pipes["overview"][0].poll():  # TODO not here?
                sys.exit(0)

            # Use lateral inhibition to combine actual and expected activations
            if hasattr(col, "inhibit"):
                col.inhibit()

            col.update_activations()

            if self.use_debug:  # TODO not here?
                self.debug_update()

        self.age += 1

        # Return outputs
        return [col.out() for col in self.O_cols]


@dataclass
class MNISTCfg(CfgBase):
    ispec: list[T.I_Base]
    ospec: list[T.O_Base]
class MNISTAgt(AgtBase):
    def __init__(self,
            cfg: MNISTCfg,
            path: str,
            skip_init: bool = False):  # For loading from save

        super().__init__(cfg, path, skip_init)

        self.ispec = ispec = cfg.ispec  # Input specification
        self.ospec = ospec = cfg.ospec  # Output specification

        assert len(ispec) == 1
        assert type(ispec[0]) is T.I_Vector
        assert ispec[0].d == 784

        assert len(ospec) == 1
        assert type(ospec[0]) is T.O_Vector
        assert ospec[0].d == 10

        if not skip_init:
            print("\nInitializing new agent...")

            self.age = 0

            # Cols
            col1 = I_VectorCol((0, 0), I_VectorColCfg(784))
            self.cols[col1.loc] = col1
            self.I_cols.append(col1)

            # Single post-ReLU hidden layer, read directly. A second (col3) readout
            # hop only bottlenecks and collapses under learning; reading the
            # post-ReLU hidden gives a controlled learn-vs-random gap that is
            # visible at width 128 but washes out by ~256 as random ReLU features
            # saturate the task (see examples/04). 128 keeps the effect visible;
            # widen for higher absolute accuracy at the cost of a smaller gap.
            col2 = BareCol((1, 0), BareColCfg(128))
            self.cols[col2.loc] = col2

            # Conns
            self.cols[col1.loc].conns[col2.loc, Dir.A] = \
                conn(self.cols[col1.loc], self.cols[col2.loc], Dir.A, 3)

            self.create_directory()

            print("done init.")

    def step(self, ipt: Inputs, use_lrn: bool = True, disable_print: bool = False) -> Outputs:
        """
        Different from Agt's step: using that would result in
        mismatched learning, with new input and old representations.

        In the general case this is unavoidable (?),
        but in this small testing case it can be prevented.
        """
        col1 = self.cols[0, 0]
        col2 = self.cols[1, 0]
        assert isinstance(col2, BareCol)

        # Feed the raw grayscale input. BCM (lrn_adaptive) is continuous, so it
        # does not need the discretized input the old discrete rule did. Tested
        # roughly a wash on accuracy (binarizing flatters kNN, raw flatters ridge);
        # raw is the principled choice; keeping binary for a nicer number would
        # be p-hacking. Read ipt[0]; never rebind it (callers reuse the list).
        # For a discrete rule (lrn_*_d / instar) re-binarize:
        # col1.ipt(torch.where(ipt[0] > 0, 1.0, 0.0)).
        col1.ipt(ipt[0])
        col1.update_activations()
        col2.a_post_ += col1.a_pre @ col1.conns[col2.loc, Dir.A]
        # Main net uses step activation function, here uses ReLU,
        # since ReLU would blow up main net, and step is overly restrictive here
        col2.a_post_ = F.relu(col2.a_post_)
        col2.update_activations(use_norm=False)

        if use_lrn:
            # BCM (adaptive): the only local rule that self-stabilizes without a
            # competition mechanism, via its sliding threshold (see examples/04).
            # Pass the full Activs (nr_1), not nr_1.actual, since BCM needs the
            # avg-of-squares threshold (avg_sq). ss=1e-4: BCM is potentiation-dominated
            # with no weight bound, and in float16 the default 1e-2 overflows to NaN
            # within a few steps; 1e-4 is the largest stable rate here.
            col1.conns[col2.loc, Dir.A] = fc.lrn_adaptive(
                col1.nr_1,
                col1.conns[col2.loc, Dir.A],
                col2.nr_1,
                ss=1e-4
            )

        if self.use_debug:
            if self.pipes["overview"][0].poll():
                sys.exit(0)
            self.debug_update()

        self.age += 1

        return [torch.zeros(10)]


DEFAULT_CIFAR_LAYERS = [(3, 96, 5, "max"), (96, 384, 3, "max"), (384, 1536, 3, "avg")]
class CIFARAgt:
    """
    Deep local-Hebbian conv stack for CIFAR: no backprop, no objective, online and
    single-sample. Reproduces the SoftHebb recipe (Moraitis et al., arXiv:2209.11883)
    under gc's online constraint with no offline oracles. Three conv layers trained
    only by a local rule; the pooled final feature map is the rep. See 09_cifar10.py.

    Rule = oja-signed (SoftHebb's update): an Oja prototype rule gated by a signed
    soft-WTA. The winner moves its weight toward the input (Hebbian), losers away
    (anti-Hebbian). The loser repulsion makes channels tile instead of collapsing onto
    one prototype; gc's own rules (BCM, plain instar/oja) were tested and fail here, so
    the anti-Hebbian gate is load-bearing. Four ingredients each matter: Triangle
    activation relu(u-mean_c(u))^p (graded, so depth doesn't collapse to one-hot); soft
    weight-norm (let ||w|| drift via the rule's own decay; hard projection makes
    learning destructive); online BatchNorm (current-image stats at train, running at
    eval: a homeostatic regularizer that keeps the rep stable under continual
    training, a transient peak without it); and no whitening (it suppresses the kNN gain).

    layers: list of (in_ch, out_ch, kernel, pool), pool in {max, avg, none}:
    max=MaxPool(4,s2,p1), avg=AvgPool(2), none=hold spatial (for extra depth).
    pool: side of the final adaptive-avg-pool; default 4 keeps the full 4x4 map
    (24576-dim rep, matching SoftHebb's readout; reducing it to 2x2 costs ridge & kNN).

    Standalone (a conv doesn't fit the vector-col Col framework); step() /
    get_representations() interface like the probe examples. `training` (BN mode)
    defaults to `use_lrn`; pass training=True for a no-learning BN warmup. No
    save/load/debugger.
    """
    def __init__(self, layers: list | None = None, base_lr: float = 0.03,
                 power: float = 0.7, signed_beta: float = 1.0, scale: float = 3.0,
                 bn_mom: float = 0.01, pool: int = 4):
        self.layers = DEFAULT_CIFAR_LAYERS if layers is None else layers
        self.base_lr = base_lr
        self.power = power              # Triangle activation exponent
        self.signed_beta = signed_beta  # soft-WTA gate inverse temperature (beta)
        self.bn_mom = bn_mom            # online-BN running-stat momentum
        self.pool = pool                # final adaptive-avg-pool side (4 -> full 4x4 = 24576-dim rep)
        # He-ish init per layer; weight (in*k*k, out) projects unfolded patches
        self.W = [scale * torch.randn(ic * k * k, oc) / (ic * k * k) ** 0.5
                  for ic, oc, k, _ in self.layers]
        self.bn_m = [torch.zeros(ic) for ic, _, _, _ in self.layers]  # online-BN running mean
        self.bn_v = [torch.ones(ic) for ic, _, _, _ in self.layers]   # online-BN running var
        self.rep: torch.Tensor | None = None

    def _bn(self, fmap: torch.Tensor, li: int, training: bool) -> torch.Tensor:
        """Online BatchNorm: normalize by current-image spatial stats at train (and
        update the running stats), by running stats at eval."""
        cur_m, cur_v = fmap.mean((1, 2)), fmap.var((1, 2))
        if training:
            self.bn_m[li] = (1 - self.bn_mom) * self.bn_m[li] + self.bn_mom * cur_m
            self.bn_v[li] = (1 - self.bn_mom) * self.bn_v[li] + self.bn_mom * cur_v
            m, v = cur_m, cur_v
        else:
            m, v = self.bn_m[li], self.bn_v[li]
        return (fmap - m[:, None, None]) / (v[:, None, None] + 1e-5).sqrt()

    def step(self, ipt: Inputs, use_lrn: bool = True, training: bool | None = None,
             disable_print: bool = False) -> Outputs:
        if training is None:  # BN trains iff we are learning (override for warmup)
            training = use_lrn
        fmap = ipt[0].to(torch.get_default_device()).to(torch.get_default_dtype())  # (C, H, W)
        for li, (_ic, oc, k, pool) in enumerate(self.layers):
            fmap = self._bn(fmap, li, training)
            h, w = fmap.shape[1], fmap.shape[2]
            x = F.unfold(fmap.unsqueeze(0), k, stride=1, padding=(k - 1) // 2)[0].T  # (h*w, ic*k*k)
            u = x @ self.W[li]
            y = fc.triangle_batched(u, self.power)  # Triangle activation (graded)
            if use_lrn:
                g = fc.softmax_wta_batched(u, self.signed_beta, signed=True)  # signed soft-WTA gate
                self.W[li] = fc.lrn_oja_gated_batched(x, self.W[li], g, u, ss=self.base_lr)  # SoftHebb update (soft norm, no hard projection)
            fmap = y.T.reshape(oc, h, w).unsqueeze(0)
            if pool == "max":  # MaxPool 4x4/s2 (early layers, halve spatial)
                fmap = F.max_pool2d(fmap, 4, stride=2, padding=1)[0]
            elif pool == "avg":  # AvgPool 2x2 (halve spatial)
                fmap = F.avg_pool2d(fmap, 2)[0]
            else:  # hold spatial (deeper layers)
                fmap = fmap[0]
        self.rep = F.adaptive_avg_pool2d(fmap.unsqueeze(0), self.pool).reshape(-1).clone()
        return [torch.zeros(10)]  # dummy output; the offline experiment reads the rep

    def get_representations(self) -> torch.Tensor:
        assert self.rep is not None
        return self.rep
