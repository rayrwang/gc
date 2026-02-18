
import random
import multiprocessing
import time
from numbers import Number
import os
import math
import shutil
import sys

from tqdm import tqdm
import numpy as np
import torch

from .debugger import nrn_debugger
from . import funcs as fc

Loc = tuple[Number, Number]


class Nrn:
    def __init__(self, loc):
        self.loc = loc

        self.x: Number  # Init in each agt
        self.conns: dict[Loc, Number] = {}


class NrnAgtBase:
    def create_directory(self):
        # Reset or create save directory
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)

        # Create directories for all cols
        for col in self.nrns.values():
            os.mkdir(f"{self.path}/{col.loc}")

    def debug_init(self):
        self.pipes = {}

        self.pipes["overview"] = multiprocessing.Pipe()
        self.pipes["nrn"] = multiprocessing.Pipe()

        # Start debug process
        self.debug_process = multiprocessing.Process(target=nrn_debugger, args=(self.path, self.pipes))
        self.debug_process.start()

        # Timestamps of most recent updates, to calculate cooldowns
        self.t_prevs = {}
        self.t_prevs["overview"] = time.time()
        self.t_prevs["nrn"] = time.time()

    def debug_update(self):
        def stats(x):
            x = x.cpu().to(torch.float64)
            shape = tuple(x.shape)
            threshold = 1.0
            d = (torch.sum(torch.where(x < threshold, 0.0, 1.0)) / x.numel()).item()
            n = torch.linalg.vector_norm(x).item()
            m = torch.mean(x).item()
            s = torch.std(x).item()
            h = None
            # Histogram
                # currently 43 bins: (-inf, -2.05), [-2.05, -1.95), ..., [-0.05, 0.05), ..., [1.95, 2.05), [2.05, inf)
            bins = torch.tensor([float("-inf")] + [0.1*i - 2.05 for i in range(42)] + [float("inf")], device="cpu", dtype=torch.float64)
            h, _ = torch.histogram(x, bins)  # NOTE issue if use lower float precision
            h = h.tolist()
            assert x.numel() == sum(h), "Failed to calculate histogram, probably because tensor is NaN"
            return (shape, d, n, m, s, h)

        COOLDOWN_OVERVIEW = self.COOLDOWN_OVERVIEW
        COOLDOWN_NRN = self.COOLDOWN_NRN

        # Send overview information #####################################################################
        pipe, _ = self.pipes["overview"]
        if time.time()-self.t_prevs["overview"] > COOLDOWN_OVERVIEW:
            self.t_prevs["overview"] = time.time()

            info = {}
            info["timestamp"] = time.time()
            nrns = []
            syns = []
            for loc, nrn in self.nrns.items():
                info[loc] = nrn.x
                nrns.append(nrn.x)
                for _, syn in nrn.conns.items():
                    syns.append(syn)

            # Activations stats and histogram
            nrns = torch.tensor(nrns)
            nrn_stats = stats(nrns)
            info["nrn_stats"] = nrn_stats

            # Weights stats and histogram
            syns = torch.tensor(syns)
            syn_stats = stats(syns)
            info["syn_stats"] = syn_stats

            pipe.send(info)

        # Send information for single nrn #####################################
        request = None
        pipe, _ = self.pipes["nrn"]
        while pipe.poll():  # Get most recent request
            request = pipe.recv()
        if request is not None and (time.time()-self.t_prevs["nrn"]) > COOLDOWN_NRN:
            self.t_prevs["col"] = time.time()

            col = self.nrns[request]
            info = {}
            info["timestamp"] = time.time()
            info["loc"] = col.loc  # for debugger to verify info is up to date
            info["conns"] = col.conns
            pipe.send(info)


class Ising(NrnAgtBase):
    def __init__(self, path, n_cols):
        self.path = path

        self.COOLDOWN_OVERVIEW = 0.1
        self.COOLDOWN_NRN = 0.1

        self.nrns = {}

        # Initialize nrns
        width = math.ceil(n_cols**(1/2))
        for i, (y, x) in tqdm(enumerate(np.ndindex((width,width))),
                              desc="Initializing bulk of cols",
                              total=n_cols):
            if i < n_cols:
                loc = (x+1, y+1)  # Offset +1 since 0 is for input/output
                nrn = Nrn(loc)
                nrn.x = random.choice((-1, 1))
                self.nrns[loc] = nrn
            else:
                break

        # Initialize connections
        for loc, nrn in self.nrns.items():
            for dir in [(0,1),(1,0),(0,-1),(-1,0)]:
                neighbor_loc = tuple([sum(x) for x in zip(loc, dir)])
                neighbor = self.nrns.get(neighbor_loc)
                if neighbor:
                    nrn.conns[neighbor_loc] = 1

        self.create_directory()

    def step(self):
        if self.pipes["overview"][0].poll():
            sys.exit()
            
        for nrn in tqdm(self.nrns.values(), disable=True):
            delta_h = 0
            for conn_loc, weight in nrn.conns.items():
                delta_h += 2*weight * nrn.x*self.nrns[conn_loc].x
            if delta_h < 0:
                nrn.x = -nrn.x
            else:
                # BETA = 1  # Low temperature
                BETA = 0.4  # Critical temperature
                # BETA = 0.01  # High temperature
                if random.random() < min(0.5, math.exp(-BETA*delta_h)):
                    nrn.x = -nrn.x
        self.debug_update()


class Oscillator(NrnAgtBase):
    def __init__(self, path):
        self.path = path

        self.COOLDOWN_OVERVIEW = 0.0
        self.COOLDOWN_NRN = 0.0

        self.nrns = {}

        # Initialize nrns
        nrn = Nrn((0,0))
        nrn.x = 1
        self.nrns[(0,0)] = nrn

        nrn = Nrn((1,0))
        nrn.x = 0
        self.nrns[(1,0)] = nrn

        # Initialize connections
        self.nrns[(0,0)].conns[(1,0)] = 1
        self.nrns[(1,0)].conns[(0,0)] = -1

        self.create_directory()

    def step(self):
        if self.pipes["overview"][0].poll():
            sys.exit()

        self.nrns[(0,0)].x += 2
        for nrn in tqdm(self.nrns.values(), disable=True):
            self.debug_update()
            time.sleep(0.2)
            for conn_loc, weight in nrn.conns.items():
                self.nrns[conn_loc].x += fc.spike(nrn.x) * weight
            nrn.x = fc.update(nrn.x)


class NrnAgt(NrnAgtBase):
    def __init__(self, path, n_cols):
        self.path = path

        self.COOLDOWN_OVERVIEW = 0.01
        self.COOLDOWN_NRN = 0.01

        self.rng = np.random.default_rng()

        self.nrns = {}

        # Initialize nrns
        width = math.ceil(n_cols**(1/2))
        for i, (y, x) in tqdm(enumerate(np.ndindex((width,width))),
                              desc="Initializing bulk of cols",
                              total=n_cols):
            if i < n_cols:
                loc = (x+1, y+1)  # Offset +1 since 0 is for input/output
                nrn = Nrn(loc)
                nrn.x = 1
                self.nrns[loc] = nrn
            else:
                break

        # Initialize connections
        for nrn in tqdm(self.nrns.values(), desc="Initializing conns"):
            # Independent probability for each possible conn
            for other_loc in self.nrns.keys():
                if other_loc != nrn.loc:
                    distance = fc.dist(nrn.loc, other_loc)
                    p = 1 / (distance + 1e-3)**2
                    if random.random() < p:
                        nrn.conns[other_loc] = random.gauss(sigma=0.4)  # With shrieker

        self.create_directory()

    def step(self):
        if self.pipes["overview"][0].poll():
            sys.exit()

        # If have low enough weights, need constantly active nrs,
            # otherwise activity would die out
        self.nrns[(20, 20)].x = 1

        for nrn in tqdm(self.nrns.values(), disable=True):
            for conn_loc, weight in nrn.conns.items():
                noise = self.rng.lognormal(sigma=0.5)
                nrn.conns[conn_loc] = fc.lrn(nrn.x, weight, self.nrns[conn_loc].x)
                self.nrns[conn_loc].x += noise * fc.spike(nrn.x) * weight
            nrn.x = fc.update(nrn.x)
        self.debug_update()

