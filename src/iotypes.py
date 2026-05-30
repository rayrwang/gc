
"""
This file contains specifications for input and output types.
The actual inputs and outputs are simply lists of tensors.
"""

from functools import reduce
import operator
from dataclasses import dataclass, asdict
from typing import Sequence, Literal
from abc import ABC

import dacite

class I_Base(ABC):
    pass
class O_Base(ABC):
    pass

# Input types #################################################################
@dataclass
class I_Video(I_Base):
    w: int  # width
    h: int  # height
    c: int = 3  # channels
    d: int = 8  # depth: bits per channel
    name: str = ""
    desc: str = ""

@dataclass
class I_Audio(I_Base):
    ...  # wave or spectrogram

@dataclass
class I_Scalar(I_Base):
    name: str = ""
    desc: str = ""
    type: str = "float16"

@dataclass
class I_Vector(I_Base):
    d: int
    name: str = ""
    desc: str = ""
    type: str = "float16"

@dataclass
class I_Tensor(I_Base):
    shape: Sequence[int]
    name: str = ""
    desc: str = ""
    type: str = "float16"
    def numel(self): return reduce(operator.mul, self.shape, 1)

@dataclass
class I_Tool(I_Base):
    ...

@dataclass
class I_Str(I_Base):  # Text
    ...

# Output types ################################################################
@dataclass
class O_Keyboard(O_Base):
    keys: Sequence[str]
    name: str = ""
    desc: str = ""

@dataclass
class O_MouseMovement(O_Base):
    name: str = ""
    desc: str = ""
    mode: Literal["rel", "abs"] = "rel"

@dataclass
class O_MouseButtons(O_Base):
    name: str = ""
    desc: str = ""
    buttons: Sequence[str] = ("left", "middle", "right")

@dataclass
class O_Scalar(O_Base):
    name: str = ""
    desc: str = ""
    type: str = "float16"

@dataclass
class O_Vector(O_Base):
    d: int
    name: str = ""
    desc: str = ""
    type: str = "float16"

@dataclass
class O_Tensor(O_Base):
    shape: Sequence[int]
    name: str = ""
    desc: str = ""
    type: str = "float16"
    def numel(self): return reduce(operator.mul, self.shape, 1)

@dataclass
class O_Tool(O_Base):
    ...

@dataclass
class O_Str(O_Base):  # Text
    ...

def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)
SPEC_REGISTRY = {c.__name__: c for base in (I_Base, O_Base) for c in _all_subclasses(base)}

def spec2dict(obj):
    return {"spec_type": type(obj).__name__, **asdict(obj)}
def dict2spec(d):
    d = dict(d)
    cls = SPEC_REGISTRY[d.pop("spec_type")]
    return dacite.from_dict(cls, d)
