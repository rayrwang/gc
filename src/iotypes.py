
"""
This file contains specifications for input and output types.
The actual inputs and outputs are simply lists of tensors.
"""

from functools import reduce
import operator
from dataclasses import dataclass
from typing import Sequence
from abc import ABC

class I_Base(ABC):
    pass
class O_Base(ABC):
    pass

# Input types #################################################################
@dataclass
class I_Video(I_Base):
    h: int  # height
    w: int  # width
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
class O_Mouse(O_Base):
    name: str = ""
    desc: str = ""
    mode: str = "rel"  # "rel" or "abs"
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
