
import pytest

import src.iotypes as T
from src.iotypes import SPEC_REGISTRY, dict2spec, spec2dict


@pytest.mark.parametrize("spec", [
    T.I_Vector(d=784, name="image", desc="flattened digit"),
    T.O_Vector(d=10),
    T.I_Scalar(name="reward"),
    T.I_Tensor(shape=[3, 32, 32], name="frame"),
    T.O_Keyboard(keys=["a", "b", "c"]),
    T.O_MouseMovement(mode="abs"),
])
def test_spec_dict_roundtrip(spec):
    """spec2dict tags the payload with the class name; dict2spec reconstructs the
    exact same spec (the tagged-union serialization used by save/load)."""
    d = spec2dict(spec)
    assert d["spec_type"] == type(spec).__name__
    assert dict2spec(d) == spec


def test_spec_registry_maps_names_to_classes():
    """Every concrete I_/O_ spec auto-registers by name (recursive subclass walk)."""
    for name in ("I_Vector", "O_Vector", "I_Tensor", "I_Video", "O_Keyboard", "O_MouseButtons"):
        assert name in SPEC_REGISTRY
    assert SPEC_REGISTRY["I_Vector"] is T.I_Vector
    assert SPEC_REGISTRY["O_Vector"] is T.O_Vector


def test_tensor_numel():
    """I_Tensor / O_Tensor report the product of their shape."""
    assert T.I_Tensor(shape=[3, 4, 5]).numel() == 60
    assert T.O_Tensor(shape=[10]).numel() == 10


def test_dict2spec_ignores_unknown_key_order():
    """dict2spec works regardless of key ordering and does not mutate its input."""
    d = spec2dict(T.I_Vector(d=128))
    original = dict(d)
    assert dict2spec(d) == T.I_Vector(d=128)
    assert d == original  # input dict left intact (dict2spec copies before popping)
