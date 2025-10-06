import dataclasses
from pathlib import Path

import numpy as np

from ndsl import Quantity, State
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Float


@dataclasses.dataclass
class CodeState(State):
    @dataclasses.dataclass
    class InnerA:
        A: Quantity = dataclasses.field(
            metadata={
                "name": "A",
                "dims": [X_DIM, Y_DIM, Z_DIM],
                "units": "kg kg-1",
                "intent": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class InnerB:
        B: Quantity = dataclasses.field(
            metadata={
                "name": "B",
                "dims": [X_DIM, Y_DIM, Z_DIM],
                "units": "1",
                "intent": "?",
                "dtype": Float,
            }
        )

    inner_A: InnerA
    inner_B: InnerB
    C: Quantity = dataclasses.field(
        metadata={
            "name": "C",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg kg-1",
            "intent": "?",
            "dtype": Float,
        }
    )


def test_basic_state(tmpdir):
    _, quantity_factory = get_factories_single_tile(
        5, 5, 3, 0, backend="dace:cpu_kfirst"
    )

    # Test allocator
    microphys_state = CodeState.ones(quantity_factory)
    assert (microphys_state.inner_A.A.field[:] == 1.0).all()

    # Test NetCDF round trip
    microphys_state.inner_A.A.field[:] = 42.42
    microphys_state.to_netcdf(Path(tmpdir))
    microphys_state2 = CodeState.zeros(quantity_factory)
    microphys_state2.update_from_netcdf(Path(tmpdir))
    assert (microphys_state2.inner_A.A.field[:] == 42.42).all()

    # Test memory move (no copy)
    a = np.ones((5, 5, 3))
    b = np.ones((5, 5, 3))
    c = np.ones((5, 5, 3))
    b[:] = 23.23
    microphys_state2.update_move_memory(
        {"inner_A": {"A": a}, "inner_B": {"B": b}, "C": c},
        check_shape_and_strides=False,
    )
    assert (microphys_state2.inner_A.A.field[:] == 1.0).all()
    assert (microphys_state2.inner_B.B.field[:] == 23.23).all()

    # Test fill
    microphys_state2.fill(18.18)
    assert (microphys_state2.inner_A.A.field[:] == 18.18).all()
    assert (microphys_state2.inner_B.B.field[:] == 18.18).all()


@dataclasses.dataclass
class CodeStateWithDDim(State):
    @dataclasses.dataclass
    class InnerA:
        ddim_A: Quantity = dataclasses.field(
            metadata={
                "name": "A",
                "dims": [X_DIM, Y_DIM, Z_DIM, "ExtraDim1"],
                "units": "kg kg-1",
                "intent": "?",
                "dtype": Float,
            }
        )

    @dataclasses.dataclass
    class InnerB:
        ddim_B: Quantity = dataclasses.field(
            metadata={
                "name": "A",
                "dims": [X_DIM, Y_DIM, Z_DIM, "ExtraDim2"],
                "units": "kg kg-1",
                "intent": "?",
                "dtype": Float,
            }
        )

    inner_A: InnerA
    inner_B: InnerB
    no_ddim: Quantity = dataclasses.field(
        metadata={
            "name": "C",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg kg-1",
            "intent": "?",
            "dtype": Float,
        }
    )


def test_state_ddim():
    _, quantity_factory = get_factories_single_tile(
        5, 5, 3, 0, backend="dace:cpu_kfirst"
    )

    # Test allocator
    microphys_state = CodeStateWithDDim.ones(
        quantity_factory,
        data_dimensions={
            "ExtraDim1": 3,
            "ExtraDim2": 4,
        },
    )
    assert (microphys_state.no_ddim.field[:] == 1.0).all()
    assert microphys_state.inner_A.ddim_A.field.shape == (5, 5, 3, 3)
    assert (microphys_state.inner_A.ddim_A.field[:] == 1.0).all()
    assert microphys_state.inner_B.ddim_B.field.shape == (5, 5, 3, 4)
    assert (microphys_state.inner_B.ddim_B.field[:] == 1.0).all()
