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


def test_state(tmpdir):
    _, quantity_factory = get_factories_single_tile(
        5, 5, 3, 0, backend="dace:cpu_kfirst"
    )

    microphys_state = CodeState.zeros(quantity_factory)
    microphys_state.inner_A.A.field[:] = 42.42
    microphys_state.to_netcdf(Path(tmpdir))
    microphys_state2 = CodeState.zeros(quantity_factory)
    microphys_state2.update_from_netcdf(Path(tmpdir))
    assert (microphys_state2.inner_A.A.field[:] == 42.42).all()
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
