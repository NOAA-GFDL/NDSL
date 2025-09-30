from ndsl import Quantity
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Float
from ndsl.boilerplate import get_factories_single_tile
import dataclasses
import numpy as np

from ndsl import State


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


def test_state():
    _, qty_factry = get_factories_single_tile(5, 5, 3, 0, backend="dace:cpu_kfirst")

    microphys_state = CodeState.zeros(qty_factry)
    microphys_state.inner_A.A.field[:] = 42.42
    microphys_state.to_netcdf()
    microphys_state2 = CodeState.zeros(qty_factry)
    microphys_state2.from_netcdf("CodeState.nc4")
    assert (microphys_state2.inner_A.A.field[:] == 42.42).all()
    a = np.ones((5, 5, 3))
    b = np.ones((5, 5, 3))
    c = np.ones((5, 5, 3))
    b[:] = 23.23
    microphys_state2.init_zero_copy(
        {"inner_A": {"A": a}, "inner_B": {"B": b}, "C": c},
        check=False,
    )
    assert (microphys_state2.inner_A.A.field[:] == 1.0).all()
    assert (microphys_state2.inner_B.B.field[:] == 23.23).all()
