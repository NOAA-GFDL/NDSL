import dataclasses

import numpy as np
import pytest

from ndsl import (
    Local,
    LocalState,
    NDSLRuntime,
    Quantity,
    QuantityFactory,
    StencilFactory,
)
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float


def test_dace_data_descriptor_is_transient() -> None:
    nx = 5
    shape = (nx,)
    local = Local(
        data=np.empty(shape),
        origin=(0,),
        extent=(nx,),
        dims=("dim_X",),
        units="n/a",
        backend="debug",
    )
    array = local.__descriptor__()
    assert array.transient


@dataclasses.dataclass
class GoodLocals(LocalState):
    my_local: Local = dataclasses.field(
        metadata={
            "name": "my_local",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "intent": "?",
            "dtype": Float,
        }
    )


@dataclasses.dataclass
class BadLocals(LocalState):
    my_local: Local = dataclasses.field(
        metadata={
            "name": "my_local",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "intent": "?",
            "dtype": Float,
        }
    )
    my_quantity: Quantity = dataclasses.field(
        metadata={
            "name": "my_local",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "?",
            "intent": "?",
            "dtype": Float,
        }
    )


class TheCode(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ) -> None:
        super().__init__(stencil_factory)
        self._quantity_factory = quantity_factory
        self._locals = GoodLocals.make_locals(quantity_factory)
        self._a_local = self.make_local(quantity_factory, [I_DIM, J_DIM, K_DIM])

    def allocate_bad_locals(self) -> None:
        self._bad = BadLocals.make_locals(self._quantity_factory)

    def check_local_right_after_init(self) -> bool:
        return (self._locals.my_local.field[:] == 123456789).all()

    def __call__(self) -> None:
        self._a_local.field[:] = 1  # legal
        self._locals.my_local.field[:] = 2  # legal


def test_proper_initialization() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(
        3, 3, 5, 0, backend="debug"
    )
    the_code = TheCode(stencil_factory, quantity_factory)
    assert the_code.check_local_right_after_init()


def test_forbidden_access_to_locals() -> None:
    stencil_factory, quantity_factory = get_factories_single_tile(
        3, 3, 5, 0, backend="debug"
    )
    the_code = TheCode(stencil_factory, quantity_factory)

    the_code()

    with pytest.raises(
        RuntimeError,
        match="Forbidden Local access: _a_local called outside of TheCode.",
    ):
        the_code._a_local.field[:] = 0

    with pytest.raises(
        RuntimeError,
        match="Forbidden Local access: my_local called outside*",
    ):
        the_code._locals.my_local.field[:] = 0

    with pytest.raises(
        RuntimeError,
        match="LocalState allocated outside of NDSLRuntime: forbidden",
    ):
        A = GoodLocals.make_locals(quantity_factory)

    with pytest.raises(
        TypeError,
        match="State contains Quantity my_quantity but is allocated as a LocalState. LocalState with Locals can _only_ contain Locals.",
    ):
        the_code.allocate_bad_locals()


def test_local_state_as_regular_state() -> None:
    _, quantity_factory = get_factories_single_tile(3, 3, 5, 0, backend="debug")
    with pytest.raises(
        RuntimeError,
        match="LocalState allocated outside of NDSLRuntime: forbidden",
    ):
        _ = GoodLocals.make_locals(quantity_factory)
    B = GoodLocals.make_as_state(quantity_factory)
    assert type(B.my_local) is Quantity
    with pytest.raises(
        RuntimeError,
        match="LocalState allocated outside of NDSLRuntime: forbidden",
    ):
        _ = GoodLocals.make_locals(quantity_factory)
