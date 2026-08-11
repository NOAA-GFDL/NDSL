from enum import IntEnum

from ndsl import Backend, NDSLRuntime, StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, enum, interval
from ndsl.dsl.typing import Int, IntField


def test_enum_runtime() -> None:
    @enum
    class MyEnum(IntEnum):
        Zero = 0
        A = 10
        B = 20
        C = 30

    def stencil_with_enum(out_field: IntField, order: MyEnum):  # type: ignore
        with computation(PARALLEL), interval(0, 1):
            out_field = 32
            if order < MyEnum.A:
                out_field = MyEnum.A

        with computation(PARALLEL), interval(1, 2):
            out_field = 23
            out_field = MyEnum.B

        with computation(PARALLEL), interval(2, None):
            out_field = 56
            out_field = MyEnum.C

    class Code(NDSLRuntime):
        def __init__(self, stencil_factory: StencilFactory) -> None:
            super().__init__(stencil_factory)

            self._stencil = stencil_factory.from_dims_halo(
                func=stencil_with_enum, compute_dims=[I_DIM, J_DIM, K_DIM]
            )

        def __call__(self, field: IntField) -> None:  # type: ignore
            self._stencil(field, MyEnum.Zero)

    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=3, ny=3, nz=5, nhalo=0, backend=Backend("st:python:cpu:IJK")
    )
    field = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="n/a", dtype=Int)

    test_code = Code(stencil_factory)
    test_code(field)

    assert field[0, 0, 0] == MyEnum.A.value
    assert field[0, 0, 1] == MyEnum.B.value
    assert (field.field[0, 0, 2:] == MyEnum.C.value).all()
