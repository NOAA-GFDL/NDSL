import pytest

from ndsl import StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils import (
    adjust_divide_stencil,
    adjustmentfactor_stencil,
    copy,
    set_value,
)
from ndsl.stencils.basic_operations import copy_defn


class Copy:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._copy_stencil = stencil_factory.from_origin_domain(
            copy,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(self, f_in: FloatField, f_out: FloatField):
        self._copy_stencil(f_in, f_out)


class AdjustmentFactor:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._adjustmentfactor_stencil = stencil_factory.from_origin_domain(
            adjustmentfactor_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        factor: FloatFieldIJ,
        f_out: FloatField,
    ):
        self._adjustmentfactor_stencil(factor, f_out)


class SetValue:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._set_value_stencil = stencil_factory.from_origin_domain(
            set_value,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        f_out: FloatField,
        value: Float,
    ):
        self._set_value_stencil(f_out, value)


class AdjustDivide:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._adjust_divide_stencil = stencil_factory.from_origin_domain(
            adjust_divide_stencil,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        factor: FloatField,
        f_out: FloatField,
    ):
        self._adjust_divide_stencil(factor, f_out)


def test_copy():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    infield = quantity_factory.zeros(
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )
    outfield = quantity_factory.ones(
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    stencil = Copy(stencil_factory)
    stencil(f_in=infield, f_out=outfield)

    assert (infield.field == outfield.field).all()


def test_copy_defn_deprecated():
    stencil_factory, _ = get_factories_single_tile(nx=20, ny=20, nz=79, nhalo=0)

    with pytest.deprecated_call(match=r"^copy_defn\(\.\.\.\) is deprecated"):
        grid_indexing = stencil_factory.grid_indexing
        stencil_factory.from_origin_domain(
            copy_defn,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )


def test_adjustment_factor():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    factor = quantity_factory.full(dims=[X_DIM, Y_DIM], units="m", value=2.0)
    outfield = quantity_factory.full(dims=[X_DIM, Y_DIM, Z_DIM], units="m", value=2.0)

    stencil = AdjustmentFactor(stencil_factory)
    stencil(factor=factor, f_out=outfield)
    assert (outfield.field == 4.0).all()


def test_setvalue():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )
    fill_value = 2.0

    outfield = quantity_factory.zeros(
        dims=[X_DIM, Y_DIM, Z_DIM],
        units="m",
    )

    stencil = SetValue(stencil_factory)
    stencil(f_out=outfield, value=fill_value)

    assert (outfield.field == fill_value).all()


def test_adjust_divide():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    factor = quantity_factory.full(dims=[X_DIM, Y_DIM, Z_DIM], units="m", value=2.0)
    outfield = quantity_factory.full(dims=[X_DIM, Y_DIM, Z_DIM], units="m", value=2.0)

    stencil = AdjustDivide(stencil_factory)
    stencil(factor=factor, f_out=outfield)

    assert (outfield.field == 1.0).all()
