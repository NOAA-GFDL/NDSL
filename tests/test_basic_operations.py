from ndsl import NDSLRuntime, StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.stencils import (
    add,
    add_2d,
    adjust_divide_stencil,
    adjustmentfactor_stencil,
    copy,
    copy_2d,
    divide,
    divide_2d,
    multiply,
    multiply_2d,
    set_value,
    subtract,
    subtract_2d,
)


class Copy(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self._copy_stencil = stencil_factory.from_dims_halo(
            func=copy, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._copy_2d_stencil = stencil_factory.from_dims_halo(
            func=copy_2d, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def __call__(
        self,
        f_in: FloatField,
        f_in_2d: FloatFieldIJ,
        f_out: FloatField,
        f_out_2d: FloatFieldIJ,
    ):
        self._copy_stencil(f_in, f_out)
        self._copy_2d_stencil(f_in_2d, f_out_2d)


class Add(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self._add_stencil = stencil_factory.from_dims_halo(
            func=add, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._add_2d_stencil = stencil_factory.from_dims_halo(
            func=add_2d, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def __call__(
        self,
        f_in_1: FloatField,
        f_in_2: FloatField,
        f_in_1_2d: FloatFieldIJ,
        f_in_2_2d: FloatFieldIJ,
        f_out: FloatField,
        f_out_2d: FloatFieldIJ,
    ):
        self._add_stencil(f_in_1, f_in_2, f_out)
        self._add_2d_stencil(f_in_1_2d, f_in_2_2d, f_out_2d)


class Subtract(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self._subtract_stencil = stencil_factory.from_dims_halo(
            func=subtract, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._subtract_2d_stencil = stencil_factory.from_dims_halo(
            func=subtract_2d, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def __call__(
        self,
        f_in_1: FloatField,
        f_in_2: FloatField,
        f_in_1_2d: FloatFieldIJ,
        f_in_2_2d: FloatFieldIJ,
        f_out: FloatField,
        f_out_2d: FloatFieldIJ,
    ):
        self._subtract_stencil(f_in_1, f_in_2, f_out)
        self._subtract_2d_stencil(f_in_1_2d, f_in_2_2d, f_out_2d)


class Multiply(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self._multiply_stencil = stencil_factory.from_dims_halo(
            func=multiply, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._multiply_2d_stencil = stencil_factory.from_dims_halo(
            func=multiply_2d, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def __call__(
        self,
        f_in_1: FloatField,
        f_in_2: FloatField,
        f_in_1_2d: FloatFieldIJ,
        f_in_2_2d: FloatFieldIJ,
        f_out: FloatField,
        f_out_2d: FloatFieldIJ,
    ):
        self._multiply_stencil(f_in_1, f_in_2, f_out)
        self._multiply_2d_stencil(f_in_1_2d, f_in_2_2d, f_out_2d)


class Divide(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
        self._divide_stencil = stencil_factory.from_dims_halo(
            func=divide, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._divide_2d_stencil = stencil_factory.from_dims_halo(
            func=divide_2d, compute_dims=[I_DIM, J_DIM, K_DIM]
        )

    def __call__(
        self,
        f_in_1: FloatField,
        f_in_2: FloatField,
        f_in_1_2d: FloatFieldIJ,
        f_in_2_2d: FloatFieldIJ,
        f_out: FloatField,
        f_out_2d: FloatFieldIJ,
    ):
        self._divide_stencil(f_in_1, f_in_2, f_out)
        self._divide_2d_stencil(f_in_1_2d, f_in_2_2d, f_out_2d)


class AdjustmentFactor(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
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


class SetValue(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
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


class AdjustDivide(NDSLRuntime):
    def __init__(self, stencil_factory: StencilFactory):
        super().__init__(stencil_factory)
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

    infield = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="m")
    infield_2d = quantity_factory.zeros(dims=[I_DIM, J_DIM], units="m")
    outfield = quantity_factory.ones(dims=[I_DIM, J_DIM, K_DIM], units="m")
    outfield_2d = quantity_factory.ones(dims=[I_DIM, J_DIM], units="m")

    stencil = Copy(stencil_factory)
    stencil(f_in=infield, f_in_2d=infield_2d, f_out=outfield, f_out_2d=outfield_2d)

    assert (infield.field == outfield.field).all()
    assert (infield_2d.field == outfield_2d.field).all()


def test_add():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    infield_1 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=1.0)
    infield_2 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)
    infield_1_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=1.0)
    infield_2_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=2.0)
    outfield = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="m")
    outfield_2d = quantity_factory.zeros(dims=[I_DIM, J_DIM], units="m")

    stencil = Add(stencil_factory)
    stencil(
        f_in_1=infield_1,
        f_in_2=infield_2,
        f_in_1_2d=infield_1_2d,
        f_in_2_2d=infield_2_2d,
        f_out=outfield,
        f_out_2d=outfield_2d,
    )

    assert (outfield.field == (infield_1.field + infield_2.field)).all()
    assert (outfield_2d.field == (infield_1_2d.field + infield_2_2d.field)).all()


def test_subtract():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    infield_1 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=1.0)
    infield_2 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)
    infield_1_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=1.0)
    infield_2_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=2.0)
    outfield = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="m")
    outfield_2d = quantity_factory.zeros(dims=[I_DIM, J_DIM], units="m")

    stencil = Subtract(stencil_factory)
    stencil(
        f_in_1=infield_1,
        f_in_2=infield_2,
        f_in_1_2d=infield_1_2d,
        f_in_2_2d=infield_2_2d,
        f_out=outfield,
        f_out_2d=outfield_2d,
    )

    assert (outfield.field == (infield_1.field - infield_2.field)).all()
    assert (outfield_2d.field == (infield_1_2d.field - infield_2_2d.field)).all()


def test_multiply():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    infield_1 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=1.0)
    infield_2 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)
    infield_1_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=1.0)
    infield_2_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=2.0)
    outfield = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="m")
    outfield_2d = quantity_factory.zeros(dims=[I_DIM, J_DIM], units="m")

    stencil = Multiply(stencil_factory)
    stencil(
        f_in_1=infield_1,
        f_in_2=infield_2,
        f_in_1_2d=infield_1_2d,
        f_in_2_2d=infield_2_2d,
        f_out=outfield,
        f_out_2d=outfield_2d,
    )

    assert (outfield.field == (infield_1.field * infield_2.field)).all()
    assert (outfield_2d.field == (infield_1_2d.field * infield_2_2d.field)).all()


def test_divide():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    infield_1 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=1.0)
    infield_2 = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)
    infield_1_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=1.0)
    infield_2_2d = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=2.0)
    outfield = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="m")
    outfield_2d = quantity_factory.zeros(dims=[I_DIM, J_DIM], units="m")

    stencil = Divide(stencil_factory)
    stencil(
        f_in_1=infield_1,
        f_in_2=infield_2,
        f_in_1_2d=infield_1_2d,
        f_in_2_2d=infield_2_2d,
        f_out=outfield,
        f_out_2d=outfield_2d,
    )

    assert (outfield.field == (infield_1.field / infield_2.field)).all()
    assert (outfield_2d.field == (infield_1_2d.field / infield_2_2d.field)).all()


def test_adjustment_factor():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    factor = quantity_factory.full(dims=[I_DIM, J_DIM], units="m", value=2.0)
    outfield = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)

    stencil = AdjustmentFactor(stencil_factory)
    stencil(factor=factor, f_out=outfield)
    assert (outfield.field == 4.0).all()


def test_setvalue():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )
    fill_value = 2.0

    outfield = quantity_factory.zeros(
        dims=[I_DIM, J_DIM, K_DIM],
        units="m",
    )

    stencil = SetValue(stencil_factory)
    stencil(f_out=outfield, value=fill_value)

    assert (outfield.field == fill_value).all()


def test_adjust_divide():
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=20, ny=20, nz=79, nhalo=0
    )

    factor = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)
    outfield = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="m", value=2.0)

    stencil = AdjustDivide(stencil_factory)
    stencil(factor=factor, f_out=outfield)

    assert (outfield.field == 1.0).all()
