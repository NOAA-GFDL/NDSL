from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField
from ndsl.quantity.field_bundle import FieldBundle, FieldBundleType


def assign_4d_field_stcl(field_4d: FieldBundleType.T("Tracers")):  # type: ignore # noqa
    with computation(PARALLEL), interval(...):
        field_4d[0, 0, 0][1] = 63.63
        field_4d[0, 0, 0][3] = 63.63


def assign_3d_field_stcl(field_3d: FloatField):
    with computation(PARALLEL), interval(...):
        field_3d = 121.121


def test_field_bundle():
    # Grid & Factories
    NX = 2
    NY = 2
    NZ = 2
    N4th = 5
    stencil_factory, quantity_factory = get_factories_single_tile(NX, NY, NZ, 1)

    # Type register
    FieldBundleType.register("Tracers", (N4th,))

    # Make stencils
    assign_4d_field = stencil_factory.from_dims_halo(
        func=assign_4d_field_stcl,
        compute_dims=[X_DIM, Y_DIM, Z_DIM],
    )
    assign_3d_field = stencil_factory.from_dims_halo(
        func=assign_3d_field_stcl,
        compute_dims=[X_DIM, Y_DIM, Z_DIM],
    )

    # "Input" data
    new_quantity_factory = FieldBundle.extend_3D_quantity_factory(
        quantity_factory, {"tracers": N4th}
    )
    data = new_quantity_factory.ones([X_DIM, Y_DIM, Z_DIM, "tracers"], units="kg/g")

    # Build Bundle
    tracers = FieldBundle(
        bundle_name="tracers",
        quantity=data,
        mapping={"vapor": 0, "cloud": 2},
    )

    # Test
    tracers.quantity.field[:, :, :, :] = 48.4
    tracers.quantity.field[:, :, :, 2] = 21.21

    assign_4d_field(tracers.quantity)

    assert (tracers.quantity.field[:, :, :, 0] == 48.4).all()
    assert (tracers.quantity.field[:, :, :, 1] == 63.63).all()
    assert (tracers.quantity.field[:, :, :, 2] == 21.21).all()
    assert (tracers.quantity.field[:, :, :, 3] == 63.63).all()
    assert (tracers.quantity.field[:, :, :, 4] == 48.4).all()

    tracers.vapor.field[:] = 1000.1000

    assert (tracers.quantity.field[:, :, :, 0] == 1000.1000).all()
    assert (tracers.quantity.field[:, :, :, 1] == 63.63).all()

    assign_3d_field(tracers.cloud)

    assert (tracers.cloud.field[:] == 121.121).all()
    assert (tracers.quantity.field[:, :, :, 2] == tracers.cloud.field[:]).all()


if __name__ == "__main__":
    test_field_bundle()
