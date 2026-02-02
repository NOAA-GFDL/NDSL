from ndsl import NDSLRuntime, QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile_orchestrated
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, interval
from ndsl.dsl.typing import FloatField, IntFieldIJ


def reset_mask(dp1: FloatField, pe1: FloatField, mask: IntFieldIJ):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1
    with computation(FORWARD), interval(0, 1):
        mask = 0


def conditional_copy(dp1: FloatField, pe1: FloatField, mask: IntFieldIJ):
    with computation(PARALLEL), interval(0, 1):
        if mask == 1:
            dp1 = pe1


class OrchestratedProgramm(NDSLRuntime):
    def __init__(
        self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory
    ):
        super().__init__(stencil_factory)

        self._reset_mask = self._stencil_factory.from_dims_halo(
            reset_mask, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._conditional_copy = self._stencil_factory.from_dims_halo(
            conditional_copy, compute_dims=[I_DIM, J_DIM, K_DIM]
        )
        self._mask = self.make_local(quantity_factory, [I_DIM, J_DIM])

    def __call__(self, dp1: FloatField, pe1: FloatField) -> None:
        self._reset_mask(dp1, pe1, self._mask)
        self._conditional_copy(dp1, pe1, self._mask)

    def mask_has_been_reset(self) -> bool:
        return (self._mask.field[:] == 0).all()


def test_conditional_copy_with_mask():
    stencil_factory, quantity_factory = get_factories_single_tile_orchestrated(
        nx=4, ny=5, nz=6, nhalo=1
    )

    dp1 = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="")
    pe1 = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="")
    code = OrchestratedProgramm(stencil_factory, quantity_factory)

    code(dp1, pe1)

    assert code.mask_has_been_reset()
