import numpy as np
from ndsl import StencilFactory, QuantityFactory, orchestrate, DaceConfig
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, X_INTERFACE_DIM, Y_INTERFACE_DIM
from ndsl import Namelist, StencilFactory
from ndsl.stencils.testing.translate import TranslateFortranData2Py
from gt4py.cartesian.gtscript import computation, FORWARD, PARALLEL, interval
from typing import Tuple

def _make_range(offset: int, domain: int):        
    return range(offset, offset+domain)

def fillq2zero1_plain_python(
    domain: Tuple[int, ...],
    offset: Tuple[int, ...],
    q: FloatField,
    mass:FloatField,
    fillq:FloatField
):
    tpw = np.sum(q*mass,2)
    for J in _make_range(offset[1], domain[1]):
        for I in _make_range(offset[0], domain[0]):
            neg_tpw = 0.
            for L in _make_range(offset[2], domain[2]):
                if(q[I,J,L] < 0.0):
                    neg_tpw = neg_tpw + (q[I,J,L]*mass[I,J,L])
                    q[I,J,L] = 0.0
            for L in _make_range(offset[2], domain[2]):
                if(q[I,J,L] >= 0.0):
                    q[I,J,L] = q[I,J,L]*(1.0 + neg_tpw/(tpw[I,J]-neg_tpw))
            fillq[I,J] = -neg_tpw

class FillQZero:
    def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory):
        orchestrate(
            obj=self,
            config=(
                stencil_factory.config.dace_config or
                DaceConfig(communicator=None, backend=stencil_factory.backend))
        )
        self._domain = quantity_factory.sizer.get_extent([X_DIM, Y_DIM, Z_DIM])
        self._offset = quantity_factory.sizer.get_origin([X_DIM, Y_DIM, Z_DIM])

    def __call__(
        self,
        q: FloatField,
        mass: FloatField,
        fillq: FloatField
    ):
        fillq2zero1_plain_python(
            domain=self._domain,
            offset=self._offset,
            q=q,
            mass=mass,
            fillq=fillq)
        

class TranslateFILLQ2ZERO1(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.max_error=1e-7
        self.compute_func = FillQZero(   # type: ignore
            self.stencil_factory,
            self.grid.quantity_factory,
        )

        fillq_info = self.grid.compute_dict()
        fillq_info["serialname"] = "fq"
        self.in_vars["data_vars"] = {
            "mass": self.grid.compute_dict(),
            "q": self.grid.compute_dict(),
            "fillq": fillq_info,
        }
        self.out_vars = {
            "fillq": fillq_info,
            "q": self.grid.compute_dict(),
        }


    def compute_from_storage(self, inputs):
        self.compute_func(**inputs)
        return inputs
