"""This module uses DaCe's op_repository feature to override symbols/AST object
during parsing and replace them with an SDFG compatible representation. This
allow custom NDSL system to be natively orchestratable."""

from ndsl.dsl.typing import Float
from dace.frontend.python.replacements import _datatype_converter, UfuncInput
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.newast import ProgramVisitor
from dace import SDFG, SDFGState, dtypes


@oprepo.replaces("Float")
def _converter(pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput):
    """Replace `Float(x)` with a typecast of `x` to the proper floating precision type"""
    return _datatype_converter(
        sdfg,
        state,
        arg,
        dtype=dtypes.dtype_to_typeclass(Float),
    )
