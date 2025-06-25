"""This module uses DaCe's op_repository feature to override symbols/AST object
during parsing and replace them with an SDFG compatible representation. This
allow custom NDSL system to be natively orchestratable."""

from dace import SDFG, SDFGState, dtypes
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.newast import ProgramVisitor
from dace.frontend.python.replacements import UfuncInput, _datatype_converter

from ndsl.dsl.typing import Float, Int


@oprepo.replaces("Float")
def _convert_Float(_pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput):
    """Replace `Float(x)` with a typecast of `x` to the proper floating precision type"""
    return _datatype_converter(
        sdfg,
        state,
        arg,
        dtype=dtypes.dtype_to_typeclass(Float),
    )


@oprepo.replaces("Int")
def _convert_Int(_pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput):
    """Replace `Int(x)` with a typecast of `x` to the proper integer precision type"""
    return _datatype_converter(
        sdfg,
        state,
        arg,
        dtype=dtypes.dtype_to_typeclass(Int),
    )
