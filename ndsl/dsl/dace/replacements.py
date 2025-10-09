"""This module uses DaCe's op_repository feature to override symbols/AST objects
during parsing and replace them with an SDFG compatible representation. This
allows custom NDSL syntax, objects and symbols to be natively orchestratable."""

from dace import SDFG, SDFGState, dtypes
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.newast import ProgramVisitor
from dace.frontend.python.replacements import (
    UfuncInput,
    UfuncOutput,
    _datatype_converter,
)

from ndsl.dsl.typing import Float, Int


@oprepo.replaces("Float")
def _convert_Float(
    _pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput
) -> UfuncOutput:
    """Replace `Float(x)` with a typecast of `x` to the proper floating precision type"""
    return _datatype_converter(
        sdfg,
        state,
        arg,
        dtype=dtypes.dtype_to_typeclass(Float),
    )


@oprepo.replaces("Int")
def _convert_Int(
    _pv: ProgramVisitor, sdfg: SDFG, state: SDFGState, arg: UfuncInput
) -> UfuncOutput:
    """Replace `Int(x)` with a typecast of `x` to the proper integer precision type"""
    return _datatype_converter(
        sdfg,
        state,
        arg,
        dtype=dtypes.dtype_to_typeclass(Int),
    )
