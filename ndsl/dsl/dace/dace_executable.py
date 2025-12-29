import dataclasses
from typing import Any

import dace
from dace.frontend.python.parser import DaceProgram

from ndsl.quantity import State


@dataclasses.dataclass
class DaceExecutable:
    """Bundle the executable (lib) and it's marshalled
    arguments for execution"""

    compiled_sdfg: dace.CompiledSDFG
    """Loaded compiled SDFG"""
    arguments: dict[str, Any] | None = None
    """Arguments as C-ready pointers"""
    arguments_hash: int = 0
    """Hash reflexting the python/C pointers arguments"""

    @staticmethod
    def hash_expected_dsl_args(args: tuple[Any], kwargs: dict[str, Any]) -> int:
        """Hash direct memory of NDSL expected types.

        Handling the following types:
        - quantity | Numpy.ndarray | cupy.ndarray: we hash the C pointer through the array interface,
        - state: called into a bespoke function,
        - everything else is passed as-is to `hash` which _can_ fail.
        """
        to_hash = []
        for arg in list(args) + list(kwargs.values()):
            if hasattr(arg, "__array_interface__"):
                to_hash.append(arg.__array_interface__["data"][0])
            elif hasattr(arg, "__cuda_array_interface__"):
                to_hash.append(arg.__cuda_array_interface__["data"][0])
            elif isinstance(arg, State):
                to_hash.append(arg._hash())
            else:
                to_hash.append(arg)

        return hash(tuple(to_hash))


DaceExecutables = dict[DaceProgram, DaceExecutable]
