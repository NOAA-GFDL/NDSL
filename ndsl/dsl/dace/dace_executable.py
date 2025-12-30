import dataclasses
import warnings
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
    _skip_hash: bool = False
    """Internal: skip hash computation because some
    arguments where detected to be un-hashable last time"""

    def hash_expected_dsl_args(self, args: tuple[Any], kwargs: dict[str, Any]) -> int:
        """Hash direct memory of NDSL expected types.

        Handling the following types:
        - quantity | Numpy.ndarray | cupy.ndarray: we hash the C pointer through the array interface,
        - state: called into a bespoke function,
        - everything else is passed as-is to `hash` which _can_ fail.
        """
        if self._skip_hash:
            self.arguments = None  # Flush arguments to force recompute
            return 0

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

        try:
            h = hash(tuple(to_hash))
        except TypeError as e:
            warnings.warn(
                f"[NDSL|Orchestration] argument type aren't hashable: {e}",
                DeprecationWarning,
                stacklevel=2,
            )
            self.arguments = None  # Flush arguments to force recompute
            self._skip_hash = True  # Skip future checks
            return 0

        return h


DaceExecutables = dict[DaceProgram, DaceExecutable]
