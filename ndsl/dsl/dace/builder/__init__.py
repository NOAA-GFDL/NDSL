"""`builder` is a pipeline that transform a python program decorated with the `orchestrate` functions
(see `orchestration` or `NDSLRuntime`) into a `DaceExecutable` (see `dace_executable`).

The pipeline logic is guided by `builder.get_dace_executable` with steps broken in `parse` and `optimize`.
The pipeline uses an always-cache approach that leads to a single-call overhead.

Step subsystems are therefore not responsible for checking the correctness of their use (e.g. backend,
orchestration mode or caching.) and should be called outside of this package with extreme care.

`parse` is responsible for turning a `DaceProgram` into an SDFG and is called either by the `orchestration`
decorator or by DaCe itself during nested parsing.

`optimize` is responsible for tuning a `parsed SDFG`, e.g. the result of a full `parse` call, into an
optimized and compiled SDFG (DaCe cache will be created as part of this step).

`builder` is responsible for giving back a ready-to-execute `DaceExecutable` by either triggering the full
pipeline or giving back the cached executable. It is also responsible for handling multi-rank compilation
on the cube-sphere and is the only piece of code here that is multi-rank safe.

`cache` carries a subset of needed queries to the caching infrastructure
TODO: this will be reworked in a larger caching rework

The rest of the code are transforms or helpers that are used in the above steps.
"""
from .builder import get_dace_executable
from .parse import parse_sdfg
from .labeler import set_label

__all__ = [
    "get_dace_executable",
    "parse_sdfg",
    "set_label"
]
