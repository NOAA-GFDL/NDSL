from __future__ import annotations

import dace.properties
from dace import library, nodes
from dace.sdfg.analysis.vector_inference import Any
from dace.transformation import transformation as xf


@library.node
class _Labeller(nodes.LibraryNode):
    implementations = ""
    default_implementation = "pure"
    unique_name = dace.properties.Property(dtype=str, desc="Unique name")

    def __init__(self, unique_name: str, **kwargs: dict[str, Any]) -> None:
        super().__init__(name="NDSLRuntime_Label", **kwargs)
        self._unique_name = unique_name


@library.register_expansion(_Labeller, "pure")
class _ExpandLabeller(xf.ExpandTransformation):
    @staticmethod
    def expansion(
        node: _Labeller,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> nodes.Tasklet:
        return nodes.Tasklet("donothing", code="pass")


def set_label(
    sdfg: dace.SDFG | dace.CompiledSDFG, qualname: str, is_top_sdfg: bool
) -> None:
    """Surround the SDFG with two state/library node combo labelling
    the code for future reference in further optimization.

    WARNING: The Label are passthrough, any use of `simplify` _will remove
    them from the SDFG_ and this is on purpose so there's no tracers of them
    in runtime.
    """
    # Cannot be applied to already compiled SDFG
    if isinstance(sdfg, dace.CompiledSDFG):
        return

    for state in sdfg.states():
        if sdfg.in_edges(state) == []:
            # With the topmost SDFG we have to skip over the
            # "init" state
            if is_top_sdfg:
                state = sdfg.add_state_after(
                    state,
                    label=f"__Label_Enter__{qualname}",
                )
            else:
                state = sdfg.add_state_before(
                    state,
                    label=f"__Label_Enter__{qualname}",
                )
            state.add_node(_Labeller(unique_name=f"Enter__{qualname}"))
        if sdfg.out_edges(state) == []:
            state = sdfg.add_state_after(
                state,
                label=f"__Label_Exit__{qualname}",
            )
            state.add_node(_Labeller(unique_name=f"Exit__{qualname}"))
