from typing import Any

import dace
from dace import library, nodes
from dace.transformation import transformation as xf

from ndsl import OptimizationConfig


@library.node
class _Labeler(nodes.LibraryNode):
    implementations: dict[str, Any] = {}
    default_implementation = "pure"
    unique_name = dace.properties.Property(dtype=str, desc="Unique name")

    def __init__(
        self,
        unique_name: str,
        local_optimization: OptimizationConfig | None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(name="NDSLRuntime_Label", **kwargs)
        # HACK to avoid state fusion of labeler states
        # MPI WaitAll block state fusion, so we just pretend to be one 🐉.
        # Keeping the labeler states non-fused is important to keep code flow consistent until we
        # get to the schedule tree.
        self.label = "_Waitall_"

        self._unique_name = unique_name
        self._local_optimizations = local_optimization

    def has_side_effects(self, sdfg: dace.SDFG) -> bool:
        # HACK
        # LibraryNodes with side effects aren't touched by simplify. This
        # keeps the library nodes alive until we get to the schedule tree
        # where we can use the information.
        return True


@library.register_expansion(_Labeler, "pure")
class _ExpandLabeler(xf.ExpandTransformation):
    environments: list[Any] = []

    @staticmethod
    def expansion(
        node: _Labeler,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> nodes.Tasklet:
        return nodes.Tasklet("donothing", code="pass")


def set_label(
    sdfg: dace.SDFG | dace.CompiledSDFG,
    qualname: str,
    is_top_sdfg: bool,
    local_optimizations: OptimizationConfig | None,
) -> None:
    """Surround the SDFG with two state/library node combo labelling
    the code for future reference in further optimization.

    WARNING: The Label are passthrough, any use of `simplify()` _will remove
    them from the SDFG_ and this is on purpose so there's no traces of them
    in runtime.
    """
    # Cannot be applied to already compiled SDFG
    if isinstance(sdfg, dace.CompiledSDFG):
        return

    for state in sdfg.nodes():
        if sdfg.in_edges(state) == []:
            # With the topmost SDFG we have to skip over the
            # "init" state
            if is_top_sdfg:
                label_state = sdfg.add_state_after(
                    state,
                    label=f"__Label_Enter__{qualname}",
                )
            else:
                label_state = sdfg.add_state_before(
                    state,
                    label=f"__Label_Enter__{qualname}",
                )
            label_state.add_node(
                _Labeler(
                    unique_name=f"Enter__{qualname}",
                    local_optimization=local_optimizations,
                )
            )
        if sdfg.out_edges(state) == []:
            label_state = sdfg.add_state_after(
                state,
                label=f"__Label_Exit__{qualname}",
            )
            label_state.add_node(
                _Labeler(
                    unique_name=f"Exit__{qualname}",
                    local_optimization=local_optimizations,
                )
            )
