from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import Backend, OptimizationConfig, ndsl_log
from ndsl.dsl.dace.stree.optimizations.cartesian_merge import CartesianMerge
from ndsl.dsl.dace.stree.optimizations.common import list_index
from ndsl.dsl.dace.stree.optimizations.kernelize_maps import KernelizeMaps
from ndsl.dsl.dace.stree.optimizations.refine_transients import (
    CartesianRefineTransients,
)
from ndsl.dsl.dace.stree.optimizations.remove_loops import InlineVertical2DWrite


class _LabeledSection(tn.ScheduleTreeScope):
    def __init__(
        self,
        *,
        children: list[tn.ScheduleTreeNode],
        parent: tn.ScheduleTreeScope,
        label: str,
        optimizations: OptimizationConfig,
    ) -> None:
        super().__init__(children=children, parent=parent)
        self.label = label
        self.optimizations = optimizations

    def as_string(self, indent: int = 0) -> str:
        result = indent * tn.INDENTATION + f"section '{self.label}':\n"
        return result + super().as_string(indent)


class _LabelSections(tn.ScheduleNodeVisitor):
    """
    Transform entry/exit labeler nodes into a `LabeledSection` (see above)
    for easier later handling in case of local optimizations. Handles nested
    labeled sections.

    Before

    ```none
    # program before

    library_node("entry my_stencil")
    map i in [...]
      map j in [...]
        map k in [...]
          # contents of "my_stencil"
    library node("exit my_stencil")

    # program continues
    ```

    After

    ```none
    # program before

    labeled_section "my_stecil":
      map i in [...]
        map j in [...]
          map k in [...]
            # contents of "my_stencil

    # program continues
    ```
    """

    _entry_nodes: list[tn.LibraryCall]
    """
    Stack of entry nodes for labeled sections. Nodes get pushed on entering the
    labeled section and are removed again upon reaching the matching exit node.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entry_nodes = []

    def __str__(self) -> str:
        return "_LabelSections"

    def visit_LibraryCall(self, node: tn.LibraryCall) -> None:
        if node.node.name != "NDSLRuntime_Label":
            # Only look at "our" label nodes.
            return

        if node.node.unique_name.startswith("Enter__"):
            # Keep taps on where we start.
            self._entry_nodes.append(node)
            return

        if node.node.unique_name.startswith("Exit__"):
            # Find the matching entry node.
            section_start = self._entry_nodes.pop()

            # sanity checks
            # - ensure we have the right section (if not, something is screwed up)
            name = section_start.node.unique_name.removeprefix("Enter__")
            exit_name = node.node.unique_name.removeprefix("Exit__")
            assert name == exit_name
            # - ensure we have the same parent (if not something is screwed up)
            parent = section_start.parent
            assert parent == node.parent

            # Grab all the nodes in-between and put them in a `LabeledSection`.
            start_index = list_index(parent.children, section_start)
            end_index = list_index(parent.children, node)
            new_node = _LabeledSection(
                children=[
                    child for child in parent.children[start_index + 1 : end_index]
                ],
                parent=parent,
                label=name,
                optimizations=node.node._local_optimizations,
            )

            # Overwrite the nodes (including the labels) with the new node.
            parent.children[start_index : end_index + 1] = [new_node]

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        # Reset the stack of entry nodes.
        self._entry_nodes = []

        self.generic_visit(node)

        # If we have nodes left, something is screwed up.
        assert len(self._entry_nodes) == 0


class _ApplyLocalOptimizations(tn.ScheduleNodeVisitor):
    """
    Applies local optimization in `LabeledSection`s in a "leaf first" approach.

    This work inline and replaces the `LabeledSection` with the results of the local
    optimization as configured in the `OptimizationConfig` of the `LabeledSection`.
    """

    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend = backend

    def __str__(self) -> str:
        return "_LabelSections"

    def visit_LabeledSection(self, node: _LabeledSection) -> None:
        # Go down into children first such that we can apply local optimization "leaf first".
        self.generic_visit(node)

        # TODO
        # The code below is basically an `StreePipeline`. I've duplicated that
        # pipeline because we need some clever engineering to not get into a
        # hell of dependency circles (where the local optimizations are pipeline pass
        # and in itself depend on the pipeline).

        config = node.optimizations
        assert config.stree.enabled

        # HACK
        # Below, we are calling `visit_ScheduleTreeRoot` with a `LabeledSection`. This works
        # because python uses duck-typing.
        # TODO
        # Clean up pipeline passes and the pipeline itself such that they can work
        # on any subtree (i.e. any `ScheduleTreeScope`).

        if self._backend.is_gpu_backend():
            if config.stree.inline_K_loops_size_one:
                gpu_inliner = InlineVertical2DWrite()
                gpu_inliner.visit_ScheduleTreeRoot(node)

            if config.stree.merger.enabled:
                gpu_merger = CartesianMerge(
                    self._backend,
                    overcompute=config.stree.merger.overcompute,
                    merge_order=config.stree.merger.order,
                )
                gpu_merger.visit_ScheduleTreeRoot(node)

            if config.stree.kernelize:
                if config.stree.merger.order not in ("IJK", "KJI"):
                    ndsl_log.warning(
                        "Can't locally kernelize maps. Unknown apply oder. Skipping this pass."
                    )
                else:
                    # Follow the merge-order for kernelization
                    gpu_kernelizer = KernelizeMaps(
                        self._backend,
                        apply_order=(
                            "JI" if config.stree.merger.order == "IJK" else "JK"
                        ),
                    )
                    gpu_kernelizer.visit_ScheduleTreeRoot(node)

            if config.stree.refine_transients:
                # TODO
                # 🐞 Transient refine can't be used because of bugs transients showing
                #    in code generation.
                # gpu_refiner = CartesianRefineTransients(self._backend)
                # gpu_refiner.visit_ScheduleTreeRoot(node)
                raise ValueError(
                    "Transient refinement is currently unavailable in the GPU pipeline."
                )
        else:
            if config.stree.inline_K_loops_size_one:
                cpu_inliner = InlineVertical2DWrite()
                cpu_inliner.visit_ScheduleTreeRoot(node)

            if config.stree.merger.enabled:
                cpu_merger = CartesianMerge(
                    self._backend,
                    overcompute=config.stree.merger.overcompute,
                    merge_order=config.stree.merger.order,
                )
                cpu_merger.visit_ScheduleTreeRoot(node)

            if config.stree.refine_transients:
                cpu_refiner = CartesianRefineTransients(self._backend)
                cpu_refiner.visit_ScheduleTreeRoot(node)

        # Replace this `LabeledSection` with just the (now transformed) children.
        for child in node.children:
            # be sure to re-parent the children of this node to the new parent
            child.parent = node.parent
        node_index = list_index(node.parent.children, node)
        node.parent.children[node_index : node_index + 1] = node.children


class LocalOptimizations(tn.ScheduleNodeVisitor):
    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend = backend

    def __str__(self) -> str:
        return "LocalOptimizations"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        # First, parse enter/exit labels into `LabeledSection`s...
        _LabelSections().visit(node)

        # .. then, apply local optimizations on children of `LabeledSection`s.
        _ApplyLocalOptimizations(self._backend).visit(node)

        # debug only
        assert node
