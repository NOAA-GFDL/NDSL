from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import Backend, OptimizationConfig, ndsl_log
from ndsl.dsl.dace.stree.optimizations.cartesian_merge import CartesianMerge
from ndsl.dsl.dace.stree.optimizations.kernelize_maps import KernelizeMaps
from ndsl.dsl.dace.stree.optimizations.remove_loops import InlineVertical2DWrite


class ScheduleTreeScopeTransformer(tn.ScheduleNodeTransformer):
    def __init__(self) -> None:
        super().__init__()

    def _breadth_first_callback(self, node: tn.ScheduleTreeScope) -> None:
        pass

    def _depth_first_callback(self, node: tn.ScheduleTreeScope) -> None:
        pass

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_GBlock(self, node: tn.GBlock) -> tn.GBlock:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_LoopScope(self, node: tn.LoopScope) -> tn.LoopScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_ForScope(self, node: tn.ForScope) -> tn.ForScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_WhileScope(self, node: tn.WhileScope) -> tn.WhileScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_DoWhileScope(self, node: tn.DoWhileScope) -> tn.DoWhileScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_IfScope(self, node: tn.IfScope) -> tn.IfScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_StateIfScope(self, node: tn.StateIfScope) -> tn.StateIfScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_ElifScope(self, node: tn.ElifScope) -> tn.ElifScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_ElseScope(self, node: tn.ElseScope) -> tn.ElseScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_MapScope(self, node: tn.MapScope) -> tn.MapScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def visit_ConsumeScope(self, node: tn.ConsumeScope) -> tn.ConsumeScope:
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node


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


class _LabelSections(ScheduleTreeScopeTransformer):
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

    labeled_section "my_stencil":
      map i in [...]
        map j in [...]
          map k in [...]
            # contents of "my_stencil

    # program continues
    ```
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "_LabelSections"

    def _depth_first_callback(self, scope: tn.ScheduleTreeScope) -> None:
        """
        This is the function that actually does all the work by going over the children of a given schedule tree
        scope and re-grouping them into labeled sections based on `NDSLRuntime_Label` entry/exit nodes.
        """
        # The stack of entry nodes. They pop when the matching exit node is reached. Using a stack adds
        # support for nested labeled sections.
        entry_nodes_stack: list[tn.LibraryCall] = []

        # The stack of children. Every new entry node pushes its children into a new stack entry. This allows
        # one pass to gather nested children.
        children_stack: list[list[tn.ScheduleTreeNode]] = []

        # Top-level stack is for the current scope.
        children_stack.append([])

        for child in scope.children:
            # Unless we are dealing with `tn.LibraryCall` nodes, we push all nodes to the stack of new children.
            if not isinstance(child, tn.LibraryCall):
                children_stack[-1].append(child)
                continue

            if not child.node.name == "NDSLRuntime_Label":
                # Leave other library call nodes alone.
                children_stack[-1].append(child)
                continue

            if child.node.unique_name.startswith("Enter__"):
                # Keep taps on where we start and open a new list of children.
                entry_nodes_stack.append(child)
                children_stack.append([])
                continue

            # Expect to find an exit node now (matching the entry node that current on top of the stack).
            if not child.node.unique_name.startswith("Exit__"):
                raise RuntimeError(
                    f"Unexpected `NDSLRuntime_Label` '{child.node.unique_name}'."
                )

            # For exit nodes, find the matching entry node and the new children.
            section_start = entry_nodes_stack.pop()
            new_children = children_stack.pop()

            # sanity checks
            # - ensure we have the right section (if not, something is screwed up)
            name = section_start.node.unique_name.removeprefix("Enter__")
            assert name == child.node.unique_name.removeprefix("Exit__")
            # - ensure we have the same parent (if not something is screwed up)
            parent = section_start.parent
            assert parent == child.parent
            # - ensure the stack of children is not empty (it will at least contain the top-level scope)
            assert len(children_stack) > 0

            # Put all the new children in a `LabeledSection` and push that into the
            # new children of the above stack of children.
            new_node = _LabeledSection(
                children=new_children,
                parent=parent,
                label=name,
                optimizations=section_start.node._local_optimizations,
            )
            # re-parent new children to new node
            for c in new_node.children:
                c.parent = new_node
            # push new node into enclosing stack of children
            children_stack[-1].append(new_node)

            # and - of course - the final book keeping
            self._labeled_sections += 1

        # set the new children on the current scope
        scope.children = children_stack.pop()

        # some sanity checks
        assert len(children_stack) == 0  # expect empty stack
        for child in scope.children:
            assert child.parent == scope  # expect correct parent

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> tn.ScheduleTreeRoot:
        self._labeled_sections = 0

        # recurse down first to label sections "leaf first"
        self.generic_visit(node)
        self._depth_first_callback(node)

        ndsl_log.debug(f"{self}: labeled {self._labeled_sections} sections.")
        return node


class _ApplyLocalOptimizations(ScheduleTreeScopeTransformer):
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

    def visit__LabeledSection(self, node: _LabeledSection) -> _LabeledSection:
        # Recurse into labeled sections to support nested labeled sections.
        self._breadth_first_callback(node)
        self.generic_visit(node)
        self._depth_first_callback(node)

        return node

    def _depth_first_callback(self, scope: tn.ScheduleTreeScope) -> None:
        new_children: list[tn.ScheduleTreeNode] = []

        for child in scope.children:
            # Any child that isn't a _LabeledSection gets directly added to the list of new children.
            if not isinstance(child, _LabeledSection):
                new_children.append(child)
                continue

            # For labeled sections, apply the local optimizations to the sections' children, then
            # append the possibly transformed children to the list of new children (without the
            # labeled section).

            # TODO
            # The code below is basically an `StreePipeline`. I've duplicated that
            # pipeline because we need some clever engineering to not get into a
            # hell of dependency circles (where the local optimizations are pipeline pass
            # and in itself depend on the pipeline).

            config = child.optimizations
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
                    gpu_inliner.visit_ScheduleTreeRoot(child)

                if config.stree.merger.enabled:
                    gpu_merger = CartesianMerge(
                        self._backend,
                        overcompute=config.stree.merger.overcompute,
                        merge_order=config.stree.merger.order,
                    )
                    gpu_merger.visit_ScheduleTreeRoot(child)

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
                        gpu_kernelizer.visit_ScheduleTreeRoot(child)

                if config.stree.refine_transients:
                    # We can't know if transients are local to the scope that we are working in.
                    # In they are not, transient refinement can generate wrong results and refine
                    # too eagerly. Global transient refinement will also work in this section.
                    ndsl_log.warning(
                        "[Local-Opt]: Transient refinement can't e applied on a local scale "
                        "because it needs the global information on where/how transient data "
                        "is used. Please enable transient refinement on your global optimization "
                        "config and disable it here. No transients will be refined on the local "
                        "scale even if this option is turned on."
                    )
            else:
                if config.stree.inline_K_loops_size_one:
                    cpu_inliner = InlineVertical2DWrite()
                    cpu_inliner.visit_ScheduleTreeRoot(child)

                if config.stree.merger.enabled:
                    cpu_merger = CartesianMerge(
                        self._backend,
                        overcompute=config.stree.merger.overcompute,
                        merge_order=config.stree.merger.order,
                    )
                    cpu_merger.run(child)

                if config.stree.refine_transients:
                    # We can't know if transients are local to the scope that we are working in.
                    # In they are not, transient refinement can generate wrong results and refine
                    # too eagerly. Global transient refinement will also work in this section.
                    ndsl_log.warning(
                        "[Local-Opt]: Transient refinement can't e applied on a local scale "
                        "because it needs the global information on where/how transient data "
                        "is used. Please enable transient refinement on your global optimization "
                        "config and disable it here. No transients will be refined on the local "
                        "scale even if this option is turned on."
                    )

            # Replace this `LabeledSection` with just the (now transformed) children.
            for c in child.children:
                # be sure to re-parent the children of this node to the new parent
                c.parent = child.parent
                new_children.append(c)

        scope.children = new_children

        # sanity checks
        for child in scope.children:
            assert child.parent == scope  # expect correct parent
            assert not isinstance(
                child, _LabeledSection
            )  # no labeled sections should be left at this point


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
