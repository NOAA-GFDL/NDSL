from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.common import (
    AxisIterator,
    get_next_node,
    get_previous_node,
    is_axis_map,
    is_last_node,
    is_off_grid_conditional,
    list_index,
    swap_node_in_tree,
)


class SimplifyConditional(tn.ScheduleNodeVisitor):
    """Turn Else and ElseIf into Ifs.

    Can restore original nodes using `restore`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.else_turned_if: list[tn.IfScope] = []

    def __str__(self) -> str:
        return "SimplifyConditional"

    def visit_ElifScope(self, node: tn.ElifScope) -> None:
        for child in node.children:
            self.visit(child)

        # Dev Note: parsing turns
        # ```
        # If y
        # ElseIf x
        # Else
        # ```
        # into
        # ```
        # If y
        # Else
        #   If x
        #   Else
        # ```
        ndsl_log.debug("ElifScope in SimplifyConditional unimplemented.")

    def visit_ElseScope(self, node: tn.ElseScope) -> None:
        assert node.parent

        # Recurse first
        for child in node.children:
            self.visit(child)

        potential_if = get_previous_node(node.parent.children, node)
        if isinstance(potential_if, tn.IfScope):
            code = potential_if.condition.as_string
            if_scope = tn.IfScope(
                condition=CodeBlock(f"not ({code})"),
                children=node.children,
                parent=node.parent,
            )
            self.else_turned_if.append(if_scope)
            swap_node_in_tree(node, if_scope)


class RevertSimplifyConditional(tn.ScheduleNodeVisitor):
    """Reverting the SimplifyConditional based on bookeeping done during SimplyConditional transform"""

    def __init__(self, original_simplify: SimplifyConditional) -> None:
        self._simplify_conditional = original_simplify

    def __str__(self) -> str:
        return "RevertSimplifyConditional"

    def visit_ScheduleTreeRoot(self, _node: tn.ScheduleTreeRoot) -> None:
        """Restore original Else.

        WARNING: no check if those still exists in the tree, if merging of conditionals
        or any other operation happened, this will create bad stree.
        """
        for if_scope in self._simplify_conditional.else_turned_if:
            assert if_scope.parent
            potential_if = get_previous_node(if_scope.parent.children, if_scope)
            if not isinstance(potential_if, tn.IfScope):
                continue
            else_scope = tn.ElseScope(
                children=if_scope.children, parent=if_scope.parent
            )
            swap_node_in_tree(if_scope, else_scope)


class InlineOffGridConditionals(tn.ScheduleNodeVisitor):
    """
    Push off-grid conditional inside their cartesian block, duplicating the
    conditional if needed.

    Turning:
    ```
    if a_flag == 0:
        map i, j, k:
            ...
        map i, j, k:
            ...
    ```
    into
    ```
    map i, j, k:
        if a_flag == 0:
            ...
    map i, j, k:
        if a_flag == 0:
            ...
    ```
    """

    _axis: AxisIterator

    def __init__(self, axis: AxisIterator) -> None:
        super().__init__()
        self._axis = axis

    def __str__(self) -> str:
        return f"InlineOffGridConditionals_{self._axis}"

    def visit_IfScope(self, node: tn.IfScope) -> None:
        assert node.parent is not None  # just to keep pyright happy
        if not is_off_grid_conditional(node):
            return

        for child in node.children:
            self.visit(child)

        # For now, skip in case there's an `elif` or `else` following.
        if not is_last_node(node.parent.children, node):
            next_node = get_next_node(node.parent.children, node)
            if isinstance(next_node, (tn.ElifScope, tn.ElseScope)):
                ndsl_log.debug(
                    "Can't handle conditionals with `elif` and `else` blocks yet :("
                )
                return

        if not all(
            isinstance(child, tn.MapScope) and is_axis_map(child, self._axis)
            for child in node.children
        ):
            return

        # If all children are maps over the correct axis, move the if inside.
        new_nodes: list[tn.MapScope] = []

        for child in node.children:
            assert isinstance(
                child, tn.MapScope
            )  # otherwise the condition above is wrong

            if_scope = tn.IfScope(
                condition=node.condition, children=child.children, parent=child
            )

            for map_child in child.children:
                map_child.parent = if_scope  # re-parent to new if_scope

            child.children = [if_scope]
            child.parent = node.parent  # re-parent to parent of old if_scope
            new_nodes.append(child)

        insert_at = list_index(node.parent.children, node)
        node.parent.children[insert_at:insert_at] = new_nodes
        node.parent.children.remove(node)


class ExtractOffGridConditionals(tn.ScheduleNodeTransformer):
    """
    Push off-grid conditional outside of their cartesian block.

    This is the inverse transform of InlineOffGridConditionals.
    """

    def __str__(self) -> str:
        return "ExtractOffGridConditionals"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        ndsl_log.debug("ExtractOffgridConditionals is not implemented yet.")


class MergeConditionals(tn.ScheduleNodeVisitor):
    """
    Merge consecutive and equal conditionals.

    Turning:
    ```
        if a_flag == 0:
            map i, j, k:
        if a_flag == 0:
            map i, j, k:
    ```
    into
    ```
        if a_flag == 0:
            map i, j, k:
            map i, j, k:
    ```

    or merge nested conditionals
    ```
    if a_flag == 0:
        if b_flag == 0:
            map i, j, k
    ```
    into
    ```
    if a_flag = 0 and b_flag == 0:
        map i,j,k
    ```

    Outside of user code, combination of ExtractOffGridConditionals,
    InlineOffGridConditionals and CartesianMapMerge can lead to this
    pattern.
    """

    def __str__(self) -> str:
        return "MergeConditionals"

    def visit_ScheduleTreeRoot(self, node: tn.ScheduleTreeRoot) -> None:
        ndsl_log.debug("MergeConditionals is not implemented yet.")
