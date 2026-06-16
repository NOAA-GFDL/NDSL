from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.optimizations.common import (
    AxisIterator,
    get_next_node,
    is_axis_map,
    last_node,
    list_index,
)


class InlineOffgridConditionals(tn.ScheduleNodeVisitor):
    """
    Push offgrid conditional inside their cartesian block, duplicating the
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
        return f"InlineOffgridConditionals_{self._axis}"

    def visit_IfScope(self, node: tn.IfScope) -> None:
        assert node.parent is not None  # just to keep pyright happy

        # For now, skip in case there's an `elif` or `else` following.
        if not last_node(node.parent.children, node):
            next_node = get_next_node(node.parent.children, node)
            if isinstance(next_node, (tn.ElifScope, tn.ElseScope)):
                ndsl_log.debug(
                    "Can't handle conditionals with `elif` and `else` blocks yet :("
                )
                return

        if not all(
            [
                isinstance(child, tn.MapScope) and is_axis_map(child, self._axis)
                for child in node.children
            ]
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


class ExtractOffgridConditionals(tn.ScheduleNodeTransformer):
    """
    Push offgrid conditional outside of their cartesian block.

    This is the inverse transform of InlineOffgridConditionals.
    """

    def __str__(self) -> str:
        return "ExtractOffgridConditionals"


class MergeConditionals(tn.ScheduleNodeTransformer):
    """
    Merge consecutive and equal conditionals.

    Turning:
    ```
        if a_flag == 0:
            map i, j, k:
                ...
        if a_flag == 0:
            map i, j, k:
                ...
    ```
    into
    ```
        if a_flag == 0:
            map i, j, k:
                ...
            map i, j, k:
                ...
    ```

    Outside of user code, combination of ExtractOffgridConditionals,
    InlineOffgridConditionals and CartesianMapMerge can lead to this
    pattern.
    """

    def __str__(self) -> str:
        return "MergeConditionals"
