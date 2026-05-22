import ast
from typing import Any

import dace
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.optimizations.common import (
    AxisIterator,
    is_axis_for,
    list_index,
)
from ndsl.dsl.dace.stree.optimizations.replace_axis_symbol import ReplaceAxisSymbol


class InlineVertical2DWrite(tn.ScheduleNodeVisitor):
    """Inline K index value for 2D write vertical while removing for loop.

    Transforming:
    ```
    for __k = 0; __k < 1; __k = __k + 1:
        map __j, __i:
            field[__i, __j] = tasklet(field_in[__i, __j, __k])
    ```

    Into
    ```
    map __j, __i:
        field[__i, __j] = tasklet(field_in[__i, __j, 0])
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self._for_scopes_removed = 0

    def __str__(self) -> str:
        return "InlineVertical2DWrite"

    def visit_ForScope(self, the_for: tn.ForScope) -> None:
        if not is_axis_for(the_for, AxisIterator._K):
            return

        assert the_for.parent is not None  # just to keep pyright happy

        # Retrieve init/bound value by executing the code and replace usage of it
        # If the code cannot be executed (no-literal variable part of the op, etc.)
        # we will _not_ inline
        try:
            exec_locals: dict[str, Any] = {}
            exec_globals: dict[str, Any] = {}
            exec(
                ast.unparse(the_for.loop.init_statement.code[0]),
                exec_globals,
                exec_locals,
            )
            init_value = exec_locals[the_for.loop.loop_variable]
            bound_value = eval(
                ast.unparse(the_for.loop.loop_condition.code[0].value.comparators)
            )
        except Exception as _:
            return
        if abs(bound_value - init_value) != 1:
            return

        ReplaceAxisSymbol(
            {dace.symbol(the_for.loop.loop_variable): str(init_value)}
        ).visit(the_for)

        # Insert children of the ForScope to parent
        insert_at = list_index(the_for.parent.children, the_for)
        for child in the_for.children:
            child.parent = the_for.parent
        the_for.parent.children[insert_at:insert_at] = the_for.children

        # Remove ForScope
        the_for.parent.children.remove(the_for)
        self._for_scopes_removed += 1
        assert len(the_for.children) > 0

    def visit_ScheduleTreeRoot(self, the_root: tn.ScheduleTreeRoot) -> None:
        self._for_scopes_removed = 0

        for child in the_root.children:
            self.visit(child)

        ndsl_log.debug(f"🚀 {self}: {self._for_scopes_removed} inlined")
