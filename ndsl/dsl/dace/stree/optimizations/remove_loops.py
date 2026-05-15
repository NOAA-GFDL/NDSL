import ast

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl import ndsl_log
from ndsl.dsl.dace.stree.optimizations.common import AxisIterator, reparent_scope_node
from ndsl.dsl.dace.stree.optimizations.replace_symbol_in_tasklet import (
    ReplaceAxisSymbolInTasklet,
)


class InlineVertical2DWrite(tn.ScheduleNodeTransformer):
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
        self._for_scope_removed = 0

    def __str__(self) -> str:
        return "InlineVertical2DWrite"

    def visit_ForScope(self, the_for: tn.ForScope) -> tn.ForScope | tn.ScheduleTreeNode:
        if (
            AxisIterator._K.is_equal(the_for.loop.loop_variable)
            and the_for.loop.executions == 1
            and the_for.parent
        ):
            # Retrieve init value by executing the code and replace usage of it
            # If the code cannot be executed (no-literal variable part of the op, etc.)
            # we will _not_ inline
            try:
                exec(ast.unparse(the_for.loop.init_statement.code[0]))
            except Exception as _:
                return the_for
            init_value = locals()[the_for.loop.loop_variable]
            ReplaceAxisSymbolInTasklet().visit(
                the_for, axis_replacements={the_for.loop.loop_variable: str(init_value)}
            )

            # Prepend children of the ForScope to parent
            # the_for.parent.children = [*the_for.children, *the_for.parent.children]
            reparent_scope_node(the_for, the_for.parent)

            # Remove ForScope
            the_for.parent.children.remove(the_for)
            self._for_scope_removed += 1
            assert len(the_for.children) > 0
            return the_for.parent.children[0]

        return the_for

    def visit_ScheduleTreeRoot(
        self, the_root: tn.ScheduleTreeRoot
    ) -> tn.ScheduleTreeRoot:

        for child in the_root.children:
            self.visit(child)

        ndsl_log.debug(f"🚀 {self}: {self._for_scope_removed} inlined")

        return the_root
