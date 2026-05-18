import itertools

from dace.sdfg.analysis.schedule_tree import treenodes as tn


class ReplaceAxisSymbolInTasklet(tn.ScheduleNodeVisitor):
    def visit_TaskletNode(
        self,
        node: tn.TaskletNode,
        axis_replacements: dict[str, str] | None = None,
    ) -> None:
        if not axis_replacements:
            # Noop if there are no replacements to do.
            return

        # Dev NOTE: We directly replace the memlet.subset because the `memlet.replace`
        #           function sometimes doesn't work
        for memlet in itertools.chain(
            node.in_memlets.values(), node.out_memlets.values()
        ):
            memlet.replace(axis_replacements)

    def __str__(self) -> str:
        return "ReplaceAxisSymbolInTasklet"
