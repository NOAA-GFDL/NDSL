from dace import SDFG, ScheduleType, nodes


def make_SDFG_CPU_sequential(sdfg: SDFG) -> None:
    """Utility to turn a CPU-based SDFG to pure serial by removing OpenMP"""
    # Disable OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False

    # Disable OpenMP maps
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.EntryNode):
            schedule = getattr(node, "schedule", False)
            if schedule in (
                ScheduleType.CPU_Multicore,
                ScheduleType.CPU_Persistent,
                ScheduleType.Default,
            ):
                node.schedule = ScheduleType.Sequential
