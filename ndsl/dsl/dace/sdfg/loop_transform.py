from dace import SDFG
from dace import ScheduleType


def _make_sequential(sdfg: SDFG):
    import dace

    # Disable OpenMP sections
    for sd in sdfg.all_sdfgs_recursive():
        sd.openmp_sections = False
    # Disable OpenMP maps
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, dace.nodes.EntryNode):
            sched = getattr(n, "schedule", False)
            if sched in (
                ScheduleType.CPU_Multicore,
                ScheduleType.CPU_Persistent,
                ScheduleType.Default,
            ):
                n.schedule = dace.ScheduleType.Sequential
