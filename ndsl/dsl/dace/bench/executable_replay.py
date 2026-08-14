from ndsl import Backend, LocalComm, PerformanceCollector
from ndsl.dsl.dace.bench.executable_recorder import DaceExecutableRecorder


class DaceExecutableReplay:
    """☢️ Replays the DaceExecutableRecorder"""

    def __init__(self, bundle_dir: str) -> None:
        self._recorder = DaceExecutableRecorder()
        self._recorder.load(bundle_dir)

        assert self._recorder._opt_sdfg
        self._opt_csdfg = self._recorder._opt_sdfg.compile()

        assert self._recorder._ready
        assert self._recorder._args
        assert self._recorder._gt4py_sdfg

    def run(self) -> None:
        self._opt_csdfg(**self._recorder._args)

    def bench(self) -> None:

        perf_collector = PerformanceCollector(
            self._recorder._gt4py_sdfg.name, LocalComm(0, 1, {})
        )
        perf_collector.start_cuda_profiler()

        self.run()

        with perf_collector.total_timer.clock("all"):
            for _ in range(1000):
                with perf_collector.clock_timestep("ts"):
                    self.run()

        perf_collector.write_out_rank_0(
            Backend.cpu(), True, dt_atmos=-1.0, sim_status="done"
        )
