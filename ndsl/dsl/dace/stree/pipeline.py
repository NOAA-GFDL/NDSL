from abc import abstractmethod
from typing import Protocol
import dace.sdfg.analysis.schedule_tree.treenodes as dace_stree
from ndsl.dsl.dace.stree.optimizations.merge import MapMerge, MergeStrategy


class StreePipeline(Protocol):
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError("Missing implementation of __hash__")

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Missing implementation of __repr__")

    @abstractmethod
    def run(
        self, stree: dace_stree.ScheduleTreeRoot, verbose=False
    ) -> dace_stree.ScheduleTreeRoot:
        raise NotImplementedError("Missing implementation of run")


class CPUPipeline(StreePipeline):
    def __init__(self) -> None:
        self.passes = [
            MapMerge(merge_strategy=MergeStrategy.Force_K),
        ]

    def __repr__(self) -> str:
        return str([type(p) for p in self.passes])

    def __hash__(self) -> int:
        return hash(repr(self))

    def run(
        self, stree: dace_stree.ScheduleTreeRoot, verbose=False
    ) -> dace_stree.ScheduleTreeRoot:
        for p in self.passes:
            if verbose:
                print(f"[Stree OPT] {type(p)}")
            p.visit(stree)

        return stree


class GPUPipeline(StreePipeline):
    def __init__(self) -> None:
        self.passes = [
            MapMerge(merge_strategy=MergeStrategy.Trivial),
        ]

    def __repr__(self) -> str:
        return str([type(p) for p in self.passes])

    def __hash__(self) -> int:
        return hash(repr(self))

    def run(
        self, stree: dace_stree.ScheduleTreeRoot, verbose=False
    ) -> dace_stree.ScheduleTreeRoot:
        for p in self.passes:
            if verbose:
                print(f"[Stree OPT] {type(p)}")
            p.visit(stree)

        return stree
