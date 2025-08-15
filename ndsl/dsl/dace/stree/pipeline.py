from abc import abstractmethod
from typing import Protocol
import dace.sdfg.analysis.schedule_tree.treenodes as dst
from ndsl.dsl.dace.stree.optimizations.merge import MapMerge, MergeStrategy
from ndsl.dsl.dace.stree.optimizations.axis_merge import (
    CartesianAxisMerge,
    AxisIterator,
)


class StreePipeline(Protocol):
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError("Missing implementation of __hash__")

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Missing implementation of __repr__")

    @abstractmethod
    def run(self, stree: dst.ScheduleTreeRoot, verbose=False) -> dst.ScheduleTreeRoot:
        raise NotImplementedError("Missing implementation of run")


class CPUPipeline(StreePipeline):
    def __init__(self) -> None:
        self.passes = [
            CartesianAxisMerge(AxisIterator._K),
        ]

    def __repr__(self) -> str:
        return str([type(p) for p in self.passes])

    def __hash__(self) -> int:
        return hash(repr(self))

    def run(self, stree: dst.ScheduleTreeRoot, verbose=False) -> dst.ScheduleTreeRoot:
        for p in self.passes:
            if verbose:
                print(f"[Stree OPT] {p}")
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

    def run(self, stree: dst.ScheduleTreeRoot, verbose=False) -> dst.ScheduleTreeRoot:
        for p in self.passes:
            if verbose:
                print(f"[Stree OPT] {p}")
            p.visit(stree)

        return stree
