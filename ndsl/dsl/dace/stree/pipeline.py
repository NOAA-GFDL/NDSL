from abc import abstractmethod
from typing import Protocol

import dace.sdfg.analysis.schedule_tree.treenodes as dst

from ndsl.dsl.dace.stree.optimizations import AxisIterator, CartesianAxisMerge


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
    def __init__(self, passes: list[dst.ScheduleNodeTransformer] | None = None) -> None:
        self.passes = (
            passes if passes is not None else [CartesianAxisMerge(AxisIterator._K)]
        )

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
    def __init__(self, passes: list[dst.ScheduleNodeTransformer] | None = None) -> None:
        self.passes = passes if passes else []

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
