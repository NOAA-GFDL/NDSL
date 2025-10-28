from abc import ABC, abstractmethod

import dace.sdfg.analysis.schedule_tree.treenodes as stree

from ndsl.dsl.dace.stree.optimizations import AxisIterator, CartesianAxisMerge


class StreePipeline(ABC):
    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError("Missing implementation of __hash__")

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("Missing implementation of __repr__")

    @abstractmethod
    def run(
        self,
        stree: stree.ScheduleTreeRoot,
        verbose: bool = False,
    ) -> stree.ScheduleTreeRoot:
        raise NotImplementedError("Missing implementation of run")


class CPUPipeline(StreePipeline):
    def __init__(
        self, passes: list[stree.ScheduleNodeTransformer] | None = None
    ) -> None:
        self.passes = (
            passes if passes is not None else [CartesianAxisMerge(AxisIterator._K)]
        )

    def __repr__(self) -> str:
        return str([type(p) for p in self.passes])

    def __hash__(self) -> int:
        return hash(repr(self))

    def run(
        self,
        stree: stree.ScheduleTreeRoot,
        verbose: bool = False,
    ) -> stree.ScheduleTreeRoot:
        for p in self.passes:
            if verbose:
                print(f"[Stree OPT] {p}")
            p.visit(stree)

        return stree


class GPUPipeline(StreePipeline):
    def __init__(
        self, passes: list[stree.ScheduleNodeTransformer] | None = None
    ) -> None:
        self.passes = passes if passes else []

    def __repr__(self) -> str:
        return str([type(p) for p in self.passes])

    def __hash__(self) -> int:
        return hash(repr(self))

    def run(
        self,
        stree: stree.ScheduleTreeRoot,
        verbose: bool = False,
    ) -> stree.ScheduleTreeRoot:
        for p in self.passes:
            if verbose:
                print(f"[Stree OPT] {p}")
            p.visit(stree)

        return stree
