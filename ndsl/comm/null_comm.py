import copy
import warnings
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from ndsl.comm.comm_abc import Comm, ReductionOperator, Request


T = TypeVar("T")


class NullAsyncResult(Request):
    def __init__(self, recvbuf: Any = None) -> None:
        self._recvbuf = recvbuf

    def wait(self) -> None:
        if self._recvbuf is not None:
            self._recvbuf[:] = 0.0


class NullComm(Comm[T]):
    """
    A class with a subset of the mpi4py Comm API, but which
    'receives' a fill value (default zero) instead of using MPI.
    """

    default_fill_value: T = cast(T, 0)

    def __init__(self, rank: int, total_ranks: int, fill_value: T = default_fill_value):
        """
        Args:
            rank: rank to mock
            total_ranks: number of total MPI ranks to mock
            fill_value: fill halos with this value when performing
                halo updates.
        """
        warnings.warn(
            "NullComm is deprecated and will be removed with the next version of NDSL. "
            "Use MPIComm or LocalComm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rank = rank
        self.total_ranks = total_ranks
        self._fill_value = fill_value
        self._split_comms: Mapping[Any, list[NullComm]] = {}

    def __repr__(self) -> str:
        return f"NullComm(rank={self.rank}, total_ranks={self.total_ranks})"

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.total_ranks

    def bcast(self, value: T | None, root: int = 0) -> T | None:
        return value

    def barrier(self) -> None:
        return

    def Barrier(self) -> None:
        return

    def Scatter(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        if recvbuf is not None:
            recvbuf[:] = self._fill_value

    def Gather(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        if recvbuf is not None:
            recvbuf[:] = self._fill_value

    def allgather(self, sendobj: T) -> list[T]:
        return [copy.deepcopy(sendobj) for _ in range(self.total_ranks)]

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        pass

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return NullAsyncResult()

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        recvbuf[:] = self._fill_value

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return NullAsyncResult(recvbuf)

    def sendrecv(self, sendbuf, dest, **kwargs: dict):  # type: ignore[no-untyped-def]
        return sendbuf

    def Split(self, color, key) -> Comm:  # type: ignore[no-untyped-def]
        # key argument is ignored, assumes we're calling the ranks from least to
        # greatest when mocking Split
        self._split_comms[color] = self._split_comms.get(color, [])  # type: ignore[index]
        rank = len(self._split_comms[color])
        total_ranks = rank + 1
        new_comm = NullComm(
            rank=rank, total_ranks=total_ranks, fill_value=self._fill_value
        )
        for comm in self._split_comms[color]:
            # won't know how many ranks there are until everything is split
            comm.total_ranks = total_ranks
        self._split_comms[color].append(new_comm)
        return new_comm

    def allreduce(
        self, sendobj: T, op: ReductionOperator = ReductionOperator.NO_OP
    ) -> T:
        return self._fill_value

    def Allreduce(self, sendobj: T, recvobj: T, op: ReductionOperator) -> T:
        # TODO: what about reduction operator `op`?
        recvobj = sendobj
        return recvobj

    def Allreduce_inplace(self, obj: T, op: ReductionOperator) -> T:
        raise NotImplementedError("NullComm.Allreduce_inplace")
