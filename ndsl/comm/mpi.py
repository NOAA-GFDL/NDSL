"""Wrapper around mpi4py.

This module defines a light-weight wrapper around mpi4py. It is the only
place where we directly import from mpi4py. This allows to potentially
swap mpi4py in the future.
"""

from typing import TypeVar, cast

from mpi4py import MPI

from ndsl.comm.comm_abc import Comm, ReductionOperator, Request


T = TypeVar("T")


class MPIComm(Comm):
    _op_mapping: dict[ReductionOperator, MPI.Op] = {
        ReductionOperator.OP_NULL: MPI.OP_NULL,
        ReductionOperator.MAX: MPI.MAX,
        ReductionOperator.MIN: MPI.MIN,
        ReductionOperator.SUM: MPI.SUM,
        ReductionOperator.PROD: MPI.PROD,
        ReductionOperator.LAND: MPI.LAND,
        ReductionOperator.BAND: MPI.BAND,
        ReductionOperator.LOR: MPI.LOR,
        ReductionOperator.BOR: MPI.BOR,
        ReductionOperator.LXOR: MPI.LXOR,
        ReductionOperator.BXOR: MPI.BXOR,
        ReductionOperator.MAXLOC: MPI.MAXLOC,
        ReductionOperator.MINLOC: MPI.MINLOC,
        ReductionOperator.REPLACE: MPI.REPLACE,
        ReductionOperator.NO_OP: MPI.NO_OP,
    }

    def __init__(self) -> None:
        if MPI is None:
            raise RuntimeError("MPI not available")
        self._comm: Comm = cast(Comm, MPI.COMM_WORLD)

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def bcast(self, value: T | None, root: int = 0) -> T | None:
        return self._comm.bcast(value, root=root)

    def barrier(self) -> None:
        self._comm.barrier()

    def Barrier(self) -> None:
        pass

    def Scatter(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Scatter(sendbuf, recvbuf, root=root, **kwargs)

    def Gather(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Gather(sendbuf, recvbuf, root=root, **kwargs)

    def allgather(self, sendobj: T) -> list[T]:
        return self._comm.allgather(sendobj)

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Send(sendbuf, dest, tag=tag, **kwargs)

    def sendrecv(self, sendbuf, dest, **kwargs: dict):  # type: ignore[no-untyped-def]
        return self._comm.sendrecv(sendbuf, dest, **kwargs)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return self._comm.Isend(sendbuf, dest, tag=tag, **kwargs)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Recv(recvbuf, source, tag=tag, **kwargs)

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return self._comm.Irecv(recvbuf, source, tag=tag, **kwargs)

    def Split(self, color, key) -> Comm:  # type: ignore[no-untyped-def]
        return self._comm.Split(color, key)

    def allreduce(
        self, sendobj: T, op: ReductionOperator = ReductionOperator.NO_OP
    ) -> T:
        return self._comm.allreduce(sendobj, self._op_mapping[op])

    def Allreduce(self, sendobj: T, recvobj: T, op: ReductionOperator) -> T:
        return self._comm.Allreduce(sendobj, recvobj, self._op_mapping[op])

    def Allreduce_inplace(self, recvobj: T, op: ReductionOperator) -> T:
        return self._comm.Allreduce(MPI.IN_PLACE, recvobj, self._op_mapping[op])
