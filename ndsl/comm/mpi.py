try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    MPI = None
from typing import Dict, List, Optional, TypeVar, cast

from ndsl.comm.comm_abc import Comm, ReductionOperator, Request
from ndsl.logging import ndsl_log


T = TypeVar("T")


class MPIComm(Comm):
    _op_mapping: Dict[ReductionOperator, mpi4py.MPI.Op] = {
        ReductionOperator.OP_NULL: mpi4py.MPI.OP_NULL,
        ReductionOperator.MAX: mpi4py.MPI.MAX,
        ReductionOperator.MIN: mpi4py.MPI.MIN,
        ReductionOperator.SUM: mpi4py.MPI.SUM,
        ReductionOperator.PROD: mpi4py.MPI.PROD,
        ReductionOperator.LAND: mpi4py.MPI.LAND,
        ReductionOperator.BAND: mpi4py.MPI.BAND,
        ReductionOperator.LOR: mpi4py.MPI.LOR,
        ReductionOperator.BOR: mpi4py.MPI.BOR,
        ReductionOperator.LXOR: mpi4py.MPI.LXOR,
        ReductionOperator.BXOR: mpi4py.MPI.BXOR,
        ReductionOperator.MAXLOC: mpi4py.MPI.MAXLOC,
        ReductionOperator.MINLOC: mpi4py.MPI.MINLOC,
        ReductionOperator.REPLACE: mpi4py.MPI.REPLACE,
        ReductionOperator.NO_OP: mpi4py.MPI.NO_OP,
    }

    def __init__(self):
        if MPI is None:
            raise RuntimeError("MPI not available")
        self._comm: Comm = cast(Comm, MPI.COMM_WORLD)

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def bcast(self, value: Optional[T], root=0) -> T:
        ndsl_log.debug("bcast from root %s on rank %s", root, self._comm.Get_rank())
        return self._comm.bcast(value, root=root)

    def barrier(self):
        ndsl_log.debug("barrier on rank %s", self._comm.Get_rank())
        self._comm.barrier()

    def Barrier(self):
        pass

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs):
        ndsl_log.debug("Scatter on rank %s with root %s", self._comm.Get_rank(), root)
        self._comm.Scatter(sendbuf, recvbuf, root=root, **kwargs)

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs):
        ndsl_log.debug("Gather on rank %s with root %s", self._comm.Get_rank(), root)
        self._comm.Gather(sendbuf, recvbuf, root=root, **kwargs)

    def allgather(self, sendobj: T) -> List[T]:
        ndsl_log.debug("allgather on rank %s", self._comm.Get_rank())
        return self._comm.allgather(sendobj)

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs):
        ndsl_log.debug("Send on rank %s with dest %s", self._comm.Get_rank(), dest)
        self._comm.Send(sendbuf, dest, tag=tag, **kwargs)

    def sendrecv(self, sendbuf, dest, **kwargs):
        ndsl_log.debug("sendrecv on rank %s with dest %s", self._comm.Get_rank(), dest)
        return self._comm.sendrecv(sendbuf, dest, **kwargs)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs) -> Request:
        ndsl_log.debug("Isend on rank %s with dest %s", self._comm.Get_rank(), dest)
        return self._comm.Isend(sendbuf, dest, tag=tag, **kwargs)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs):
        ndsl_log.debug("Recv on rank %s with source %s", self._comm.Get_rank(), source)
        self._comm.Recv(recvbuf, source, tag=tag, **kwargs)

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs) -> Request:
        ndsl_log.debug("Irecv on rank %s with source %s", self._comm.Get_rank(), source)
        return self._comm.Irecv(recvbuf, source, tag=tag, **kwargs)

    def Split(self, color, key) -> "Comm":
        ndsl_log.debug(
            "Split on rank %s with color %s, key %s", self._comm.Get_rank(), color, key
        )
        return self._comm.Split(color, key)

    def allreduce(self, sendobj: T, op: Optional[ReductionOperator] = None) -> T:
        ndsl_log.debug(
            "allreduce on rank %s with operator %s", self._comm.Get_rank(), op
        )
        return self._comm.allreduce(sendobj, self._op_mapping[op])

    def Allreduce(self, sendobj_or_inplace: T, recvobj: T, op: ReductionOperator) -> T:
        ndsl_log.debug(
            "Allreduce on rank %s with operator %s", self._comm.Get_rank(), op
        )
        return self._comm.Allreduce(sendobj_or_inplace, recvobj, self._op_mapping[op])

    def Allreduce_inplace(self, recvobj: T, op: ReductionOperator) -> T:
        ndsl_log.debug(
            "Allreduce (in place) on rank %s with operator %s",
            self._comm.Get_rank(),
            op,
        )
        return self._comm.Allreduce(mpi4py.MPI.IN_PLACE, recvobj, self._op_mapping[op])
