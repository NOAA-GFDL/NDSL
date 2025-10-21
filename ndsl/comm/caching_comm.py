from __future__ import annotations

import copy
import dataclasses
import pickle
from typing import Any, BinaryIO, TypeVar

import numpy as np

from ndsl.comm.comm_abc import Comm, ReductionOperator, Request


T = TypeVar("T")


class CachingRequestWriter(Request):
    def __init__(
        self, req: Request, buffer: np.ndarray, buffer_list: list[np.ndarray]
    ) -> None:
        self._req = req
        self._buffer = buffer
        self._buffer_list = buffer_list

    def wait(self) -> None:
        self._req.wait()
        self._buffer_list.append(copy.deepcopy(self._buffer))


class CachingRequestReader(Request):
    def __init__(self, recvbuf: Any, data: Any) -> None:
        self._recvbuf = recvbuf
        self._data = data

    def wait(self) -> None:
        self._recvbuf[:] = self._data


class NullRequest(Request):
    def wait(self) -> None:
        pass


@dataclasses.dataclass
class CachingCommData:
    """
    Data required to restore a CachingCommReader.

    Usually you will not want to initialize this class directly, but instead
    use the CachingCommReader.load method.
    """

    rank: int
    size: int
    bcast_objects: list[Any] = dataclasses.field(default_factory=list)
    received_buffers: list[np.ndarray] = dataclasses.field(default_factory=list)
    generic_obj_buffers: list[Any] = dataclasses.field(default_factory=list)
    split_data: list[CachingCommData] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self._i_bcast = 0
        self._i_buffers = 0
        self._i_split = 0
        self._i_generic_obj = 0

    def get_bcast(self) -> Any:
        return_value = self.bcast_objects[self._i_bcast]
        self._i_bcast += 1
        return return_value

    def get_buffer(self) -> np.ndarray:
        return_value = self.received_buffers[self._i_buffers]
        self._i_buffers += 1
        return return_value

    def get_generic_obj(self) -> Any:
        return_value = self.generic_obj_buffers[self._i_generic_obj]
        self._i_generic_obj += 1
        return return_value

    def get_split(self) -> CachingCommData:
        return_value = self.split_data[self._i_split]
        self._i_split += 1
        return return_value

    def dump(self, file: BinaryIO) -> None:
        pickle.dump(self, file)

    @classmethod
    def load(self, file: BinaryIO) -> CachingCommData:
        return pickle.load(file)


class CachingCommReader(Comm[T]):
    """
    mpi4py Comm-like object which replays stored communications.
    """

    def __init__(self, data: CachingCommData) -> None:
        """
        Initialize a CachingCommReader.

        Usually you will not want to initialize this class directly, but instead
        use the CachingCommReader.load method.

        Args:
            data: contains all data needed for mocked communication
        """
        self._data = data

    def Get_rank(self) -> int:
        return self._data.rank

    def Get_size(self) -> int:
        return self._data.size

    def bcast(self, value: T | None, root: int = 0) -> T | None:
        return self._data.get_bcast()

    def barrier(self) -> None:
        pass

    def Barrier(self) -> None:
        pass

    def Scatter(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        recvbuf[:] = self._data.get_buffer()

    def Gather(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        if recvbuf is not None:
            recvbuf[:] = self._data.get_buffer()

    def allgather(self, sendobj: T) -> list[T]:
        raise NotImplementedError("allgather not yet implemented for CachingCommReader")

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        pass

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return NullRequest()

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        recvbuf[:] = self._data.get_buffer()

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return CachingRequestReader(recvbuf, self._data.get_buffer())

    def sendrecv(self, sendbuf, dest, **kwargs: dict):  # type: ignore[no-untyped-def]
        raise NotImplementedError("CachingCommReader.sendrecv")

    def Split(self, color, key) -> CachingCommReader:  # type: ignore[no-untyped-def]
        new_data = self._data.get_split()
        return CachingCommReader(data=new_data)

    def allreduce(
        self, sendobj: T, op: ReductionOperator = ReductionOperator.NO_OP
    ) -> T:
        return self._data.get_generic_obj()

    def Allreduce(self, sendobj: T, recvobj: T, op: ReductionOperator) -> T:
        raise NotImplementedError("CachingCommReader.Allreduce")

    def Allreduce_inplace(self, obj: T, op: ReductionOperator) -> T:
        raise NotImplementedError("CachingCommReader.Allreduce_inplace")

    @classmethod
    def load(cls, file: BinaryIO) -> CachingCommReader:
        data = CachingCommData.load(file)
        return cls(data)


class CachingCommWriter(Comm[T]):
    """
    Wrapper around a mpi4py Comm object which can be serialized and then loaded
    as a CachingCommReader.
    """

    def __init__(self, comm: Comm[T]) -> None:
        """
        Args:
            comm: underlying mpi4py comm-like object
        """
        self._comm = comm
        self._data = CachingCommData(
            rank=comm.Get_rank(),
            size=comm.Get_size(),
        )

    def Get_rank(self) -> int:
        return self._comm.Get_rank()

    def Get_size(self) -> int:
        return self._comm.Get_size()

    def bcast(self, value: T | None, root: int = 0) -> T | None:
        result = self._comm.bcast(value=value, root=root)
        self._data.bcast_objects.append(copy.deepcopy(result))
        return result

    def barrier(self) -> None:
        return self._comm.barrier()

    def Barrier(self) -> None:
        pass

    def Scatter(self, sendbuf, recvbuf, root=0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Scatter(sendbuf=sendbuf, recvbuf=recvbuf, root=root, **kwargs)
        self._data.received_buffers.append(copy.deepcopy(recvbuf))

    def Gather(self, sendbuf, recvbuf, root=0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Gather(sendbuf=sendbuf, recvbuf=recvbuf, root=root, **kwargs)
        self._data.received_buffers.append(copy.deepcopy(recvbuf))

    def allgather(self, sendobj: T) -> list[T]:
        raise NotImplementedError("allgather not yet implemented for CachingCommReader")

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Send(sendbuf=sendbuf, dest=dest, tag=tag, **kwargs)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        return self._comm.Isend(sendbuf, dest, tag=tag, **kwargs)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        self._comm.Recv(recvbuf=recvbuf, source=source, tag=tag, **kwargs)
        self._data.received_buffers.append(copy.deepcopy(recvbuf))

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict) -> Request:  # type: ignore[no-untyped-def]
        req = self._comm.Irecv(recvbuf, source, tag=tag, **kwargs)
        return CachingRequestWriter(
            req=req, buffer=recvbuf, buffer_list=self._data.received_buffers
        )

    def sendrecv(self, sendbuf, dest, **kwargs: dict):  # type: ignore[no-untyped-def]
        raise NotImplementedError("CachingCommWriter.sendrecv")

    def Split(self, color, key) -> CachingCommWriter:  # type: ignore[no-untyped-def]
        new_comm = self._comm.Split(color=color, key=key)
        new_wrapper = CachingCommWriter(new_comm)
        self._data.split_data.append(new_wrapper._data)
        return new_wrapper

    def dump(self, file: BinaryIO) -> None:
        self._data.dump(file)

    def allreduce(
        self, sendobj: T, op: ReductionOperator = ReductionOperator.NO_OP
    ) -> T:
        result = self._comm.allreduce(sendobj, op)
        self._data.generic_obj_buffers.append(copy.deepcopy(result))
        return result

    def Allreduce(self, sendobj: T, recvobj: T, op: ReductionOperator) -> T:
        raise NotImplementedError("CachingCommWriter.Allreduce")

    def Allreduce_inplace(self, obj: T, op: ReductionOperator) -> T:
        raise NotImplementedError("CachingCommWriter.Allreduce_inplace")
