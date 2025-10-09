from __future__ import annotations

import copy
from typing import Any, TypeVar

from ndsl.comm.comm_abc import Comm, ReductionOperator
from ndsl.logging import ndsl_log
from ndsl.utils import ensure_contiguous, safe_assign_array


T = TypeVar("T")


class ConcurrencyError(Exception):
    """Exception to denote that a rank cannot proceed because it is waiting on a
    call from another rank."""

    pass


class AsyncResult:
    def __init__(self, result) -> None:  # type: ignore[no-untyped-def]
        self._result = result

    def wait(self):  # type: ignore[no-untyped-def]
        return self._result()


class LocalComm(Comm[T]):
    def __init__(self, rank: int, total_ranks: int, buffer_dict: dict) -> None:
        self.rank = rank
        self.total_ranks = total_ranks
        self._buffer = buffer_dict
        self._i_buffer: dict = {}

    @property
    def _split_comms(self) -> dict:
        self._buffer["split_comms"] = self._buffer.get("split_comms", {})
        return self._buffer["split_comms"]

    @property
    def _split_buffers(self) -> dict:
        self._buffer["split_buffers"] = self._buffer.get("split_buffers", {})
        return self._buffer["split_buffers"]

    def __repr__(self) -> str:
        return f"LocalComm(rank={self.rank}, total_ranks={self.total_ranks})"

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.total_ranks

    def _get_buffer(self, buffer_type: str, in_value: T) -> T:
        i_buffer = self._i_buffer.get(buffer_type, 0)
        self._i_buffer[buffer_type] = i_buffer + 1
        if buffer_type not in self._buffer:
            self._buffer[buffer_type] = []
        if self.rank == 0:
            self._buffer[buffer_type].append(in_value)
        return self._buffer[buffer_type][i_buffer]

    def _get_send_recv(self, from_rank, tag: int):  # type: ignore[no-untyped-def]
        key = (from_rank, self.rank, tag)
        if "send_recv" not in self._buffer:
            raise ConcurrencyError(
                "buffer not initialized for send_recv, likely recv called before send"
            )
        elif key not in self._buffer["send_recv"]:
            raise ConcurrencyError(
                f"rank-specific buffer not initialized for send_recv, likely "
                f"recv called before send from rank {from_rank} to rank {self.rank}"
            )
        return_value = self._buffer["send_recv"][key].pop(0)
        return return_value

    def _put_send_recv(self, value, to_rank, tag: int) -> None:  # type: ignore[no-untyped-def]
        key = (self.rank, to_rank, tag)
        self._buffer["send_recv"] = self._buffer.get("send_recv", {})
        self._buffer["send_recv"][key] = self._buffer["send_recv"].get(key, [])
        self._buffer["send_recv"][key].append(copy.deepcopy(value))

    @property
    def _bcast_buffer(self) -> list:
        if "bcast" not in self._buffer:
            self._buffer["bcast"] = []
        return self._buffer["bcast"]

    @property
    def _scatter_buffer(self) -> list:
        if "scatter" not in self._buffer:
            self._buffer["scatter"] = []
        return self._buffer["scatter"]

    @property
    def _gather_buffer(self) -> list:
        if "gather" not in self._buffer:
            self._buffer["gather"] = [None for i in range(self.total_ranks)]
        return self._buffer["gather"]

    def bcast(self, value: T | None, root: int = 0) -> T | None:
        if root != 0:
            raise NotImplementedError(
                "LocalComm assumes ranks are called in order, so root must be "
                "the bcast source"
            )
        assert value is not None
        value = self._get_buffer("bcast", value)
        ndsl_log.debug(f"bcast {value} to rank {self.rank}")
        return value

    def Barrier(self) -> None:
        return

    def barrier(self) -> None:
        return

    def Scatter(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        ensure_contiguous(sendbuf)
        ensure_contiguous(recvbuf)
        if root != 0:
            raise NotImplementedError(
                "LocalComm assumes ranks are called in order, so root must be "
                "the scatter source"
            )
        if sendbuf is not None:
            sendbuf = self._get_buffer("scatter", copy.deepcopy(sendbuf))
        else:
            sendbuf = self._get_buffer("scatter", None)  # type: ignore[arg-type]
        safe_assign_array(recvbuf, sendbuf[self.rank])

    def Gather(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        ensure_contiguous(sendbuf)
        ensure_contiguous(recvbuf)
        gather_buffer = self._gather_buffer
        gather_buffer[self.rank] = copy.deepcopy(sendbuf)
        if self.rank == root:
            # ndarrays are finnicky, have to check for None like this:
            if any(item is None for item in gather_buffer):
                uncalled_ranks = [
                    i for i, val in enumerate(gather_buffer) if val is None
                ]
                raise ConcurrencyError(
                    f"gather called on root rank before ranks {uncalled_ranks}"
                )
            for i, sendbuf in enumerate(gather_buffer):
                safe_assign_array(recvbuf[i, :], sendbuf)

    def allgather(self, sendobj: T) -> list[T]:
        raise NotImplementedError(
            "cannot implement allgather on local comm due to its inherent parallelism"
        )

    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        ensure_contiguous(sendbuf)
        self._put_send_recv(sendbuf, dest, tag)

    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        result = self.Send(sendbuf, dest, tag)

        def send():  # type: ignore[no-untyped-def]
            return result

        return AsyncResult(send)

    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        ensure_contiguous(recvbuf)
        safe_assign_array(recvbuf, self._get_send_recv(source, tag))

    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict):  # type: ignore[no-untyped-def]
        def receive():  # type: ignore[no-untyped-def]
            return self.Recv(recvbuf, source, tag)

        return AsyncResult(receive)

    def sendrecv(self, sendbuf, dest, **kwargs: dict):  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "sendrecv fundamentally cannot be written for LocalComm, "
            "as it requires synchronicity"
        )

    def Split(self, color, key) -> LocalComm:  # type: ignore[no-untyped-def]
        # key argument is ignored, assumes we're calling the ranks from least to
        # greatest when mocking Split
        self._split_comms[color] = self._split_comms.get(color, [])
        self._split_buffers[color] = self._split_buffers.get(color, {})
        rank = len(self._split_comms[color])
        total_ranks = rank + 1
        new_comm: LocalComm = LocalComm(
            rank=rank, total_ranks=total_ranks, buffer_dict=self._split_buffers[color]
        )
        for comm in self._split_comms[color]:
            comm.total_ranks = total_ranks
        self._split_comms[color].append(new_comm)
        return new_comm

    def allreduce(self, sendobj, op=None, recvobj=None) -> Any:  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "allreduce fundamentally cannot be written for LocalComm, "
            "as it requires synchronicity"
        )

    def Allreduce(self, sendobj, recvobj, op) -> Any:  # type: ignore[no-untyped-def]
        raise NotImplementedError(
            "Allreduce fundamentally cannot be written for LocalComm, "
            "as it requires synchronicity"
        )

    def Allreduce_inplace(self, obj: Any, op: ReductionOperator) -> Any:
        raise NotImplementedError(
            "Allreduce_inplace fundamentally cannot be written for LocalComm, "
            "as it requires synchronicity"
        )
