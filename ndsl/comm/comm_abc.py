from __future__ import annotations

import abc
import enum
from typing import Generic, TypeVar


T = TypeVar("T")


@enum.unique
class ReductionOperator(enum.Enum):
    OP_NULL = enum.auto()
    MAX = enum.auto()
    MIN = enum.auto()
    SUM = enum.auto()
    PROD = enum.auto()
    LAND = enum.auto()
    BAND = enum.auto()
    LOR = enum.auto()
    BOR = enum.auto()
    LXOR = enum.auto()
    BXOR = enum.auto()
    MAXLOC = enum.auto()
    MINLOC = enum.auto()
    REPLACE = enum.auto()
    NO_OP = enum.auto()


class Request(abc.ABC):
    @abc.abstractmethod
    def wait(self) -> None: ...


class Comm(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def Get_rank(self) -> int: ...

    @abc.abstractmethod
    def Get_size(self) -> int: ...

    @abc.abstractmethod
    def bcast(self, value: T | None, root: int = 0) -> T | None: ...

    @abc.abstractmethod
    def barrier(self) -> None: ...

    @abc.abstractmethod
    def Barrier(self) -> None: ...

    @abc.abstractmethod
    def Scatter(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict): ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def Gather(self, sendbuf, recvbuf, root: int = 0, **kwargs: dict): ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def allgather(self, sendobj: T) -> list[T]: ...

    @abc.abstractmethod
    def Send(self, sendbuf, dest, tag: int = 0, **kwargs: dict): ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def sendrecv(self, sendbuf, dest, **kwargs: dict): ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def Isend(self, sendbuf, dest, tag: int = 0, **kwargs: dict) -> Request: ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def Recv(self, recvbuf, source, tag: int = 0, **kwargs: dict): ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def Irecv(self, recvbuf, source, tag: int = 0, **kwargs: dict) -> Request: ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def Split(self, color, key) -> Comm: ...  # type: ignore[no-untyped-def]

    @abc.abstractmethod
    def allreduce(
        self, sendobj: T, op: ReductionOperator = ReductionOperator.NO_OP
    ) -> T: ...

    @abc.abstractmethod
    def Allreduce(self, sendobj: T, recvobj: T, op: ReductionOperator) -> T: ...

    @abc.abstractmethod
    def Allreduce_inplace(self, obj: T, op: ReductionOperator) -> T: ...
