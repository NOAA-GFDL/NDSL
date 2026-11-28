from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np
import xarray as xr

from ndsl import Quantity
from ndsl.comm.comm_abc import Comm as CommABC
from ndsl.comm.communicator import Communicator
from ndsl.comm.mpi import get_mpi_type

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

T = TypeVar("T", bound="DataclassInstance")


def _shape_list_gen(
    var_shape: tuple,
    pelist_size: int,
    locale: str,
    nhalo: int = 0,
    i_adj: int = 0,
    j_adj: int = 0,
    k_adj: int = 0,
) -> list:
    shape_list: list[Union[tuple[Any, int], tuple[Any, int, Any]]] = []
    for n in range(pelist_size):
        match locale:
            case "right" | "left":
                i = var_shape[0] - i_adj
                j = nhalo
            case "top" | "bottom":
                i = nhalo
                j = (
                    var_shape[1] - j_adj - nhalo
                    if n in (0, pelist_size - 1)
                    else var_shape[1] - j_adj
                )
        if len(var_shape) == 3:
            k = var_shape[2] - k_adj
            shape_list.append((i, j, k))
        else:
            shape_list.append((i, j))
    return shape_list


def _get_displs(counts: list) -> list:
    displs = [0]
    for n in range(1, len(counts)):
        displs.append(displs[n - 1] + counts[n - 1])
    return displs


class BoundaryConditionCommunicator:
    _main_comm: Communicator
    _sub_comm: CommABC
    _color: int
    _location: str

    def __init__(self, main_comm: Communicator, location: str, layout: tuple):
        self._main_comm = main_comm
        pelist = []
        self._location = location.lower()
        match self._location:
            case "top":
                for n in range(layout[1]):
                    pelist.append(layout[1] * (layout[1] - 1) + n)
                self._color = 1 if self._main_comm.rank in pelist else 0
            case "bottom":
                for n in range(layout[1]):
                    pelist.append(n)
                self._color = 1 if self._main_comm.rank in pelist else 0
            case "right":
                for n in range(layout[0]):
                    pelist.append(layout[0] * (n + 1) - 1)
                self._color = 1 if self._main_comm.rank in pelist else 0
            case "left":
                for n in range(layout[0]):
                    pelist.append(layout[0] * n)
                self._color = 1 if self._main_comm.rank in pelist else 0
            case _:
                raise ValueError(f"{location} is not an edge position")
        self._sub_comm = self._main_comm.comm.Split(
            color=self._color, key=self._main_comm.rank
        )

    @property
    def rank(self) -> int:
        return self._sub_comm.Get_rank()

    @property
    def size(self) -> int:
        return self._sub_comm.Get_size()

    @property
    def location(self) -> str:
        return self._location

    def sub_scatterv(
        self,
        var: Quantity,
        var_name: str,
        dataset: xr.Dataset,
    ) -> None:
        if self._color == 1:
            var_shape = var.shape
            n_halo = var.metadata.n_halo
            iadj = 1 if var.dims[0] == "i" else 0
            jadj = 1 if var.dims[1] == "j" else 0
            kadj = 1 if (len(var.dims) == 3 and var.dims[2] == "k") else 0
            shape_list = _shape_list_gen(
                var_shape=var_shape,
                pelist_size=self.size,
                locale=self.location,
                nhalo=n_halo,
                i_adj=iadj,
                j_adj=jadj,
                k_adj=kadj,
            )
            recv_buf = np.empty(shape=shape_list[self.rank], dtype=var.dtype).flatten()
            if self.rank == 0:
                da = np.ascontiguousarray(dataset[var_name].data).flatten()
                sendcounts = [np.prod(shape_list[n]) for n in range(self.size)]
                displs = _get_displs(sendcounts)
                temp = np.empty(shape=sum(sendcounts), dtype=da.dtype)
                datatype = get_mpi_type(da)
                m = 0
                for n in range(self.size):
                    temp[m : m + sendcounts[n]] = da[m : m + sendcounts[n]]
                    m += sendcounts[n]
            else:
                temp = None
                sendcounts = None
                displs = None
                datatype = None
            self._sub_comm.Scatterv(
                [temp, sendcounts, displs, datatype], recv_buf, root=0
            )

            match self.location:
                case "top":
                    js = 0
                    je = shape_list[self.rank][1]
                    if self.rank == 0:
                        js = n_halo
                        je = shape_list[self.rank][1] + n_halo
                    if len(var_shape) == 2:
                        var[:n_halo, js:je] = recv_buf[:].reshape(shape_list[self.rank])
                    if len(var_shape) == 3:
                        var[:n_halo, js:je, : var_shape[2] - kadj] = recv_buf[
                            :
                        ].reshape(shape_list[self.rank])
                case "bottom":
                    js = 0
                    je = shape_list[self.rank][1]
                    if self.rank == 0:
                        js = n_halo
                        je = shape_list[self.rank][1] + n_halo
                    if len(var_shape) == 2:
                        var[var_shape[0] - n_halo - 1 : var_shape[0] - 1, js:je] = (
                            recv_buf[:].reshape(shape_list[self.rank])
                        )
                    if len(var_shape) == 3:
                        var[
                            var_shape[0] - n_halo - 1 : var_shape[0] - 1,
                            js:je,
                            : var_shape[2] - kadj,
                        ] = recv_buf[:].reshape(shape_list[self.rank])
                case "right":
                    if len(var_shape) == 2:
                        var[
                            : var_shape[0] - iadj,
                            var_shape[1] - n_halo - 1 : var_shape[1] - 1,
                        ] = recv_buf[:].reshape(shape_list[self.rank])
                    if len(var_shape) == 3:
                        var[
                            : var_shape[0] - iadj,
                            var_shape[1] - n_halo - 1 : var_shape[1] - 1,
                            : var_shape[2] - kadj,
                        ] = recv_buf[:].reshape(shape_list[self.rank])
                case "left":
                    if len(var_shape) == 2:
                        var[
                            : var_shape[0] - iadj,
                            :n_halo,
                        ] = recv_buf[
                            :
                        ].reshape(shape_list[self.rank])
                    if len(var_shape) == 3:
                        var[
                            : var_shape[0] - iadj,
                            :n_halo,
                            : var_shape[2] - kadj,
                        ] = recv_buf[:].reshape(shape_list[self.rank])

    def create_bc_data(
        self,
        var: Quantity,
        var_name: str,
    ) -> xr.DataArray | None:
        sub_da = None
        if self._color == 1:
            var_shape = var.shape
            n_halo = var.metadata.n_halo
            iadj = 1 if var.dims[0] == "i" else 0
            jadj = 1 if var.dims[1] == "j" else 0
            kadj = 1 if (len(var.dims) == 3 and var.dims[2] == "k") else 0
            shape_list = _shape_list_gen(
                var_shape=var_shape,
                pelist_size=self.size,
                locale=self.location,
                nhalo=n_halo,
                i_adj=iadj,
                j_adj=jadj,
                k_adj=kadj,
            )
            send_buf = np.empty(shape=shape_list[self.rank], dtype=var.dtype).flatten()
            match self.location:
                case "top":
                    js = 0
                    je = shape_list[self.rank][1]
                    if self.rank == 0:
                        js = n_halo
                        je = shape_list[self.rank][1] + n_halo
                    if len(var_shape) == 2:
                        send_buf[:] = var[:n_halo, js:je].flatten()
                        dim = "size_2D_top"
                    if len(var_shape) == 3:
                        send_buf[:] = var[
                            :n_halo, js:je, : var_shape[2] - kadj
                        ].flatten()
                        dim = "size_3D_top"
                    var_name += "_top"
                case "bottom":
                    js = 0
                    je = shape_list[self.rank][1]
                    if self.rank == 0:
                        js = n_halo
                        je = shape_list[self.rank][1] + n_halo
                    if len(var_shape) == 2:
                        send_buf[:] = var[
                            var_shape[0] - n_halo - 1 : var_shape[0] - 1,
                            js:je,
                        ].flatten()
                        dim = "size_2D_bottom"
                    if len(var_shape) == 3:
                        send_buf[:] = var[
                            var_shape[0] - n_halo - 1 : var_shape[0] - 1,
                            js:je,
                            : var_shape[2] - kadj,
                        ].flatten()
                        dim = "size_3D_bottom"
                    var_name += "_bottom"
                case "left":
                    if len(var_shape) == 2:
                        send_buf[:] = var[
                            : var_shape[0] - iadj,
                            :n_halo,
                        ].flatten()
                        dim = "size_2D_left"
                    if len(var_shape) == 3:
                        send_buf[:] = var[
                            : var_shape[0] - iadj,
                            :n_halo,
                            : var_shape[2] - kadj,
                        ].flatten()
                        dim = "size_3D_left"
                    var_name += "_left"
                case "right":
                    if len(var_shape) == 2:
                        send_buf[:] = var[
                            : var_shape[0] - iadj,
                            var_shape[1] - n_halo - 1 : var_shape[1] - 1,
                        ].flatten()
                        dim = "size_2D_right"
                    if len(var_shape) == 3:
                        send_buf[:] = var[
                            : var_shape[0] - iadj,
                            var_shape[1] - n_halo - 1 : var_shape[1] - 1,
                            : var_shape[2] - kadj,
                        ].flatten()
                        dim = "size_3D_right"
                    var_name += "_right"
            if self.rank == 0:
                sendcounts = [np.prod(shape_list[n]) for n in range(self.size)]
                displs = _get_displs(sendcounts)
                temp = np.empty(shape=sum(sendcounts), dtype=var.dtype)
                datatype = get_mpi_type(var[:])
            else:
                sendcounts = None
                displs = None
                temp = None
                datatype = None
            self._sub_comm.Gatherv(
                send_buf, [temp, sendcounts, displs, datatype], root=0
            )
            if self.rank == 0:
                sub_da = xr.DataArray(data=temp, dims=[dim], name=var_name)
            return sub_da
        else:
            return None


class BoundaryCondition:
    file: Path
    dataset: xr.Dataset
    var_list: list[str]
    _location: str
    _color: int
    _main_comm: Communicator
    sub_comm_top: BoundaryConditionCommunicator
    sub_comm_bottom: BoundaryConditionCommunicator
    sub_comm_left: BoundaryConditionCommunicator
    sub_comm_right: BoundaryConditionCommunicator

    def __init__(
        self,
        comm: Communicator,
        layout: tuple,
        file: str | None = None,
    ):
        self._main_comm = comm
        self.sub_comm_top = BoundaryConditionCommunicator(
            main_comm=comm, location="top", layout=layout
        )
        self.sub_comm_bottom = BoundaryConditionCommunicator(
            main_comm=comm, location="bottom", layout=layout
        )
        self.sub_comm_left = BoundaryConditionCommunicator(
            main_comm=comm, location="left", layout=layout
        )
        self.sub_comm_right = BoundaryConditionCommunicator(
            main_comm=comm, location="right", layout=layout
        )
        self.var_list = []
        file_bytes = None
        if file is not None:
            self.file = Path(file)
            if self.file.is_file():
                if self._main_comm.rank == 0:
                    with open(file, "rb") as f:
                        file_bytes = f.read()
                file_bytes = self._main_comm.comm.bcast(file_bytes) or None
        if file_bytes is not None:
            with io.BytesIO(file_bytes) as buf:
                self.dataset = xr.open_dataset(buf, engine="h5netcdf").load()
            if self._main_comm.rank == 0:
                self.var_list = list(self.dataset.keys())
            self.var_list = self._main_comm.comm.bcast(self.var_list) or []

    def scatter_bcs(self, state: T, timestep: int) -> None:
        for field_obj in dataclasses.fields(state):
            var_name = field_obj.name
            if any(var_name in name for name in self.var_list):
                var = getattr(state, var_name)
                self.sub_comm_top.sub_scatterv(
                    var=var,
                    var_name=var_name + "_top",
                    dataset=self.dataset,
                )
                self.sub_comm_bottom.sub_scatterv(
                    var=var, var_name=var_name + "_bottom", dataset=self.dataset
                )
                self.sub_comm_left.sub_scatterv(
                    var=var, var_name=var_name + "_left", dataset=self.dataset
                )
                self.sub_comm_right.sub_scatterv(
                    var=var, var_name=var_name + "_right", dataset=self.dataset
                )
                setattr(state, var_name, var)

    def write_out_bcs(
        self,
        state: T,
        bc_file_name: Path,
    ) -> None:
        self.dataset = xr.Dataset()
        for field_obj in dataclasses.fields(state):
            var_name = field_obj.name
            var = getattr(state, var_name)
            if var_name not in self.var_list:
                for direction in ["_top", "_bottom", "_left", "_right"]:
                    self.var_list.append(var_name + direction)
                top_da = self.sub_comm_top.create_bc_data(
                    var=var,
                    var_name=var_name,
                )
                bottom_da = self.sub_comm_bottom.create_bc_data(
                    var=var,
                    var_name=var_name,
                )
                left_da = self.sub_comm_left.create_bc_data(
                    var=var,
                    var_name=var_name,
                )
                right_da = self.sub_comm_right.create_bc_data(
                    var=var,
                    var_name=var_name,
                )
                gather_top = self._main_comm.comm.gather(top_da, root=0)
                gather_bottom = self._main_comm.comm.gather(bottom_da, root=0)
                gather_right = self._main_comm.comm.gather(right_da, root=0)
                gather_left = self._main_comm.comm.gather(left_da, root=0)
                if self._main_comm.rank == 0:
                    gather_total = (
                        gather_top + gather_bottom + gather_left + gather_right
                    )
                    for da in gather_total:
                        if da is not None:
                            da.to_netcdf(bc_file_name, mode="a")
