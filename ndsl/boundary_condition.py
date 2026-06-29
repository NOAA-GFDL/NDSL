from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import xarray as xr

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
) -> list:
    i = 1
    j = 1
    k = 1
    shape_list = []
    for n in range(pelist_size):
        if len(var_shape) == 3:
            k = var_shape[2] - 1
        match locale:
            case "east" | "west":
                i = var_shape[0] - i_adj
                j = nhalo
            case "north" | "south":
                i = nhalo
                j = (
                    var_shape[1] - j_adj - nhalo
                    if n in (0, pelist_size - 1)
                    else var_shape[1] - j_adj
                )
            case _:
                raise ValueError(f"{locale} is not an edge position")
        shape_list.append((k, i, j))
    return shape_list


def _get_displs(counts: list) -> list:
    displs = [0]
    for n in range(1, len(counts)):
        displs.append(displs[n - 1] + counts[n - 1])
    return displs


def _get_data_indices(npoints: int, nhalo: int, size: int, locale: str) -> list:
    span = npoints // 2
    match locale:
        case "east" | "west":
            indices_list = [(0, npoints)]
            for n in range(1, size):
                indices_list.append(
                    (
                        indices_list[n - 1][0] + span,
                        indices_list[n - 1][0] + span + npoints,
                    )
                )
        case "north" | "south":
            indices_list = [(0, npoints - nhalo)]
            indices_list.append(
                (
                    indices_list[0][0] + span - nhalo,
                    indices_list[0][0] + span - nhalo + npoints,
                )
            )
            for n in range(2, size - 1):
                indices_list.append(
                    (
                        indices_list[n - 1][0] + span,
                        indices_list[n - 1][0] + span + npoints,
                    )
                )
            indices_list.append(
                (
                    indices_list[-1][0] + span,
                    indices_list[-1][0] + span + npoints - nhalo,
                )
            )
        case _:
            raise ValueError(f"{locale} is not an edge position")
    return indices_list


class BoundaryCondition:
    file: Path
    dataset: xr.Dataset
    var_list: list[str]
    _location: str
    _color: int
    _main_comm: Communicator
    _sub_com: CommABC
    _sub_com_rank: int
    _sub_com_size: int
    _is_root: bool = False

    def __init__(
        self,
        file: Path,
        comm: Communicator,
        layout: tuple,
        locale: str,
    ):
        self.file = file
        self._main_comm = comm
        pelist = []
        self._location = locale.lower()
        match self._location:
            case "north":
                for n in range(layout[1]):
                    pelist.append(layout[1] * (layout[1] - 1) + n)
                self._color = 1 if comm.rank in pelist else 0
            case "south":
                for n in range(layout[1]):
                    pelist.append(n)
                self._color = 1 if comm.rank in pelist else 0
            case "east":
                for n in range(layout[0]):
                    pelist.append(layout[0] * (n + 1) - 1)
                self._color = 1 if comm.rank in pelist else 0
            case "west":
                for n in range(layout[0]):
                    pelist.append(layout[0] * n)
                self._color = 1 if comm.rank in pelist else 0
            case _:
                raise ValueError(f"{locale} is not an edge position")
        self._sub_com = self._main_comm.comm.Split(
            color=self._color, key=self._main_comm.rank
        )
        self._sub_com_rank = self._sub_com.Get_rank()
        self._sub_com_size = self._sub_com.Get_size()
        self.var_list = []
        if self._sub_com_rank == 0 and self._color == 1:
            self._is_root = True
            self.dataset = xr.open_dataset(file)
            whole_var_list = list(self.dataset.keys())
            self.var_list = [
                var for var in whole_var_list if self._location in var.lower()
            ]
        self.var_list = self._sub_com.bcast(self.var_list) or []

    def scatter_bcs(self, state: T, timestep: int) -> None:
        if self._color == 1:
            for field_obj in dataclasses.fields(state):
                var_name = field_obj.name
                if var_name in self.var_list:
                    var = getattr(state, var_name)
                    var_shape = var.shape
                    n_halo = var.metadata.n_halo
                    iadj = 1 if var.dims[0] == "i" else 0
                    jadj = 1 if var.dims[1] == "j" else 0
                    kadj = 1 if (len(var.dims) == 3 and var.dims[2] == "k") else 0
                    shape_list = _shape_list_gen(
                        var_shape=var_shape,
                        pelist_size=self._sub_com_size,
                        locale=self._location,
                        nhalo=n_halo,
                        i_adj=iadj,
                        j_adj=jadj,
                    )
                    recv_buf = np.empty(
                        shape=shape_list[self._sub_com_rank], dtype=var.dtype
                    )
                    if self._sub_com_rank == 0:
                        da = np.ascontiguousarray(self.dataset[var_name].data)
                        if len(da.shape) != 4:
                            da = da[:, np.newaxis, :, :]
                        sendcounts = [
                            np.prod(shape_list[n]) for n in range(self._sub_com_size)
                        ]
                        displs = _get_displs(sendcounts)
                        temp = np.empty(shape=sum(sendcounts), dtype=da.dtype)
                        datatype = get_mpi_type(da)
                    else:
                        temp = None
                        sendcounts = None
                        displs = None
                        datatype = None
                    match self._location:
                        case "north":
                            indices = _get_data_indices(
                                npoints=shape_list[1][2],
                                nhalo=n_halo,
                                size=self._sub_com_size,
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                assert indices is not None
                                assert sendcounts is not None
                                for n in range(self._sub_com_size):
                                    temp[m : m + sendcounts[n]] = da[
                                        timestep, :, :, indices[n][0] : indices[n][1]
                                    ].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv(
                                [temp, sendcounts, displs, datatype], recv_buf, root=0
                            )
                            js = 0
                            je = shape_list[self._sub_com_rank][2]
                            if self._sub_com_rank == 0:
                                js = n_halo
                                je = shape_list[self._sub_com_rank][2] + n_halo
                            if len(var_shape) == 2:
                                var[:n_halo, js:je] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .squeeze(axis=0)
                                )
                            if len(var_shape) == 3:
                                var[:n_halo, js:je, : var_shape[2] - kadj] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .transpose(1, 2, 0)
                                )
                            setattr(state, var_name, var)
                        case "south":
                            indices = _get_data_indices(
                                npoints=shape_list[1][2],
                                nhalo=n_halo,
                                size=self._sub_com_size,
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                assert indices is not None
                                assert sendcounts is not None
                                for n in range(self._sub_com_size):
                                    temp[m : m + sendcounts[n]] = da[
                                        timestep, :, :, indices[n][0] : indices[n][1]
                                    ].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv(
                                [temp, sendcounts, displs, datatype], recv_buf, root=0
                            )
                            js = 0
                            je = shape_list[self._sub_com_rank][2]
                            if self._sub_com_rank == 0:
                                js = n_halo
                                je = shape_list[self._sub_com_rank][2] + n_halo
                            if len(var_shape) == 2:
                                var[
                                    var_shape[0] - n_halo - iadj : var_shape[0] - iadj,
                                    js:je,
                                ] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .squeeze(axis=0)
                                )
                            if len(var_shape) == 3:
                                var[
                                    var_shape[0] - n_halo - iadj : var_shape[0] - iadj,
                                    js:je,
                                    : var_shape[2] - kadj,
                                ] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .transpose(1, 2, 0)
                                )
                            setattr(state, var_name, var)
                        case "east":
                            indices = _get_data_indices(
                                npoints=shape_list[0][1],
                                nhalo=n_halo,
                                size=self._sub_com_size,
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                assert indices is not None
                                assert sendcounts is not None
                                for n in range(self._sub_com_size):
                                    temp[m : m + sendcounts[n]] = da[
                                        timestep, :, indices[n][0] : indices[n][1], :
                                    ].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv(
                                [temp, sendcounts, displs, datatype], recv_buf, root=0
                            )
                            if len(var_shape) == 2:
                                var[
                                    : var_shape[0] - iadj,
                                    var_shape[1] - n_halo - jadj : var_shape[1] - jadj,
                                ] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .squeeze(axis=0)[::-1]
                                )
                            if len(var_shape) == 3:
                                var[
                                    : var_shape[0] - iadj,
                                    var_shape[1] - n_halo - jadj : var_shape[1] - jadj,
                                    : var_shape[2] - kadj,
                                ] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .transpose(1, 2, 0)[::-1]
                                )
                            setattr(state, var_name, var)
                        case "west":
                            indices = _get_data_indices(
                                npoints=shape_list[0][1],
                                nhalo=n_halo,
                                size=self._sub_com_size,
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                assert indices is not None
                                assert sendcounts is not None
                                for n in range(self._sub_com_size):
                                    temp[m : m + sendcounts[n]] = da[
                                        timestep, :, indices[n][0] : indices[n][1], :
                                    ].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv(
                                [temp, sendcounts, displs, datatype], recv_buf, root=0
                            )
                            if len(var_shape) == 2:
                                var[
                                    : var_shape[0] - iadj,
                                    :n_halo,
                                ] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .squeeze(axis=0)[::-1]
                                )
                            if len(var_shape) == 3:
                                var[
                                    : var_shape[0] - iadj,
                                    :n_halo,
                                    : var_shape[2] - kadj,
                                ] = (
                                    recv_buf[:]
                                    .reshape(shape_list[self._sub_com_rank])
                                    .transpose(1, 2, 0)[::-1]
                                )
                            setattr(state, var_name, var)

    def write_out_bcs(
        self,
        bc_file_name: Path,
    ) -> None:
        pass
