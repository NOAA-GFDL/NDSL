import xarray as xr
from pathlib import Path
from typing import TypeVar
import dataclasses
import numpy as np

from ndsl.quantity import Quantity
from ndsl.comm.communicator import Communicator
from ndsl.comm.mpi import get_mpi_type

T = TypeVar("T", bound=dataclasses._is_dataclass_instance)

def _shape_list_gen(var_shape: tuple, pelist_size: int, locale: str, nhalo: int = 0, i_adj: int = 0, j_adj: int = 0)->list:
    i = 1
    j = 1
    k = 1
    shape_list = []
    for n in range(pelist_size):
        if len(var_shape) == 3:
            k = var_shape[2] - 1
        match locale:
            case "right" | "left":
                i = var_shape[0] - i_adj
                j = nhalo
            case "top" | "bottom":
                i = nhalo
                j = var_shape[1] - j_adj - nhalo if n in (0, pelist_size-1) else var_shape[1] - j_adj
            case _:
                raise ValueError(f"{locale} is not an edge position")
        shape_list.append((k,i,j))
    return shape_list
    
def _get_displs(counts: list) -> list:
    displs = [0]
    for n in range(1,len(counts)):
        displs.append(displs[n-1]+counts[n-1])
    return displs
    
def _get_data_indices(npoints: int, nhalo: int, size: int, locale: str = None)->list:
    span = npoints//2
    match locale:
        case "right" | "left":
            indices_list = [(0,npoints)]
            for n in range(1,size):
                indices_list.append((indices_list[n-1][0]+span, indices_list[n-1][0]+span+npoints))
        case "top" | "bottom":
            indices_list = [(0,npoints-nhalo)]
            indices_list.append((indices_list[0][0]+span-nhalo, indices_list[0][0]+span-nhalo+npoints))
            for n in range(2,size-1):
                indices_list.append((indices_list[n-1][0]+span, indices_list[n-1][0]+span+npoints))
            indices_list.append((indices_list[-1][0]+span,indices_list[-1][0]+span+npoints-nhalo))
        case _:
            raise ValueError(f"{locale} is not an edge position")
    return indices_list

class BoundaryCondition:
    file: Path = None
    dataset: xr.Dataset = None
    var_list: list[str] = None
    _location: str = None
    _color: int = None
    _main_comm: Communicator = None
    _sub_com: Communicator = None
    _sub_com_rank: int = None
    _sub_com_size: int = None
    _is_root: bool = False

    def __init__(
            self, 
            file: Path, 
            comm: Communicator,
            layout: tuple,
            locale: str = None,
    ):
        self.file = file
        self._main_comm = comm
        pelist = []
        match locale:
            case "top" | "Top" | "TOP":
                self._location = "top"
                for n in range(layout[1]):
                    pelist.append(layout[1]*(layout[1]-1)+n)
                self._color = 1 if comm.rank in pelist else 0
            case "bottom" | "Bottom" | "BOTTOM":
                self._location = "bottom"
                for n in range(layout[1]):
                    pelist.append(n)
                self._color = 1 if comm.rank in pelist else 0
            case "right" | "Right" | "RIGHT":
                self._location = "right"
                for n in range(layout[0]):
                    pelist.append(layout[0]*(n+1)-1)
                self._color = 1 if comm.rank in pelist else 0
            case "left" | "Left" | "LEFT":
                self._location = "left"
                for n in range(layout[0]):
                    pelist.append(layout[0]*n)
                self._color = 1 if comm.rank in pelist else 0
            case _:
                raise ValueError(f"{locale} is not an edge position") 
        self._sub_com = self._main_comm.comm.Split(color=self._color, key=self._main_comm.rank)
        self._sub_com_rank = self._sub_com.rank
        self._sub_com_size = self._sub_com.size
        if self._sub_com_rank == 0 and self._color == 1:
            self._is_root = True
            self.dataset = xr.open_dataset(file)
            whole_var_list = list(self.dataset.keys())
            self.var_list = [var for var in whole_var_list if self._location in var]
        self.var_list = self._sub_com.bcast(self.var_list)
    
    def scatter_bcs(self, state: T, timestep: int):
        if self.color == 1:
            for var in dataclasses.fields(state):
                if var in self.var_names:
                    shape = state.var.data.shape
                    n_halo = state.var.metadata.n_halo
                    iadj = 1 if state.var.dims[0] == "i" else 0
                    jadj = 1 if state.var.dims[1] == "j" else 0
                    shape_list = _shape_list_gen(
                        var_shape=shape,
                        pelist_size=self._sub_com_size,
                        locale=self._location,
                        nhalo=n_halo,
                        i_adj=iadj,
                        j_adj=jadj
                    )
                    recv_buf = np.empty(shape=shape_list[self._sub_com_rank])
                    if self._sub_com_rank == 0:
                        ds = xr.open_dataset(self.file)
                        da = np.ascontiguousarray(ds[self._location].data)
                        if len(da.shape) != 4:
                            da = da[:,np.newaxis,:,:]
                        sendcounts = [np.prod(shape_list[n]) for n in range(self._sub_com_size)]
                        displs = _get_displs(sendcounts)
                        temp = np.empty(shape=sum(sendcounts), dtype=da.dtype)
                        datatype = get_mpi_type(da)
                    else:
                        temp = None
                        sendcounts = None
                        displs = None
                        datatype = None
                    match self.location:
                        case "top":
                            indices = _get_data_indices(
                                npoints=shape_list[1][2], 
                                nhalo=n_halo, 
                                size=self._sub_com_size, 
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                for n in range(self._sub_com_size):
                                    temp[m:m+sendcounts[n]] = da[timestep,:,:,indices[n][0]:indices[n][1]].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv([temp, sendcounts, displs, datatype], recv_buf, root=0)
                        case "bottom":
                            indices = _get_data_indices(
                                npoints=shape_list[1][2], 
                                nhalo=n_halo, 
                                size=self._sub_com_size, 
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                for n in range(self._sub_com_size):
                                    temp[m:m+sendcounts[n]] = da[timestep,:,:,indices[n][0]:indices[n][1]].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv([temp, sendcounts, displs, datatype], recv_buf, root=0)
                        case "right":
                            indices = _get_data_indices(
                                npoints=shape_list[0][1], 
                                nhalo=n_halo, 
                                size=self._sub_com_size, 
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                for n in range(self._sub_com_size):
                                    temp[m:m+sendcounts[n]] = da[timestep,:,indices[n][0]:indices[n][1],:].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv([temp, sendcounts, displs, datatype], recv_buf, root=0)
                        case "left":
                            indices = _get_data_indices(
                                npoints=shape_list[0][1], 
                                nhalo=n_halo, 
                                size=self._sub_com_size, 
                                locale=self._location,
                            )
                            if self._sub_com_rank == 0:
                                m = 0
                                for n in range(self._sub_com_size):
                                    temp[m:m+sendcounts[n]] = da[timestep,:,indices[n][0]:indices[n][1],:].flatten()
                                    m += sendcounts[n]
                            self._sub_com.Scatterv([temp, sendcounts, displs, datatype], recv_buf, root=0)     

    def write_out_bcs(
        self,
        bc_file_name: Path,
    ):
      self.to_netcdf(bc_file_name)

    def free(self):
        self._sub_com.Free()