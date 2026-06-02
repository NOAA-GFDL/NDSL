import xarray as xr
from pathlib import Path
from typing import Any
import numpy as np

from ndsl.quantity import Quantity
from ndsl.comm.communicator import Communicator

class RegionalBoundaryConditions:
    bc_file: Path
    dataset: xr.Dataset
    var_names: list[str]
    location: str
    pelist: list[int]
    comm: Communicator
    root_pe: int
    color: int
    is_root: bool = False

    def __init__(self, file: Path, locale: str, comm: Communicator):
        self.comm = comm
        self.pelist = []
        #TODO: Replace with try-except block
        if locale in ("north", "North", "NORTH"):
            self.location = "north"
            for i in range(comm.partitioner.layout[0]):
                self.pelist.append(i)
        elif locale in ("south", "South", "SOUTH"):
            self.location = "south"
            for i in range(comm.partitioner.layout[0]):
                self.pelist.append(comm.partitioner.layout[0]*(comm.partitioner.layout[0]-1)+i)
        elif locale in ("west", "West", "WEST"):
            self.location = "west"
            for i in range(comm.partitioner.layout[1]):
                self.pelist.append(comm.partitioner.layout[1]*i)
        elif locale in ("east", "East", "EAST"):
            self.location = "east"
            for i in range(comm.partitioner.layout[1]):
                self.pelist.append(comm.partitioner.layout[1]*(i+1)-1)
        else:
            print(f"{locale} is not an edge position")
        self.root_pe = self.pelist[0]
        if comm.rank is self.pelist[0]:
            self.is_root = True
            self.dataset = xr.open_dataset(file)
            self.var_names = list(self.dataset.keys())
        #     comm.comm.bcast(self.var_names, root=self.pelist[0])
        self.color = 1 if comm.rank in self.pelist else 0
        

    def write_out_bcs(
        self,
        bc_file_name: Path,
    ):
      self.to_netcdf(bc_file_name)

    def scatter_bcs(self):
       