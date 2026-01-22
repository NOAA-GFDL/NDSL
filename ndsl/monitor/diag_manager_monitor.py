import os
from pathlib import Path
from warnings import warn

import numpy as np
import xarray as xr

from ndsl.comm.communicator import Communicator
from ndsl.dsl.typing import Float, get_precision
from ndsl.logging import ndsl_log
from ndsl.monitor.convert import to_numpy
from ndsl.quantity import Quantity
from ndsl.monitor.netcdf_monitor import NetCDFMonitor

from ndsl.constants import N_HALO_DEFAULT
from ndsl import GridSizer
from ndsl.quantity.state import State
    

from pyfms import diag_manager, fms, mpp_domains

from mpi4py import MPI


class DiagManagerMonitor(NetCDFMonitor):
    """
    sympl.Monitor-style object for sending diagnostics to FMS's diag manager
    """

    _CONSTANT_FILENAME = "constants"

    def __init__(  # type: ignore[no-untyped-def]
        self,
        path: str,
        communicator: Communicator,
        domain_id: int,
        time_chunk_size: int = 1,
        precision=Float,
    ) -> None:
        """Create a DiagManagerMonitor. Generates its own namelist that requires  

        Args:
            path: directory in which to store data
            communicator: provides global communication to gather state
            time_chunk_size: number of times per file
        """
        # then initialize diag_manager and this object, mpp domain should be created prior to this
        mpp_domains.set_current_domain(domain_id=domain_id)
        diag_manager.init(diag_model_subset=diag_manager.DIAG_ALL)
        self.domain_id = domain_id
        self.fields = {}
        self.axes = {}
        super(DiagManagerMonitor, self).__init__(path, communicator, time_chunk_size, precision)


    def store(self, state: dict) -> None:
        """Append the model state dictionary to the netcdf files.

        Will only write to disk when a full time chunk has been accumulated,
        or when .cleanup() is called.

        Requires the state contain the same quantities with the same metadata as the
        first time this is called. Dimension order metadata may change between calls
        so long as the set of dimensions is the same. Quantities are stored with
        dimensions [time, tile] followed by the dimensions included in the first
        state snapshot. The one exception is "time" which is stored with dimensions
        [time].
        """
        if self._expected_vars is None:
            self._expected_vars = set(state.keys())
        elif self._expected_vars != set(state.keys()):
            raise ValueError(
                "state keys must be the same each time store is called, "
                "got {} but previously got {}".format(
                    set(state.keys()), self._expected_vars
                )
            )
        # diag manager expects data from each PE, shouldn't need to gather state but we'll see
        #state = self._communicator.tile.gather_state(
        #    state, transfer_type=self._transfer_type
        #)

        # if no state then we have nothing to do
        if state is None:
            return

        # get the associated quantities/axis for each field that has been registered
        if state is not None:
            for field_name, field_id in self.fields:
               field_quantity = getattr(state, field_name) # this may not work
               axis_ids = list(map(lambda dimname: axis_ids[dimname], field_quantity.dims))
               # TODO data conversion may not be correct here
               diag_manager.send_data(diag_field_id=field_id, field=np.ascontiguousarray(field_quantity.data), convert_cf_order=True)
               diag_manager.send_complete(field_id)
               diag_manager.advance_field_time(field_id)


    def cleanup(self):
        diag_manager.end()

    # sets the end/initial times that the run is using in the diag manager 
    def set_model_times(self, init_time, end_time):
        diag_manager.set_field_init_time(
            year=init_time.year,
            month=init_time.month,
            day=init_time.day,
            hour=init_time.hour,
            minute=init_time.minute,
            second=init_time.second,
        )
        diag_manager.set_time_end(
            year=end_time.year,
            month=end_time.month,
            day=end_time.day,
            hour=end_time.hour,
            minute=end_time.minute,
            second=end_time.second,
        )

    # registers a diag_manager field/variable for a given Quantity, adds name:field_id to self.fields
    def register_field(self, name: str, quantity: Quantity, domain_id: int, timestep):
        self.fields[name] = diag_manager.field_init(
            name=name,
            long_name=name,
            units=quantity.units,
            domain_id=domain_id,
            set_name="atm",
            cart_name=name,
            precision=get_precision(quantity.data),
        )

    # registers a axis in diag_manager, adds name:axis_id to self.axes 
    def register_axis(self, name: str, axis_data: np.ndarray, cart_name: str = None,
                      long_name: str = None, units: str = None):
        self.axes[name] = diag_manager.axis_init(
            name=name,
            long_name=long_name,
            axis_data=axis_data,
            cart_name=cart_name,
            domain_id=self.domain_id,
            set_name="atm",
            units="radians"
        )

        
# this is a copy of the _TimeChunkedVariable from netcdf_monitor.py for now, intent is to modify for fms/fortran formatting later as needed 
class _FortranFormattedVariable:
    '''
    Handles adjusting variable data from a Quantity to be passed into Fortran routines for pyFMS
    '''
    def __init__(self, initial: Quantity, time_chunk_size: int):
        self._data = np.zeros(
            (time_chunk_size, *initial.extent), dtype=initial.data.dtype
        )
        self._data[0, ...] = to_numpy(initial.view[:])
        self._dims = initial.dims
        self._units = initial.units
        self._i_time = 1
        self._backend = initial.backend

    def append(self, quantity: Quantity) -> None:
        # Allow mismatch precision here since this is I/O
        self._data[self._i_time, ...] = to_numpy(
            quantity.transpose(self._dims, allow_mismatch_float_precision=True).view[:]
        )
        self._i_time += 1

    @property
    def data(self) -> Quantity:
        # Allow mismatch precision here since this is I/O
        return Quantity(
            data=self._data[: self._i_time, ...],
            dims=("time",) + tuple(self._dims),
            units=self._units,
            allow_mismatch_float_precision=True,
            backend=self._backend,
        )

