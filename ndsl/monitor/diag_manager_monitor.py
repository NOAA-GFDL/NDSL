import datetime
import os
from pathlib import Path
from warnings import warn

from ndsl.quantity.metadata import QuantityMetadata
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

from datetime import datetime, timedelta


class DiagManagerMonitor(NetCDFMonitor):
    """
    sympl.Monitor-style object for sending diagnostics to FMS's diag manager
    """
    _diag_initialized = False

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
        if not DiagManagerMonitor._diag_initialized:
            diag_manager.init(diag_model_subset=diag_manager.DIAG_ALL)
            DiagManagerMonitor._diag_initialized = True
        self.domain_id = domain_id
        self.fields = {}
        self.axes = {}
        self.precision = precision
        super().__init__(path, communicator, time_chunk_size, precision)


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
            for field_name, field_id in self.fields.items():
                field_quantity = state[field_name] # this may not work
                # TODO: this step may need to be skipped in certain scenarios, will jump off that bridge when we get there
                diag_manager.advance_field_time(field_id)
                print(f"sending data for {field_name} id: {field_id}\ntime: {state['time']}\nquantity value: {field_quantity}")
                # TODO data conversion may not be correct here
                diag_manager.send_data(diag_field_id=field_id, field=np.ascontiguousarray(field_quantity.view[:]), convert_cf_order=True)
                diag_manager.send_complete(field_id)


    def cleanup(self):
        diag_manager.end()

    # sets the end/initial times that the run is using in the diag manager 
    def set_model_times(self, init_time: datetime, end_time: datetime):
        print(f"setting model times in diag_manager: init_time={init_time}, end_time={end_time}")
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

    # sets the timestep for a given diag_manager field/variable
    # TODO would be nice to check the diag_table.yaml for the file freq and ensure the timestep is valid 
    def _set_field_timestep(self, field_id: int, timestep: timedelta, ticks_per_second: int = 1):
        print(f"setting field timestep in diag_manager: field_id={field_id}, timestep= days={timestep.days}, seconds={timestep.seconds}, ticks_per_second={ticks_per_second}")
        diag_manager.set_field_timestep(
            diag_field_id=field_id,
            ddays=timestep.days,
            dseconds=timestep.seconds,
            dticks=ticks_per_second,
        )

    # registers a diag_manager field/variable for a given Quantity, adds name:field_id to self.fields
    # TODO: 
    def register_field(self, module_name: str, field_name: str, dims: list[str], units: str, timestep: timedelta = None,
                       long_name: str = None, ticks_per_seconds: int = 1):

        field_axes = [self.axes[dim] for dim in dims]
        if any(field_axes) is None:
            raise ValueError(f"All axes for field {field_name} must be registered before registering the field.")

        print(f"registering field {field_name} with axes {field_axes}, timestep {timestep}")
        # putting all the arguments here for now, only module_name, field_name, precision are required
        field_id = diag_manager.register_field_array(
            module_name=module_name,
            field_name=field_name,
            axes = field_axes,
            long_name= long_name,
            units= units,
            dtype = "float32" if self.precision == np.float32 else "float64", #TODO
            missing_value = None,
            range_data = None,
            mask_variant = None,
            standard_name = None,
            verbose = True,
            do_not_log = None,
            interp_method = None,
            tile_count = None,
            area = None,
            volume = None, 
            realm = None,
            multiple_send_data = None
        )
        self.fields[field_name] = field_id 
        if timestep is not None:
            self._set_field_timestep(field_id=field_id, timestep=timestep, ticks_per_second=ticks_per_seconds)
        else:
            raise NotImplementedError("Static fields not yet implemented.")
        

    # registers a axis in diag_manager, adds name:axis_id to self.axes 
    # TODO: might be able to make this private, intialize from quantity dims when registering field
    def register_axis(self, name: str, axis_data: np.ndarray, cart_name: str = None,
                      long_name: str = None, not_xy: bool = False, units: str = None):
        self.axes[name] = diag_manager.axis_init(
            name=name,
            long_name=long_name,
            axis_data=axis_data,
            cart_name=cart_name,
            domain_id=self.domain_id,
            set_name="atm",
            not_xy=not_xy,
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

