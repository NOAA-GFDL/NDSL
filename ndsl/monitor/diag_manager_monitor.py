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
        """Create a DiagManagerMonitor.

        Args:
            path: directory in which to store data
            communicator: provides global communication to gather state
            time_chunk_size: number of times per file
        """
        # initialize diag_manager and this object, mpp domain should be created prior to this
        mpp_domains.set_current_domain(domain_id=domain_id)
        if not DiagManagerMonitor._diag_initialized:
            diag_manager.init(diag_model_subset=diag_manager.DIAG_ALL)
            DiagManagerMonitor._diag_initialized = True
        self.domain_id = domain_id
        self.fields = {}
        self.axes = {}
        self.diag_end_time = None
        self.precision = precision
        super().__init__(path, communicator, time_chunk_size, precision)


    def store(self, state: dict) -> None:
        """
        Send the data to the diag manager. Sends data for each field in the state. 
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
            time = state["time"]
            for field_name, field_id in self.fields.items():
                field_quantity = state[field_name]
                print(f"sending data for {field_name} id: {field_id}\ntime: {state['time']}\nquantity value: {field_quantity}")
                # TODO data conversion may not be correct here
                diag_manager.send_data(
                    diag_field_id=field_id,
                    field=np.ascontiguousarray(field_quantity.view[:]),
                    convert_cf_order=True,
                    time=time,
                )
            # TODO: need some way to get the timestep for this, potentially save the last time that was passed in
            try:
                diag_manager.send_complete( timestep=self.timestep)
            except:
                raise NameError("no timestep set via set_timestep")

    # close the file
    def cleanup(self):
        if self.diag_end_time is None:
            raise RuntimeError("End time was not set via set_end_time prior to cleanup call")
        diag_manager.end(end_time=self.diag_end_time)

    # sets the end time to stop recieving data. Must be called prior to cleanup/diag_manager.end()
    def set_end_time(self, end_time: datetime):
        diag_manager.set_time_end(end_time)
        self.diag_end_time = end_time

    def set_timestep(self, timestep: timedelta):
        self.timestep = timestep

    # registers a diag_manager field/variable for a given Quantity, adds name:field_id to self.fields
    # TODO: add args
    def register_field(self, module_name: str, field_name: str, dims: list[str], units: str, init_time: datetime,
                       long_name: str = None, ticks: int = 0):

        field_axes = [self.axes[dim] for dim in dims]
        if any(field_axes) is None:
            raise ValueError(f"All axes for field {field_name} must be registered before registering the field.")

        print(f"registering field {field_name} with axes {field_axes}, init time: {init_time}")
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
            multiple_send_data = None,
            init_time = init_time,
            ticks_per_second = ticks, 
        )
        if field_id < 0:
            raise RuntimeError(f"Failed to register field {field_name} in diag_manager, got field_id={field_id}")
        self.fields[field_name] = field_id 


    # registers a axis in diag_manager, adds name:axis_id to self.axes 
    # TODO: might be able to make this private, intialize from quantity dims when registering field
    def register_axis(self, name: str, axis_data: np.ndarray, cart_name: str = None,
                      long_name: str = None, not_xy: bool = False, units: str = None,
                      domain_id: int = None, set_name: str = "atm"):
        self.axes[name] = diag_manager.axis_init(
            name=name,
            long_name=long_name,
            axis_data=axis_data,
            cart_name=cart_name,
            domain_id=domain_id,
            set_name=set_name,
            not_xy=not_xy,
            units=units,
        )
