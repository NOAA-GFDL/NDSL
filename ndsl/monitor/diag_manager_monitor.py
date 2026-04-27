from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt

from ndsl.monitor.protocol import Monitor


try:
    from pyfms import diag_manager

    HAS_PYFMS = True
except ImportError:
    HAS_PYFMS = False


class DiagManagerMonitor(Monitor):
    """
    sympl.Monitor-style object for sending diagnostics to FMS's diag manager
    """

    def __init__(
        self,
        domain_id: int,
    ) -> None:
        """Create a DiagManagerMonitor.

        Args:
            domain_id: integer domain-decomposition identifier as returned by mpp_define_domain
        """
        if not HAS_PYFMS:
            raise RuntimeError(
                "pyFMS not installed, install ndsl[pyfms] to use the diag manager monitor"
            )
        diag_manager.init(diag_model_subset=diag_manager.DIAG_ALL)
        self.fields: dict[str, int] = {}
        self.axes: dict[str, int] = {}
        self.diag_end_time: datetime | None = None
        self.domain_id = domain_id

    def store(self, state: dict) -> None:
        """
        Sends data from quantities in the state to be written by the diag_manager.
        All state variables must be registered beforehand via register_field.
        """
        # get the associated quantities/axis for each field that has been registered
        if state is not None:
            time = state["time"]
            for field_name, field_id in self.fields.items():
                field_quantity = state[field_name]
                success = diag_manager.send_data(
                    diag_field_id=field_id,
                    field=field_quantity.field,
                    convert_cf_order=True,
                    time=time,
                )
                if not success:
                    raise RuntimeError(
                        f"Failed to send data for field {field_name} at time {time} to diag_manager"
                    )
            try:
                diag_manager.send_complete(timestep=self.timestep)
            except NameError:
                raise RuntimeError("no timestep set via set_timestep")

    # close the file
    def cleanup(self) -> None:
        if self.diag_end_time is None:
            raise RuntimeError(
                "End time was not set via set_end_time prior to cleanup call"
            )
        diag_manager.end(end_time=self.diag_end_time)

    def set_end_time(self, end_time: datetime) -> None:
        """
        Sets the end time to stop recieving data. Must be called prior to cleanup/diag_manager.end()
        """
        diag_manager.set_time_end(end_time)
        self.diag_end_time = end_time

    def set_timestep(self, timestep: timedelta) -> None:
        """
        Sets the timestep to increment by after data is sent.
        """
        self.timestep = timestep

    def register_field(
        self,
        module_name: str,
        field_name: str,
        units: str,
        dtype: str,
        init_time: datetime,
        dims: list[str] | None = None,  # if none, static field
        missing_value: float | None = None,
        long_name: str | None = None,
        range_data: npt.NDArray | None = None,
    ) -> None:
        """
        Register a diagnostic field with the FMS diag_manager via the pyFMS interface for fortran
        This corresponds to a variable/field in the output netcdf file.
        Any axis/dimensions used by this variable should be registered prior to this function.
        """
        if dims is not None:
            field_axes = [self.axes[dim] for dim in dims]
            if any(field_axes) is None:
                raise ValueError(
                    f"All axes for field {field_name} must be registered before registering the field."
                )

        field_id = diag_manager.register_field_array(
            module_name=module_name,
            field_name=field_name,
            axes=field_axes,
            long_name=long_name,
            units=units,
            dtype=dtype,
            missing_value=missing_value,
            range_data=range_data,
            init_time=init_time,
        )
        if field_id < 0:
            raise RuntimeError(
                f"Failed to register field {field_name} in diag_manager, got field_id={field_id}"
            )
        self.fields[field_name] = field_id

    def register_axis(
        self,
        name: str,
        axis_data: np.ndarray,
        not_xy: bool,
        cart_name: str | None = None,
        long_name: str | None = None,
        units: str | None = None,
        domain_id: int | None = None,
        set_name: str | None = None,
    ) -> None:
        """
        Registers an axis with the FMS diag_manager via the pyFMS interface for fortran
        This corresponds to a axis/dimension in the output netcdf file.
        Time axis will be added as an unlimited dimension automatically,
        so does not need to be explicitly registered.
        """
        if not_xy:
            self.axes[name] = diag_manager.axis_init(
                name=name,
                long_name=long_name,
                axis_data=axis_data,
                cart_name=cart_name,
                set_name=set_name,
                not_xy=not_xy,
                units=units,
            )
        else:
            self.axes[name] = diag_manager.axis_init(
                name=name,
                long_name=long_name,
                axis_data=axis_data,
                cart_name=cart_name,
                domain_id=domain_id,
                set_name=set_name,
                units=units,
            )
