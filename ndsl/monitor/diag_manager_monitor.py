from collections.abc import Mapping
from dataclasses import Field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt

from ndsl import MPIComm
from ndsl.monitor.protocol import Monitor
from ndsl.quantity import Quantity

try:
    from pyfms import diag_manager, fms, mpp_domains, py_mpp

    HAS_PYFMS = True
except ImportError:
    HAS_PYFMS = False


CURRENT_DOMAIN_ID: int | None = None


def set_current_pyfms_domain_id(domain_id: int) -> None:
    global CURRENT_DOMAIN_ID
    CURRENT_DOMAIN_ID = domain_id


def get_current_pyfms_domain_id() -> int | None:
    return CURRENT_DOMAIN_ID


def initialize_pyfms(
    *,
    nx_tile: int,
    layout: list[int],
    ntiles: int,
    halo: int,
    use_cubic_mosaic: bool,
) -> int:
    """Initialize pyFMS communication/domain decomposition and return domain_id."""
    if not HAS_PYFMS:
        raise RuntimeError(
            "pyFMS not installed, install ndsl[pyfms] to use the diag manager monitor"
        )

    text_content = "&diag_manager_nml\nuse_modern_diag=.true.\n/"
    with open("input.nml", "w", encoding="utf-8") as f:
        f.write(text_content)

    # py2f exists on MPIComm at runtime but is not visible to static typing.
    localcomm = MPIComm()._comm.py2f()  # type: ignore[attr-defined]
    fms.init(
        localcomm=localcomm,
        calendar_type=fms.NOLEAP,
    )

    if use_cubic_mosaic:
        domain_id = mpp_domains.define_cubic_mosaic(
            ni=[nx_tile for _ in range(ntiles)],
            nj=[nx_tile for _ in range(ntiles)],
            global_indices=[0, nx_tile - 1, 0, nx_tile - 1],
            layout=layout,
            ntiles=ntiles,
            halo=halo,
            use_memsize=False,
        )
    else:
        domain = mpp_domains.define_domains(
            global_indices=[0, nx_tile - 1, 0, nx_tile - 1],
            layout=layout,
            xhalo=halo,
            yhalo=halo,
        )
        domain_id = domain.domain_id

    mpp_domains.set_current_domain(domain_id=domain_id)
    mpp_domains.define_io_domain(
        domain_id=domain_id,
        io_layout=[1, 1],
    )
    set_current_pyfms_domain_id(domain_id)
    return domain_id


def register_diag_manager_fields(
    *,
    dataclass_fields: Mapping[str, Field[Any]],
    monitor: Any,
    init_time: datetime,
    field_names: list[str],
    module_name: str,
    dtype: Any,
    use_metadata_name: bool = False,
) -> None:
    """Register selected dataclass fields with a diag_manager monitor.

    The input list is updated in place by removing names that are registered.
    """
    for field_name in list(field_names):
        dataclass_field = dataclass_fields.get(field_name)
        if dataclass_field is None:
            continue

        dims = dataclass_field.metadata.get("dims", "unknown")
        units = dataclass_field.metadata.get("units", "unknown")
        if use_metadata_name:
            diag_field_name = dataclass_field.metadata.get("name", field_name)
        else:
            diag_field_name = field_name

        monitor.register_field(
            module_name=module_name,
            field_name=diag_field_name,
            dims=dims,
            units=units,
            init_time=init_time,
            dtype=dtype,
        )
        field_names.remove(field_name)


class DiagManagerMonitor(Monitor):
    """
    sympl.Monitor-style object for sending diagnostics to FMS's diag manager
    """

    def __init__(
        self,
        domain_id: int | None = None,
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
        if domain_id is None:
            domain_id = get_current_pyfms_domain_id()
        if domain_id is None:
            raise RuntimeError(
                "pyFMS domain id is not set. "
                "Call initialize_pyfms before creating DiagManagerMonitor, "
                "or pass domain_id explicitly."
            )
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

    def cleanup(self) -> None:
        """
        Calls diag_manager.end after simulation ends to ensure all data is written.
        """

        if self.diag_end_time is None:
            raise RuntimeError(
                "End time was not set via set_end_time prior to cleanup call"
            )
        diag_manager.end(end_time=self.diag_end_time)

    def store_constant(self, state: dict[str, Quantity]) -> None:
        """diag_manager does not use the Monitor.store_constant API."""
        return

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
        extend_domain_direction: str | None = None,
    ) -> None:
        """
        Registers an axis with the FMS diag_manager via the pyFMS interface for fortran
        This corresponds to a axis/dimension in the output netcdf file.
        Time axis will be added as an unlimited dimension automatically,
        so does not need to be explicitly registered.
        """
        domain_pos = None
        if extend_domain_direction is not None:
            if extend_domain_direction.lower() == "north":
                domain_pos = py_mpp.mpp_domains.NORTH
            elif extend_domain_direction.lower() == "east":
                domain_pos = py_mpp.mpp_domains.EAST
            else:
                raise RuntimeError(
                    "extend_domain_direction must be either 'north' or 'east'."
                )
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
            resolved_domain_id = self.domain_id if domain_id is None else domain_id
            self.axes[name] = diag_manager.axis_init(
                name=name,
                long_name=long_name,
                axis_data=axis_data,
                cart_name=cart_name,
                domain_id=resolved_domain_id,
                set_name=set_name,
                units=units,
                domain_position=domain_pos,
            )
