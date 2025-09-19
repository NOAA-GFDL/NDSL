import copy
from typing import Any

from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.initialization.allocator import Quantity, QuantityFactory
from ndsl.quantity.tracer_bundle_type import TracerBundleTypeRegistry


class Tracer(Quantity):
    """A Tracer is a specialized Quantity, grouped together in a TracerBundle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


_TracerName = str
_TracerIndex = int
_TracerMapping = dict[_TracerName, _TracerIndex]
_TracerDataMapping = dict[_TracerIndex, Tracer]


class TracerBundle:
    """A TracerBundle groups a given set of named/nameless tracers into a single
    four-dimensional Quantity.

    All tracers can be accessed by index, e.g. `tracer[1]`. Named tracers can be
    accessed by name too, e.g. `tracer.vapor` assuming `vapor` is defined in the
    `mapping` of names to tracer indices. `len(tracers)` returns the size of this
    TracerBundle.
    """

    def __init__(
        self,
        *,
        type_name: str,
        quantity_factory: QuantityFactory,
        mapping: _TracerMapping = {},
        unit: str = "g/kg",
    ) -> None:
        """
        Initialize a TracerBundle of a given size.

        Args:
            type_name (str): name under which this bundle's type is registered.
            quantity_factory: QuantityFactory to build tracers with.
            mapping: Optional mapping of names to tracer ids, e.g. `{"vapor": 3}`.
            unit: Optional unit of the tracers (one for all).
        """
        type: Any = TracerBundleTypeRegistry.T(type_name, do_markup=False)
        size = type.data_dims[0]
        factory = _tracer_quantity_factory(quantity_factory, size)

        # TODO: zeros() or empty()? should this be an option?
        self._quantity = factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, "tracers"], dtype=type.dtype, units=unit
        )
        self._size = size
        self._name_mapping = mapping
        self._data_mapping: _TracerDataMapping = {}

    @property
    def __array_interface__(self):
        """Memory interface for CPU."""
        return self._quantity.__array_interface__

    @property
    def __cuda_array_interface__(self):
        """Memory interface for GPU memory as defined by cupy."""
        return self._quantity.__cuda_array_interface__

    def __descriptor__(self) -> Any:
        """Data descriptor for DaCe."""
        return self._quantity.__descriptor__()

    def __len__(self) -> int:
        """Number of tracers in this bundle."""
        return self._size

    def __getattr__(self, name: _TracerName) -> Tracer | None:
        """Access tracers by name, e.g. `tracers.ice`."""

        index = self._name_mapping.get(name, None)
        if index is None:
            # This replicates as close possible the default behavior of getattr
            # without breaking orchestration
            return None

        return self._by_index(index)

    def __getitem__(self, index: _TracerIndex) -> Tracer:
        """Access tracers by index, e.g. `tracers[i]`."""
        return self._by_index(index)

    def _by_index(self, index: _TracerIndex) -> Tracer:
        if index < 0 or index >= self._size:
            raise ValueError(f"You can only select tracers in range [0, {self._size}).")

        # Memoize tracers accessed such that we always return the same instance
        # regardless of whether users access through __getattr__() or __getitem__().
        if index not in self._data_mapping:
            self._data_mapping[index] = Tracer(
                data=self._quantity.data[:, :, :, index],
                dims=self._quantity.dims[:-1],
                origin=self._quantity.origin[:-1],
                extent=self._quantity.extent[:-1],
                units=self._quantity.units,
                # Ensure we never copy data into a tracer
                raise_on_data_copy=True,
            )

        return self._data_mapping[index]


def _tracer_quantity_factory(
    quantity_factory: QuantityFactory, number_of_tracers: int
) -> QuantityFactory:
    """Create a tracer factory from a given cartesian quantity factory.

    Args:
        quantity_factory: Cartesian 3D factory to start from.
        number_of_tracers: number of tracers in this bundle.
    """
    tracer_factory = copy.copy(quantity_factory)
    tracer_factory.set_extra_dim_lengths(tracers=number_of_tracers)
    return tracer_factory
