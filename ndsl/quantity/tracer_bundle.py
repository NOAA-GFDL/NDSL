import copy
from enum import Enum, auto
from typing import Any

from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.initialization.allocator import Quantity, QuantityFactory
from ndsl.quantity.tracer_bundle_type import TracerBundleTypeRegistry


class Region(Enum):
    compute_domain = auto()


class Tracer(Quantity):
    """A Tracer is a specialized Quantity, grouped together in a TracerBundle."""

    class Restriction(Enum):
        compute_domain = auto()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fill(self, value: Any, *, restrict_to: Region | None = None) -> None:
        if restrict_to is None:
            super().data[:] = value
        elif restrict_to is Region.compute_domain:
            super().field[:] = value
        else:
            raise NotImplementedError(f"Unknown restriction {restrict_to}.")


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
        types: Any = TracerBundleTypeRegistry.T(type_name, do_markup=False)

        size = types[0].data_dims[0]
        factory = _tracer_quantity_factory(quantity_factory, size)

        # TODO: zeros() or empty()? should this be an option?
        self._quantity = factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, "tracers"], dtype=types[0].dtype, units=unit
        )
        self._size = size
        self._name_mapping = mapping
        self._data_mapping: _TracerDataMapping = {}

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
            # Note: it is important to raise an IndexError to support iterations of
            #       the form `for tracer in tracers`.
            raise IndexError(f"You can only select tracers in range [0, {self._size}).")

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

    def fill_tracer(
        self, index: _TracerIndex, *, value: Any, compute_domain_only: bool = False
    ) -> None:
        if compute_domain_only:
            self._quantity.field[:, :, :, index] = value
        else:
            self._quantity.data[:, :, :, index] = value

    def fill_tracer_by_name(
        self, name: str, *, value: Any, compute_domain_only: bool = False
    ) -> None:
        index = self._name_mapping[name]

        if compute_domain_only:
            self._quantity.field[:, :, :, index] = value
        else:
            self._quantity.data[:, :, :, index] = value


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
