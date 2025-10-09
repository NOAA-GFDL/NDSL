import copy
from dataclasses import dataclass
from typing import Any

from gt4py.cartesian import gtscript

from ndsl.dsl.typing import Float
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity.quantity import Quantity


# ToDo: This is 4th dimensions restricted. We need a concept
#       of data dimensions index here to be able to extend to N dimensions
_DataDimensionIndex = int
_FieldBundleIndexer = dict[str, _DataDimensionIndex]


class FieldBundle:
    """Field Bundle wraps a nD array (3D + n Data Dimensions) into a complex
    indexing scheme that allows a dual name-based and index-based
    access to the underlying memory. It is paired with the `FieldBundleType`
    which provides a way to type hint parameters for stencils in the `gtscript`.

    WARNING: The present implementation only allows for 4D array.
    """

    _quantity: Quantity
    _indexer: _FieldBundleIndexer = {}

    def __init__(
        self,
        bundle_name: str,
        quantity: Quantity,
        mapping: _FieldBundleIndexer = {},
        register_type: bool = False,
    ):
        """
        Initialize a bundle from a nD quantity.

        Dev note: current implementation limits to 4D inputs.

        Args:
            bundle_name: name of the bundle, accessible via `name`.
            quantity: data inputs as a nD array.
            mapping: sparse dict of [name, index] to be able to call tracers by name.
            register_type: boolean to register the type as part of initialization.
        """
        if len(quantity.shape) != 4:
            raise NotImplementedError("FieldBundle implementation restricted to 4D")

        self.name = bundle_name
        self._quantity = quantity
        self._indexer = mapping
        if register_type:
            # ToDo: extend the dims below to work with more than 4 dims
            assert len(quantity.shape) == 4
            FieldBundleType.register(bundle_name, quantity.shape[3:])

    def map(self, index: _DataDimensionIndex, name: str) -> None:
        """Map a single `index` to ` name`"""
        self._indexer[name] = index

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def shape(self) -> tuple[int, ...]:
        return self._quantity.shape

    def groupby(self, name: str) -> Quantity:
        """Not implemented"""
        raise NotImplementedError

    def __getattr__(self, name: str) -> Quantity:
        """Allow to reference sub-array using `field.a_name`"""
        if name not in self._indexer.keys():
            # This replicates as close possible the default behavior of getattr
            # without breaking orchestration
            return None  # type: ignore
        # ToDo: extend the dims below to work with more than 4 dims
        assert len(self._quantity.data.shape) == 4
        return Quantity(
            data=self._quantity.data[:, :, :, self.index(name)],
            dims=self._quantity.dims[:-1],
            units=self._quantity.units,
            origin=self._quantity.origin[:-1],
            extent=self._quantity.extent[:-1],
        )

    def index(self, name: str) -> int:
        """Get index from name."""
        return self._indexer[name]

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

    @staticmethod
    def extend_3D_quantity_factory(
        quantity_factory: QuantityFactory,
        extra_dims: dict[str, int],
    ) -> QuantityFactory:
        """Create a nD quantity factory from a cartesian 3D factory.

        Args:
            quantity_factory: Cartesian 3D factory.
            extra_dims: dict of [name, size] of the data dimensions to add.
        """
        new_factory = copy.copy(quantity_factory)
        new_factory.set_extra_dim_lengths(
            **{
                **extra_dims,
            }
        )
        return new_factory


@dataclass
class MarkupFieldBundleType:
    """Markup a field bundle to delay specialization.

    Properties:
        name: name of the future type to look into the registrar.
    """

    name: str


class FieldBundleType:
    """Field Bundle Types to help with static sizing of Data Dimensions.

    Methods:
        register: Register a type by sizing its data dimensions
        T: access any registered types for type hinting.
    """

    _field_type_registrar: dict[str, gtscript._FieldDescriptor] = {}

    @classmethod
    def register(  # type: ignore
        cls, name: str, data_dims: tuple[int], dtype=Float
    ) -> gtscript._FieldDescriptor:
        """Register a name type by name by giving the size of its data dimensions.

        The same type cannot be registered twice and will error out.

        Args:
            name: Type name, to be re-used with `T`.
            data_dims: tuple of int giving size of each data dimensions.
            dtype: Inner data type, defaults to Float.
        """
        if name in cls._field_type_registrar.keys():
            raise RuntimeError(f"Registering {name} a second time!")
        cls._field_type_registrar[name] = gtscript.Field[
            gtscript.IJK, (dtype, (data_dims))
        ]
        return cls._field_type_registrar[name]

    @classmethod
    def T(
        cls, name: str, do_markup: bool = True
    ) -> gtscript._FieldDescriptor | MarkupFieldBundleType:
        """
        Get registered type.

        Dev note: The markup feature is to allow early parsing (at file import)
        to go ahead - while we will resolve the full type when calling the stencil.

        Args:
            name: name of the type as registered via `register`
            do_markup: if name not registered, markup for a future specialization
                at stencil call time
        """
        if name not in cls._field_type_registrar:
            if do_markup:
                return MarkupFieldBundleType(name)
            raise RuntimeError(f"FieldBundle type {name} as not been registered!")
        return cls._field_type_registrar[name]
