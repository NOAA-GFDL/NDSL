import copy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ndsl.dsl.typing import Float
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity.quantity import Quantity


import gt4py.cartesian.gtscript as gtscript  # isort: skip


# ToDo: This is 4th dimensions restricted. We need a concept
#       of data dimensions index here to be able to extend to N dimensions
_DataDimensionIndex = int
_FieldBundleIndexer = Dict[str, _DataDimensionIndex]


class FieldBundle:
    """Field Bundle wraps a nD array (3D + n Data Dimensions) into a complex
    indexing scheme that allows a dual name-based and index-based
    access to the underlying memory. It is paired with the `FieldBundleType`
    which provides a way to

    WARNING: The present implementation only allows for 4D array.
    """

    _quantity: Quantity
    _indexer: _FieldBundleIndexer = {}

    def __init__(
        self,
        bundle_name: str,
        quantity: Quantity,
        mapping: _FieldBundleIndexer = {},
        do_register_type: bool = False,
    ):
        if len(quantity.shape) != 4:
            raise NotImplementedError("FieldBundle implementation restricted to 4D")

        self.name = bundle_name
        self._quantity = quantity
        self._indexer = mapping
        if do_register_type:
            FieldBundleType.register(bundle_name, quantity.shape[3:])

    def map(self, index: _DataDimensionIndex, name: str):
        self._indexer[name] = index

    @property
    def quantity(self) -> Quantity:
        return self._quantity

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._quantity.shape

    def groupby(self, name: str) -> Quantity:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Quantity:
        if name not in self._indexer.keys():
            # This replicates as close possible the default behavior of getattr
            # without breaking orchestration
            return None  # type: ignore
        # ToDo: extend the dims below to work with more than 4 dims
        assert len(self._quantity.data.shape) == 4
        return Quantity(
            data=self._quantity.data[:, :, :, self._indexer[name]],
            dims=self._quantity.dims[:-1],
            units=self._quantity.units,
            origin=self._quantity.origin[:-1],
            extent=self._quantity.extent[:-1],
        )

    @property
    def __array_interface__(self):
        return self._quantity.__array_interface__

    @property
    def __cuda_array_interface__(self):
        return self._quantity.__array_interface__

    def __descriptor__(self) -> Any:
        return self._quantity.__descriptor__()

    @staticmethod
    def extend_3D_quantity_factory(
        quantity_factory: QuantityFactory,
        extra_dims: dict[str, int],
    ) -> QuantityFactory:
        new_factory = copy.copy(quantity_factory)
        new_factory.set_extra_dim_lengths(
            **{
                **extra_dims,
            }
        )
        return new_factory

    def index(self, name: str) -> int:
        return self._indexer[name]


@dataclass
class MarkupFieldBundleType:
    """Markup a field bundle to delay specialization

    Properties:
        name: name of the future type to look into the registrat
    """

    name: str


class FieldBundleType:
    """Field Bundle Types to help with static sizing of Data Dimensions

    Methods:
        register: Register a type by sizing it's data dimensions
        T: access any registered types for type hinting
    """

    _field_type_registrar: dict[str, gtscript._FieldDescriptor] = {}

    @classmethod
    def register(cls, name: str, data_dims: tuple[int]) -> gtscript._FieldDescriptor:
        if name in cls._field_type_registrar.keys():
            raise RuntimeError(f"Registering {name} a second time!")
        cls._field_type_registrar[name] = gtscript.Field[
            gtscript.IJK, (Float, (data_dims))
        ]
        return cls._field_type_registrar[name]

    @classmethod
    def T(
        cls, name: str, do_markup: bool = True
    ) -> gtscript._FieldDescriptor | MarkupFieldBundleType:
        if name not in cls._field_type_registrar:
            if do_markup:
                return MarkupFieldBundleType(name)
            else:
                raise RuntimeError(f"FieldBundle type {name} as not been registered!")
        return cls._field_type_registrar[name]
