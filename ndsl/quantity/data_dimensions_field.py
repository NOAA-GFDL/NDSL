import collections
import inspect
from dataclasses import dataclass

import numpy.typing as npt
from dace import SDFG, SDFGState
from dace.frontend.common import op_repository as oprepo
from dace.frontend.python.common import StringLiteral
from dace.frontend.python.newast import ProgramVisitor

from ndsl import QuantityFactory
from ndsl.dsl.typing import Float, Int
from ndsl.internal.deferred_type import (
    StencilDeferredType,
    StencilTypeRegistrar,
    get_lhs_name,
)


from gt4py.cartesian import gtscript  # isort: skip
from gt4py.cartesian.gtscript import _FieldDescriptorMaker  # isort: skip


DataDimensionIndex = int
SparseNameMapping = dict[str, DataDimensionIndex]


class _DataDimensionsFieldDescriptor(gtscript._FieldDescriptor):
    """Extension to the gt4py.cartesian.Field to account for sparsly
    named indexed data dimensions.
    """

    def __init__(
        self,
        dtype: npt.DTypeLike,
        axes: gtscript.Axis | collections.abc.Collection,
        data_dims: tuple[int] | collections.abc.Collection = tuple(),
    ) -> None:
        super().__init__(dtype, axes, data_dims)
        self._mapping: SparseNameMapping = {}

    def range(self, dimension_index: DataDimensionIndex) -> range:
        return range(self.data_dims[dimension_index])

    @property
    def mapping(self) -> SparseNameMapping:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: SparseNameMapping) -> None:
        self._mapping = mapping

    def index(self, name: str) -> int:
        if len(self.data_dims) != 1:
            raise NotImplementedError("Unimplemtened index on a multiple data dims")
        return self._mapping[name]

    def size(self, data_dim_index: int) -> int:
        return self.data_dims[data_dim_index]


class _DataDimensionFieldMaker(_FieldDescriptorMaker):
    """Factory for DataDimensionsField"""

    def __getitem__(
        self, field_spec: gtscript.Axis | collections.abc.Collection
    ) -> _DataDimensionsFieldDescriptor:
        field_descriptor = super().__getitem__(field_spec)
        return _DataDimensionsFieldDescriptor(
            field_descriptor.dtype,
            field_descriptor.axes,
            field_descriptor.data_dims,
        )


_DataDimensionDescriptor = _DataDimensionFieldMaker()


class DataDimensionsField(StencilTypeRegistrar):
    """Type allowing semi-dynamic sizing of field with data dimensions.

    Methods:
        register: Register a type by sizing its data dimensions
        declare: declare a type for future registration
    """

    _type_registrar: dict[str, _DataDimensionsFieldDescriptor] = {}

    @classmethod
    def register(
        cls,
        pre_registration_type: "DataDimensionsMarkupType",
        quantity_factory: QuantityFactory,
        data_dimensions_names: list[str],
        name_mapping: SparseNameMapping | None = None,
        dtype: npt.DTypeLike = Float,
    ) -> _DataDimensionsFieldDescriptor:
        """Register a type by name by giving the size of its data dimensions and
        optionally a sparse mapping of name/index.

        The same type cannot be registered twice and will error out.

        Args:
            pre_registration_type: Type returned by the "declare" function.
            quantity_factory: Factory carrying the proper data dimensions axis described
                in `data_dimensions_names`.
            data_dimensions_names: list of name of data dimension axis.
            name_mapping: for each dimensions, a sparse dictionnary giving a name/index
                to retrieve 3D fields by name.
            dtype: Inner data type, defaults to Float.
        """
        name = pre_registration_type.name
        if name in cls._type_registrar.keys():
            raise RuntimeError(f"Registering {name} a second time!")

        data_dims_size = []
        for ddim_name in data_dimensions_names:
            if ddim_name not in quantity_factory.sizer.data_dimensions:
                raise KeyError(
                    f'Data dimension axis "{ddim_name}" is not present in QuantityFactory. '
                    "Use QuantityFactory.add_data_dimensions prior to registering field."
                )
            data_dims_size.append(quantity_factory.sizer.data_dimensions[ddim_name])

        cls._type_registrar[name] = _DataDimensionDescriptor[
            gtscript.IJK, (dtype, tuple(data_dims_size))
        ]
        if name_mapping is not None:
            cls._type_registrar[name].mapping = name_mapping

        # Dynamic op replacement fo Type.index() function
        # Requires the _locals to get `name` - do not pull out of `register`
        @oprepo.replaces(f"{name}.index")
        def _data_dimensions_index(
            pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            index_name: StringLiteral,
        ) -> slice:
            if len(cls._type_registrar[name].data_dims) != 1:
                raise NotImplementedError("Unimplemented index on a multiple data dims")
            index = cls._type_registrar[name].index(str(index_name))
            return slice(index, index + 1)

        # Dynamic op replacement for Type.size() function
        # Requires the _locals to get `name` - do not pull out of `register`
        @oprepo.replaces(f"{name}.size")
        def _data_dimensions_size(
            pv: ProgramVisitor,
            sdfg: SDFG,
            state: SDFGState,
            data_dim_index: int,
        ) -> Int:
            size = cls._type_registrar[name].size(data_dim_index)
            return Int(size)

        return cls._type_registrar[name]

    @classmethod
    def declare(cls) -> "DataDimensionsMarkupType":
        """
        Declare a data dimension field - which will need to be properly
        registered later.

        Args:
            name: name of the type as registered via `register`
            do_markup: if name not registered, markup for a future specialization
                at stencil call time
        """
        name = get_lhs_name(inspect.currentframe())
        return DataDimensionsMarkupType(name)

    @classmethod
    def declare_and_register(
        cls,
        quantity_factory: QuantityFactory,
        data_dimensions_names: list[str],
        name_mapping: SparseNameMapping | None = None,
        dtype: npt.DTypeLike = Float,
    ) -> "DataDimensionsMarkupType":
        """Declare a data dimension field and register it's size

        Args:
            quantity_factory: Factory carrying the proper data dimensions axis described
                in `data_dimensions_names`.
            data_dimensions_names: list of name of data dimension axis.
            name_mapping: for each dimensions, a sparse dictionnary giving a name/index
                to retrieve 3D fields by name.
            dtype: Inner data type, defaults to Float.
        """

        name = get_lhs_name(inspect.currentframe())
        markup_type = DataDimensionsMarkupType(name)
        cls.register(
            markup_type,
            quantity_factory,
            data_dimensions_names,
            name_mapping,
            dtype,
        )
        return markup_type

    @classmethod
    def get(cls, name: str) -> _DataDimensionsFieldDescriptor:
        """
        Declare a data dimension field - which will need to be properly
        registered later.

        Args:
            name: name of the type as registered via `register`
        """
        if name not in cls._type_registrar:
            raise RuntimeError(f"Data dimension field {name} as not been registered!")
        return cls._type_registrar[name]


@dataclass
class DataDimensionsMarkupType(StencilDeferredType):
    """Markup a future data dimensions type.

    Dev note: The markup feature is to allow early parsing (at file import)
    to go ahead - while we will resolve the full type when calling the stencil.

    Properties:
        name: name of the future type to look into the registrar.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def range(self, dimension_index: DataDimensionIndex) -> range:
        if self.name not in DataDimensionsField._type_registrar:
            raise RuntimeError()
        return DataDimensionsField._type_registrar[self.name].range(dimension_index)

    def _get_true_type(self) -> _DataDimensionsFieldDescriptor:
        if self.name not in DataDimensionsField._type_registrar:
            raise KeyError(
                f"Data dimension field {self.name} is not registered. "
                f"Call DataDimensionsField.register({self.name})."
            )

        return DataDimensionsField._type_registrar[self.name]

    @property
    def mapping(self) -> SparseNameMapping:
        return self._get_true_type().mapping

    def index(self, name: str) -> int:
        return self._get_true_type().index(name)

    def size(self, data_dims_index: int) -> int:
        return self._get_true_type().size(data_dims_index)

    @classmethod
    def resolve(cls) -> type[StencilTypeRegistrar]:
        return DataDimensionsField
