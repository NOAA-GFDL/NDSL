from dataclasses import dataclass

from gt4py.cartesian import gtscript

from ndsl.dsl.typing import Float


@dataclass
class MarkupTracerBundleType:
    """Markup a `TracerBundle` to delay specialization.

    Properties:
        name: name of the future type to retrieve from the registry.
    """

    name: str


class TracerBundleTypeRegistry:
    """Class to register and retrieve TraceBundle types.

    Methods:
        register: Register a type
        T: access to any registered type for type hinting.
    """

    _type_registry: dict[str, gtscript._FieldDescriptor] = {}

    @classmethod
    def register(cls, name: str, size: int, dtype=Float) -> gtscript._FieldDescriptor:
        """Register a name type by name by giving the size of its data dimensions.

        The same type cannot be registered twice and will error out.

        Args:
            name: Unique name for this `TracerBundle` type.
            size: Number of tracers in the `TracerBundle`.
            dtype: Data type of `TracerBundle`.
        """
        if name in cls._type_registry:
            raise RuntimeError(
                f"Names of `TracerBundle` types must be unique. `{name}` is already taken."
            )

        cls._type_registry[name] = gtscript.Field[gtscript.IJK, (dtype, ((size,)))]
        return cls._type_registry[name]

    @classmethod
    def T(
        cls, name: str, *, do_markup: bool = True
    ) -> gtscript._FieldDescriptor | MarkupTracerBundleType:
        """
        Retrieve a previously registered type.

        Args:
            name: name of the type as registered via `register`
            do_markup: if name not registered, markup for a future specialization
                at stencil call time
        """
        if name not in cls._type_registry:
            # Dev note: The markup feature is to allow early parsing (at file import)
            # to go ahead - while we will resolve the full type when calling the stencil.
            if do_markup:
                return MarkupTracerBundleType(name)

            raise ValueError(f"TracerBundle type `{name}` has not been registered!")

        return cls._type_registry[name]
