import dataclasses

from ndsl.quantity import Quantity, State


@dataclasses.dataclass
class Temporaries(State):
    """Base class to collect temporaries Quantities.

    You _cannot_ expect the temporaries memory to be available outside of
    the class it has been defined in.

    Shares the `ndsl.quantity.State` API, see `State` docs.
    """

    def __post_init__(self):
        def _post_init_recursive(dataclass: Temporaries):
            for _field in dataclasses.fields(dataclass):
                if dataclasses.is_dataclass(_field.type):
                    _post_init_recursive(dataclass.__getattribute__(_field.name))
                elif _field.type == Quantity:
                    dataclass.__getattribute__(_field.name).transient = True

        _post_init_recursive(self)
