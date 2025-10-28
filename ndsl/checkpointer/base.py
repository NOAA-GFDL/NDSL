import abc
from typing import TypeAlias

import numpy as np

from ndsl import Quantity


SavepointName: TypeAlias = str
VariableName: TypeAlias = str
ArrayLike: TypeAlias = Quantity | np.ndarray


class Checkpointer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, savepoint_name: SavepointName, **kwargs: ArrayLike) -> None: ...
