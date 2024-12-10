from typing import Sequence, Tuple, Union

import numpy as np

import ndsl.constants as constants
from ndsl.comm._boundary_utils import bound_default_slice, shift_boundary_slice_tuple


class BoundaryArrayView:
    def __init__(self, data, boundary_type, dims, origin, extent):
        self._data = data
        self._boundary_type = boundary_type
        self._dims = dims
        self._origin = origin
        self._extent = extent

    def __getitem__(self, index):
        if len(self._origin) == 0:
            if isinstance(index, tuple) and len(index) > 0:
                raise IndexError("more than one index given for a zero-dimension array")
            elif isinstance(index, slice) and index != slice(None, None, None):
                raise IndexError("cannot slice a zero-dimension array")
            else:
                return self._data  # array[()] does not return an ndarray
        else:
            return self._data[self._get_array_index(index)]

    def __setitem__(self, index, value):
        self._data[self._get_array_index(index)] = value

    def _get_array_index(self, index):
        if isinstance(index, list):
            index = tuple(index)
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(self._dims):
            raise IndexError(
                f"{len(index)} is too many indices for a "
                f"{len(self._dims)}-dimensional quantity"
            )
        if len(index) < len(self._dims):
            index = index + (slice(None, None),) * (len(self._dims) - len(index))
        return shift_boundary_slice_tuple(
            self._dims, self._origin, self._extent, self._boundary_type, index
        )

    def sel(self, **kwargs: Union[slice, int]) -> np.ndarray:
        """Convenience method to perform indexing using dimension names
        without knowing dimension order.

        Args:
            **kwargs: slice/index to retrieve for a given dimension name

        Returns:
            view_selection: an ndarray-like selection of the given indices
                on `self.view`
        """
        return self[tuple(kwargs.get(dim, slice(None, None)) for dim in self._dims)]


class BoundedArrayView:
    """
    A container of objects which provide indexing relative to corners and edges
    of the computational domain for convenience.

    Default start and end indices for all dimensions are modified to be the
    start and end of the compute domain. When using edge and corner attributes, it is
    recommended to explicitly write start and end offsets to avoid confusion.

    Indexing on the object itself (view[:]) is offset by the origin, and default
    start and end indices are modified to be the start and end of the compute domain.

    For corner attributes e.g. `northwest`, modified indexing is done for the two
    axes according to the edges which make up the corner. In other words, indexing
    is offset relative to the intersection of the two edges which make the corner.

    For `interior`, start indices of the horizontal dimensions are relative to the
    origin, and end indices are relative to the origin + extent. For example,
    view.interior[0:0, 0:0, :] would retrieve the entire compute domain for an x/y/z
    array, while view.interior[-1:1, -1:1, :] would also include one halo point.
    """

    def __init__(
        self, array, dims: Sequence[str], origin: Sequence[int], extent: Sequence[int]
    ):
        self._data = array
        self._dims = tuple(dims)
        self._origin = tuple(origin)
        self._extent = tuple(extent)
        self._northwest = BoundaryArrayView(
            array, constants.NORTHWEST, dims, origin, extent
        )
        self._northeast = BoundaryArrayView(
            array, constants.NORTHEAST, dims, origin, extent
        )
        self._southwest = BoundaryArrayView(
            array, constants.SOUTHWEST, dims, origin, extent
        )
        self._southeast = BoundaryArrayView(
            array, constants.SOUTHEAST, dims, origin, extent
        )
        self._interior = BoundaryArrayView(
            array, constants.INTERIOR, dims, origin, extent
        )

    @property
    def origin(self) -> Tuple[int, ...]:
        """the start of the computational domain"""
        return self._origin

    @property
    def extent(self) -> Tuple[int, ...]:
        """the shape of the computational domain"""
        return self._extent

    def __getitem__(self, index):
        if len(self.origin) == 0:
            if isinstance(index, tuple) and len(index) > 0:
                raise IndexError("more than one index given for a zero-dimension array")
            elif isinstance(index, slice) and index != slice(None, None, None):
                raise IndexError("cannot slice a zero-dimension array")
            else:
                return self._data  # array[()] does not return an ndarray
        else:
            return self._data[self._get_compute_index(index)]

    def __setitem__(self, index, value):
        self._data[self._get_compute_index(index)] = value

    def _get_compute_index(self, index):
        if not isinstance(index, (tuple, list)):
            index = (index,)
        if len(index) > len(self._dims):
            raise IndexError(
                f"{len(index)} is too many indices for a "
                f"{len(self._dims)}-dimensional quantity"
            )
        index = _fill_index(index, len(self._data.shape))
        shifted_index = []
        for entry, origin, extent in zip(index, self.origin, self.extent):
            if isinstance(entry, slice):
                shifted_slice = _shift_slice(entry, origin, extent)
                shifted_index.append(
                    bound_default_slice(shifted_slice, origin, origin + extent)
                )
            elif entry is None:
                shifted_index.append(entry)
            else:
                shifted_index.append(entry + origin)
        return tuple(shifted_index)

    @property
    def northwest(self) -> BoundaryArrayView:
        return self._northwest

    @property
    def northeast(self) -> BoundaryArrayView:
        return self._northeast

    @property
    def southwest(self) -> BoundaryArrayView:
        return self._southwest

    @property
    def southeast(self) -> BoundaryArrayView:
        return self._southeast

    @property
    def interior(self) -> BoundaryArrayView:
        return self._interior


def _fill_index(index, length):
    return tuple(index) + (slice(None, None, None),) * (length - len(index))


def _shift_slice(slice_in, shift, extent):
    start = _shift_index(slice_in.start, shift, extent)
    stop = _shift_index(slice_in.stop, shift, extent)
    return slice(start, stop, slice_in.step)


def _shift_index(current_value, shift, extent):
    if current_value is None:
        new_value = None
    else:
        new_value = current_value + shift
        if new_value < 0:
            new_value = extent + new_value
    return new_value
