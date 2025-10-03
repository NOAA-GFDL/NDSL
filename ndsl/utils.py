from enum import EnumMeta
from pathlib import Path
from typing import Iterable, Sequence, Tuple, TypeVar, Union

import f90nml
import numpy as np

import ndsl.constants as constants
from ndsl.optional_imports import cupy as cp
from ndsl.types import Allocator


# Run a deviceSynchronize() to check
# that the GPU is present and ready to run
if cp is not None:
    try:
        cp.cuda.runtime.deviceSynchronize()
        GPU_AVAILABLE = True
    except cp.cuda.runtime.CUDARuntimeError:
        GPU_AVAILABLE = False
else:
    GPU_AVAILABLE = False

T = TypeVar("T")


class MetaEnumStr(EnumMeta):
    def __contains__(cls, item) -> bool:
        return item in cls.__members__.keys()


def list_by_dims(
    dims: Sequence[str], horizontal_list: Sequence[T], non_horizontal_value: T
) -> Tuple[T, ...]:
    """Take in a list of dimensions, a (y, x) set of values, and a value for any
    non-horizontal dimensions. Return a list of length len(dims) with the value for
    each dimension.
    """
    return_list = []
    for dim in dims:
        if dim in constants.Y_DIMS:
            return_list.append(horizontal_list[0])
        elif dim in constants.X_DIMS:
            return_list.append(horizontal_list[1])
        else:
            return_list.append(non_horizontal_value)
    return tuple(return_list)


def is_contiguous(array: np.ndarray) -> bool:
    return array.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]


def is_c_contiguous(array: np.ndarray) -> bool:
    return array.flags["C_CONTIGUOUS"]


def ensure_contiguous(maybe_array: Union[np.ndarray, None]) -> None:
    if maybe_array is not None and not is_contiguous(maybe_array):
        raise BufferError("dlpack: buffer is not contiguous")


def safe_assign_array(to_array: np.ndarray, from_array: np.ndarray):
    """Failproof assignment for array on different devices.

    The memory will be downloaded/uploaded from GPU if need be.

    Args:
        to_array: destination ndarray
        from_array: source ndarray
    """
    try:
        to_array[:] = from_array
    except (ValueError, TypeError):
        if cp and isinstance(to_array, cp.ndarray):
            to_array[:] = cp.asarray(from_array)
        elif cp and isinstance(from_array, cp.ndarray):
            to_array[:] = cp.asnumpy(from_array)
        else:
            raise


def device_synchronize():
    """Synchronize all memory communication"""
    if GPU_AVAILABLE:
        cp.cuda.runtime.deviceSynchronize()


def safe_mpi_allocate(
    allocator: Allocator, shape: Iterable[int], dtype: type
) -> np.ndarray:
    """Make sure the allocation use an allocator that works with MPI

    For G2G transfer, MPICH requires the allocation to not be done
    with managed memory. Since we can't know what state `cupy` is in
    with switch for the default pooled allocator.

    If allocator comes from cupy, it must be cupy.empty or cupy.zeros.
    We raise a RuntimeError if a cupy array is allocated outside of
    the safe code path.

    Though the allocation _might_ be safe, the MPI crash that result
    from a managed memory allocation is non trivial and should be
    tightly controlled.
    """
    if cp and (allocator is cp.empty or allocator is cp.zeros):
        original_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
        array = allocator(shape, dtype=dtype)  # type: np.ndarray
        cp.cuda.set_allocator(original_allocator)
    else:
        array = allocator(shape, dtype=dtype)
        if __debug__ and cp and isinstance(array, cp.ndarray):
            raise RuntimeError("cupy allocation might not be MPI-safe")
    return array


########################################################
# Helpers for loading and working with Fortran Namelists
# TODO: Consider moving these to a separate utils/namelist.py

DEFAULT_GRID_NML_GROUPS = ["fv_core_nml"]


def flatten_nml_to_dict(nml: f90nml.Namelist) -> dict:
    """Returns a flattened dict version of a f90nml.namelist.Namelist

    Args:
        nml: f90nml.Namelist
    """
    nml_dict = dict(nml)
    for name, value in nml_dict.items():
        if isinstance(value, f90nml.Namelist):
            nml_dict[name] = flatten_nml_to_dict(value)
    flatter_namelist = {}
    for key, value in nml_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey in flatter_namelist:
                    raise ValueError(
                        "Cannot flatten this namelist, duplicate keys: " + subkey
                    )
                flatter_namelist[subkey] = subvalue
        else:
            flatter_namelist[key] = value
    return flatter_namelist


def load_f90nml(namelist_path: Path) -> f90nml.Namelist:
    """Loads a Fortran namelist given its path and return a f90nml.Namelist

    Args:
        namelist_path: Path to the Fortran namelist file
    """
    return f90nml.read(namelist_path)


def load_f90nml_as_dict(
    namelist_path: Path,
    flatten: bool = True,
    target_groups=None,
) -> dict:
    """Loads a Fortran namelist given its path and returns a
    dict representation. If target_groups are specified, then
    the dict is created using only those groups.

    Args:
        namelist_path: Path to the Fortran namelist file
        flatten: If True, flattens the loaded namelist (without groups) before
                 returning it. (Default: True) Otherwise, it returns the f90nml.Namelist
                 dict representation.
        target_groups: If 'None' is specified, then all groups are
                       considered. (Default: None) Otherwise, only parameters
                       from the specified groups are considered.
    """
    nml = load_f90nml(namelist_path)
    return f90nml_as_dict(nml, flatten=flatten, target_groups=target_groups)


def f90nml_as_dict(
    nml: f90nml.Namelist,
    flatten: bool = True,
    target_groups=None,
) -> dict:
    """Uses a f90nml.Namelist and returns a dict representation.
    If target_groups are specified, then the dict is created using only those
    groups. The return dicts can be flattened further to remove the group
    information or keep the group information.

    Args:
        namelist_path: Path to the Fortran namelist file
        flatten: If True, flattens the loaded namelist (without groups) before
                 returning it. (Default: True) Otherwise, it returns the f90nml.Namelist
                 dict representation.
        target_groups: If 'None' is specified, then all groups are
                       considered. (Default: None) Otherwise, only parameters
                       from the specified groups are considered.
    """
    if target_groups is not None:
        extracted_groups = f90nml.Namelist()
        for group in target_groups:
            if group in nml:
                extracted_groups[group] = nml[group]
    else:
        extracted_groups = nml

    if flatten:
        return flatten_nml_to_dict(extracted_groups)
    return extracted_groups.todict()


def grid_params_from_f90nml(nml):
    return f90nml_as_dict(nml, flatten=True, target_groups=DEFAULT_GRID_NML_GROUPS)
