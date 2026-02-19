"""Test of the GPU to GPU communication strategy.

Those test use halo_update but are separated from the entire
"""

import contextlib
import functools

import numpy as np
import pytest

from ndsl import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    LocalComm,
    Quantity,
    TilePartitioner,
)
from ndsl.config import Backend
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.optional_imports import cupy as cp
from ndsl.performance import Timer


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request) -> tuple[int, int]:
    return request.param


@pytest.fixture
def tile_partitioner(layout: tuple[int, int]) -> TilePartitioner:
    return TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner: TilePartitioner) -> CubedSpherePartitioner:
    return CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def cpu_communicators(
    cube_partitioner: CubedSpherePartitioner,
) -> list[CubedSphereCommunicator]:
    shared_buffer = {}
    communicators = []
    for rank in range(cube_partitioner.total_ranks):
        communicators.append(
            CubedSphereCommunicator(
                comm=LocalComm(
                    rank=rank,
                    total_ranks=cube_partitioner.total_ranks,
                    buffer_dict=shared_buffer,
                ),
                force_cpu=True,
                partitioner=cube_partitioner,
                timer=Timer(),
            )
        )
    return communicators


@pytest.fixture
def gpu_communicators(
    cube_partitioner: CubedSpherePartitioner,
) -> list[CubedSphereCommunicator]:
    shared_buffer = {}
    communicators = []
    for rank in range(cube_partitioner.total_ranks):
        communicators.append(
            CubedSphereCommunicator(
                comm=LocalComm(
                    rank=rank,
                    total_ranks=cube_partitioner.total_ranks,
                    buffer_dict=shared_buffer,
                ),
                partitioner=cube_partitioner,
                force_cpu=False,
                timer=Timer(),
            )
        )
    return communicators


# To record the calls to cp.ZEROS/np.ZEROS we use a global
# dict indexed on the functions
global N_ZEROS_CALLS
N_ZEROS_CALLS = {}


@contextlib.contextmanager
def module_count_calls_to_zeros(module):
    # case: cupy is not installed in a cpu-only environment
    if module is None:
        yield
        return

    def count_calls(func):
        """Count function calls"""

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            global N_ZEROS_CALLS  # noqa: F824 global ... is unused
            N_ZEROS_CALLS[func] += 1
            return func(*args, **kwargs)

        return wrapped

    global N_ZEROS_CALLS  # noqa: F824 global ... is unused
    N_ZEROS_CALLS[module.zeros] = 0

    try:
        original = module.zeros
        module.zeros = count_calls(module.zeros)
        yield
    finally:
        module.zeros = original


@pytest.mark.gpu
@pytest.mark.parallel
def test_halo_update_only_communicate_on_gpu(
    gpu_communicators: list[CubedSphereCommunicator],
) -> None:
    with module_count_calls_to_zeros(np), module_count_calls_to_zeros(cp):
        shape = (10, 10, 79)
        dims = (I_DIM, J_DIM, K_DIM)
        data = cp.ones(shape, dtype=float)
        quantity = Quantity(
            data,
            dims=dims,
            units="m",
            origin=(3, 3, 1),
            extent=(3, 3, 1),
            backend=Backend("st:gt:gpu:KJI"),
        )
        halo_updater_list = []
        for communicator in gpu_communicators:
            halo_updater = communicator.start_halo_update(quantity, 3)
            halo_updater_list.append(halo_updater)
        for halo_updater in halo_updater_list:
            halo_updater.wait()

    # We expect no np calls and several cp calls
    global N_ZEROS_CALLS  # noqa: F824 global ... is unused
    assert N_ZEROS_CALLS[cp.zeros] > 0
    assert N_ZEROS_CALLS[np.zeros] == 0


@pytest.mark.parallel
def test_halo_update_communicate_though_cpu(
    cpu_communicators: list[CubedSphereCommunicator],
) -> None:
    with module_count_calls_to_zeros(np), module_count_calls_to_zeros(cp):
        shape = (10, 10, 79)
        data = np.ones(shape, dtype=float)
        quantity = Quantity(
            data,
            dims=(
                I_DIM,
                J_DIM,
                K_DIM,
            ),
            units="m",
            origin=(3, 3, 0),
            extent=(3, 3, 0),
            backend=Backend("st:numpy:cpu:IJK"),
        )
        halo_updater_list = []
        for communicator in cpu_communicators:
            halo_updater = communicator.start_halo_update(quantity, 3)
            halo_updater_list.append(halo_updater)
        for halo_updater in halo_updater_list:
            halo_updater.wait()

    # We expect several np calls and several cp calls
    global N_ZEROS_CALLS  # noqa: F824 global ... is unused
    assert N_ZEROS_CALLS[np.zeros] > 0
    assert len(N_ZEROS_CALLS) == 1 or N_ZEROS_CALLS[cp.zeros] == 0, (
        "no calls to cupy.zeros logged"
    )
