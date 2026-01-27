import pytest

from ndsl import LocalComm, Quantity, TileCommunicator, TilePartitioner
from ndsl.constants import I_DIM, I_INTERFACE_DIM, J_DIM, J_INTERFACE_DIM


def rank_scatter_results(communicator_list, quantity):
    for rank, tile_communicator in enumerate(communicator_list):
        if rank == 0:
            array = quantity
        else:
            array = None
        yield (tile_communicator, tile_communicator.scatter(send_quantity=array))


def get_tile_communicator_list(partitioner):
    total_ranks = partitioner.total_ranks
    shared_buffer = {}
    tile_communicator_list = []
    for rank in range(total_ranks):
        tile_communicator_list.append(
            TileCommunicator(
                comm=LocalComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=partitioner,
            )
        )
    return tile_communicator_list


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
def test_interface_state_two_by_two_per_rank_scatter_tile(layout, numpy):
    state = {
        "pos_j": Quantity(
            numpy.empty([layout[0] + 1, layout[1] + 1]),
            dims=[J_INTERFACE_DIM, I_INTERFACE_DIM],
            units="dimensionless",
            backend="debug",
        ),
        "pos_i": Quantity(
            numpy.empty([layout[0] + 1, layout[1] + 1], dtype=numpy.int32),
            dims=[J_INTERFACE_DIM, I_INTERFACE_DIM],
            units="dimensionless",
            backend="debug",
        ),
    }

    state["pos_j"].view[:, :] = numpy.arange(0, layout[0] + 1)[:, None]
    state["pos_i"].view[:, :] = numpy.arange(0, layout[1] + 1)[None, :]

    partitioner = TilePartitioner(layout)
    tile_communicator_list = get_tile_communicator_list(partitioner)
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["pos_j"]
    ):
        assert rank_array.extent == (2, 2)
        j, i = partitioner.subtile_index(communicator.rank)
        assert rank_array.view[0, 0] == j
        assert rank_array.view[0, 1] == j
        assert rank_array.view[1, 0] == j + 1
        assert rank_array.view[1, 1] == j + 1
        assert rank_array.data.dtype == state["pos_j"].data.dtype

    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["pos_i"]
    ):
        assert rank_array.extent == (2, 2)
        j, i = partitioner.subtile_index(communicator.rank)
        assert rank_array.view[0, 0] == i
        assert rank_array.view[1, 0] == i
        assert rank_array.view[0, 1] == i + 1
        assert rank_array.view[1, 1] == i + 1
        assert rank_array.data.dtype == state["pos_i"].data.dtype


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
def test_centered_state_one_item_per_rank_scatter_tile(layout, numpy):
    total_ranks = layout[0] * layout[1]
    state = {
        "rank": Quantity(
            numpy.empty([layout[0], layout[1]]),
            dims=[J_DIM, I_DIM],
            units="dimensionless",
            backend="debug",
        ),
        "rank_pos_j": Quantity(
            numpy.empty([layout[0], layout[1]]),
            dims=[J_DIM, I_DIM],
            units="dimensionless",
            backend="debug",
        ),
        "rank_pos_i": Quantity(
            numpy.empty([layout[0], layout[1]]),
            dims=[J_DIM, I_DIM],
            units="dimensionless",
            backend="debug",
        ),
    }

    partitioner = TilePartitioner(layout)
    for rank in range(total_ranks):
        rank = numpy.asarray([rank])
        state["rank"].view[numpy.unravel_index(rank, state["rank"].extent)] = rank
        j, i = partitioner.subtile_index(rank)
        state["rank_pos_j"].view[
            numpy.unravel_index(rank, state["rank_pos_j"].extent)
        ] = j
        state["rank_pos_i"].view[
            numpy.unravel_index(rank, state["rank_pos_i"].extent)
        ] = i

    partitioner = TilePartitioner(layout)
    tile_communicator_list = get_tile_communicator_list(partitioner)
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["rank"]
    ):
        assert rank_array.extent == (1, 1)
        assert rank_array.view[0, 0] == communicator.rank
        assert rank_array.data.dtype == state["rank"].data.dtype
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["rank_pos_j"]
    ):
        assert rank_array.extent == (1, 1)
        assert rank_array.view[0, 0] == partitioner.subtile_index(communicator.rank)[0]
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["rank_pos_i"]
    ):
        assert rank_array.extent == (1, 1)
        assert rank_array.view[0, 0] == partitioner.subtile_index(communicator.rank)[1]


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
@pytest.mark.parametrize("n_halo", [0, 1, 3])
def test_centered_state_one_item_per_rank_with_halo_scatter_tile(layout, n_halo, numpy):
    extent = layout
    total_ranks = layout[0] * layout[1]
    state = {
        "rank": Quantity(
            numpy.empty([layout[0] + 2 * n_halo, layout[1] + 2 * n_halo]),
            dims=[J_DIM, I_DIM],
            units="dimensionless",
            origin=(n_halo, n_halo),
            extent=extent,
            backend="debug",
        ),
        "rank_pos_j": Quantity(
            numpy.empty([layout[0] + 2 * n_halo, layout[1] + 2 * n_halo]),
            dims=[J_DIM, I_DIM],
            units="dimensionless",
            origin=(n_halo, n_halo),
            extent=extent,
            backend="debug",
        ),
        "rank_pos_i": Quantity(
            numpy.empty([layout[0] + 2 * n_halo, layout[1] + 2 * n_halo]),
            dims=[J_DIM, I_DIM],
            units="dimensionless",
            origin=(n_halo, n_halo),
            extent=extent,
            backend="debug",
        ),
    }

    partitioner = TilePartitioner(layout)
    for rank in range(total_ranks):
        rank = numpy.asarray([rank])
        state["rank"].view[numpy.unravel_index(rank, state["rank"].extent)] = rank
        j, i = partitioner.subtile_index(rank)
        state["rank_pos_j"].view[
            numpy.unravel_index(rank, state["rank_pos_j"].extent)
        ] = j
        state["rank_pos_i"].view[
            numpy.unravel_index(rank, state["rank_pos_i"].extent)
        ] = i

    tile_communicator_list = get_tile_communicator_list(partitioner)
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["rank"]
    ):
        assert rank_array.extent == (1, 1)
        assert rank_array.data[0, 0] == communicator.rank
        assert rank_array.data.dtype == state["rank"].data.dtype
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["rank_pos_j"]
    ):
        assert rank_array.extent == (1, 1)
        assert rank_array.view[0, 0] == partitioner.subtile_index(communicator.rank)[0]
    for communicator, rank_array in rank_scatter_results(
        tile_communicator_list, state["rank_pos_i"]
    ):
        assert rank_array.extent == (1, 1)
        assert rank_array.view[0, 0] == partitioner.subtile_index(communicator.rank)[1]
