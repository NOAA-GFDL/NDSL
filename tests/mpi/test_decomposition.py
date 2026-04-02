from unittest.mock import MagicMock

import pytest

from ndsl import MPIComm
from ndsl.comm.decomposition import block_waiting_for_compilation, unblock_waiting_tiles


@pytest.mark.parallel
def test_unblock_waiting_tiles():
    comm = MPIComm()
    compilation_config = MagicMock(compiling_equivalent=0)

    if comm.Get_rank() == 0:
        unblock_waiting_tiles(comm._comm)
    else:
        block_waiting_for_compilation(comm._comm, compilation_config)
