from ndsl import TilePartitioner
from ndsl.dsl.caches import FV3CodePath, identify_code_path


def test_single_code_path() -> None:
    partitioner = TilePartitioner((1, 1))
    for rank in range(0, 6):
        assert (
            identify_code_path(rank, partitioner, single_code_path=True)
            == FV3CodePath.All
        )

    partitioner = TilePartitioner((2, 2))
    for rank in range(0, 24):
        assert (
            identify_code_path(rank, partitioner, single_code_path=True)
            == FV3CodePath.All
        )

    partitioner = TilePartitioner((3, 3))
    for rank in range(0, 54):
        assert (
            identify_code_path(rank, partitioner, single_code_path=True)
            == FV3CodePath.All
        )


def test_1x1_layout() -> None:
    partitioner = TilePartitioner((1, 1))
    for rank in range(0, 6):
        assert identify_code_path(rank, partitioner) == FV3CodePath.All


def test_2x2_layout() -> None:
    partitioner = TilePartitioner((2, 2))
    for rank in range(0, 24):
        match rank % 4:
            case 0:
                assert identify_code_path(rank, partitioner) == FV3CodePath.BottomLeft
            case 1:
                assert identify_code_path(rank, partitioner) == FV3CodePath.BottomRight
            case 2:
                assert identify_code_path(rank, partitioner) == FV3CodePath.TopLeft
            case 3:
                assert identify_code_path(rank, partitioner) == FV3CodePath.TopRight


def test_3x3_layout() -> None:
    partitioner = TilePartitioner((3, 3))
    for rank in range(0, 54):
        match rank % 9:
            case 0:
                assert identify_code_path(rank, partitioner) == FV3CodePath.BottomLeft
            case 1:
                assert identify_code_path(rank, partitioner) == FV3CodePath.Bottom
            case 2:
                assert identify_code_path(rank, partitioner) == FV3CodePath.BottomRight
            case 3:
                assert identify_code_path(rank, partitioner) == FV3CodePath.Left
            case 4:
                assert identify_code_path(rank, partitioner) == FV3CodePath.Center
            case 5:
                assert identify_code_path(rank, partitioner) == FV3CodePath.Right
            case 6:
                assert identify_code_path(rank, partitioner) == FV3CodePath.TopLeft
            case 7:
                assert identify_code_path(rank, partitioner) == FV3CodePath.Top
            case 8:
                assert identify_code_path(rank, partitioner) == FV3CodePath.TopRight
