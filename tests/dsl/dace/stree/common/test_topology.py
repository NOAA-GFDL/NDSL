import pytest
from dace import dtypes
from dace.data import Data
from dace.sdfg import nodes
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from ndsl.dsl.dace.stree.common.topology import (
    detect_cycle,
    get_next_node,
    get_previous_node,
    is_first_node,
    is_last_node,
    list_index,
    remove_from_tree,
    replace_node_in_tree,
    swap_node_position_in_tree,
)


@pytest.fixture
def schedule_tree() -> tn.ScheduleTreeRoot:
    data = Data(
        dtypes.float32,
        (3, 3, 3),
        False,
        dtypes.StorageType.Default,
        {},
        dtypes.AllocationLifetime.Global,
        dtypes.DebugInfo(0),
    )
    return tn.ScheduleTreeRoot(
        name="my_tree",
        containers={"field": data},
        children=[
            tn.MapScope(
                node=nodes.MapEntry(nodes.Map("map_i", ["__i"], [(0, 3, 1)])),
                children=[
                    tn.MapScope(
                        node=nodes.MapEntry(nodes.Map("map_j", ["__j"], [(0, 3, 1)])),
                        children=[
                            tn.MapScope(
                                node=nodes.MapEntry(
                                    nodes.Map("map_k", ["__k"], [(0, 3, 1)])
                                ),
                                children=[
                                    tn.TaskletNode(
                                        node=nodes.Tasklet("my_code"),
                                        in_memlets={},
                                        out_memlets={},
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
            tn.MapScope(
                node=nodes.MapEntry(nodes.Map("map2_i", ["__i"], [(0, 3, 1)])),
                children=[
                    tn.MapScope(
                        node=nodes.MapEntry(nodes.Map("map2_j", ["__j"], [(0, 3, 1)])),
                        children=[
                            tn.MapScope(
                                node=nodes.MapEntry(
                                    nodes.Map("map2_k", ["__k"], [(0, 3, 1)])
                                ),
                                children=[
                                    tn.TaskletNode(
                                        node=nodes.Tasklet("my_code"),
                                        in_memlets={},
                                        out_memlets={},
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    )


@pytest.fixture
def bad_schedule_tree() -> tn.ScheduleTreeRoot:
    data = Data(
        dtypes.float32,
        (3, 3, 3),
        False,
        dtypes.StorageType.Default,
        {},
        dtypes.AllocationLifetime.Global,
        dtypes.DebugInfo(0),
    )
    # Bad parent/child
    map_cycling_i = tn.MapScope(
        node=nodes.MapEntry(nodes.Map("map_i", ["__i"], [(0, 3, 1)])), children=[]
    )
    map_cycling_j = tn.MapScope(
        node=nodes.MapEntry(nodes.Map("map_i", ["__i"], [(0, 3, 1)])),
        children=[map_cycling_i],
    )
    map_cycling_i.children.append(map_cycling_j)
    return tn.ScheduleTreeRoot(
        name="my_faulty_tree", containers={"field": data}, children=[map_cycling_i]
    )


def test_swap_position_in_tree(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert schedule_tree.children[0].node.label == "map_i"
    assert schedule_tree.children[0].children[0].node.label == "map_j"

    swap_node_position_in_tree(
        schedule_tree.children[0],
        schedule_tree.children[0].children[0],
    )

    assert schedule_tree.children[0].node.label == "map_j"
    assert schedule_tree.children[0].children[0].node.label == "map_i"


def test_replace_node_in_tree(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert schedule_tree.children[0].node.label == "map_i"
    assert schedule_tree.children[1].node.label == "map2_i"

    old_map = schedule_tree.children[0]
    replace_node_in_tree(
        old_map,
        schedule_tree.children[1],
    )

    assert old_map.parent is None
    assert schedule_tree.children[0].node.label == "map2_i"
    assert schedule_tree.children[1].node.label == "map2_i"


def test_detect_cycle(
    schedule_tree: tn.ScheduleTreeRoot, bad_schedule_tree: tn.ScheduleTreeRoot
):
    assert not detect_cycle(schedule_tree.children, set())
    with pytest.raises(ValueError, match="Cycle detected"):
        assert detect_cycle(bad_schedule_tree.children, set())


def test_list_index(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert list_index(schedule_tree.children[1]) == 1
    assert list_index(schedule_tree.children[1].children[0]) == 0


def test_get_previous_node(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert get_previous_node(schedule_tree.children[1]) == schedule_tree.children[0]
    assert get_previous_node(schedule_tree.children[0]) is None


def test_get_next_node(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert get_next_node(schedule_tree.children[1]) is None
    assert get_next_node(schedule_tree.children[0]) == schedule_tree.children[1]


def test_is_last_node(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert is_last_node(schedule_tree.children[1])
    assert not is_last_node(schedule_tree.children[0])


def test_is_first_node(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert not is_first_node(schedule_tree.children[1])
    assert is_first_node(schedule_tree.children[0])


def test_remove_from_tree(schedule_tree: tn.ScheduleTreeRoot) -> None:
    assert len(schedule_tree.children) == 2
    to_remove = schedule_tree.children[1]
    remove_from_tree(to_remove)
    assert to_remove.parent is None
    assert len(schedule_tree.children) == 1
