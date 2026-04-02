from dace import nodes
from dace.properties import CodeBlock
from dace.sdfg.analysis.schedule_tree import treenodes as tn
from dace.sdfg.state import LoopRegion
from dace.subsets import Range

from ndsl.dsl.dace.stree.optimizations import CleanUpScheduleTree


def test_if_scope() -> None:
    stree = tn.ScheduleTreeRoot(
        name="tester",
        children=[
            tn.IfScope(
                condition=CodeBlock("True"),
                children=[tn.StateBoundaryNode()],
            ),
        ],
    )

    cleaner = CleanUpScheduleTree()
    cleaner.visit(stree)

    assert [type(node) for node in stree.children] == [tn.IfScope]
    assert len(stree.children[0].children) == 0


def test_for_scope() -> None:
    stree = tn.ScheduleTreeRoot(
        name="tester",
        children=[
            tn.ForScope(
                loop=LoopRegion("test"),
                children=[tn.StateBoundaryNode()],
            ),
        ],
    )

    cleaner = CleanUpScheduleTree()
    cleaner.visit(stree)

    assert [type(node) for node in stree.children] == [tn.ForScope]
    assert len(stree.children[0].children) == 0


def test_while_scope() -> None:
    stree = tn.ScheduleTreeRoot(
        name="tester",
        children=[
            tn.WhileScope(
                loop=LoopRegion("test"),
                children=[tn.StateBoundaryNode()],
            ),
        ],
    )

    cleaner = CleanUpScheduleTree()
    cleaner.visit(stree)

    assert [type(node) for node in stree.children] == [tn.WhileScope]
    assert len(stree.children[0].children) == 0


def test_map_scope() -> None:
    stree = tn.ScheduleTreeRoot(
        name="tester",
        children=[
            tn.MapScope(
                node=nodes.MapEntry(map=nodes.Map("asdf", ["i"], Range([]))),
                children=[tn.StateBoundaryNode()],
            ),
        ],
    )

    cleaner = CleanUpScheduleTree()
    cleaner.visit(stree)

    assert [type(node) for node in stree.children] == [tn.MapScope]
    assert len(stree.children[0].children) == 0
