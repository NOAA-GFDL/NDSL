from typing import Collection

import dace.sdfg.analysis.schedule_tree.treenodes as tn


def swap_node_position_in_tree(
    top_node: tn.ScheduleTreeScope, child_node: tn.ScheduleTreeScope
) -> None:
    """Top node becomes child, child becomes top node."""
    # Ensue parent/children relationship is valid
    tn.validate_children_and_parents_align(top_node)

    # Take refs before swap
    top_children = top_node.parent.children
    top_level_parent = top_node.parent

    # Swap children
    top_node.children = child_node.children
    child_node.children = [top_node]
    top_children.insert(list_index(top_children, top_node), child_node)

    # Re-parent
    top_node.parent = child_node
    child_node.parent = top_level_parent

    # Remove now-pushed original node
    top_children.remove(top_node)

    # Reset parent/child relationship
    for child in top_node.children:
        child.parent = top_node
    for child in child_node.children:
        child.parent = child_node


def detect_cycle(nodes: list[tn.ScheduleTreeNode], visited: set) -> None:
    """Detect the cycles in the tree."""
    # Dev note: isn't there a DaCe tool for this?!
    for node in nodes:
        if id(node) in visited:
            breakpoint()
        visited.add(id(node))
        if isinstance(node, tn.ScheduleTreeScope):
            detect_cycle(node.children, visited)


def list_index(
    collection: Collection[tn.ScheduleTreeNode],
    node: tn.ScheduleTreeNode,
) -> int:
    """Check if node is in list with "is" operator."""
    # compare with "is" to get memory comparison. ".index()" uses value comparison
    return next(index for index, element in enumerate(collection) if element is node)
