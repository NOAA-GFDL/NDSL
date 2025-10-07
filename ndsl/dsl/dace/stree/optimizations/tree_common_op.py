from typing import Collection

import dace.sdfg.analysis.schedule_tree.treenodes as dst


def swap_node_position_in_tree(top_node, child_node):
    """Top node becomes child, child becomes top node"""
    # Take refs before swap
    top_children = top_node.parent.children
    top_level_parent = top_node.parent

    # Swap childrens
    top_node.children = child_node.children
    child_node.children = [top_node]
    top_children.insert(list_index(top_children, top_node), child_node)

    # Re-parent
    top_node.parent = child_node
    child_node.parent = top_level_parent

    # Remove now-pushed original node
    top_children.remove(top_node)


def detect_cycle(nodes, visited: set):
    """Detect the cycles in the tree."""
    # Dev note: isn't there a DaCe tool for this?!
    for n in nodes:
        if id(n) in visited:
            breakpoint()
        visited.add(id(n))
        if hasattr(n, "children"):
            detect_cycle(n.children, visited)


def list_index(
    collection: Collection[dst.ScheduleTreeNode],
    node: dst.ScheduleTreeNode,
) -> int:
    """Check if node is in list with "is" operator."""
    # compare with "is" to get memory comparison. ".index()" uses value comparison
    return next(index for index, element in enumerate(collection) if element is node)
