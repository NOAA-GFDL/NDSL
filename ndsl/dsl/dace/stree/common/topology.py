from collections.abc import Collection

import dace.sdfg.analysis.schedule_tree.treenodes as tn


def swap_node_position_in_tree(
    top_node: tn.ScheduleTreeScope, child_node: tn.ScheduleTreeScope
) -> None:
    """Top node becomes child, child becomes top node."""
    # Ensue parent/children relationship is valid
    tn.validate_children_and_parents_align(top_node)
    assert top_node.parent is not None

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


def swap_node_in_tree(
    old_node: tn.ScheduleTreeNode, new_node: tn.ScheduleTreeNode
) -> None:
    """
    Replace `old_node`  with `new_node`  in the children of the old nodes' parent.

    Used when children (downstream) changes that cannot be covered with the ScheduleNodeTransformer.
    """

    assert old_node.parent and old_node.parent.children
    index = list_index(old_node.parent.children, old_node)
    old_node.parent.children[index] = new_node
    new_node.parent = old_node.parent
    old_node.parent = None


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


def get_previous_node(
    nodes: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode
) -> tn.ScheduleTreeNode | None:
    """Get previous node in the children, return None if first node"""
    index = list_index(nodes, node)
    if index == 0:
        return None
    return nodes[index - 1]


def get_next_node(
    nodes: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode
) -> tn.ScheduleTreeNode | None:
    """Get next node in the children from given node, return None if last node"""
    index = list_index(nodes, node)
    if index == len(nodes) - 1:
        return None
    return nodes[index + 1]


def is_last_node(nodes: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode) -> bool:
    """Check if the node is the last node of the list."""
    return list_index(nodes, node) >= len(nodes) - 1


def remove_from_tree(node: tn.ScheduleTreeNode) -> None:
    """Remove a node from the tree. DO NOT take care of children of to-be-delete node"""
    if node.parent:
        node.parent.children.remove(node)
        node.parent = None
