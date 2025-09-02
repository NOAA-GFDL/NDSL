import dace.sdfg.analysis.schedule_tree.treenodes as dst


def _list_index(list: list[dst.ScheduleTreeNode], node: dst.ScheduleTreeNode) -> int:
    """Check if node is in list with "is" operator."""
    index = 0
    for element in list:
        # compare with "is" to get memory comparison. ".index()" uses value comparison
        if element is node:
            return index
        index += 1

    raise StopIteration


def _swap_node_position_in_tree(top_node, child_node):
    """Top node becomes child, child becomes top node"""
    # Take refs before swap
    top_children = top_node.parent.children
    top_level_parent = top_node.parent

    # Swap childrens
    top_node.children = child_node.children
    child_node.children = [top_node]
    top_children.insert(_list_index(top_children, top_node), child_node)

    # Re-parent
    top_node.parent = child_node
    child_node.parent = top_level_parent

    # Remove now-pushed original node
    top_children.remove(top_node)


class PushVerticalMapDown(dst.ScheduleNodeVisitor):
    def visit_MapScope(self, node: dst.MapScope):
        if node.node.map.params[0].startswith("__k"):
            # take refs before moving things around
            parent = node
            grandparent = node.parent
            grandparent_children = node.parent.children
            k_loop_index = _list_index(grandparent_children, parent)

            for child in node.children:
                if not isinstance(child, dst.MapScope):
                    raise NotImplementedError(
                        "We don't expect anything else than (IJ)-MapScopes here."
                    )

                # New loop with MapEntry (`node`) from parent and children from `child`
                new_loop = dst.MapScope(
                    node=parent.node,
                    children=child.children,
                )
                new_loop.parent = grandparent
                child.children = [new_loop]
                child.parent = grandparent
                grandparent_children.insert(k_loop_index, child)
                k_loop_index += 1

            # delete old (now unused) node
            grandparent_children.remove(node)
