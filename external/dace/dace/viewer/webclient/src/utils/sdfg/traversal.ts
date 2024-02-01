// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import {
    Edge as DagreEdge
} from 'dagre';
import {
    Edge,
    EntryNode,
    ExitNode,
    SDFGNode,
} from '../../renderer/renderer_elements';
import {
    DagreSDFG,
    JsonSDFG,
    JsonSDFGEdge,
    JsonSDFGNode,
    JsonSDFGState,
} from '../../index';

/**
 * Receives a callback that accepts (node, parent graph) and returns a value.
 * This function is invoked recursively per scope (including scope nodes),
 * unless the return value is false, upon which the sub-scope will not be
 * visited. The function also accepts an optional post-subscope callback (same
 * signature as `func`).
 **/
 export function traverse_sdfg_scopes(
     sdfg: DagreSDFG,
     func: CallableFunction,
     post_subscope_func: CallableFunction | null = null
): void {
    function scopes_recursive(
        graph: DagreSDFG,
        nodes: any[],
        processed_nodes: Set<string> | null = null
    ): void {
        if (processed_nodes === null)
            processed_nodes = new Set();

        for (const nodeid of nodes) {
            const node = graph.node(nodeid);

            if (node === undefined || processed_nodes.has(node.id?.toString()))
                continue;

            // Invoke function
            const result = func(node, graph);

            // Skip in case of e.g., collapsed nodes
            if (result !== false) {
                // Traverse scopes recursively (if scope_dict provided)
                if (node.type().endsWith('Entry') && node.parent_id !== null &&
                    node.id !== null) {
                    const state = node.sdfg.nodes[node.parent_id];
                    if (state.scope_dict[node.id] !== undefined)
                        scopes_recursive(
                            graph, state.scope_dict[node.id], processed_nodes
                        );
                }

                // Traverse states or nested SDFGs
                if (node.data.graph) {
                    const state = node.data.state;
                    if (state !== undefined &&
                        state.scope_dict[-1] !== undefined)
                        scopes_recursive(node.data.graph, state.scope_dict[-1]);
                    else // No scope_dict, traverse all nodes as a flat hierarchy
                        scopes_recursive(
                            node.data.graph, node.data.graph.nodes()
                        );
                }
            }

            if (post_subscope_func)
                post_subscope_func(node, graph);

            processed_nodes.add(node.id?.toString());
        }
    }
    scopes_recursive(sdfg, sdfg.nodes());
}



/**
 * Returns a partial memlet tree from a given edge, from the root node
 * through all children (without siblings). Calling this function with
 * the root edge returns the entire memlet tree.
 **/
export function memlet_tree(
    graph: DagreSDFG,
    edge: Edge,
    root_only: boolean = false
): any[] {
    const result = [];
    const graph_edges: any = {};
    graph.edges().forEach((e: DagreEdge) => {
        if (e.name)
            graph_edges[e.name] = e;
    });


    function src(e: any): SDFGNode {
        const ge = graph_edges[e.id];
        return graph.node(ge.v);
    }
    function dst(e: any): SDFGNode {
        const ge = graph_edges[e.id];
        return graph.node(ge.w);
    }

    // Determine direction
    let propagate_forward = false, propagate_backward = false;
    if ((edge.src_connector && src(edge) instanceof EntryNode) ||
        (edge.dst_connector && dst(edge) instanceof EntryNode &&
            edge.dst_connector.startsWith('IN_')))
        propagate_forward = true;
    if ((edge.src_connector && src(edge) instanceof ExitNode) ||
        (edge.dst_connector && dst(edge) instanceof ExitNode))
        propagate_backward = true;

    result.push(edge);

    // If either both are false (no scopes involved) or both are true
    // (invalid SDFG), we return only the current edge as a degenerate tree
    if (propagate_forward == propagate_backward)
        return result;

    // Ascend (find tree root) while prepending
    let curedge: any = edge;
    if (propagate_forward) {
        let source = src(curedge);
        while (source instanceof EntryNode && curedge && curedge.src_connector) {
            if (source.attributes().is_collapsed)
                break;

            const cname = curedge.src_connector.substring(4);  // Remove OUT_
            curedge = null;
            graph.inEdges(source.id.toString())?.forEach(e => {
                const ge = graph.edge(e);
                if (ge.dst_connector == 'IN_' + cname)
                    curedge = ge;
            });
            if (curedge) {
                result.unshift(curedge);
                source = src(curedge);
            }
        }
    } else if (propagate_backward) {
        let dest = dst(curedge);
        while (dest instanceof ExitNode && curedge && curedge.dst_connector) {
            const cname = curedge.dst_connector.substring(3);  // Remove IN_
            curedge = null;
            graph.outEdges(dest.id.toString())?.forEach(e => {
                const ge = graph.edge(e);
                if (ge.src_connector == 'OUT_' + cname)
                    curedge = ge;
            });
            if (curedge) {
                result.unshift(curedge);
                dest = dst(curedge);
            }
        }
    }

    if (root_only)
        return [result[0]];

    // Descend recursively
    function add_children(edge: any) {
        const children: any[] = [];
        if (propagate_forward) {
            const next_node = dst(edge);
            if (!(next_node instanceof EntryNode) ||
                !edge.dst_connector || !edge.dst_connector.startsWith('IN_'))
                return;
            if (next_node.attributes().is_collapsed)
                return;
            const conn = edge.dst_connector.substring(3);
            graph.outEdges(next_node.id.toString())?.forEach(e => {
                const ge = graph.edge(e);
                if (ge.src_connector == 'OUT_' + conn) {
                    children.push(ge);
                    result.push(ge);
                }
            });
        } else if (propagate_backward) {
            const next_node = src(edge);
            if (!(next_node instanceof ExitNode) || !edge.src_connector)
                return;
            const conn = edge.src_connector.substring(4);
            graph.inEdges(next_node.id.toString())?.forEach(e => {
                const ge = graph.edge(e);
                if (ge.dst_connector == 'IN_' + conn) {
                    children.push(ge);
                    result.push(ge);
                }
            });
        }

        for (const child of children)
            add_children(child);
    }

    // Start from current edge
    add_children(edge);

    return result;
}

/**
 * Returns a partial memlet tree from a given edge. It descends into nested SDFGs.
 * @param visited_edges is used to speed up the computation of the memlet trees
 **/
export function memlet_tree_nested(
    sdfg: JsonSDFG,
    state: JsonSDFGState,
    edge: JsonSDFGEdge,
    visited_edges: JsonSDFGEdge[] = []
): any[] {
    if (visited_edges.includes(edge) ||
        edge.attributes.data.attributes.shortcut) {
        return [];
    }

    visited_edges.push(edge);

    let result: any[] = [];

    function src(e: JsonSDFGEdge): JsonSDFGNode {
        return state.nodes[parseInt(e.src)];
    }
    function dst(e: JsonSDFGEdge): JsonSDFGNode {
        return state.nodes[parseInt(e.dst)];
    }
    function isview(node: JsonSDFGNode) {
        if (node.type == 'AccessNode') {
            const nodedesc = sdfg.attributes._arrays[node.attributes.data];
            return (nodedesc && nodedesc.type === 'View');
        }
        return false;
    }

    // Determine direction
    let propagate_forward = false, propagate_backward = false;
    if ((edge.src_connector && src(edge).type.endsWith('Entry')) ||
        (edge.dst_connector && dst(edge).type.endsWith('Entry') &&
            edge.dst_connector.startsWith('IN_')) ||
        dst(edge).type == 'NestedSDFG' ||
        isview(dst(edge)))
        propagate_forward = true;
    if ((edge.src_connector && src(edge).type.endsWith('Exit')) ||
        (edge.dst_connector && dst(edge).type.endsWith('Exit')) ||
        src(edge).type == 'NestedSDFG' ||
        isview(src(edge)))
        propagate_backward = true;

    result.push(edge);

    // If either both are false (no scopes involved), we 
    // return only the current edge as a degenerate tree
    if (propagate_forward == propagate_backward && propagate_backward === false)
        return result;

    // Descend recursively
    function add_children(edge: JsonSDFGEdge) {
        const children: JsonSDFGEdge[] = [];

        if (propagate_forward) {
            const next_node = dst(edge);

            // Descend into nested SDFG
            if (next_node.type == 'NestedSDFG') {
                const name = edge.dst_connector;
                const nested_sdfg = next_node.attributes.sdfg;

                nested_sdfg.nodes.forEach((nstate: any) => {
                    nstate.edges.forEach((e: any) => {
                        const node = nstate.nodes[e.src];
                        if (node.type == 'AccessNode' &&
                            node.attributes.data === name) {
                            result = result.concat(
                                memlet_tree_nested(
                                    nested_sdfg, nstate, e, visited_edges
                                )
                            );
                        }
                    });
                });
            }

            if (isview(next_node)) {
                state.edges.forEach((e: JsonSDFGEdge) => {
                    if (parseInt(e.src) == next_node.id) {
                        children.push(e);
                        if (!e.attributes.data.attributes.shortcut) {
                            result.push(e);
                        }
                    }
                });
            } else {
                if (!(next_node.type.endsWith('Entry')) ||
                    !edge.dst_connector ||
                    !edge.dst_connector.startsWith('IN_'))
                    return;
                if (next_node.attributes.is_collapsed)
                    return;
                const conn = edge.dst_connector.substring(3);
                state.edges.forEach((e: JsonSDFGEdge) => {
                    if (parseInt(e.src) == next_node.id &&
                        e.src_connector == 'OUT_' + conn) {
                        children.push(e);
                        if (!e.attributes.data.attributes.shortcut) {
                            result.push(e);
                        }
                    }
                });
            }
        }
        if (propagate_backward) {
            const next_node = src(edge);

            // Descend into nested SDFG
            if (next_node.type == 'NestedSDFG') {
                const name = edge.src_connector;
                const nested_sdfg = next_node.attributes.sdfg;

                nested_sdfg.nodes.forEach((nstate: JsonSDFGState) => {
                    nstate.edges.forEach((e: JsonSDFGEdge) => {
                        const node = nstate.nodes[parseInt(e.dst)];
                        if (node.type == 'AccessNode' &&
                            node.attributes.data == name) {
                            result = result.concat(
                                memlet_tree_nested(
                                    nested_sdfg, nstate, e, visited_edges
                                )
                            );
                        }
                    });
                });
            }

            if (isview(next_node)) {
                state.edges.forEach((e: JsonSDFGEdge) => {
                    if (parseInt(e.dst) == next_node.id) {
                        children.push(e);
                        result.push(e);
                    }
                });
            } else {
                if (!(next_node.type.endsWith('Exit')) || !edge.src_connector)
                    return;

                const conn = edge.src_connector.substring(4);
                state.edges.forEach((e: JsonSDFGEdge) => {
                    if (parseInt(e.dst) == next_node.id &&
                        e.dst_connector == 'IN_' + conn) {
                        children.push(e);
                        result.push(e);
                    }
                });
            }
        }

        for (const child of children)
            add_children(child);
    }

    // Start from current edge
    add_children(edge);

    return result;
}

/**
 * Calls memlet_tree_nested for every nested SDFG and its edges and returns a
 * list with all memlet trees. As edges are visited only in one direction (from
 * outer SDFGs to inner SDFGs) a memlet can be split into several arrays.
 */
export function memlet_tree_recursive(root_sdfg: JsonSDFG): any[] {
    let trees: any[] = [];
    const visited_edges: JsonSDFGEdge[] = [];

    root_sdfg.nodes.forEach((state: JsonSDFGState) => {

        state.edges.forEach((e: JsonSDFGEdge) => {
            const tree = memlet_tree_nested(root_sdfg, state, e, visited_edges);
            if (tree.length > 1) {
                trees.push(tree);
            }
        });

        state.nodes.forEach((n: JsonSDFGNode) => {
            if (n.type == 'NestedSDFG') {
                const t = memlet_tree_recursive(n.attributes.sdfg);
                trees = trees.concat(t);
            }
        });

    });

    return trees;
}

/**
 * Returns all memlet trees as sets for the given graph.
 * 
 * @param {Graph} root_graph The top level graph.
 */
export function memlet_tree_complete(sdfg: JsonSDFG): any[] {
    const all_memlet_trees: any[] = [];
    const memlet_trees = memlet_tree_recursive(sdfg);

    // combine trees as memlet_tree_recursive does not necessarily return the
    // complete trees (they might be split into several trees)
    memlet_trees.forEach(tree => {
        let common_edge = false;
        for (const mt of all_memlet_trees) {
            for (const edge of tree) {
                if (mt.has(edge)) {
                    tree.forEach((e: JsonSDFGEdge) => mt.add(e));
                    common_edge = true;
                    break;
                }
            }
            if (common_edge)
                break;
        }
        if (!common_edge)
            all_memlet_trees.push(new Set(tree));
    });

    return all_memlet_trees;
}
