import { DagreSDFG, JsonSDFG, Point2D, SimpleRect } from '../index';
import { SDFGRenderer } from '../renderer/renderer';
import {
    NestedSDFG,
    SDFGNode,
    SDFGElement,
    State,
} from '../renderer/renderer_elements';
import { SDFV } from '../sdfv';
import { GenericSdfgOverlay } from './generic_sdfg_overlay';

export type LogicalGroup = {
    name: string,
    color: string,
    nodes: [number, number][],
    states: number[],
};

export class LogicalGroupOverlay extends GenericSdfgOverlay {

    public constructor(renderer: SDFGRenderer) {
        super(renderer);

        this.refresh();
    }

    public refresh(): void {
        this.renderer.draw_async();
    }

    public shade_node(
        node: SDFGNode, groups: LogicalGroup[], ctx: CanvasRenderingContext2D
    ): void {
        const all_groups: LogicalGroup[] = [];
        if (node instanceof State) {
            groups.forEach(group => {
                if (group.states.includes(node.id)) {
                    node.shade(this.renderer, ctx, group.color, 0.3);
                    all_groups.push(group);
                }
            });
        } else {
            groups.forEach(group => {
                group.nodes.forEach(n => {
                    if (n[0] === node.parent_id && n[1] === node.id) {
                        node.shade(this.renderer, ctx, group.color, 0.3);
                        all_groups.push(group);
                    }
                });
            });
        }

        const mousepos = this.renderer.get_mousepos();
        if (all_groups.length > 0 && mousepos &&
            node.intersect(mousepos.x, mousepos.y)) {
            // Show the corresponding group.
            this.renderer.set_tooltip(() => {
                const tt_cont = this.renderer.get_tooltip_container();
                if (tt_cont) {
                    if (all_groups.length === 1) {
                        tt_cont.innerText = 'Group: ' + all_groups[0].name;
                    } else {
                        let group_string = 'Groups: ';
                        all_groups.forEach((group, i) => {
                            group_string += group.name;
                            if (i < all_groups.length - 1)
                                group_string += ', ';
                        });
                        tt_cont.innerText = group_string;
                    }
                }
            });
        }
    }

    public recursively_shade_sdfg(
        sdfg: JsonSDFG,
        graph: DagreSDFG,
        ctx: CanvasRenderingContext2D,
        ppp: number,
        visible_rect: SimpleRect
    ): void {
        // First go over visible states, skipping invisible ones. We only draw
        // something if the state is collapsed or we're zoomed out far enough.
        // In that case, we overlay the correct grouping color(s).
        // If it's expanded or zoomed in close enough, we traverse inside.
        const sdfg_groups = sdfg.attributes.logical_groups;
        if (sdfg_groups === undefined)
            return;

        graph.nodes().forEach(v => {
            const state = graph.node(v);

            // If the node's invisible, we skip it.
            if ((ctx as any).lod && !state.intersect(
                visible_rect.x, visible_rect.y,
                visible_rect.w, visible_rect.h
            ))
                return;

            if (((ctx as any).lod && (ppp >= SDFV.STATE_LOD ||
                state.width / ppp <= SDFV.STATE_LOD)) ||
                state.data.state.attributes.is_collapsed) {
                this.shade_node(state, sdfg_groups, ctx);
            } else {
                const state_graph = state.data.graph;
                if (state_graph) {
                    state_graph.nodes().forEach((v: string) => {
                        const node = state_graph.node(v);

                        // Skip the node if it's not visible.
                        if ((ctx as any).lod && !node.intersect(visible_rect.x,
                            visible_rect.y, visible_rect.w, visible_rect.h))
                            return;

                        if (node.data.node.attributes.is_collapsed ||
                            ((ctx as any).lod && ppp >= SDFV.NODE_LOD)) {
                            this.shade_node(node, sdfg_groups, ctx);
                        } else {
                            if (node instanceof NestedSDFG) {
                                this.recursively_shade_sdfg(
                                    node.data.node.attributes.sdfg,
                                    node.data.graph, ctx, ppp, visible_rect
                                );
                            } else {
                                this.shade_node(node, sdfg_groups, ctx);
                            }
                        }
                    });
                }
            }
        });
    }

    public draw(): void {
        const sdfg = this.renderer.get_sdfg();
        const graph = this.renderer.get_graph();
        const ppp = this.renderer.get_canvas_manager()?.points_per_pixel();
        const context = this.renderer.get_context();
        const visible_rect = this.renderer.get_visible_rect();
        if (graph && ppp !== undefined && context && visible_rect)
            this.recursively_shade_sdfg(
                sdfg, graph, context, ppp, visible_rect
            );
    }

    public on_mouse_event(
        type: string,
        _ev: MouseEvent,
        _mousepos: Point2D,
        _elements: SDFGElement[],
        foreground_elem: SDFGElement | undefined,
        ends_drag: boolean
    ): boolean {
        return false;
    }

}
