import { NestedSDFG, SDFGNode } from '../renderer/renderer_elements';
import { GenericSdfgOverlay } from './generic_sdfg_overlay';
import { mean, median } from 'mathjs';
import { getTempColor } from '../renderer/renderer_elements';
import { SDFGRenderer } from '../renderer/renderer';
import { DagreSDFG, SimpleRect } from '../index';
import { SDFV } from '../sdfv';
import { get_element_uuid } from '../utils/utils';


export class RuntimeMicroSecondsOverlay extends GenericSdfgOverlay {

    private criterium: string = 'mean';
    private runtime_map: { [uuids: string]: any } = {}

    public constructor(renderer: SDFGRenderer) {
        super(renderer);
        this.badness_scale_center = 0;
    }

    public refresh(): void {
        this.badness_scale_center = 5;

        const micros_values = [0];

        for (const key of Object.keys(this.runtime_map)) {
            // Make sure the overall SDFG's runtime isn't included in this.
            if (key !== '0/-1/-1/-1')
                micros_values.push(this.runtime_map[key][this.criterium]);
        }

        switch (this.overlay_manager.get_badness_scale_method()) {
            case 'mean':
                this.badness_scale_center = mean(micros_values);
                break;
            case 'median':
            default:
                this.badness_scale_center = median(micros_values);
                break;
        }

        this.renderer.draw_async();
    }

    public pretty_print_micros(micros: number): string {
        let unit = 'µs';
        let value = micros;
        if (micros > 1000) {
            unit = 'ms';
            const millis = micros / 1000;
            value = millis;
            if (millis > 1000) {
                unit = 's';
                const seconds = millis / 1000;
                value = seconds;
            }
        }

        value = Math.round((value + Number.EPSILON) * 100) / 100;
        return value.toString() + ' ' + unit;
    }

    public shade_node(node: SDFGNode, ctx: CanvasRenderingContext2D): void {
        const rt_summary = this.runtime_map[get_element_uuid(node)];

        if (rt_summary === undefined)
            return;

        const mousepos = this.renderer.get_mousepos();
        if (mousepos && node.intersect(mousepos.x, mousepos.y)) {
            // Show the measured runtime.
            if (rt_summary['min'] === rt_summary['max'])
                this.renderer.set_tooltip(() => {
                    const tt_cont = this.renderer.get_tooltip_container();
                    if (tt_cont)
                        tt_cont.innerText = this.pretty_print_micros(
                            rt_summary['min']
                        );
                });

            else
                this.renderer.set_tooltip(() => {
                    const tt_cont = this.renderer.get_tooltip_container();
                    if (tt_cont)
                        tt_cont.innerText = (
                            'Min: ' +
                            this.pretty_print_micros(rt_summary['min']) +
                            '\nMax: ' +
                            this.pretty_print_micros(rt_summary['max']) +
                            '\nMean: ' +
                            this.pretty_print_micros(rt_summary['mean']) +
                            '\nMedian: ' +
                            this.pretty_print_micros(rt_summary['med']) +
                            '\nCount: ' +
                            rt_summary['count']
                        );
                });
        }

        // Calculate the 'badness' color.
        const micros = rt_summary[this.criterium];
        let badness = (1 / (this.badness_scale_center * 2)) * micros;
        if (badness < 0)
            badness = 0;
        if (badness > 1)
            badness = 1;
        const color = getTempColor(badness);

        node.shade(this.renderer, ctx, color);
    }

    public recursively_shade_sdfg(
        graph: DagreSDFG,
        ctx: CanvasRenderingContext2D,
        ppp: number,
        visible_rect: SimpleRect
    ): void {
        // First go over visible states, skipping invisible ones. We only draw
        // something if the state is collapsed or we're zoomed out far enough.
        // In that case, we draw the measured runtime for the entire state.
        // If it's expanded or zoomed in close enough, we traverse inside.
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
                this.shade_node(state, ctx);
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
                            this.shade_node(node, ctx);
                        } else {
                            if (node instanceof NestedSDFG) {
                                this.recursively_shade_sdfg(
                                    node.data.graph, ctx, ppp, visible_rect
                                );
                            } else {
                                this.shade_node(node, ctx);
                            }
                        }
                    });
                }
            }
        });
    }

    public draw(): void {
        const graph = this.renderer.get_graph();
        const ppp = this.renderer.get_canvas_manager()?.points_per_pixel();
        const context = this.renderer.get_context();
        const visible_rect = this.renderer.get_visible_rect();
        if (graph && ppp !== undefined && context && visible_rect)
            this.recursively_shade_sdfg(graph, context, ppp, visible_rect);
    }

    public set_runtime_map(runtime_map: { [uuids: string]: any }): void {
        this.runtime_map = runtime_map;
    }

    public set_criterium(criterium: string): void {
        this.criterium = criterium;
    }

    public get_criterium(): string {
        return this.criterium;
    }

}
