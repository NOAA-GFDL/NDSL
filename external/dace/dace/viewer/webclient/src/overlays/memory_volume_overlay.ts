import {
    Edge,
    NestedSDFG,
    SDFGElement,
    SDFGNode,
    State
} from '../renderer/renderer_elements';
import { GenericSdfgOverlay } from './generic_sdfg_overlay';
import { mean, median } from 'mathjs';
import { getTempColor } from '../renderer/renderer_elements';
import { SDFGRenderer } from '../renderer/renderer';
import { DagreSDFG, Point2D, SimpleRect, SymbolMap } from '../index';
import { SDFV } from '../sdfv';

export class MemoryVolumeOverlay extends GenericSdfgOverlay {

    public constructor(renderer: SDFGRenderer) {
        super(renderer);

        this.refresh();
    }

    public clear_cached_volume_values(): void {
        this.renderer.for_all_elements(0, 0, 0, 0, (
            type: string, e: MouseEvent, obj: any,
        ) => {
            if (obj.data) {
                if (obj.data.volume !== undefined)
                    obj.data.volume = undefined;
            }
        });
    }

    public calculate_volume_edge(
        edge: Edge,
        symbol_map: SymbolMap,
        volume_values: number[]
    ): number | undefined {
        let volume_string = undefined;
        if (edge.data && edge.data.attributes) {
            volume_string = edge.data.attributes.volume;
            if (volume_string !== undefined) {
                volume_string = volume_string.replace(/\*\*/g, '^');
                volume_string = volume_string.replace(/ceiling/g, 'ceil');
            }
        }
        let volume = undefined;
        if (volume_string !== undefined)
            volume = this.symbol_resolver.parse_symbol_expression(
                volume_string,
                symbol_map
            );

        edge.data.volume = volume;

        if (volume !== undefined && volume > 0)
            volume_values.push(volume);

        return volume;
    }

    public calculate_volume_graph(
        g: DagreSDFG,
        symbol_map: SymbolMap,
        volume_values: number[]
    ): void {
        const that = this;
        g.nodes().forEach((v: string) => {
            const state = g.node(v);
            const state_graph = state.data.graph;
            if (state_graph) {
                state_graph.edges().forEach((e: number) => {
                    const edge = state_graph.edge(e);
                    if (edge instanceof Edge)
                        that.calculate_volume_edge(
                            edge,
                            symbol_map,
                            volume_values
                        );
                });

                state_graph.nodes().forEach((v: number) => {
                    const node = state_graph.node(v);
                    if (node instanceof NestedSDFG) {
                        const nested_symbols_map: SymbolMap = {};
                        const mapping = node.data.node.attributes.symbol_mapping;
                        // Translate the symbol mappings for the nested SDFG
                        // based on the mapping described on the node.
                        Object.keys(mapping).forEach((symbol) => {
                            nested_symbols_map[symbol] =
                                that.symbol_resolver.parse_symbol_expression(
                                    mapping[symbol],
                                    symbol_map
                                );
                        });
                        // Merge in the parent mappings.
                        Object.keys(symbol_map).forEach((symbol) => {
                            if (!(symbol in nested_symbols_map))
                                nested_symbols_map[symbol] = symbol_map[symbol];
                        });

                        that.calculate_volume_graph(
                            node.data.graph,
                            nested_symbols_map,
                            volume_values
                        );
                    }
                });
            }
        });
    }

    public recalculate_volume_values(graph: DagreSDFG): void {
        this.badness_scale_center = 5;

        const volume_values = [0];
        this.calculate_volume_graph(
            graph,
            this.symbol_resolver.get_symbol_value_map(),
            volume_values
        );

        switch (this.overlay_manager.get_badness_scale_method()) {
            case 'mean':
                this.badness_scale_center = mean(volume_values);
                break;
            case 'median':
            default:
                this.badness_scale_center = median(volume_values);
                break;
        }
    }

    public refresh(): void {
        this.clear_cached_volume_values();
        const graph = this.renderer.get_graph();
        if (graph)
            this.recalculate_volume_values(graph);

        this.renderer.draw_async();
    }

    public shade_edge(edge: Edge, ctx: CanvasRenderingContext2D): void {
        const volume = edge.data.volume;
        if (volume !== undefined) {
            // Only draw positive volumes.
            if (volume <= 0)
                return;

            let badness = (1 / (this.badness_scale_center * 2)) * volume;
            if (badness < 0)
                badness = 0;
            if (badness > 1)
                badness = 1;
            const color = getTempColor(badness);

            edge.shade(this.renderer, ctx, color);
        }
    }

    public recursively_shade_sdfg(
        graph: DagreSDFG, ctx: CanvasRenderingContext2D, ppp: number,
        visible_rect: SimpleRect
    ): void {
        graph.nodes().forEach(v => {
            const state: State = graph.node(v);

            // If we're zoomed out enough that the contents aren't visible, we
            // skip the state.
            if ((ctx as any).lod && (
                ppp >= SDFV.STATE_LOD || state.width / ppp < SDFV.STATE_LOD
            ))
                return;

            // If the node's invisible, we skip it.
            if ((ctx as any).lod && !state.intersect(
                visible_rect.x, visible_rect.y,
                visible_rect.w, visible_rect.h
            ))
                return;

            const state_graph = state.data.graph;
            if (state_graph && !state.data.state.attributes.is_collapsed) {
                state_graph.nodes().forEach((v: string) => {
                    const node: SDFGNode = state_graph.node(v);

                    // Skip the node if it's not visible.
                    if ((ctx as any).lod && !node.intersect(
                        visible_rect.x, visible_rect.y,
                        visible_rect.w, visible_rect.h
                    ))
                        return;

                    // If we're zoomed out enough that the node's contents
                    // aren't visible or the node is collapsed, we skip it.
                    if (node.data.node.attributes.is_collapsed ||
                        ((ctx as any).lod && ppp >= SDFV.NODE_LOD))
                        return;

                    if (node instanceof NestedSDFG)
                        this.recursively_shade_sdfg(
                            node.data.graph, ctx, ppp, visible_rect
                        );
                });

                state_graph.edges().forEach((e: any) => {
                    const edge: Edge = state_graph.edge(e);

                    if ((ctx as any).lod && !edge.intersect(visible_rect.x,
                        visible_rect.y, visible_rect.w, visible_rect.h))
                        return;

                    this.shade_edge(edge, ctx);
                });
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

    public on_mouse_event(
        type: string,
        _ev: MouseEvent,
        _mousepos: Point2D,
        _elements: SDFGElement[],
        foreground_elem: SDFGElement | undefined,
        ends_drag: boolean
    ): boolean {
        if (type === 'click' && !ends_drag) {
            if (foreground_elem !== undefined &&
                foreground_elem instanceof Edge) {
                if (foreground_elem.data.volume === undefined) {
                    if (foreground_elem.data.attributes.volume) {
                        const that = this;
                        this.symbol_resolver.parse_symbol_expression(
                            foreground_elem.data.attributes.volume,
                            that.symbol_resolver.get_symbol_value_map(),
                            true,
                            () => {
                                const graph = that.renderer.get_graph();
                                if (graph) {
                                    that.clear_cached_volume_values();
                                    that.recalculate_volume_values(graph);
                                }
                            }
                        );
                    }
                }
            }
        }
        return false;
    }

}
