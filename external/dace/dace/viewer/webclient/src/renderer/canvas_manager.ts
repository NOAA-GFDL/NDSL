// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { Edge, EntryNode, SDFGElement, SDFGNode } from './renderer_elements';
import { lerpMatrix } from '../utils/lerp_matrix';
import { updateEdgeBoundingBox } from '../utils/bounding_box';
import {
    get_positioning_info,
    initialize_positioning_info,
} from '../utils/sdfg/sdfg_utils';
import { SDFGRenderer, SDFGListType } from './renderer';
import { DagreSDFG, intersectRect, Point2D } from '../index';

const animation_duration = 1000;

// cubic ease out
const animation_function = (t: number) => 1 - Math.pow(1 - t, 3);

let _canvas_manager_counter = 0;

export class CanvasManager {
    // Manages translation and scaling of canvas rendering

    private anim_id: number | null = null;
    private prev_time: number | null = null;
    private drawables: any[] = [];
    private indices: any[] = [];

    private animation_start: number | null = null;
    private animation_end: number | null = null;
    private animation_target: DOMMatrix | null = null;

    // Takes a number [0, 1] and returns a transformation matrix.
    private animation: ((t: any) => DOMMatrix) | null = null;

    private request_scale: boolean = false;
    private scalef: number = 1.0;

    private _destroying: boolean = false;

    private scale_origin: Point2D = { x: 0, y: 0 };

    private contention: number = 0;

    private _svg: SVGSVGElement;
    private user_transform: DOMMatrix;

    public static counter(): number {
        return _canvas_manager_counter++;
    }

    public constructor(
        private ctx: CanvasRenderingContext2D,
        private renderer: SDFGRenderer,
        private canvas: HTMLCanvasElement
    ) {
        (this.ctx as any).lod = true;

        this._svg = document.createElementNS(
            'http://www.w3.org/2000/svg', 'svg'
        );

        this.user_transform = this._svg.createSVGMatrix();

        this.addCtxTransformTracking();
    }

    public stopAnimation(): void {
        this.animation_start = null;
        this.animation_end = null;
        this.animation = null;
        this.animation_target = null;
    }

    public alreadyAnimatingTo(new_transform: DOMMatrix): boolean {
        if (this.animation_target) {
            let result = true;
            result = result && (this.animation_target.a == new_transform.a);
            result = result && (this.animation_target.b == new_transform.b);
            result = result && (this.animation_target.c == new_transform.c);
            result = result && (this.animation_target.d == new_transform.d);
            result = result && (this.animation_target.e == new_transform.e);
            result = result && (this.animation_target.f == new_transform.f);
            return result;
        } else {
            return false;
        }
    }

    public animateTo(new_transform: DOMMatrix): void {
        // If was already animating to the same target, jump to it directly
        if (this.alreadyAnimatingTo(new_transform)) {
            this.stopAnimation();
            this.user_transform = new_transform;
            return;
        }

        this.stopAnimation();
        this.animation = lerpMatrix(this.user_transform, new_transform);
        this.animation_target = new_transform;
    }

    public svgPoint(x: number, y: number): DOMPoint {
        const pt = this._svg.createSVGPoint();
        pt.x = x; pt.y = y;
        return pt;
    }

    public applyUserTransform(): void {
        const ut = this.user_transform;
        this.ctx.setTransform(ut.a, ut.b, ut.c, ut.d, ut.e, ut.f);
    }

    public get translation(): Point2D {
        return { x: this.user_transform.e, y: this.user_transform.f };
    }

    public addCtxTransformTracking(): void {
        /* This function is a hack to provide the non-standardized functionality
        of getting the current transform from a RenderingContext.
        When (if) this is standardized, the standard should be used instead.
        This is made for "easy" transforms and does not support saving/restoring
        */

        const svg = document.createElementNS(
            'http://www.w3.org/2000/svg', 'svg'
        );
        (this.ctx as any)._custom_transform_matrix = svg.createSVGMatrix();
        // Save/Restore is not supported.

        const checker = () => {
            console.assert(
                !isNaN((this.ctx as any)._custom_transform_matrix.f)
            );
        };
        const _ctx = this.ctx;
        const scale_func = _ctx.scale;
        _ctx.scale = function (sx, sy) {
            (_ctx as any)._custom_transform_matrix =
                (_ctx as any)._custom_transform_matrix.scaleNonUniform(sx, sy);
            checker();
            return scale_func.call(_ctx, sx, sy);
        };
        const translate_func = _ctx.translate;
        _ctx.translate = function (sx, sy) {
            (_ctx as any)._custom_transform_matrix =
                (_ctx as any)._custom_transform_matrix.translate(sx, sy);
            checker();
            return translate_func.call(_ctx, sx, sy);
        };
        const rotate_func = _ctx.rotate;
        _ctx.rotate = function (r) {
            (_ctx as any)._custom_transform_matrix =
                (_ctx as any)._custom_transform_matrix.rotate(
                    r * 180.0 / Math.PI
                );
            checker();
            return rotate_func.call(_ctx, r);
        };
        const transform_func = _ctx.scale;
        _ctx.transform = function (a, b, c, d, e, f) {
            const m2 = svg.createSVGMatrix();
            m2.a = a; m2.b = b; m2.c = c; m2.d = d; m2.e = e; m2.f = f;
            (_ctx as any)._custom_transform_matrix =
                (_ctx as any)._custom_transform_matrix.multiply(m2);
            checker();
            return (transform_func as any).call(_ctx, a, b, c, d, e, f);
        };

        const setTransform_func = _ctx.setTransform;
        (_ctx as any).setTransform = function (
            a: number, b: number, c: number, d: number, e: number, f: number
        ) {
            const ctxref: any = _ctx;
            ctxref._custom_transform_matrix.a = a;
            ctxref._custom_transform_matrix.b = b;
            ctxref._custom_transform_matrix.c = c;
            ctxref._custom_transform_matrix.d = d;
            ctxref._custom_transform_matrix.e = e;
            ctxref._custom_transform_matrix.f = f;
            checker();
            return (setTransform_func as any).call(_ctx, a, b, c, d, e, f);
        };

        (_ctx as any).custom_inverseTransformMultiply =
            function (x: number, y: number) {
                const pt = svg.createSVGPoint();
                pt.x = x; pt.y = y;
                checker();
                return pt.matrixTransform(
                    (_ctx as any)._custom_transform_matrix.inverse()
                );
            };
    }

    public destroy(): void {
        this._destroying = true;
        this.clearDrawables();
    }

    public addDrawable(obj: any): void {
        this.drawables.push(obj);
        this.indices.push({ 'c': CanvasManager.counter(), 'd': obj });
    }

    public removeDrawable(drawable: any): void {
        this.drawables = this.drawables.filter(x => x != drawable);
    }

    public clearDrawables(): void {
        for (const x of this.drawables) {
            x.destroy();
        }
        this.drawables = [];
        this.indices = [];
    }

    public isBlank(): boolean {
        const ctx = this.canvas.getContext('2d');
        if (!ctx)
            return true;

        const topleft = ctx.getImageData(0, 0, 1, 1).data;
        if (topleft[0] != 0 || topleft[1] != 0 || topleft[2] != 0 ||
            topleft[3] != 255)
            return false;

        const pixelBuffer = new Uint32Array(
            ctx.getImageData(
                0, 0, this.canvas.width, this.canvas.height
            ).data.buffer
        );

        return !pixelBuffer.some(color => color !== 0xff000000);
    }

    public scale(diff: number, x: number = 0, y: number = 0): void {
        this.stopAnimation();
        if (this.request_scale || this.contention > 0) {
            return;
        }
        this.contention++;
        this.request_scale = true;
        if (this.isBlank()) {
            this.renderer.set_bgcolor('black');
            this.renderer.zoom_to_view(null, false);
            diff = 0.01;
        }

        this.scale_origin.x = x;
        this.scale_origin.y = y;

        const sv = diff;
        const pt = this.svgPoint(
            this.scale_origin.x, this.scale_origin.y
        ).matrixTransform(this.user_transform.inverse());
        this.user_transform = this.user_transform.translate(pt.x, pt.y);
        this.user_transform = this.user_transform.scale(sv, sv, 1, 0, 0, 0);
        this.scalef *= sv;
        this.user_transform = this.user_transform.translate(-pt.x, -pt.y);

        this.contention--;
    }

    // Sets the view to the square around the input rectangle
    public set_view(rect: DOMRect, animate: boolean = false): void {
        const canvas_w = this.canvas.width;
        const canvas_h = this.canvas.height;
        if (canvas_w == 0 || canvas_h == 0)
            return;

        let scale = 1, tx = 0, ty = 0;
        if (rect.width > rect.height) {
            scale = canvas_w / rect.width;
            tx = -rect.x;
            ty = -rect.y - (rect.height / 2) + (canvas_h / scale / 2);

            // Now other dimension does not fit, scale it as well
            if (rect.height * scale > canvas_h) {
                scale = canvas_h / rect.height;
                tx = -rect.x - (rect.width / 2) + (canvas_w / scale / 2);
                ty = -rect.y;
            }
        } else {
            scale = canvas_h / rect.height;
            tx = -rect.x - (rect.width / 2) + (canvas_w / scale / 2);
            ty = -rect.y;

            // Now other dimension does not fit, scale it as well
            if (rect.width * scale > canvas_w) {
                scale = canvas_w / rect.width;
                tx = -rect.x;
                ty = -rect.y - (rect.height / 2) + (canvas_h / scale / 2);
            }
        }

        // Uniform scaling
        const new_transform = this._svg.createSVGMatrix().scale(
            scale, scale, 1, 0, 0, 0
        ).translate(tx, ty);

        if (animate && this.prev_time !== null) {
            this.animateTo(new_transform);
        } else {
            this.stopAnimation();
            this.user_transform = new_transform;
        }

        this.scale_origin = { x: 0, y: 0 };
        this.scalef = 1.0;
    }

    public translate(x: number, y: number): void {
        this.stopAnimation();
        this.user_transform = this.user_transform.translate(
            x / this.user_transform.a, y / this.user_transform.d
        );
    }

    /**
     * Move/translate an element in the graph by a change in x and y.
     * @param {*} el                Element to move
     * @param {*} old_mousepos      Old mouse position in canvas coordinates
     * @param {*} new_mousepos      New mouse position in canvas coordinates
     * @param {*} entire_graph      Reference to the entire graph
     * @param {*} sdfg_list         List of SDFGs and nested SDFGs
     * @param {*} state_parent_list List of parent elements to SDFG states
     * @param {*} drag_start        Drag starting event, undefined if no drag
     * @param {*} update_position_info Whether to update positioning information
     * @param {*} move_entire_edge  Whether to move the entire edge for edges
     * @param {*} edge_dpoints      List of edge points for edges
     */
    public translate_element(
        el: SDFGElement,
        old_mousepos: Point2D,
        new_mousepos: Point2D,
        entire_graph: DagreSDFG,
        sdfg_list: SDFGListType,
        state_parent_list: any[],
        drag_start: any,
        update_position_info: boolean = true,
        move_entire_edge: boolean = false,
        edge_dpoints: any[] | undefined = undefined
    ): void {
        this.stopAnimation();

        // Edges connected to the moving element
        const out_edges: any[] = [];
        const in_edges: any[] = [];

        // Find the parent graph in the list of available SDFGs
        let parent_graph = sdfg_list[el.sdfg.sdfg_list_id];
        let parent_element: SDFGElement | null = null;

        if (
            entire_graph !== parent_graph &&
            (el.data.state || el.data.type == 'InterstateEdge')
        ) {
            // If the parent graph and the entire SDFG shown are not the same,
            // we're currently in a nested SDFG. If we're also moving a state,
            // this means that its parent element is found in the list of
            // parents to states (state_parent_list)
            parent_element = state_parent_list[el.sdfg.sdfg_list_id];
        } else if (el.parent_id !== null && parent_graph) {
            // If the parent_id isn't null and there is a parent graph, we can
            // look up the parent node via the element's parent_id
            parent_element = parent_graph.node(el.parent_id);
            // If our parent element is a state, we want the state's graph
            if (parent_element && parent_element.data.state)
                parent_graph = parent_element.data.graph;
        }

        if (parent_graph && !(el instanceof Edge)) {
            // Find all the edges connected to the moving node
            parent_graph.outEdges(el.id).forEach((edge_id: number) => {
                out_edges.push(parent_graph.edge(edge_id));
            });
            parent_graph.inEdges(el.id).forEach((edge_id: number) => {
                in_edges.push(parent_graph.edge(edge_id));
            });
        }

        // Compute theoretical initial displacement/movement
        let dx = new_mousepos.x - old_mousepos.x;
        let dy = new_mousepos.y - old_mousepos.y;

        // If edge, find closest point to drag start position
        let pt = -1;
        if (el instanceof Edge) {
            if (move_entire_edge) {
                pt = -2;
            } else if (edge_dpoints && edge_dpoints.length > 0) {
                pt = -3;
            } else if (drag_start) {
                // Find closest point to old mouse position
                if (drag_start.edge_point === undefined) {
                    let dist: number | null = null;
                    el.get_points().forEach((p, i) => {
                        // Only allow dragging if the memlet has more than two
                        // points
                        if (i == 0 || i == el.get_points().length - 1)
                            return;
                        const ddx = p.x - drag_start.cx;
                        const ddy = p.y - drag_start.cy;
                        const curdist = ddx*ddx + ddy*ddy;
                        if (dist === null || curdist < dist) {
                            dist = curdist;
                            pt = i;
                        }
                    });
                    drag_start.edge_point = pt;
                } else {
                    pt = drag_start.edge_point;
                }
            }
        }

        if (parent_element) {
            // Calculate the box to bind the element to. This is given by
            // the parent element, i.e. the element where out to-be-moved
            // element is contained within
            const parent_left_border =
                (parent_element.x - (parent_element.width / 2));
            const parent_right_border =
                parent_left_border + parent_element.width;
            const parent_top_border =
                (parent_element.y - (parent_element.height / 2));
            const parent_bottom_border =
                parent_top_border + parent_element.height;

            let el_h_margin = el.height / 2;
            let el_w_margin = el.width / 2;
            if (el instanceof Edge) {
                el_h_margin = el_w_margin = 0;
            }
            const min_x = parent_left_border + el_w_margin;
            const min_y = parent_top_border + el_h_margin;
            const max_x = parent_right_border - el_w_margin;
            const max_y = parent_bottom_border - el_h_margin;

            // Make sure we don't move our element outside its parent's
            // bounding box. If either the element or the mouse pointer are
            // outside the parent, we clamp movement in that direction
            if (el instanceof Edge) {
                if (pt > 0) {
                    const points = el.get_points();
                    const target_x = points[pt].x + dx;
                    const target_y = points[pt].y + dy;
                    if (target_x <= min_x ||
                        new_mousepos.x <= parent_left_border) {
                        dx = min_x - points[pt].x;
                    } else if (target_x >= max_x ||
                        new_mousepos.x >= parent_right_border) {
                        dx = max_x - points[pt].x;
                    }
                    if (target_y <= min_y ||
                        new_mousepos.y <= parent_top_border) {
                        dy = min_y - points[pt].y;
                    } else if (target_y >= max_y ||
                        new_mousepos.y >= parent_bottom_border) {
                        dy = max_y - points[pt].y;
                    }
                }
            } else {
                const target_x = el.x + dx;
                const target_y = el.y + dy;
                if (target_x <= min_x ||
                    new_mousepos.x <= parent_left_border) {
                    dx = min_x - el.x;
                } else if (target_x >= max_x ||
                    new_mousepos.x >= parent_right_border) {
                    dx = max_x - el.x;
                }
                if (target_y <= min_y ||
                    new_mousepos.y <= parent_top_border) {
                    dy = min_y - el.y;
                } else if (target_y >= max_y ||
                    new_mousepos.y >= parent_bottom_border) {
                    dy = max_y - el.y;
                }
            }
        }

        if (el instanceof Edge) {
            const points = el.get_points();
            let position;
            if (update_position_info) {
                position = get_positioning_info(el);

                if (!position)
                    position = initialize_positioning_info(el);
            }
            if (pt > 0) {
                // Move point
                points[pt].x += dx;
                points[pt].y += dy;

                // Move edge bounding box
                updateEdgeBoundingBox(el);

                if (update_position_info) {
                    if (!position.points) {
                        position.points = Array(points.length);
                        for (let i = 0; i < points.length; i++) {
                            position.points[i] = {dx: 0, dy: 0};
                        }
                    }

                    position.points[pt].dx += dx;
                    position.points[pt].dy += dy;
                }
            } else if (pt == -2) {
                // Don't update first and last point (the connectors)
                for (let i = 1; i < points.length - 1; i++) {
                    points[i].x += dx;
                    points[i].y += dy;
                }

                if (update_position_info) {
                    for (let i = 1; i < points.length - 1; i++) {
                        position.points[i].dx += dx;
                        position.points[i].dy += dy;
                    }
                }
            } else if (pt == -3 && edge_dpoints) {
                for (let i = 1; i < points.length - 1; i++) {
                    points[i].x += edge_dpoints[i].dx;
                    points[i].y += edge_dpoints[i].dy;
                }
            }
            // The rest of the method doesn't apply to Edges
            return;
        }

        // Move a node together with its connectors if it has any
        function move_node_and_connectors(node: SDFGNode) {
            node.x += dx;
            node.y += dy;
            if (node.data.node && node.data.node.type === 'NestedSDFG')
                translate_recursive(node.data.graph);
            if (node.in_connectors)
                node.in_connectors.forEach(c => {
                    c.x += dx;
                    c.y += dy;
                });
            if (node.out_connectors)
                node.out_connectors.forEach(c => {
                    c.x += dx;
                    c.y += dy;
                });
        }

        // Allow recursive translation of nested SDFGs
        function translate_recursive(ng: DagreSDFG) {
            ng.nodes().forEach((state_id: string) => {
                const state = ng.node(state_id);
                state.x += dx;
                state.y += dy;
                const g = state.data.graph;
                if (g) {
                    g.nodes().forEach((node_id: string) => {
                        const node = g.node(node_id);
                        move_node_and_connectors(node);
                    });

                    g.edges().forEach((edge_id: number) => {
                        const edge = g.edge(edge_id);
                        edge.x += dx;
                        edge.y += dy;
                        edge.points.forEach((point: Point2D) => {
                            point.x += dx;
                            point.y += dy;
                        });
                        updateEdgeBoundingBox(edge);
                    });
                }
            });
            ng.edges().forEach(edge_id => {
                const edge = ng.edge(edge_id);
                edge.x += dx;
                edge.y += dy;
                edge.points.forEach(point => {
                    point.x += dx;
                    point.y += dy;
                });
                updateEdgeBoundingBox(edge);
            });
        }

        // Move the node
        move_node_and_connectors(el);

        // Store movement information in element (for relayouting)
        if (update_position_info) {
            let position = get_positioning_info(el);
            if (!position)
                position = initialize_positioning_info(el);

            position.dx += dx;
            position.dy += dy;

            // Store movement information if EntryNode for other nodes of the
            // same scope
            if (el instanceof EntryNode &&
                el.data.node.attributes.is_collapsed) {
                if (!position.scope_dx) {
                    position.scope_dx = 0;
                    position.scope_dy = 0;
                }

                position.scope_dx += dx;
                position.scope_dy += dy;
            }
        }
        
        if (el.data.state && !el.data.state.attributes.is_collapsed) {
            // We're moving a state, move all its contained elements
            const graph = el.data.graph;
            graph.nodes().forEach((node_id: string) => {
                const node = graph.node(node_id);
                move_node_and_connectors(node);
            });

            // Drag all the edges along
            graph.edges().forEach((edge_id: number) => {
                const edge = graph.edge(edge_id);
                edge.x += dx;
                edge.y += dy;
                edge.points.forEach((point: Point2D) => {
                    point.x += dx;
                    point.y += dy;
                });
                updateEdgeBoundingBox(edge);
            });
        }

        // Move the connected edges along with the element
        out_edges.forEach((edge: Edge) => {
            const points = edge.points;
            const n = points.length - 1;
            let moved = false;
            if (edge.src_connector !== null) {
                for (let i = 0; i < el.out_connectors.length; i++) {
                    if (el.out_connectors[i].data.name === edge.src_connector) {
                        points[0] = intersectRect(
                            el.out_connectors[i], points[1]
                        );
                        moved = true;
                        break;
                    }
                }
            }
            if (!moved) {
                points[0].x += dx;
                points[0].y += dy;
            }
            // Also update destination point of edge
            if (edge.dst_connector !== null) {
                const e = parent_element?.data?.state?.edges[edge.id];
                const dst_el = parent_graph.node(e?.dst);
                if (dst_el) {
                    for (let i = 0; i < dst_el.in_connectors.length; i++) {
                        const dst_name = dst_el.in_connectors[i].data.name;
                        if (dst_name === edge.dst_connector) {
                            points[n] = intersectRect(
                                dst_el.in_connectors[i], points[n - 1]
                            );
                            break;
                        }
                    }
                }
            }
            updateEdgeBoundingBox(edge);
        });
        in_edges.forEach((edge: Edge) => {
            const points = edge.points;
            const n = points.length - 1;
            let moved = false;
            if (edge.dst_connector !== null) {
                for (let i = 0; i < el.in_connectors.length; i++) {
                    if (el.in_connectors[i].data.name === edge.dst_connector) {
                        points[n] = intersectRect(
                            el.in_connectors[i], points[n-1]
                        );
                        moved = true;
                        break;
                    }
                }
            }
            if (!moved) {
                points[n].x += dx;
                points[n].y += dy;
            }
            // Also update source point of edge
            if (edge.src_connector !== null) {
                const e = parent_element?.data?.state?.edges[edge.id];
                const src_el = parent_graph.node(e?.src);
                if (src_el) {
                    for (let i = 0; i < src_el.out_connectors.length; i++) {
                        const out_name = src_el.out_connectors[i].data.name;
                        if (out_name === edge.src_connector) {
                            points[0] = intersectRect(
                                src_el.out_connectors[i], points[1]
                            );
                            break;
                        }
                    }
                }
            }
            updateEdgeBoundingBox(edge);
        });
    }

    public mapPixelToCoordsX(xpos: number): number {
        return this.svgPoint(xpos, 0).matrixTransform(
            this.user_transform.inverse()
        ).x;
    }

    public mapPixelToCoordsY(ypos: number): number {
        return this.svgPoint(0, ypos).matrixTransform(
            this.user_transform.inverse()
        ).y;
    }

    public noJitter(x: number): number {
        x = parseFloat(x.toFixed(3));
        x = Math.round(x * 100) / 100;
        return x;
    }

    public points_per_pixel(): number {
        // Since we are using uniform scaling, (bottom-top)/height and
        // (right-left)/width should be equivalent
        const left = this.mapPixelToCoordsX(0);
        const right = this.mapPixelToCoordsX(this.canvas.width);
        return (right - left) / this.canvas.width;
    }

    public animation_step(now: number): void {
        if (this.animation === null) {
            return;
        }

        if (this.animation_start === null) {
            this.animation_start = now;
            this.animation_end = now + animation_duration;
        }

        if (this.animation_end === null || now >= this.animation_end) {
            this.user_transform = this.animation(1);
            this.stopAnimation();
            return;
        }

        const start = this.animation_start;
        const end = this.animation_end;
        this.user_transform = this.animation(
            animation_function((now - start) / (end - start))
        );
    }

    public draw_now(now: number): void {
        if (this._destroying)
            return;

        let dt: number | null = null;
        if (!this.prev_time)
            dt = null;
        else
            dt = now - this.prev_time;
        this.prev_time = now;

        if (this.contention > 0)
            return;
        this.contention += 1;
        const ctx = this.ctx;

        // Clear with default transform
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.fillStyle = this.renderer.get_bgcolor();
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        this.animation_step(now);

        this.applyUserTransform();
        if (this.request_scale)
            this.request_scale = this.contention !== 1;

        this.renderer.draw(dt);
        this.contention -= 1;

        if (this.animation_end !== null && now < this.animation_end)
            this.draw_async();
    }

    public draw_async(): void {
        this.anim_id = window.requestAnimationFrame(
            (now) => this.draw_now(now)
        );
    }

    public get_user_transform(): DOMMatrix {
        return this.user_transform;
    }

    public set_user_transform(user_transform: DOMMatrix): void {
        this.user_transform = user_transform;
    }

}
