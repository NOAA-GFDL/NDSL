// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { SDFV } from '../sdfv';
import {
    DagreSDFG,
    JsonSDFG,
    JsonSDFGEdge,
    JsonSDFGNode,
    JsonSDFGState,
    Point2D,
} from '../index';
import {
    sdfg_range_elem_to_string,
    sdfg_consume_elem_to_string,
} from '../utils/sdfg/display';
import { sdfg_property_to_string } from '../utils/sdfg/display';
import { check_and_redirect_edge } from '../utils/sdfg/sdfg_utils';
import { SDFGRenderer } from './renderer';

export class SDFGElement {

    public in_connectors: Connector[] = [];
    public out_connectors: Connector[] = [];

    // Indicate special drawing conditions based on interactions.
    public selected: boolean = false;
    public highlighted: boolean = false;
    public hovered: boolean = false;

    public x: number = 0;
    public y: number = 0;
    public width: number = 0;
    public height: number = 0;

    // Parent ID is the state ID, if relevant
    public constructor(
        public data: any,
        public id: number,
        public sdfg: JsonSDFG,
        public parent_id: number | null = null
    ) {
        this.set_layout();
    }

    public set_layout(): void {
        // dagre does not work well with properties, only fields
        this.width = this.data.layout.width;
        this.height = this.data.layout.height;
    }

    public draw(
        _renderer: SDFGRenderer, _ctx: CanvasRenderingContext2D,
        _mousepos: Point2D | null
    ): void {
        return;
    }

    public simple_draw(
        _renderer: SDFGRenderer, _ctx: CanvasRenderingContext2D,
        _mousepos: Point2D | null
    ): void {
        return;
    }

    public shade(
        _renderer: SDFGRenderer, _ctx: CanvasRenderingContext2D, _color: string,
        _alpha: number = 0.6
    ): void {
        return;
    }

    public debug_draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D
    ): void {
        if (renderer.debug_draw) {
            // Print the center and bounding box in debug mode.
            ctx.beginPath();
            ctx.arc(this.x, this.y, 1, 0, 2 * Math.PI, false);
            ctx.fillStyle = 'red';
            ctx.fill();
            ctx.strokeStyle = 'red';
            ctx.stroke();
            ctx.strokeRect(
                this.x - (this.width / 2.0), this.y - (this.height / 2.0),
                this.width, this.height
            );
        }
    }

    public attributes(): any {
        return this.data.attributes;
    }

    public type(): string {
        return this.data.type;
    }

    public label(): string {
        return this.data.label;
    }

    // Produces HTML for a hover-tooltip
    public tooltip(container: HTMLElement): void {
        container.className = 'sdfvtooltip';
    }

    public topleft(): Point2D {
        return { x: this.x - this.width / 2, y: this.y - this.height / 2 };
    }

    public strokeStyle(renderer: SDFGRenderer | undefined = undefined): string {
        if (!renderer)
            return 'black';

        if (this.selected) {
            if (this.hovered)
                return this.getCssProperty(
                    renderer, '--color-selected-hovered'
                );
            else if (this.highlighted)
                return this.getCssProperty(
                    renderer, '--color-selected-highlighted'
                );
            else
                return this.getCssProperty(renderer, '--color-selected');
        } else {
            if (this.hovered)
                return this.getCssProperty(renderer, '--color-hovered');
            else if (this.highlighted)
                return this.getCssProperty(renderer, '--color-highlighted');
        }
        return this.getCssProperty(renderer, '--color-default');
    }

    // General bounding-box intersection function. Returns true iff point or
    // rectangle intersect element.
    public intersect(
        x: number, y: number, w: number = 0, h: number = 0
    ): boolean {
        if (w == 0 || h == 0) {  // Point-element intersection
            return (x >= this.x - this.width / 2.0) &&
                (x <= this.x + this.width / 2.0) &&
                (y >= this.y - this.height / 2.0) &&
                (y <= this.y + this.height / 2.0);
        } else {                 // Box-element intersection
            return (x <= this.x + this.width / 2.0) &&
                (x + w >= this.x - this.width / 2.0) &&
                (y <= this.y + this.height / 2.0) &&
                (y + h >= this.y - this.height / 2.0);
        }
    }

    public contained_in(
        x: number, y: number, w: number = 0, h: number = 0
    ): boolean {
        if (w === 0 || h === 0)
            return false;

        const box_start_x = x;
        const box_end_x = x + w;
        const box_start_y = y;
        const box_end_y = y + h;

        const el_start_x = this.x - (this.width / 2.0);
        const el_end_x = this.x + (this.width / 2.0);
        const el_start_y = this.y - (this.height / 2.0);
        const el_end_y = this.y + (this.height / 2.0);

        return box_start_x <= el_start_x &&
            box_end_x >= el_end_x &&
            box_start_y <= el_start_y &&
            box_end_y >= el_end_y;
    }

    public getCssProperty(
        renderer: SDFGRenderer, propertyName: string
    ): string {
        return renderer.getCssProperty(propertyName);
    }
}

// SDFG as an element (to support properties)
export class SDFG extends SDFGElement {

    public constructor(sdfg: JsonSDFG) {
        super(sdfg, -1, sdfg);
    }

    public set_layout(): void {
        return;
    }

    public label(): string {
        return this.data.attributes.name;
    }

}

export class State extends SDFGElement {

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        const topleft = this.topleft();
        const visible_rect = renderer.get_visible_rect();
        let clamped;
        if (visible_rect)
            clamped = {
                x: Math.max(topleft.x, visible_rect.x),
                y: Math.max(topleft.y, visible_rect.y),
                x2: Math.min(
                    topleft.x + this.width, visible_rect.x + visible_rect.w
                ),
                y2: Math.min(
                    topleft.y + this.height, visible_rect.y + visible_rect.h
                ),
                w: 0,
                h: 0,
            };
        else
            clamped = {
                x: topleft.x,
                y: topleft.y,
                x2: topleft.x + this.width,
                y2: topleft.y + this.height,
                w: 0,
                h: 0,
            };
        clamped.w = clamped.x2 - clamped.x;
        clamped.h = clamped.y2 - clamped.y;
        if (!(ctx as any).lod)
            clamped = {
                x: topleft.x, y: topleft.y, x2: 0, y2: 0,
                w: this.width, h: this.height
            };

        ctx.fillStyle = this.getCssProperty(
            renderer, '--state-background-color'
        );
        ctx.fillRect(clamped.x, clamped.y, clamped.w, clamped.h);
        ctx.fillStyle = this.getCssProperty(
            renderer, '--state-foreground-color'
        );

        if (visible_rect && visible_rect.x <= topleft.x &&
            visible_rect.y <= topleft.y + SDFV.LINEHEIGHT)
            ctx.fillText(this.label(), topleft.x, topleft.y + SDFV.LINEHEIGHT);

        // If this state is selected or hovered
        if ((this.selected || this.highlighted || this.hovered) &&
            (clamped.x === topleft.x ||
                clamped.y === topleft.y ||
                clamped.x2 === topleft.x + this.width ||
                clamped.y2 === topleft.y + this.height)) {
            ctx.strokeStyle = this.strokeStyle(renderer);
            ctx.strokeRect(clamped.x, clamped.y, clamped.w, clamped.h);
        }

        // If collapsed, draw a "+" sign in the middle
        if (this.data.state.attributes.is_collapsed) {
            ctx.beginPath();
            ctx.moveTo(this.x, this.y - SDFV.LINEHEIGHT);
            ctx.lineTo(this.x, this.y + SDFV.LINEHEIGHT);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(this.x - SDFV.LINEHEIGHT, this.y);
            ctx.lineTo(this.x + SDFV.LINEHEIGHT, this.y);
            ctx.stroke();
        }

        ctx.strokeStyle = 'black';
    }

    public simple_draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, mousepos: Point2D
    ): void {
        // Fast drawing function for small states
        const topleft = this.topleft();

        ctx.fillStyle = this.getCssProperty(
            renderer, '--state-background-color'
        );
        ctx.fillRect(topleft.x, topleft.y, this.width, this.height);
        ctx.fillStyle = this.getCssProperty(renderer, '--state-text-color');

        if (mousepos && this.intersect(mousepos.x, mousepos.y))
            renderer.set_tooltip((c) => this.tooltip(c));

        // Draw state name in center without contents (does not look good)
        /*
        let FONTSIZE = Math.min(
            renderer.canvas_manager.points_per_pixel() * 16, 100
        );
        let label = this.label();

        let oldfont = ctx.font;
        ctx.font = FONTSIZE + "px Arial";

        let textmetrics = ctx.measureText(label);
        ctx.fillText(
            label, this.x - textmetrics.width / 2.0,
            this.y - this.height / 6.0 + FONTSIZE / 2.0
        );

        ctx.font = oldfont;
        */
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        const topleft = this.topleft();
        ctx.fillRect(topleft.x, topleft.y, this.width, this.height);

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

    public tooltip(container: HTMLElement): void {
        container.innerText = 'State: ' + this.label();
    }

    public attributes(): any {
        return this.data.state.attributes;
    }

    public label(): string {
        return this.data.state.label;
    }

    public type(): string {
        return this.data.state.type;
    }

}

export class SDFGNode extends SDFGElement {

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D,
        fgstyle: string = '--node-foreground-color',
        bgstyle: string = '--node-background-color'
    ): void {
        const topleft = this.topleft();
        const visible_rect = renderer.get_visible_rect();
        let clamped;
        if (visible_rect)
            clamped = {
                x: Math.max(topleft.x, visible_rect.x),
                y: Math.max(topleft.y, visible_rect.y),
                x2: Math.min(
                    topleft.x + this.width, visible_rect.x + visible_rect.w
                ),
                y2: Math.min(
                    topleft.y + this.height, visible_rect.y + visible_rect.h
                ),
                w: 0,
                h: 0,
            };
        else
            clamped = {
                x: topleft.x,
                y: topleft.y,
                x2: topleft.x + this.width,
                y2: topleft.y + this.height,
                w: 0,
                h: 0,
            };
        clamped.w = clamped.x2 - clamped.x;
        clamped.h = clamped.y2 - clamped.y;
        if (!(ctx as any).lod)
            clamped = {
                x: topleft.x, y: topleft.y, x2: 0, y2: 0,
                w: this.width, h: this.height
            };

        ctx.fillStyle = this.getCssProperty(renderer, bgstyle);
        ctx.fillRect(clamped.x, clamped.y, clamped.w, clamped.h);
        if (clamped.x === topleft.x &&
            clamped.y === topleft.y &&
            clamped.x2 === topleft.x + this.width &&
            clamped.y2 === topleft.y + this.height) {
            ctx.strokeStyle = this.strokeStyle(renderer);
            ctx.strokeRect(clamped.x, clamped.y, clamped.w, clamped.h);
        }
        if (this.label()) {
            ctx.fillStyle = this.getCssProperty(renderer, fgstyle);
            const textw = ctx.measureText(this.label()).width;
            if (!visible_rect)
                ctx.fillText(
                    this.label(), this.x - textw / 2, this.y + SDFV.LINEHEIGHT / 4
                );
            else if (visible_rect && visible_rect.x <= topleft.x &&
                visible_rect.y <= topleft.y + SDFV.LINEHEIGHT)
                ctx.fillText(
                    this.label(), this.x - textw / 2, this.y + SDFV.LINEHEIGHT / 4
                );
        }
    }

    public simple_draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        // Fast drawing function for small nodes
        const topleft = this.topleft();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-background-color'
        );
        ctx.fillRect(topleft.x, topleft.y, this.width, this.height);
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-foreground-color'
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        const topleft = this.topleft();
        ctx.fillRect(topleft.x, topleft.y, this.width, this.height);

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

    public label(): string {
        return this.data.node.label;
    }

    public attributes(): any {
        return this.data.node.attributes;
    }

    public type(): string {
        return this.data.node.type;
    }

    public set_layout(): void {
        this.width = this.data.node.attributes.layout.width;
        this.height = this.data.node.attributes.layout.height;
    }

}

export class Edge extends SDFGElement {

    public points: any[] = [];
    public src_connector: any;
    public dst_connector: any;

    public get_points(): any[] {
        return this.points;
    }

    public create_arrow_line(ctx: CanvasRenderingContext2D): void {
        ctx.beginPath();
        ctx.moveTo(this.points[0].x, this.points[0].y);
        if (this.points.length === 2) {
            // Straight line can be drawn
            ctx.lineTo(this.points[1].x, this.points[1].y);
        } else {
            let i;
            for (i = 1; i < this.points.length - 2; i++) {
                const xm = (this.points[i].x + this.points[i + 1].x) / 2.0;
                const ym = (this.points[i].y + this.points[i + 1].y) / 2.0;
                ctx.quadraticCurveTo(
                    this.points[i].x, this.points[i].y, xm, ym
                );
            }
            ctx.quadraticCurveTo(this.points[i].x, this.points[i].y,
                this.points[i + 1].x, this.points[i + 1].y);
        }
    }

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        const edge = this;

        this.create_arrow_line(ctx);

        let style = this.strokeStyle(renderer);
        if (this.hovered)
            renderer.set_tooltip((c) => this.tooltip(c, renderer));
        // Interstate edge
        if (this.parent_id == null &&
            style === this.getCssProperty(renderer, '--color-default')) {
            style = this.getCssProperty(renderer, '--interstate-edge-color');
        }
        ctx.fillStyle = ctx.strokeStyle = style;

        // CR edges have dashed lines
        if (this.parent_id != null && this.data.attributes.wcr != null)
            ctx.setLineDash([2, 2]);
        else
            ctx.setLineDash([1, 0]);

        ctx.stroke();

        ctx.setLineDash([1, 0]);

        if (edge.points.length < 2)
            return;


        // Show anchor points for moving
        if (this.selected && renderer.get_mouse_mode() === 'move') {
            let i;
            for (i = 1; i < this.points.length - 1; i++)
                ctx.strokeRect(
                    this.points[i].x - 5, this.points[i].y - 5, 8, 8
                );
        }

        drawArrow(
            ctx, edge.points[edge.points.length - 2],
            edge.points[edge.points.length - 1], 3
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        this.create_arrow_line(ctx);

        // Save current style properties.
        const orig_stroke_style = ctx.strokeStyle;
        const orig_fill_style = ctx.fillStyle;
        const orig_line_cap = ctx.lineCap;
        const orig_line_width = ctx.lineWidth;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.lineWidth = orig_line_width + 1;
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineCap = 'round';

        ctx.stroke();

        if (this.points.length < 2)
            return;
        drawArrow(ctx, this.points[this.points.length - 2],
            this.points[this.points.length - 1], 3, 0, 2);

        // Restore previous stroke style, width, and opacity.
        ctx.strokeStyle = orig_stroke_style;
        ctx.fillStyle = orig_fill_style;
        ctx.lineCap = orig_line_cap;
        ctx.lineWidth = orig_line_width;
        ctx.globalAlpha = orig_alpha;
    }

    public tooltip(
        container: HTMLElement, renderer: SDFGRenderer | undefined = undefined
    ): void {
        if (!renderer)
            return;

        super.tooltip(container);
        const dsettings = renderer.view_settings();
        const attr = this.attributes();
        // Memlet
        if (attr.subset !== undefined) {
            if (attr.subset === null) {  // Empty memlet
                container.style.display = 'none';
                return;
            }
            let contents = attr.data;
            contents += sdfg_property_to_string(attr.subset, dsettings);

            if (attr.other_subset)
                contents += ' -> ' + sdfg_property_to_string(
                    attr.other_subset, dsettings
                );

            if (attr.wcr)
                contents += '<br /><b>CR: ' + sdfg_property_to_string(
                    attr.wcr, dsettings
                ) + '</b>';

            let num_accesses = null;
            if (attr.volume)
                num_accesses = sdfg_property_to_string(attr.volume, dsettings);
            else
                num_accesses = sdfg_property_to_string(
                    attr.num_accesses, dsettings
                );

            if (attr.dynamic) {
                if (num_accesses == '0' || num_accesses == '-1')
                    num_accesses = '<b>Dynamic (unbounded)</b>';
                else
                    num_accesses = '<b>Dynamic</b> (up to ' +
                        num_accesses + ')';
            } else if (num_accesses == '-1') {
                num_accesses = '<b>Dynamic (unbounded)</b>';
            }

            contents += '<br /><font style="font-size: 14px">Volume: ' +
                num_accesses + '</font>';
            container.innerHTML = contents;
        } else {  // Interstate edge
            container.classList.add('sdfvtooltip--interstate-edge');
            container.innerText = this.label();
            if (!this.label())
                container.style.display = 'none';
        }
    }

    public set_layout(): void {
        // NOTE: Setting this.width/height will disrupt dagre in self-edges
    }

    public label(): string {
        // Memlet
        if (this.data.attributes.subset !== undefined)
            return '';
        return super.label();
    }

    public intersect(
        x: number, y: number, w: number = 0, h: number = 0
    ): boolean {
        // First, check bounding box
        if (!super.intersect(x, y, w, h))
            return false;

        // Then (if point), check distance from line
        if (w == 0 || h == 0) {
            for (let i = 0; i < this.points.length - 1; i++) {
                const dist = ptLineDistance(
                    { x: x, y: y }, this.points[i], this.points[i + 1]
                );
                if (dist <= 5.0)
                    return true;
            }
            return false;
        }
        return true;
    }

}

export class Connector extends SDFGElement {
    public custom_label: string | null = null;

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D | null, edge: Edge | null = null
    ): void {
        const scope_connector = (
            this.data.name.startsWith('IN_') ||
            this.data.name.startsWith('OUT_')
        );
        const topleft = this.topleft();
        ctx.beginPath();
        drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
        ctx.closePath();
        ctx.strokeStyle = this.strokeStyle(renderer);
        let fillColor;
        if (scope_connector) {
            let cname = this.data.name;
            if (cname.startsWith('IN_'))
                cname = cname.substring(3);
            else
                cname = cname.substring(4);

            ctx.lineWidth = 0.4;
            ctx.stroke();
            ctx.lineWidth = 1.0;
            fillColor = this.getCssProperty(
                renderer, '--connector-scoped-color'
            );
            this.custom_label = null;
        } else if (!edge) {
            ctx.stroke();
            fillColor = this.getCssProperty(renderer, '--node-missing-background-color');
            this.custom_label = "No edge connected";
        } else {
            ctx.stroke();
            fillColor = this.getCssProperty(
                renderer, '--connector-unscoped-color'
            );
            this.custom_label = null;
        }

        // PDFs do not support transparent fill colors
        if ((ctx as any).pdf)
            fillColor = fillColor.substr(0, 7);

        ctx.fillStyle = fillColor;

        // PDFs do not support stroke and fill on the same object
        if ((ctx as any).pdf) {
            ctx.beginPath();
            drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
            ctx.closePath();
        }
        ctx.fill();

        if (this.strokeStyle(renderer) !== this.getCssProperty(renderer, '--color-default'))
            renderer.set_tooltip((c) => this.tooltip(c));
    }

    public attributes(): any {
        return {};
    }

    public set_layout(): void {
        return;
    }

    public label(): string {
        if (this.custom_label)
            return this.data.name + ': ' + this.custom_label;
        return this.data.name;
    }

    public tooltip(container: HTMLElement): void {
        super.tooltip(container);
        if (this.custom_label)
            container.classList.add('sdfvtooltip--error');
        else
            container.classList.add('sdfvtooltip--connector');

        container.innerText = this.label();
    }

}

export class AccessNode extends SDFGNode {

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        const topleft = this.topleft();
        ctx.beginPath();
        drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
        ctx.closePath();
        ctx.strokeStyle = this.strokeStyle(renderer);

        const name = this.data.node.attributes.data;
        const nodedesc = this.sdfg.attributes._arrays[name];
        // Streams have dashed edges
        if (nodedesc && nodedesc.type === 'Stream') {
            ctx.setLineDash([5, 3]);
        } else {
            ctx.setLineDash([1, 0]);
        }

        // Non-transient (external) data is thicker
        if (nodedesc && nodedesc.attributes.transient === false) {
            ctx.lineWidth = 3.0;
        } else {
            ctx.lineWidth = 1.0;
        }
        ctx.stroke();
        ctx.lineWidth = 1.0;
        ctx.setLineDash([1, 0]);

        // Views are colored like connectors
        if (nodedesc && nodedesc.type === 'View') {
            ctx.fillStyle = this.getCssProperty(
                renderer, '--connector-unscoped-color'
            );
        } else if (nodedesc && this.sdfg.attributes.constants_prop[name] !== undefined) {
            ctx.fillStyle = this.getCssProperty(
                renderer, '--connector-scoped-color'
            );
        } else if (nodedesc) {
            ctx.fillStyle = this.getCssProperty(
                renderer, '--node-background-color'
            );
        } else {
            ctx.fillStyle = this.getCssProperty(
                renderer, '--node-missing-background-color'
            );
        }

        // PDFs do not support stroke and fill on the same object
        if ((ctx as any).pdf) {
            ctx.beginPath();
            drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
            ctx.closePath();
        }
        ctx.fill();
        if (nodedesc) {
            ctx.fillStyle = this.getCssProperty(
                renderer, '--node-foreground-color'
            );
        } else {
            ctx.fillStyle = this.getCssProperty(
                renderer, '--node-missing-foreground-color'
            );
            if (this.strokeStyle(renderer) !== this.getCssProperty(renderer, '--color-default'))
                renderer.set_tooltip((c) => this.tooltip(c));
        }
        const textmetrics = ctx.measureText(this.label());
        ctx.fillText(
            this.label(), this.x - textmetrics.width / 2.0,
            this.y + SDFV.LINEHEIGHT / 4.0
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        const topleft = this.topleft();
        ctx.beginPath();
        drawEllipse(ctx, topleft.x, topleft.y, this.width, this.height);
        ctx.closePath();
        ctx.fill();

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

    public tooltip(container: HTMLElement): void {
        super.tooltip(container);
        const nodedesc = this.sdfg.attributes._arrays[this.data.node.attributes.data];
        if (nodedesc)
            return;
        container.classList.add('sdfvtooltip--error');
        container.innerText = 'Undefined array';
    }

}

export class ScopeNode extends SDFGNode {

    private cached_far_label: string | null = null;
    private cached_close_label: string | null = null;

    private schedule_label_dict: { [key: string]: string } = {
        'Default': 'Default',
        'Sequential': 'Seq',
        'MPI': 'MPI',
        'CPU_Multicore': 'OMP',
        'Unrolled': 'Unroll',
        'SVE_Map': 'SVE',
        'GPU_Default': 'GPU',
        'GPU_Device': 'GPU',
        'GPU_ThreadBlock': 'Block',
        'GPU_ThreadBlock_Dynamic': 'Block-Dyn',
        'GPU_Persistent': 'GPU-P',
        'FPGA_Device': 'FPGA',
    };

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        let draw_shape;
        if (this.data.node.attributes.is_collapsed) {
            draw_shape = () => {
                drawHexagon(ctx, this.x, this.y, this.width, this.height, {
                    x: 0,
                    y: 0,
                });
            };
        } else {
            draw_shape = () => {
                drawTrapezoid(ctx, this.topleft(), this, this.scopeend());
            };
        }
        ctx.strokeStyle = this.strokeStyle(renderer);

        // Consume scopes have dashed edges
        if (this.data.node.type.startsWith('Consume'))
            ctx.setLineDash([5, 3]);
        else
            ctx.setLineDash([1, 0]);

        draw_shape();
        ctx.stroke();
        ctx.setLineDash([1, 0]);
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-background-color'
        );
        // PDFs do not support stroke and fill on the same object
        if ((ctx as any).pdf)
            draw_shape();
        ctx.fill();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-foreground-color'
        );

        drawAdaptiveText(
            ctx, renderer, this.far_label(renderer),
            this.close_label(renderer), this.x, this.y,
            this.width, this.height,
            SDFV.SCOPE_LOD
        );

        drawAdaptiveText(
            ctx, renderer, '', this.schedule_label(), this.x, this.y,
            this.width, this.height,
            SDFV.SCOPE_LOD, SDFV.DEFAULT_MAX_FONTSIZE, 0.7,
            SDFV.DEFAULT_FAR_FONT_MULTIPLIER, true,
            TextVAlign.BOTTOM, TextHAlign.RIGHT, {
            bottom: 2.0,
            right: this.height,
        }
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        if (this.data.node.attributes.is_collapsed)
            drawHexagon(ctx, this.x, this.y, this.width, this.height, {
                x: 0,
                y: 0,
            });
        else
            drawTrapezoid(ctx, this.topleft(), this, this.scopeend());
        ctx.fill();

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

    public schedule_label(): string {
        let attrs = this.attributes();
        if (this.scopeend() && this.parent_id !== null) {
            const entry = this.sdfg.nodes[this.parent_id].nodes[
                this.data.node.scope_entry
            ];
            if (entry !== undefined)
                attrs = entry.attributes;
        }

        let label = attrs.schedule;
        try {
            label = this.schedule_label_dict[attrs.schedule];
        } catch (_err) {
        }

        return label;
    }

    public far_label(
        renderer: SDFGRenderer, recompute: boolean = false
    ): string {
        if (!recompute && this.cached_far_label)
            return this.cached_far_label;

        let result = '[';

        let attrs = this.attributes();
        if (this.scopeend() && this.parent_id !== null) {
            const entry = this.sdfg.nodes[this.parent_id].nodes[
                this.data.node.scope_entry
            ];
            if (entry !== undefined)
                attrs = entry.attributes;
            else
                return this.label();
        }

        if (this instanceof ConsumeEntry || this instanceof ConsumeExit) {
            result += sdfg_consume_elem_to_string(
                attrs.num_pes, renderer.view_settings()
            );
        } else {
            for (let i = 0; i < attrs.params.length; ++i)
                result += sdfg_range_elem_to_string(
                    attrs.range.ranges[i], renderer.view_settings()
                ) + ', ';
            // Remove trailing comma
            result = result.substring(0, result.length - 2);
        }
        result += ']';

        this.cached_far_label = result;

        return result;
    }

    public close_label(
        renderer: SDFGRenderer, recompute: boolean = false
    ): string {
        if (!recompute && this.cached_close_label)
            return this.cached_close_label;

        let attrs = this.attributes();

        let result = '';
        if (this.scopeend() && this.parent_id !== null) {
            const entry = this.sdfg.nodes[this.parent_id].nodes[
                this.data.node.scope_entry
            ];
            if (entry !== undefined)
                attrs = entry.attributes;
        }

        result += '[';
        if (this instanceof ConsumeEntry || this instanceof ConsumeExit) {
            result += attrs.pe_index + '=' + sdfg_consume_elem_to_string(
                attrs.num_pes, renderer.view_settings()
            );
        } else {
            for (let i = 0; i < attrs.params.length; ++i) {
                result += attrs.params[i] + '=';
                result += sdfg_range_elem_to_string(
                    attrs.range.ranges[i], renderer.view_settings()
                ) + ', ';
            }
            // Remove trailing comma
            result = result.substring(0, result.length - 2);
        }
        result += ']';

        this.cached_close_label = result;

        return result;
    }

    public scopeend(): boolean {
        return false;
    }

    public clear_cached_labels(): void {
        this.cached_close_label = null;
        this.cached_far_label = null;
    }

}

export class EntryNode extends ScopeNode {

    public scopeend(): boolean {
        return false;
    }

}

export class ExitNode extends ScopeNode {

    public scopeend(): boolean {
        return true;
    }

}

export class MapEntry extends EntryNode {

    public stroketype(ctx: CanvasRenderingContext2D): void {
        ctx.setLineDash([1, 0]);
    }

}

export class MapExit extends ExitNode {

    public stroketype(ctx: CanvasRenderingContext2D): void {
        ctx.setLineDash([1, 0]);
    }

}

export class ConsumeEntry extends EntryNode {

    public stroketype(ctx: CanvasRenderingContext2D): void {
        ctx.setLineDash([5, 3]);
    }

}

export class ConsumeExit extends ExitNode {

    public stroketype(ctx: CanvasRenderingContext2D): void {
        ctx.setLineDash([5, 3]);
    }

}

export class PipelineEntry extends EntryNode {

    public stroketype(ctx: CanvasRenderingContext2D): void {
        ctx.setLineDash([10, 3]);
    }

}
export class PipelineExit extends ExitNode {

    public stroketype(ctx: CanvasRenderingContext2D): void {
        ctx.setLineDash([10, 3]);
    }

}

export class Tasklet extends SDFGNode {

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        const canvas_manager = renderer.get_canvas_manager();
        if (!canvas_manager)
            return;

        const topleft = this.topleft();
        drawOctagon(ctx, topleft, this.width, this.height);
        ctx.strokeStyle = this.strokeStyle(renderer);
        ctx.stroke();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-background-color'
        );

        // PDFs do not support stroke and fill on the same object
        if ((ctx as any).pdf)
            drawOctagon(ctx, topleft, this.width, this.height);
        ctx.fill();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-foreground-color'
        );

        const ppp = canvas_manager.points_per_pixel();
        if (!(ctx as any).lod || ppp < SDFV.TASKLET_LOD) {
            // If we are close to the tasklet, show its contents
            const code = this.attributes().code.string_data;
            const lines = code.split('\n');
            let maxline = 0, maxline_len = 0;
            for (let i = 0; i < lines.length; i++) {
                if (lines[i].length > maxline_len) {
                    maxline = i;
                    maxline_len = lines[i].length;
                }
            }
            const oldfont = ctx.font;
            ctx.font = '10px courier new';
            const textmetrics = ctx.measureText(lines[maxline]);

            // Fit font size to 80% height and width of tasklet
            const height = lines.length * SDFV.LINEHEIGHT * 1.05;
            const width = textmetrics.width;
            const TASKLET_WRATIO = 0.9, TASKLET_HRATIO = 0.5;
            const hr = height / (this.height * TASKLET_HRATIO);
            const wr = width / (this.width * TASKLET_WRATIO);
            const FONTSIZE = Math.min(10 / hr, 10 / wr);
            const text_yoffset = FONTSIZE / 4;

            ctx.font = FONTSIZE + 'px courier new';
            // Set the start offset such that the middle row of the text is in
            // this.y
            const y = this.y + text_yoffset - (
                (lines.length - 1) / 2
            ) * FONTSIZE * 1.05;
            for (let i = 0; i < lines.length; i++)
                ctx.fillText(
                    lines[i], this.x - (this.width * TASKLET_WRATIO) / 2.0,
                    y + i * FONTSIZE * 1.05
                );

            ctx.font = oldfont;
            return;
        }

        const textmetrics = ctx.measureText(this.label());
        ctx.fillText(
            this.label(), this.x - textmetrics.width / 2.0,
            this.y + SDFV.LINEHEIGHT / 2.0
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        drawOctagon(ctx, this.topleft(), this.width, this.height);
        ctx.fill();

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

}

export class Reduce extends SDFGNode {

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        const topleft = this.topleft();
        const draw_shape = () => {
            ctx.beginPath();
            ctx.moveTo(topleft.x, topleft.y);
            ctx.lineTo(topleft.x + this.width / 2, topleft.y + this.height);
            ctx.lineTo(topleft.x + this.width, topleft.y);
            ctx.lineTo(topleft.x, topleft.y);
            ctx.closePath();
        };
        ctx.strokeStyle = this.strokeStyle(renderer);
        draw_shape();
        ctx.stroke();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-background-color'
        );
        // PDFs do not support stroke and fill on the same object
        if ((ctx as any).pdf)
            draw_shape();
        ctx.fill();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-foreground-color'
        );

        const far_label = this.label().substring(4, this.label().indexOf(','));
        drawAdaptiveText(
            ctx, renderer, far_label,
            this.label(), this.x, this.y - this.height * 0.2,
            this.width, this.height,
            SDFV.SCOPE_LOD
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        const topleft = this.topleft();
        ctx.beginPath();
        ctx.moveTo(topleft.x, topleft.y);
        ctx.lineTo(topleft.x + this.width / 2, topleft.y + this.height);
        ctx.lineTo(topleft.x + this.width, topleft.y);
        ctx.lineTo(topleft.x, topleft.y);
        ctx.closePath();
        ctx.fill();

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

}

export class NestedSDFG extends SDFGNode {

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        mousepos: Point2D
    ): void {
        if (this.data.node.attributes.is_collapsed) {
            const topleft = this.topleft();
            drawOctagon(ctx, topleft, this.width, this.height);
            ctx.strokeStyle = this.strokeStyle(renderer);
            ctx.stroke();
            drawOctagon(
                ctx, { x: topleft.x + 2.5, y: topleft.y + 2.5 }, this.width - 5,
                this.height - 5
            );
            ctx.strokeStyle = this.strokeStyle(renderer);
            ctx.stroke();
            ctx.fillStyle = this.getCssProperty(
                renderer, '--node-background-color'
            );
            // PDFs do not support stroke and fill on the same object
            if ((ctx as any).pdf)
                drawOctagon(
                    ctx, { x: topleft.x + 2.5, y: topleft.y + 2.5 },
                    this.width - 5, this.height - 5
                );
            ctx.fill();
            ctx.fillStyle = this.getCssProperty(
                renderer, '--node-foreground-color'
            );
            const label = this.data.node.attributes.label;
            const textmetrics = ctx.measureText(label);
            ctx.fillText(
                label, this.x - textmetrics.width / 2.0,
                this.y + SDFV.LINEHEIGHT / 4.0
            );
            return;
        }

        // Draw square around nested SDFG
        super.draw(
            renderer, ctx, mousepos, '--nested-sdfg-foreground-color',
            '--nested-sdfg-background-color'
        );

        // Draw nested graph
        draw_sdfg(renderer, ctx, this.data.graph, mousepos);
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        if (this.data.node.attributes.is_collapsed) {
            // Save the current style properties.
            const orig_fill_style = ctx.fillStyle;
            const orig_alpha = ctx.globalAlpha;

            ctx.globalAlpha = alpha;
            ctx.fillStyle = color;

            drawOctagon(ctx, this.topleft(), this.width, this.height);
            ctx.fill();

            // Restore the previous style properties.
            ctx.fillStyle = orig_fill_style;
            ctx.globalAlpha = orig_alpha;
        } else {
            super.shade(renderer, ctx, color, alpha);
        }
    }

    public set_layout(): void {
        if (this.data.node.attributes.is_collapsed) {
            const labelsize =
                this.data.node.attributes.label.length * SDFV.LINEHEIGHT * 0.8;
            const inconnsize = 2 * SDFV.LINEHEIGHT * Object.keys(
                this.data.node.attributes.in_connectors
            ).length - SDFV.LINEHEIGHT;
            const outconnsize = 2 * SDFV.LINEHEIGHT * Object.keys(
                this.data.node.attributes.out_connectors
            ).length - SDFV.LINEHEIGHT;
            const maxwidth = Math.max(labelsize, inconnsize, outconnsize);
            let maxheight = 2 * SDFV.LINEHEIGHT;
            maxheight += 4 * SDFV.LINEHEIGHT;

            const size = { width: maxwidth, height: maxheight };
            size.width += 2.0 * (size.height / 3.0);
            size.height /= 1.75;

            this.width = size.width;
            this.height = size.height;
        } else {
            this.width = this.data.node.attributes.layout.width;
            this.height = this.data.node.attributes.layout.height;
        }
    }

    public label(): string {
        return '';
    }

}

export class LibraryNode extends SDFGNode {

    private _path(ctx: CanvasRenderingContext2D): void {
        const hexseg = this.height / 6.0;
        const topleft = this.topleft();
        ctx.beginPath();
        ctx.moveTo(topleft.x, topleft.y);
        ctx.lineTo(topleft.x + this.width - hexseg, topleft.y);
        ctx.lineTo(topleft.x + this.width, topleft.y + hexseg);
        ctx.lineTo(topleft.x + this.width, topleft.y + this.height);
        ctx.lineTo(topleft.x, topleft.y + this.height);
        ctx.closePath();
    }

    private _path2(ctx: CanvasRenderingContext2D): void {
        const hexseg = this.height / 6.0;
        const topleft = this.topleft();
        ctx.beginPath();
        ctx.moveTo(topleft.x + this.width - hexseg, topleft.y);
        ctx.lineTo(topleft.x + this.width - hexseg, topleft.y + hexseg);
        ctx.lineTo(topleft.x + this.width, topleft.y + hexseg);
    }

    public draw(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
        _mousepos: Point2D
    ): void {
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-background-color'
        );
        this._path(ctx);
        ctx.fill();
        ctx.strokeStyle = this.strokeStyle(renderer);
        this._path(ctx);
        ctx.stroke();
        this._path2(ctx);
        ctx.stroke();
        ctx.fillStyle = this.getCssProperty(
            renderer, '--node-foreground-color'
        );
        const textw = ctx.measureText(this.label()).width;
        ctx.fillText(
            this.label(), this.x - textw / 2, this.y + SDFV.LINEHEIGHT / 4
        );
    }

    public shade(
        renderer: SDFGRenderer, ctx: CanvasRenderingContext2D, color: string,
        alpha: number = 0.6
    ): void {
        // Save the current style properties.
        const orig_fill_style = ctx.fillStyle;
        const orig_alpha = ctx.globalAlpha;

        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        this._path(ctx);
        ctx.fill();

        // Restore the previous style properties.
        ctx.fillStyle = orig_fill_style;
        ctx.globalAlpha = orig_alpha;
    }

}

//////////////////////////////////////////////////////

// Draw an entire SDFG
export function draw_sdfg(
    renderer: SDFGRenderer, ctx: CanvasRenderingContext2D,
    sdfg_dagre: DagreSDFG, mousepos: Point2D | null
): void {
    const canvas_manager = renderer.get_canvas_manager();
    if (!canvas_manager)
        return;

    const ppp = canvas_manager.points_per_pixel();

    // Render state machine
    const g = sdfg_dagre;
    if (!(ctx as any).lod || ppp < SDFV.EDGE_LOD)
        g.edges().forEach(e => {
            const edge = g.edge(e);
            edge.draw(renderer, ctx, mousepos);
            edge.debug_draw(renderer, ctx);
        });


    const visible_rect = renderer.get_visible_rect();

    // Render each visible state's contents
    g.nodes().forEach((v: string) => {
        const node = g.node(v);

        if ((ctx as any).lod &&
            (Math.max(node.width, node.height) / ppp < SDFV.STATE_LOD)) {
            node.simple_draw(renderer, ctx, mousepos);
            node.debug_draw(renderer, ctx);
            return;
        }
        // Skip invisible states
        if ((ctx as any).lod && visible_rect && !node.intersect(
            visible_rect.x, visible_rect.y, visible_rect.w, visible_rect.h
        ))
            return;

        node.draw(renderer, ctx, mousepos);
        node.debug_draw(renderer, ctx);

        const ng = node.data.graph;

        if (!node.data.state.attributes.is_collapsed && ng) {
            ng.nodes().forEach((v: any) => {
                const n = ng.node(v);

                if ((ctx as any).lod && visible_rect && !n.intersect(
                    visible_rect.x, visible_rect.y, visible_rect.w,
                    visible_rect.h
                ))
                    return;
                if ((ctx as any).lod && node.height / ppp < SDFV.NODE_LOD) {
                    n.simple_draw(renderer, ctx, mousepos);
                    n.debug_draw(renderer, ctx);
                    return;
                }

                n.draw(renderer, ctx, mousepos);
                n.debug_draw(renderer, ctx);
                n.in_connectors.forEach((c: Connector) => {
                    let edge: Edge | null = null;
                    ng.inEdges(v).forEach((e: JsonSDFGEdge) => {
                        const eobj: Edge = ng.edge(e);
                        if (eobj.dst_connector == c.data.name)
                            edge = eobj;
                    });

                    c.draw(renderer, ctx, mousepos, edge);
                    c.debug_draw(renderer, ctx);
                });
                n.out_connectors.forEach((c: Connector) => {
                    let edge: Edge | null = null;
                    ng.outEdges(v).forEach((e: JsonSDFGEdge) => {
                        const eobj: Edge = ng.edge(e);
                        if (eobj.src_connector == c.data.name)
                            edge = eobj;
                    });

                    c.draw(renderer, ctx, mousepos, edge);
                    c.debug_draw(renderer, ctx);
                });
            });
            if ((ctx as any).lod && ppp >= SDFV.EDGE_LOD)
                return;
            ng.edges().forEach((e: any) => {
                const edge = ng.edge(e);
                if ((ctx as any).lod && visible_rect && !edge.intersect(
                    visible_rect.x, visible_rect.y, visible_rect.w,
                    visible_rect.h
                ))
                    return;
                edge.draw(renderer, ctx, mousepos);
                edge.debug_draw(renderer, ctx);
            });
        }
    });
}

// Translate an SDFG by a given offset
export function offset_sdfg(
    sdfg: JsonSDFG, sdfg_graph: DagreSDFG, offset: Point2D
): void {
    sdfg.nodes.forEach((state: JsonSDFGState, id: number) => {
        const g = sdfg_graph.node(id.toString());
        g.x += offset.x;
        g.y += offset.y;
        if (!state.attributes.is_collapsed)
            offset_state(state, g, offset);
    });
    sdfg.edges.forEach((e: JsonSDFGEdge, _eid: number) => {
        const edge = sdfg_graph.edge(e.src, e.dst);
        edge.x += offset.x;
        edge.y += offset.y;
        edge.points.forEach((p) => {
            p.x += offset.x;
            p.y += offset.y;
        });
    });
}

// Translate nodes, edges, and connectors in a given SDFG state by an offset
export function offset_state(
    state: JsonSDFGState, state_graph: State, offset: Point2D
): void {
    const drawn_nodes: Set<string> = new Set();

    state.nodes.forEach((_n: JsonSDFGNode, nid: number) => {
        const node = state_graph.data.graph.node(nid);
        if (!node) return;
        drawn_nodes.add(nid.toString());

        node.x += offset.x;
        node.y += offset.y;
        node.in_connectors.forEach((c: Connector) => {
            c.x += offset.x;
            c.y += offset.y;
        });
        node.out_connectors.forEach((c: Connector) => {
            c.x += offset.x;
            c.y += offset.y;
        });

        if (node.data.node.type === 'NestedSDFG')
            offset_sdfg(
                node.data.node.attributes.sdfg, node.data.graph, offset
            );
    });
    state.edges.forEach((e: JsonSDFGEdge, eid: number) => {
        const ne = check_and_redirect_edge(e, drawn_nodes, state);
        if (!ne) return;
        e = ne;
        const edge = state_graph.data.graph.edge(e.src, e.dst, eid);
        if (!edge) return;
        edge.x += offset.x;
        edge.y += offset.y;
        edge.points.forEach((p: Point2D) => {
            p.x += offset.x;
            p.y += offset.y;
        });
    });
}


///////////////////////////////////////////////////////

enum TextVAlign {
    TOP,
    MIDDLE,
    BOTTOM,
}

enum TextHAlign {
    LEFT,
    CENTER,
    RIGHT,
}

type AdaptiveTextPadding = {
    left?: number,
    top?: number,
    right?: number,
    bottom?: number,
}

export function drawAdaptiveText(
    ctx: CanvasRenderingContext2D, renderer: SDFGRenderer, far_text: string,
    close_text: string, x: number, y: number, w: number, h: number,
    ppp_thres: number,
    max_font_size: number = SDFV.DEFAULT_MAX_FONTSIZE,
    close_font_multiplier: number = 1.0,
    far_font_multiplier: number = SDFV.DEFAULT_FAR_FONT_MULTIPLIER,
    bold: boolean = false,
    valign: TextVAlign = TextVAlign.MIDDLE,
    halign: TextHAlign = TextHAlign.CENTER,
    padding: AdaptiveTextPadding = {}
): void {
    // Save font.
    const oldfont = ctx.font;

    const ppp = renderer.get_canvas_manager()?.points_per_pixel();
    if (ppp === undefined)
        return;

    const is_far: boolean = (ctx as any).lod && ppp >= ppp_thres;
    const label = is_far ? far_text : close_text;

    let font_size = Math.min(
        SDFV.DEFAULT_CANVAS_FONTSIZE * close_font_multiplier, max_font_size
    );
    if (is_far)
        font_size = Math.min(ppp * far_font_multiplier, max_font_size);
    ctx.font = font_size + 'px sans-serif';

    const label_metrics = ctx.measureText(label);

    let label_width = Math.abs(label_metrics.actualBoundingBoxLeft) +
        Math.abs(label_metrics.actualBoundingBoxRight);
    let label_height = Math.abs(label_metrics.actualBoundingBoxDescent) +
        Math.abs(label_metrics.actualBoundingBoxAscent);

    const padding_left = padding.left !== undefined ? padding.left : 1.0;
    const padding_top = padding.top !== undefined ? padding.top : 0.0;
    const padding_right = padding.right !== undefined ? padding.right : 1.0;
    const padding_bottom = padding.bottom !== undefined ? padding.bottom : 4.0;

    // Ensure text is not resized beyond the bounds of the box
    if (is_far && label_width > w) {
        const old_font_size = font_size;
        font_size = font_size / (label_width / w);
        label_width /= (label_width / w);
        label_height /= (old_font_size / font_size);
        ctx.font = font_size + 'px sans-serif';
    }

    let text_center_x;
    let text_center_y;
    switch (valign) {
        case TextVAlign.TOP:
            text_center_y = y - (h / 2.0) + (label_height + padding_top);
            break;
        case TextVAlign.BOTTOM:
            text_center_y = y + (h / 2.0) - padding_bottom;
            break;
        case TextVAlign.MIDDLE:
        default:
            text_center_y = y + (label_height / 2.0);
            break;
    }
    switch (halign) {
        case TextHAlign.LEFT:
            text_center_x = (x - (w / 2.0)) + padding_left;
            break;
        case TextHAlign.RIGHT:
            text_center_x = (x + (w / 2.0)) - (label_width + padding_right);
            break;
        case TextHAlign.CENTER:
        default:
            text_center_x = x - (label_width / 2.0);
            break;
    }

    if (bold)
        ctx.font = 'bold ' + ctx.font;

    ctx.fillText(label, text_center_x, text_center_y);

    // Restore previous font.
    ctx.font = oldfont;
}

export function drawHexagon(
    ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number,
    offset: Point2D
): void {
    const topleft = { x: x - w / 2.0, y: y - h / 2.0 };
    const hexseg = h / 3.0;
    ctx.beginPath();
    ctx.moveTo(topleft.x, y);
    ctx.lineTo(topleft.x + hexseg, topleft.y);
    ctx.lineTo(topleft.x + w - hexseg, topleft.y);
    ctx.lineTo(topleft.x + w, y);
    ctx.lineTo(topleft.x + w - hexseg, topleft.y + h);
    ctx.lineTo(topleft.x + hexseg, topleft.y + h);
    ctx.lineTo(topleft.x, y);
    ctx.closePath();
}

export function drawOctagon(
    ctx: CanvasRenderingContext2D, topleft: Point2D, width: number,
    height: number
): void {
    const octseg = height / 3.0;
    ctx.beginPath();
    ctx.moveTo(topleft.x, topleft.y + octseg);
    ctx.lineTo(topleft.x + octseg, topleft.y);
    ctx.lineTo(topleft.x + width - octseg, topleft.y);
    ctx.lineTo(topleft.x + width, topleft.y + octseg);
    ctx.lineTo(topleft.x + width, topleft.y + 2 * octseg);
    ctx.lineTo(topleft.x + width - octseg, topleft.y + height);
    ctx.lineTo(topleft.x + octseg, topleft.y + height);
    ctx.lineTo(topleft.x, topleft.y + 2 * octseg);
    ctx.lineTo(topleft.x, topleft.y + 1 * octseg);
    ctx.closePath();
}

// Adapted from https://stackoverflow.com/a/2173084/6489142
export function drawEllipse(
    ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number
): void {
    const kappa = .5522848,
        ox = (w / 2) * kappa, // control point offset horizontal
        oy = (h / 2) * kappa, // control point offset vertical
        xe = x + w,           // x-end
        ye = y + h,           // y-end
        xm = x + w / 2,       // x-middle
        ym = y + h / 2;       // y-middle

    ctx.moveTo(x, ym);
    ctx.bezierCurveTo(x, ym - oy, xm - ox, y, xm, y);
    ctx.bezierCurveTo(xm + ox, y, xe, ym - oy, xe, ym);
    ctx.bezierCurveTo(xe, ym + oy, xm + ox, ye, xm, ye);
    ctx.bezierCurveTo(xm - ox, ye, x, ym + oy, x, ym);
}

export function drawArrow(
    ctx: CanvasRenderingContext2D, p1: Point2D, p2: Point2D, size: number,
    offset: number = 0, padding: number = 0
): void {
    // Rotate the context to point along the path
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const rot = Math.atan2(dy, dx);
    ctx.translate(p2.x, p2.y);
    ctx.rotate(rot);

    // arrowhead
    ctx.beginPath();
    ctx.moveTo(0 + padding + offset, 0);
    ctx.lineTo(((-2 * size) - padding) - offset, -(size + padding));
    ctx.lineTo(((-2 * size) - padding) - offset, (size + padding));
    ctx.closePath();
    ctx.fill();

    // Restore context
    ctx.rotate(-rot);
    ctx.translate(-p2.x, -p2.y);

}

export function drawTrapezoid(
    ctx: CanvasRenderingContext2D, topleft: Point2D, node: SDFGNode,
    inverted: boolean = false
): void {
    ctx.beginPath();
    if (inverted) {
        ctx.moveTo(topleft.x, topleft.y);
        ctx.lineTo(topleft.x + node.width, topleft.y);
        ctx.lineTo(
            topleft.x + node.width - node.height, topleft.y + node.height
        );
        ctx.lineTo(topleft.x + node.height, topleft.y + node.height);
        ctx.lineTo(topleft.x, topleft.y);
    } else {
        ctx.moveTo(topleft.x, topleft.y + node.height);
        ctx.lineTo(topleft.x + node.width, topleft.y + node.height);
        ctx.lineTo(topleft.x + node.width - node.height, topleft.y);
        ctx.lineTo(topleft.x + node.height, topleft.y);
        ctx.lineTo(topleft.x, topleft.y + node.height);
    }
    ctx.closePath();
}

// Returns the distance from point p to line defined by two points
// (line1, line2)
export function ptLineDistance(
    p: Point2D, line1: Point2D, line2: Point2D
): number {
    const dx = (line2.x - line1.x);
    const dy = (line2.y - line1.y);
    const res = dy * p.x - dx * p.y + line2.x * line1.y - line2.y * line1.x;

    return Math.abs(res) / Math.sqrt(dy * dy + dx * dx);
}

/**
 * Get the color on a green-red temperature scale based on a fractional value.
 * @param {Number} val Value between 0 and 1, 0 = green, .5 = yellow, 1 = red
 * @returns            HSL color string
 */
export function getTempColor(val: number): string {
    if (val < 0)
        val = 0;
    if (val > 1)
        val = 1;
    const hue = ((1 - val) * 120).toString(10);
    return 'hsl(' + hue + ',100%,50%)';
}

export const SDFGElements: { [name: string]: typeof SDFGElement } = {
    SDFGElement,
    SDFG,
    State,
    SDFGNode,
    Edge,
    Connector,
    AccessNode,
    ScopeNode,
    EntryNode,
    ExitNode,
    MapEntry,
    MapExit,
    ConsumeEntry,
    ConsumeExit,
    Tasklet,
    Reduce,
    PipelineEntry,
    PipelineExit,
    NestedSDFG,
    LibraryNode
};
