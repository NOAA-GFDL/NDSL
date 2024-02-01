// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { OverlayManager, SymbolResolver } from '../overlay_manager';
import { SDFGRenderer } from '../renderer/renderer';
import { SDFGElement } from '../renderer/renderer_elements';
import { Point2D } from '../index';

declare const vscode: any;

export class GenericSdfgOverlay {

    protected symbol_resolver: SymbolResolver;
    protected vscode: any;
    protected badness_scale_center: number;
    protected overlay_manager: OverlayManager;

    public constructor(
        protected renderer: SDFGRenderer
    ) {
        this.overlay_manager = renderer.get_overlay_manager();
        this.symbol_resolver = this.overlay_manager.get_symbol_resolver();
        this.vscode = typeof vscode !== 'undefined' && vscode;
        this.badness_scale_center = 5;
    }

    public draw(): void {
        return;
    }

    public on_mouse_event(
        _type: string,
        _ev: MouseEvent,
        _mousepos: Point2D,
        _elements: SDFGElement[],
        _foreground_elem: SDFGElement | undefined,
        _ends_drag: boolean
    ): boolean {
        return false;
    }

    public refresh(): void {
        return;
    }

}
