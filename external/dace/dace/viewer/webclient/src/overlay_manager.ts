// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { createElement } from './utils/utils';
import { MathNode, parse } from 'mathjs';
import { SDFGRenderer } from './renderer/renderer';
import { Point2D, SymbolMap } from './index';
import { GenericSdfgOverlay } from './overlays/generic_sdfg_overlay';
import { SDFGElement } from './renderer/renderer_elements';

// Some global functions and variables which are only accessible within VSCode:
declare const vscode: any;

export class SymbolResolver {

    private sdfg: any;
    private symbol_value_map: SymbolMap = {};
    private symbols_to_define: string[] = [];
    private popup_dialogue: any = undefined;

    public constructor(
        private readonly renderer: SDFGRenderer
    ) {
        this.sdfg = this.renderer.get_sdfg();

        // Initialize the symbol mapping to the graph's symbol table.
        Object.keys(this.sdfg.attributes.symbols).forEach((s) => {
            if (this.sdfg.attributes.constants_prop !== undefined &&
                Object.keys(this.sdfg.attributes.constants_prop).includes(s) &&
                this.sdfg.attributes.constants_prop[s][0]['type'] === 'Scalar')
                this.symbol_value_map[s] = this.sdfg.attributes.constants_prop[
                    s
                ][1];
            else
                this.symbol_value_map[s] = undefined;
        });

        this.init_overlay_popup_dialogue();
    }

    public symbol_value_changed(
        symbol: string, value: number | undefined
    ): void {
        if (symbol in this.symbol_value_map)
            this.symbol_value_map[symbol] = value;
    }

    public parse_symbol_expression(
        expression_string: string,
        mapping: SymbolMap,
        prompt_completion: boolean = false,
        callback: CallableFunction | undefined = undefined
    ): number | undefined {
        let result: number | undefined = undefined;
        try {
            const expression_tree = parse(expression_string);
            if (prompt_completion) {
                this.recursive_find_undefined_symbol(expression_tree, mapping);
                this.prompt_define_symbol(mapping, callback);
            } else {
                try {
                    const evaluated =
                        expression_tree.evaluate(mapping);
                    if (evaluated !== undefined &&
                        !isNaN(evaluated) &&
                        Number.isInteger(+evaluated))
                        result = +evaluated;
                    else
                        result = undefined;
                } catch (e) {
                    result = undefined;
                }
            }
            return result;
        } catch (exception) {
            console.error(exception);
            return result;
        }
    }

    public prompt_define_symbol(
        mapping: SymbolMap, callback: CallableFunction | undefined = undefined
    ): void {
        if (this.symbols_to_define.length > 0) {
            const symbol = this.symbols_to_define.pop();
            if (symbol === undefined)
                return;
            const that = this;
            this.popup_dialogue._show(
                symbol,
                mapping,
                () => {
                    if (this.renderer.get_in_vscode())
                        vscode.postMessage({
                            type: 'analysis.define_symbol',
                            symbol: symbol,
                            definition: mapping[symbol],
                        });
                    if (callback !== undefined)
                        callback();
                    that.prompt_define_symbol(mapping, callback);
                }
            );
        }
    }

    public recursive_find_undefined_symbol(
        expression_tree: MathNode, mapping: SymbolMap
    ): void {
        expression_tree.forEach((
            node: MathNode, _path: string, _parent: MathNode
        ) => {
            switch (node.type) {
                case 'SymbolNode':
                    if (node.name && node.name in mapping &&
                        mapping[node.name] === undefined &&
                        !this.symbols_to_define.includes(node.name)) {
                        // This is an undefined symbol.
                        // Ask for it to be defined.
                        this.symbols_to_define.push(node.name);
                    }
                    break;
                case 'OperatorNode':
                case 'ParenthesisNode':
                    this.recursive_find_undefined_symbol(node, mapping);
                    break;
                default:
                    // Ignore
                    break;
            }
        });
    }

    public init_overlay_popup_dialogue(): void {
        const dialogue_background: any = createElement(
            'div', '', ['sdfv_modal_background'], document.body
        );
        dialogue_background._show = function () {
            this.style.display = 'block';
        };
        dialogue_background._hide = function () {
            this.style.display = 'none';
        };

        const popup_dialogue: any = createElement(
            'div', 'sdfv_overlay_dialogue', ['sdfv_modal'], dialogue_background
        );
        popup_dialogue.addEventListener('click', (ev: Event) => {
            ev.stopPropagation();
        });
        popup_dialogue.style.display = 'none';
        this.popup_dialogue = popup_dialogue;

        const header_bar = createElement(
            'div', '', ['sdfv_modal_title_bar'], this.popup_dialogue
        );
        this.popup_dialogue._title = createElement(
            'span', '', ['sdfv_modal_title'], header_bar
        );
        const close_button = createElement(
            'div', '', ['modal_close'], header_bar
        );
        close_button.innerHTML = '<i class="material-icons">close</i>';
        close_button.addEventListener('click', () => {
            popup_dialogue._hide();
        });

        const content_box = createElement(
            'div', '', ['sdfv_modal_content_box'], this.popup_dialogue
        );
        this.popup_dialogue._content = createElement(
            'div', '', ['sdfv_modal_content'], content_box
        );
        this.popup_dialogue._input = createElement(
            'input', 'symbol_input', ['sdfv_modal_input_text'],
            this.popup_dialogue._content
        );
        
        function set_val(): void {
            if (popup_dialogue._map && popup_dialogue._symbol) {
                const val = popup_dialogue._input.value;
                if (val && !isNaN(val) && Number.isInteger(+val) && val > 0) {
                    popup_dialogue._map[popup_dialogue._symbol] = val;
                    popup_dialogue._hide();
                    if (popup_dialogue._callback)
                        popup_dialogue._callback();
                    return;
                }
            }
            popup_dialogue._input.setCustomValidity('Invalid, not an integer');
        }
        this.popup_dialogue._input.addEventListener(
            'keypress', (ev: KeyboardEvent) => {
                if (ev.code === 'Enter')
                    set_val();
            }
        );

        const footer_bar = createElement(
            'div', '', ['sdfv_modal_footer_bar'], this.popup_dialogue
        );
        const confirm_button = createElement(
            'div', '', ['button', 'sdfv_modal_confirm_button'], footer_bar
        );
        confirm_button.addEventListener('click', (_ev: MouseEvent) => {
            set_val();
        });
        const confirm_button_text = createElement(
            'span', '', [], confirm_button
        );
        confirm_button_text.innerText = 'Confirm';
        createElement('div', '', ['clearfix'], footer_bar);

        this.popup_dialogue._show = function (
            symbol: string, map: SymbolMap, callback: CallableFunction
        ) {
            this.style.display = 'block';
            popup_dialogue._title.innerText = 'Define symbol ' + symbol;
            popup_dialogue._symbol = symbol;
            popup_dialogue._map = map;
            popup_dialogue._callback = callback;
            dialogue_background._show();
        };
        this.popup_dialogue._hide = function () {
            this.style.display = 'none';
            popup_dialogue._title.innerText = '';
            popup_dialogue._input.value = '';
            popup_dialogue._input.setCustomValidity('');
            dialogue_background._hide();
        };
        dialogue_background.addEventListener('click', (_ev: MouseEvent) => {
            popup_dialogue._hide();
        });
    }

    public get_symbol_value_map(): SymbolMap {
        return this.symbol_value_map;
    }

}

export class OverlayManager {

    private badness_scale_method: string = 'median';
    private overlays: GenericSdfgOverlay[] = [];
    private symbol_resolver: SymbolResolver;

    public constructor(private readonly renderer: SDFGRenderer) {
        this.symbol_resolver = new SymbolResolver(this.renderer);
    }

    public register_overlay(type: typeof GenericSdfgOverlay): void {
        this.overlays.push(new type(this.renderer));
        this.renderer.draw_async();
    }

    public deregister_overlay(type: typeof GenericSdfgOverlay): void {
        this.overlays = this.overlays.filter(overlay => {
            return !(overlay instanceof type);
        });
        this.renderer.draw_async();
    }

    public is_overlay_active(type: typeof GenericSdfgOverlay): boolean {
        return this.overlays.filter(overlay => {
            return overlay instanceof type;
        }).length > 0;
    }

    public get_overlay(
        type: typeof GenericSdfgOverlay
    ): GenericSdfgOverlay | undefined {
        let overlay = undefined;
        this.overlays.forEach(ol => {
            if (ol instanceof type) {
                overlay = ol;
                return;
            }
        });
        return overlay;
    }

    public on_symbol_value_changed(
        symbol: string, value: number | undefined
    ): void {
        this.symbol_resolver.symbol_value_changed(symbol, value);
        this.overlays.forEach(overlay => {
            overlay.refresh();
        });
    }

    public update_badness_scale_method(method: string): void {
        this.badness_scale_method = method;
        this.overlays.forEach(overlay => {
            overlay.refresh();
        });
    }

    public draw(): void {
        this.overlays.forEach(overlay => {
            overlay.draw();
        });
    }

    public refresh(): void {
        this.overlays.forEach(overlay => {
            overlay.refresh();
        });
    }

    public on_mouse_event(
        type: string,
        ev: MouseEvent,
        mousepos: Point2D,
        elements: SDFGElement[],
        foreground_elem: SDFGElement | undefined,
        ends_drag: boolean
    ): boolean {
        let dirty = false;
        this.overlays.forEach(overlay => {
            dirty = dirty || overlay.on_mouse_event(
                type, ev, mousepos, elements,
                foreground_elem, ends_drag
            );
        });
        return dirty;
    }

    public get_badness_scale_method(): string {
        return this.badness_scale_method;
    }

    public get_symbol_resolver(): SymbolResolver {
        return this.symbol_resolver;
    }

    public get_overlays(): GenericSdfgOverlay[] {
        return this.overlays;
    }

}
