// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dagre from 'dagre';
import {
    find_exit_for_entry,
    get_positioning_info,
    delete_positioning_info,
    get_uuid_graph_element,
    find_graph_element_by_uuid,
    find_root_sdfg,
    delete_sdfg_nodes,
    delete_sdfg_states,
    check_and_redirect_edge,
} from '../utils/sdfg/sdfg_utils';
import { deepCopy, intersectRect } from '../utils/utils';
import { traverse_sdfg_scopes } from '../utils/sdfg/traversal';
import { ContextMenu } from '../utils/context_menu';
import {
    Connector,
    Edge,
    offset_state,
    SDFGElements,
    draw_sdfg,
    offset_sdfg,
    NestedSDFG,
    SDFGElement,
    SDFGNode,
    State,
    AccessNode,
    EntryNode,
} from './renderer_elements';
import { memlet_tree_complete } from '../utils/sdfg/traversal';
import { CanvasManager } from './canvas_manager';
import {
    boundingBox,
    calculateBoundingBox,
    calculateEdgeBoundingBox,
} from '../utils/bounding_box';
import { OverlayManager } from '../overlay_manager';
import { SDFV } from '../sdfv';
import { MemoryVolumeOverlay } from '../overlays/memory_volume_overlay';
import {
    DagreSDFG,
    JsonSDFG,
    JsonSDFGEdge,
    JsonSDFGNode,
    JsonSDFGState,
    ModeButtons,
    Point2D,
    SDFVTooltipFunc,
    SimpleRect,
    stringify_sdfg,
} from '../index';
import { LogicalGroupOverlay } from '../overlays/logical_group_overlay';

// External, non-typescript libraries which are presented as previously loaded
// scripts and global javascript variables:
declare const blobStream: any;
declare const canvas2pdf: any;

// Some global functions and variables which are only accessible within VSCode:
declare const vscode: any | null;

type SDFGElementType = 'states' | 'nodes' | 'edges' | 'isedges';
// If type is explicitly set, dagre typecheck fails with integer node ids
export type SDFGListType = any[];//{ [key: number]: DagreSDFG };

function check_valid_add_position(
    type: string | null, foreground_elem: SDFGElement | undefined | null,
    lib: any, mousepos: any
): boolean {
    if (type !== null) {
        switch (type) {
            case 'SDFGState':
                return (foreground_elem instanceof NestedSDFG ||
                    foreground_elem === null);
            case 'Edge':
                return (foreground_elem instanceof SDFGNode ||
                    foreground_elem instanceof State);
            case 'LibraryNode':
                return (foreground_elem instanceof State && lib);
            case 'State':
            default:
                return foreground_elem instanceof State;
        }
    }
    return false;
}

export class SDFGRenderer {

    protected sdfg_list: any = {};
    protected graph: DagreSDFG | null = null;
    protected sdfg_tree: { [key: number]: number } = {};  // Parent-pointing SDFG tree
    // List of all state's parent elements.
    protected state_parent_list: any = {};
    protected in_vscode: boolean = false;
    protected dace_daemon_connected: boolean = false;

    // Rendering related fields.
    protected ctx: CanvasRenderingContext2D | null = null;
    protected canvas: HTMLCanvasElement | null = null;
    protected canvas_manager: CanvasManager | null = null;
    protected last_dragged_element: SDFGElement | null = null;
    protected tooltip: SDFVTooltipFunc | null = null;
    protected tooltip_container: HTMLElement | null = null;
    protected overlay_manager: OverlayManager;
    protected bgcolor: string | null = null;
    protected visible_rect: SimpleRect | null = null;
    protected cssProps: { [key: string]: string } = {};


    // Toolbar related fields.
    protected menu: ContextMenu | null = null;
    protected toolbar: HTMLElement | null = null;
    protected panmode_btn: HTMLElement | null = null;
    protected movemode_btn: HTMLElement | null = null;
    protected selectmode_btn: HTMLElement | null = null;
    protected filter_btn: HTMLElement | null = null;
    protected addmode_btns: HTMLElement[] = [];
    protected add_type: string | null = null;
    protected add_mode_lib: string | null = null;
    protected mode_selected_bg_color: string = '#CCCCCC';
    protected mouse_follow_svgs: any = null;
    protected mouse_follow_element: any = null;
    protected overlays_menu: any = null;

    // Memlet-Tree related fields.
    protected all_memlet_trees_sdfg: Set<any>[] = [];
    protected all_memlet_trees: Set<any>[] = [];

    // View options.
    protected inclusive_ranges: boolean = false;
    protected omit_access_nodes: boolean = false;

    // Mouse-related fields.
    // Mouse mode - pan, move, select.
    protected mouse_mode: string = 'pan';
    protected box_select_rect: any = null;
    // Last position of the mouse pointer (in canvas coordinates).
    protected mousepos: Point2D | null = null;
    // Last position of the mouse pointer (in pixel coordinates).
    protected realmousepos: Point2D | null = null;
    protected dragging: boolean = false;
    // Null if the mouse/touch is not activated.
    protected drag_start: any = null;
    protected external_mouse_handler: ((...args: any[]) => boolean) | null = null;
    protected ctrl_key_selection: boolean = false;
    protected shift_key_movement: boolean = false;
    protected add_position: Point2D | null = null;
    protected add_edge_start: any = null;

    // Information window fields.
    protected error_popover_container: HTMLElement | null = null;
    protected error_popover_text: HTMLElement | null = null;
    protected interaction_info_box: HTMLElement | null = null;
    protected interaction_info_text: HTMLElement | null = null;
    protected dbg_info_box: HTMLElement | null = null;
    protected dbg_mouse_coords: HTMLElement | null = null;

    // Selection related fields.
    protected selected_elements: SDFGElement[] = [];

    protected ext_event_handlers: ((type: string, data: any | null) => any)[] =
        [];

    public constructor(
        protected sdfv_instance: SDFV,
        protected sdfg: JsonSDFG,
        protected container: HTMLElement,
        on_mouse_event: ((...args: any[]) => boolean) | null = null,
        user_transform: DOMMatrix | null = null,
        public debug_draw = false,
        background: string | null = null,
        mode_buttons: any = null
    ) {
        this.external_mouse_handler = on_mouse_event;

        this.overlay_manager = new OverlayManager(this);

        // Register overlays that are turned on by default.
        this.overlay_manager.register_overlay(LogicalGroupOverlay);

        this.in_vscode = false;
        try {
            vscode;
            if (vscode)
                this.in_vscode = true;
        } catch (ex) { }

        this.init_elements(user_transform, background, mode_buttons);

        this.set_sdfg(sdfg, false);

        this.all_memlet_trees_sdfg = memlet_tree_complete(this.sdfg);

        this.update_fast_memlet_lookup();
    }

    public destroy(): void {
        try {
            this.menu?.destroy();
            this.canvas_manager?.destroy();
            if (this.canvas)
                this.container.removeChild(this.canvas);
            if (this.toolbar)
                this.container.removeChild(this.toolbar);
            if (this.tooltip_container)
                this.container.removeChild(this.tooltip_container);
        } catch (ex) {
            // Do nothing
        }
    }

    public clearCssPropertyCache(): void {
        this.cssProps = {};
    }

    public getCssProperty(property_name: string): string {
        if (this.cssProps[property_name])
            return this.cssProps[property_name];

        if (this.canvas) {
            const prop_val: string = window.getComputedStyle(this.canvas).getPropertyValue(property_name).trim();
            this.cssProps[property_name] = prop_val;
            return prop_val;
        }
        return '';
    }

    public view_settings(): any {
        return { inclusive_ranges: this.inclusive_ranges };
    }

    // Updates buttons based on cursor mode
    public update_toggle_buttons(): void {
        // First clear out of all modes, then jump in to the correct mode.
        if (this.canvas)
            this.canvas.style.cursor = 'default';
        if (this.interaction_info_box)
            this.interaction_info_box.style.display = 'none';
        if (this.interaction_info_text)
            this.interaction_info_text.innerHTML = '';

        if (this.panmode_btn) {
            this.panmode_btn.style.paddingBottom = '0px';
            this.panmode_btn.style.userSelect = 'none';
            this.panmode_btn.style.background = '';
        }
        if (this.movemode_btn) {
            this.movemode_btn.style.paddingBottom = '0px';
            this.movemode_btn.style.userSelect = 'none';
            this.movemode_btn.style.background = '';
        }
        if (this.selectmode_btn) {
            this.selectmode_btn.style.paddingBottom = '0px';
            this.selectmode_btn.style.userSelect = 'none';
            this.selectmode_btn.style.background = '';
        }

        this.mouse_follow_element.innerHTML = null;

        for (const add_btn of this.addmode_btns) {
            const btn_type = add_btn.getAttribute('type');
            if (btn_type === this.add_type && this.add_type) {
                add_btn.style.userSelect = 'none';
                add_btn.style.background = this.mode_selected_bg_color;
                this.mouse_follow_element.innerHTML =
                    this.mouse_follow_svgs[this.add_type];
            } else {
                add_btn.style.userSelect = 'none';
                add_btn.style.background = '';
            }
        }

        switch (this.mouse_mode) {
            case 'move':
                if (this.movemode_btn)
                    this.movemode_btn.style.background =
                        this.mode_selected_bg_color;
                if (this.interaction_info_box)
                    this.interaction_info_box.style.display = 'block';
                if (this.interaction_info_text)
                    this.interaction_info_text.innerHTML =
                        'Middle Mouse: Pan view<br>' +
                        'Right Click: Reset position';
                break;
            case 'select':
                if (this.selectmode_btn)
                    this.selectmode_btn.style.background =
                        this.mode_selected_bg_color;
                if (this.interaction_info_box)
                    this.interaction_info_box.style.display = 'block';
                if (this.interaction_info_text) {
                    if (this.ctrl_key_selection)
                        this.interaction_info_text.innerHTML =
                            'Middle Mouse: Pan view';
                    else
                        this.interaction_info_text.innerHTML =
                            'Shift: Add to selection<br>' +
                            'Ctrl: Remove from selection<br>' +
                            'Middle Mouse: Pan view';
                }
                break;
            case 'add':
                if (this.interaction_info_box)
                    this.interaction_info_box.style.display = 'block';
                if (this.interaction_info_text) {
                    if (this.add_type === 'Edge') {
                        if (this.add_edge_start)
                            this.interaction_info_text.innerHTML =
                                'Left Click: Select second element (to)<br>' +
                                'Middle Mouse: Pan view<br>' +
                                'Right Click / Esc: Abort';
                        else
                            this.interaction_info_text.innerHTML =
                                'Left Click: Select first element (from)<br>' +
                                'Middle Mouse: Pan view<br>' +
                                'Right Click / Esc: Abort';
                    } else {
                        this.interaction_info_text.innerHTML =
                            'Left Click: Place element<br>' +
                            'Ctrl + Left Click: Place and stay in Add ' +
                            'Mode<br>' +
                            'Middle Mouse: Pan view<br>' +
                            'Right Click / Esc: Abort';
                    }
                }
                break;
            case 'pan':
            default:
                if (this.panmode_btn)
                    this.panmode_btn.style.background =
                        this.mode_selected_bg_color;
                break;
        }
    }

    // Initializes the DOM
    public init_elements(
        user_transform: DOMMatrix | null,
        background: string | null,
        mode_buttons: ModeButtons | undefined | null
    ): void {

        this.canvas = document.createElement('canvas');
        this.canvas.classList.add('sdfg_canvas');
        if (background)
            this.canvas.style.backgroundColor = background;
        else
            this.canvas.style.backgroundColor = 'inherit';
        this.container.append(this.canvas);

        if (this.debug_draw) {
            this.dbg_info_box = document.createElement('div');
            this.dbg_info_box.style.position = 'absolute';
            this.dbg_info_box.style.bottom = '.5rem';
            this.dbg_info_box.style.right = '.5rem';
            this.dbg_info_box.style.backgroundColor = 'black';
            this.dbg_info_box.style.padding = '.3rem';
            this.dbg_mouse_coords = document.createElement('span');
            this.dbg_mouse_coords.style.color = 'white';
            this.dbg_mouse_coords.style.fontSize = '1rem';
            this.dbg_mouse_coords.innerText = 'x: N/A / y: N/A';
            this.dbg_info_box.appendChild(this.dbg_mouse_coords);
            this.container.appendChild(this.dbg_info_box);
        }

        // Add an info box for interaction hints to the bottom left of the
        // canvas.
        this.interaction_info_box = document.createElement('div');
        this.interaction_info_box.style.position = 'absolute';
        this.interaction_info_box.style.bottom = '.5rem',
            this.interaction_info_box.style.left = '.5rem',
            this.interaction_info_box.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        this.interaction_info_box.style.borderRadius = '5px';
        this.interaction_info_box.style.padding = '.3rem';
        this.interaction_info_box.style.display = 'none';
        this.interaction_info_text = document.createElement('span');
        this.interaction_info_text.style.color = '#eeeeee';
        this.interaction_info_text.innerHTML = '';
        this.interaction_info_box.appendChild(this.interaction_info_text);
        this.container.appendChild(this.interaction_info_box);

        // Add buttons
        this.toolbar = document.createElement('div');
        this.toolbar.style.position = 'absolute';
        this.toolbar.style.top = '10px';
        this.toolbar.style.left = '10px';
        let d;

        // Menu bar
        try {
            ContextMenu;
            const menu_button = document.createElement('button');
            menu_button.className = 'button';
            menu_button.innerHTML = '<i class="material-icons">menu</i>';
            menu_button.style.paddingBottom = '0px';
            menu_button.style.userSelect = 'none';
            const that = this;
            menu_button.onclick = function () {
                if (that.menu && that.menu.visible()) {
                    that.menu.destroy();
                    return;
                }
                const rect = menu_button.getBoundingClientRect();
                const cmenu = new ContextMenu();
                if (!that.in_vscode) {
                    cmenu.addOption(
                        'Save SDFG as...', (_x: any) => that.save_sdfg()
                    );
                }
                cmenu.addOption(
                    'Save view as PNG', (_x: any) => that.save_as_png()
                );
                if (that.has_pdf()) {
                    cmenu.addOption(
                        'Save view as PDF', (_x: any) => that.save_as_pdf()
                    );
                    cmenu.addOption(
                        'Save all as PDF', (_x: any) => that.save_as_pdf(true)
                    );
                }
                cmenu.addCheckableOption(
                    'Inclusive ranges',
                    that.inclusive_ranges,
                    (_x: any, checked: boolean) => {
                        that.inclusive_ranges = checked;
                    }
                );
                cmenu.addCheckableOption(
                    'Adaptive content hiding',
                    (that.ctx as any).lod,
                    (_x: any, checked: boolean) => {
                        (that.ctx as any).lod = checked;
                    }
                );
                if (that.in_vscode) {
                    cmenu.addCheckableOption(
                        'Show Logical Groups',
                        that.overlay_manager ?
                            that.overlay_manager.is_overlay_active(
                                LogicalGroupOverlay
                            ) : false,
                        (x: any, checked: boolean) => {
                            if (checked)
                                that.overlay_manager?.register_overlay(
                                    LogicalGroupOverlay
                                );
                            else
                                that.overlay_manager?.deregister_overlay(
                                    LogicalGroupOverlay
                                );
                            that.draw_async();
                            that.emit_event(
                                'active_overlays_changed', null
                            );
                        }
                    );
                } else {
                    cmenu.addOption(
                        'Overlays',
                        () => {
                            if (
                                that.overlays_menu &&
                                that.overlays_menu.visible()
                            ) {
                                that.overlays_menu.destroy();
                                return;
                            }
                            const rect =
                                cmenu.get_cmenu_elem()?.getBoundingClientRect();
                            const overlays_cmenu = new ContextMenu();
                            overlays_cmenu.addCheckableOption(
                                'Memory volume analysis',
                                that.overlay_manager ?
                                    that.overlay_manager.is_overlay_active(
                                        MemoryVolumeOverlay
                                    ) : false,
                                (x: any, checked: boolean) => {
                                    if (checked)
                                        that.overlay_manager?.register_overlay(
                                            MemoryVolumeOverlay
                                        );
                                    else
                                        that.overlay_manager?.deregister_overlay(
                                            MemoryVolumeOverlay
                                        );
                                    that.draw_async();
                                    that.emit_event(
                                        'active_overlays_changed', null
                                    );
                                }
                            );
                            overlays_cmenu.addCheckableOption(
                                'Logical Groups',
                                that.overlay_manager ?
                                    that.overlay_manager.is_overlay_active(
                                        LogicalGroupOverlay
                                    ) : false,
                                (x: any, checked: boolean) => {
                                    if (checked)
                                        that.overlay_manager?.register_overlay(
                                            LogicalGroupOverlay
                                        );
                                    else
                                        that.overlay_manager?.deregister_overlay(
                                            LogicalGroupOverlay
                                        );
                                    that.draw_async();
                                    that.emit_event(
                                        'active_overlays_changed', null
                                    );
                                }
                            );
                            that.overlays_menu = overlays_cmenu;
                            that.overlays_menu.show(rect?.left, rect?.top);
                        }
                    );
                }
                cmenu.addCheckableOption(
                    'Hide Access Nodes',
                    that.omit_access_nodes,
                    (_: any, checked: boolean) => {
                        that.omit_access_nodes = checked;
                        that.relayout();
                        that.draw_async();
                    }
                );
                cmenu.addOption(
                    'Reset positions', () => that.reset_positions()
                );
                that.menu = cmenu;
                that.menu.show(rect.left, rect.bottom);
            };
            menu_button.title = 'Menu';
            this.toolbar.appendChild(menu_button);
        } catch (ex) { }

        // Zoom to fit
        d = document.createElement('button');
        d.className = 'button';
        d.innerHTML = '<i class="material-icons">filter_center_focus</i>';
        d.style.paddingBottom = '0px';
        d.style.userSelect = 'none';
        d.onclick = () => this.zoom_to_view();
        d.title = 'Zoom to fit SDFG';
        this.toolbar.appendChild(d);

        // Collapse all
        d = document.createElement('button');
        d.className = 'button';
        d.innerHTML = '<i class="material-icons">unfold_less</i>';
        d.style.paddingBottom = '0px';
        d.style.userSelect = 'none';
        d.onclick = () => this.collapse_all();
        d.title = 'Collapse all elements';
        this.toolbar.appendChild(d);

        // Expand all
        d = document.createElement('button');
        d.className = 'button';
        d.innerHTML = '<i class="material-icons">unfold_more</i>';
        d.style.paddingBottom = '0px';
        d.style.userSelect = 'none';
        d.onclick = () => this.expand_all();
        d.title = 'Expand all elements';
        this.toolbar.appendChild(d);

        if (mode_buttons) {
            // If we get the "external" mode buttons we are in vscode and do
            // not need to create them.
            this.panmode_btn = mode_buttons.pan;
            this.movemode_btn = mode_buttons.move;
            this.selectmode_btn = mode_buttons.select;
            this.addmode_btns = mode_buttons.add_btns;
            for (const add_btn of this.addmode_btns) {
                if (add_btn.getAttribute('type') === 'LibraryNode') {
                    add_btn.onclick = () => {
                        const libnode_callback = () => {
                            this.mouse_mode = 'add';
                            this.add_type = 'LibraryNode';
                            this.add_edge_start = null;
                            this.update_toggle_buttons();
                        };
                        this.emit_event(
                            'libnode_select',
                            {
                                callback: libnode_callback,
                            }
                        );
                    };
                } else {
                    add_btn.onclick = () => {
                        if (!this.dace_daemon_connected) {
                            this.emit_event('warn_no_daemon', null);
                        } else {
                            this.mouse_mode = 'add';
                            this.add_type = add_btn.getAttribute('type');
                            this.add_mode_lib = null;
                            this.add_edge_start = null;
                            this.update_toggle_buttons();
                        }
                    };
                }
            }
            this.mode_selected_bg_color = '#22A4FE';
        } else {
            // Mode buttons are empty in standalone SDFV
            this.addmode_btns = [];

            // Create pan mode button
            const pan_mode_btn = document.createElement('button');
            pan_mode_btn.className = 'button';
            pan_mode_btn.innerHTML = '<i class="material-icons">pan_tool</i>';
            pan_mode_btn.style.paddingBottom = '0px';
            pan_mode_btn.style.userSelect = 'none';
            pan_mode_btn.style.background = this.mode_selected_bg_color;
            pan_mode_btn.title = 'Pan mode';
            this.panmode_btn = pan_mode_btn;
            this.toolbar.appendChild(pan_mode_btn);

            // Create move mode button
            const move_mode_btn = document.createElement('button');
            move_mode_btn.className = 'button';
            move_mode_btn.innerHTML = '<i class="material-icons">open_with</i>';
            move_mode_btn.style.paddingBottom = '0px';
            move_mode_btn.style.userSelect = 'none';
            move_mode_btn.title = 'Object moving mode';
            this.movemode_btn = move_mode_btn;
            this.toolbar.appendChild(move_mode_btn);

            // Create select mode button
            const box_select_btn = document.createElement('button');
            box_select_btn.className = 'button';
            box_select_btn.innerHTML =
                '<i class="material-icons">border_style</i>';
            box_select_btn.style.paddingBottom = '0px';
            box_select_btn.style.userSelect = 'none';
            box_select_btn.title = 'Select mode';
            this.selectmode_btn = box_select_btn;
            this.toolbar.appendChild(box_select_btn);
        }

        // Enter pan mode
        if (this.panmode_btn)
            this.panmode_btn.onclick = () => {
                this.mouse_mode = 'pan';
                this.add_type = null;
                this.add_mode_lib = null;
                this.add_edge_start = null;
                this.update_toggle_buttons();
            };

        // Enter object moving mode
        if (this.movemode_btn) {
            this.movemode_btn.onclick = (
                _: MouseEvent, shift_click: boolean | undefined = undefined
            ): void => {
                // shift_click is false if shift key has been released and
                // undefined if it has been a normal mouse click
                if (this.shift_key_movement && shift_click === false)
                    this.mouse_mode = 'pan';
                else
                    this.mouse_mode = 'move';
                this.add_type = null;
                this.add_mode_lib = null;
                this.add_edge_start = null;
                this.shift_key_movement = (
                    shift_click === undefined ? false : shift_click
                );
                this.update_toggle_buttons();
            };
        }

        // Enter box selection mode
        if (this.selectmode_btn)
            this.selectmode_btn.onclick = (
                _: MouseEvent, ctrl_click: boolean | undefined = undefined
            ): void => {
                // ctrl_click is false if ctrl key has been released and
                // undefined if it has been a normal mouse click
                if (this.ctrl_key_selection && ctrl_click === false)
                    this.mouse_mode = 'pan';
                else
                    this.mouse_mode = 'select';
                this.add_type = null;
                this.add_mode_lib = null;
                this.add_edge_start = null;
                this.ctrl_key_selection = (
                    ctrl_click === undefined ? false : ctrl_click
                );
                this.update_toggle_buttons();
            };

        // React to ctrl and shift key presses
        document.addEventListener('keydown', (e) => this.on_key_event(e));
        document.addEventListener('keyup', (e) => this.on_key_event(e));
        document.addEventListener("visibilitychange", () => {
            if (document.hidden) {
                this.clear_key_events();
            } else {
                // Tab is visible, do nothing
            }
        });

        // Filter graph to selection
        d = document.createElement('button');
        d.className = 'button hidden';
        d.innerHTML = '<i class="material-icons">content_cut</i>';
        d.style.paddingBottom = '0px';
        d.style.userSelect = 'none';
        d.onclick = () => this.cutout_selection();
        d.title = 'Filter selection (cutout)';
        this.filter_btn = d;
        this.toolbar.appendChild(d);

        // Exit previewing mode
        if (this.in_vscode) {
            const exit_preview_btn = document.createElement('button');
            exit_preview_btn.id = 'exit-preview-button';
            exit_preview_btn.className = 'button hidden';
            exit_preview_btn.innerHTML = '<i class="material-icons">close</i>';
            exit_preview_btn.style.paddingBottom = '0px';
            exit_preview_btn.style.userSelect = 'none';
            exit_preview_btn.onclick = () => {
                exit_preview_btn.className = 'button hidden';
                this.emit_event('exit_preview', null);
                if (vscode) {
                    vscode.postMessage({
                        type: 'sdfv.get_current_sdfg',
                        preventRefreshes: true,
                    });
                    vscode.postMessage({
                        type: 'transformation_history.refresh',
                        resetActive: true,
                    });
                }
            };
            exit_preview_btn.title = 'Exit preview';
            this.toolbar.appendChild(exit_preview_btn);
        }

        this.container.append(this.toolbar);
        // End of buttons

        // Tooltip HTML container
        this.tooltip_container = document.createElement('div');
        this.tooltip_container.innerHTML = '';
        this.tooltip_container.className = 'sdfvtooltip';
        this.tooltip_container.onmouseover = () => {
            if (this.tooltip_container)
                this.tooltip_container.style.display = 'none';
        };
        this.container.appendChild(this.tooltip_container);

        // HTML container for error popovers with invalid SDFGs
        this.error_popover_container = document.createElement('div');
        this.error_popover_container.innerHTML = '';
        this.error_popover_container.className = 'invalid_popup';
        this.error_popover_text = document.createElement('div');
        const error_popover_dismiss = document.createElement('button');
        const that = this;
        error_popover_dismiss.onclick = () => {
            that.sdfg.error = undefined;
            if (that.error_popover_container && that.error_popover_text) {
                that.error_popover_text.innerText = '';
                that.error_popover_container.style.display = 'none';
            }
        };
        error_popover_dismiss.style.float = 'right';
        error_popover_dismiss.style.cursor = 'pointer';
        error_popover_dismiss.style.color = 'white';
        error_popover_dismiss.innerHTML = '<i class="material-icons">close</i>';
        this.error_popover_container.appendChild(error_popover_dismiss);
        this.error_popover_container.appendChild(this.error_popover_text);
        this.container.appendChild(this.error_popover_container);

        this.ctx = this.canvas.getContext('2d');
        if (!this.ctx) {
            console.error('Failed to get canvas context, aborting');
            return;
        }

        // Translation/scaling management
        this.canvas_manager = new CanvasManager(this.ctx, this, this.canvas);
        if (user_transform !== null)
            this.canvas_manager.set_user_transform(user_transform);

        // Resize event for container
        const observer = new MutationObserver((mutations) => {
            this.onresize();
            this.draw_async();
        });
        observer.observe(this.container, { attributes: true });

        // Set inherited properties
        if (background)
            this.bgcolor = background;
        else
            this.bgcolor = window.getComputedStyle(this.canvas).backgroundColor;

        // Create the initial SDFG layout
        this.relayout();

        // Set mouse event handlers
        this.set_mouse_handlers();

        // Set initial zoom, if not already set
        if (user_transform === null)
            this.zoom_to_view();

        const svgs: { [key: string]: string } = {};
        svgs['Map'] =
            `<svg width="8rem" height="2rem" viewBox="0 0 800 200" stroke="black" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <line x1="10" x2="190" y1="190" y2="10"/>
                <line x1="190" x2="600" y1="10" y2="10"/>
                <line x1="600" x2="790" y1="10" y2="190"/>
                <line x1="790" x2="10" y1="190" y2="190"/>
            </svg>`;
        svgs['Consume'] =
            `<svg width="8rem" height="2rem" viewBox="0 0 800 200" stroke="black" stroke-width="10" stroke-dasharray="60,25" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <line x1="10"x2="190" y1="190" y2="10"/>
                <line x1="190" x2="600" y1="10" y2="10"/>
                <line x1="600" x2="790" y1="10" y2="190"/>
                <line x1="790" x2="10" y1="190" y2="190"/>
            </svg>`;
        svgs['Tasklet'] =
            `<svg width="2.6rem" height="1.3rem" viewBox="0 0 400 200" stroke="black" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <line x1="10" x2="70" y1="130" y2="190"/>
                <line x1="70" x2="330" y1="190" y2="190"/>
                <line x1="330" x2="390" y1="190" y2="130"/>
                <line x1="390" x2="390" y1="130" y2="70"/>
                <line x1="390" x2="330" y1="70" y2="10"/>
                <line x1="330" x2="70" y1="10" y2="10"/>
                <line x1="70" x2="10" y1="10" y2="70"/>
                <line x1="10" x2="10" y1="70" y2="130"/>
            </svg>`;
        svgs['NestedSDFG'] =
            `<svg width="2.6rem" height="1.3rem" viewBox="0 0 400 200" stroke="black" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <line x1="40" x2="80" y1="120" y2="160"/>
                <line x1="80" x2="320" y1="160" y2="160"/>
                <line x1="320" x2="360" y1="160" y2="120"/>
                <line x1="360" x2="360" y1="120" y2="80"/>
                <line x1="360" x2="320" y1="80" y2="40"/>
                <line x1="320" x2="80" y1="40" y2="40"/>
                <line x1="80" x2="40" y1="40" y2="80"/>
                <line x1="40" x2="40" y1="80" y2="120"/>
                
                <line x1="10" x2="70" y1="130" y2="190"/>
                <line x1="70" x2="330" y1="190" y2="190"/>
                <line x1="330" x2="390" y1="190" y2="130"/>
                <line x1="390" x2="390" y1="130" y2="70"/>
                <line x1="390" x2="330" y1="70" y2="10"/>
                <line x1="330" x2="70" y1="10" y2="10"/>
                <line x1="70" x2="10" y1="10" y2="70"/>
                <line x1="10" x2="10" y1="70" y2="130"/>
            </svg>`;
        svgs['LibraryNode'] =
            `<svg width="2.6rem" height="1.3rem" viewBox="0 0 400 200" stroke="white" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                        <line x1="10" x2="10" y1="10" y2="190"/>
                        <line x1="10" x2="390" y1="190" y2="190"/>
                        <line x1="390" x2="390" y1="190" y2="55"/>
                        <line x1="390" x2="345" y1="55" y2="10"/>
                        <line x1="345" x2="10" y1="10" y2="10"/>
                        <line x1="345" x2="345" y1="10" y2="55"/>
                        <line x1="345" x2="390" y1="55" y2="55"/>
            </svg>`;
        svgs['AccessNode'] =
            `<svg width="1.3rem" height="1.3rem" viewBox="0 0 200 200" stroke="black" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="100" r="90" fill="none"/>
            </svg>`;
        svgs['Stream'] =
            `<svg width="1.3rem" height="1.3rem" viewBox="0 0 200 200" stroke="black" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="100" r="90" fill="none" stroke-dasharray="60,25"/>
            </svg>`;
        svgs['SDFGState'] =
            `<svg width="1.3rem" height="1.3rem" viewBox="0 0 200 200" stroke="black" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <rect x="20" y="20" width="160" height="160" style="fill:#deebf7;" />
            </svg>`;
        svgs['Connector'] =
            `<svg width="1.3rem" height="1.3rem" viewBox="0 0 200 200" stroke="white" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <circle cx="100" cy="100" r="40" fill="none"/>
            </svg>`;
        svgs['Edge'] =
            `<svg width="1.3rem" height="1.3rem" viewBox="0 0 200 200" stroke="white" stroke-width="10" version="1.1" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7"  refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" />
                    </marker>
                </defs>
                <line x1="20" y1="20" x2="180" y2="180" marker-end="url(#arrowhead)" />
            </svg>`;

        const el = document.createElement('div');
        el.style.position = 'absolute';
        el.style.top = '0px';
        el.style.left = '0px';
        el.style.userSelect = 'none';
        el.style.pointerEvents = 'none';

        this.container.appendChild(el);

        this.mouse_follow_element = el;
        this.mouse_follow_svgs = svgs;

        this.update_toggle_buttons();

        // Queue first render
        this.draw_async();
    }

    public draw_async(): void {
        this.clearCssPropertyCache();
        this.canvas_manager?.draw_async();
    }

    public emit_event(type: string, data: any | null): void {
        for (const handler of this.ext_event_handlers)
            handler(type, data);
    }

    public set_sdfg(new_sdfg: JsonSDFG, layout: boolean = true): void {
        this.sdfg = new_sdfg;

        if (layout) {
            this.relayout();
            this.draw_async();
        }

        // Update info box
        if (this.selected_elements.length == 1) {
            const uuid = get_uuid_graph_element(this.selected_elements[0]);
            if (this.graph)
                this.sdfv_instance.fill_info(
                    find_graph_element_by_uuid(this.graph, uuid).element
                );
        }

        // Update SDFG metadata
        this.sdfg_tree = {};
        this.for_all_sdfg_elements((otype: SDFGElementType, odict: any, obj: any) => {
            if (obj.type === 'NestedSDFG')
                this.sdfg_tree[obj.attributes.sdfg.sdfg_list_id] = odict.sdfg.sdfg_list_id;
        });
    }

    // Set mouse events (e.g., click, drag, zoom)
    public set_mouse_handlers(): void {
        const canvas = this.canvas;
        const br = () => canvas?.getBoundingClientRect();

        const comp_x = (event: any): number | undefined => {
            const left = br()?.left;
            return this.canvas_manager?.mapPixelToCoordsX(
                event.clientX - (left ? left : 0)
            );
        };
        const comp_y = (event: any): number | undefined => {
            const top = br()?.top;
            return this.canvas_manager?.mapPixelToCoordsY(
                event.clientY - (top ? top : 0)
            );
        };

        // Mouse handler event types
        for (const evtype of [
            'mousedown', 'mousemove', 'mouseup', 'touchstart', 'touchmove',
            'touchend', 'wheel', 'click', 'dblclick', 'contextmenu'
        ]) {
            canvas?.addEventListener(evtype, x => {
                const cancelled = this.on_mouse_event(
                    x, comp_x, comp_y, evtype
                );
                if (cancelled)
                    return;
                if (!this.in_vscode) {
                    x.stopPropagation();
                    x.preventDefault();
                }
            });
        }
    }

    public onresize(): void {
        // Set canvas size
        if (this.canvas) {
            this.canvas.style.width = '99%';
            this.canvas.style.height = '99%';
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        }
    }

    // Update memlet tree collection for faster lookup
    public update_fast_memlet_lookup(): void {
        this.all_memlet_trees = [];
        for (const tree of this.all_memlet_trees_sdfg) {
            const s = new Set<any>();
            for (const edge of tree) {
                s.add(edge.attributes.data.edge);
            }
            this.all_memlet_trees.push(s);
        }
    }

    // Re-layout graph and nested graphs
    public relayout(): DagreSDFG {
        this.sdfg_list = {};
        this.graph = relayout_sdfg(
            this.ctx, this.sdfg, this.sdfg_list,
            this.state_parent_list, this.omit_access_nodes
        );
        this.onresize();

        this.update_fast_memlet_lookup();

        // Move the elements based on its positioning information
        this.translate_moved_elements();

        // Make sure all visible overlays get recalculated if there are any.
        if (this.overlay_manager !== null)
            this.overlay_manager.refresh();

        // If we're in a VSCode context, we also want to refresh the outline.
        if (this.in_vscode)
            this.sdfv_instance.outline(this, this.graph);

        return this.graph;
    }

    public translate_moved_elements(): void {
        if (!this.graph)
            return;

        traverse_sdfg_scopes(this.graph, (node: any, graph: any) => {
            let scope_dx = 0;
            let scope_dy = 0;

            function add_scope_movement(n: any) {
                if (n.data.node.scope_entry) {
                    const scope_entry_node = graph.node(
                        n.data.node.scope_entry
                    );
                    const sp = get_positioning_info(scope_entry_node);
                    if (sp && Number.isFinite(sp.scope_dx) &&
                        Number.isFinite(sp.scope_dy)) {
                        scope_dx += sp.scope_dx;
                        scope_dy += sp.scope_dy;
                    }
                    if (scope_entry_node) {
                        add_scope_movement(scope_entry_node);
                    }
                }
            }

            // Only add scope movement for nodes (and not states)
            if (node instanceof SDFGNode)
                add_scope_movement(node);

            let dx = scope_dx;
            let dy = scope_dy;

            const position = get_positioning_info(node);
            if (position) {
                dx += position.dx;
                dy += position.dy;
            }

            if (dx || dy) {
                // Move the element
                if (this.graph)
                    this.canvas_manager?.translate_element(
                        node, { x: node.x, y: node.y },
                        { x: node.x + dx, y: node.y + dy }, this.graph,
                        this.sdfg_list, this.state_parent_list, undefined, false
                    );
            }

            // Move edges (outgoing only)
            graph.inEdges(node.id)?.forEach((e_id: number) => {
                const edge = graph.edge(e_id);
                const edge_pos = get_positioning_info(edge);

                let final_pos_d;
                // If edges are moved within a given scope, update the point
                // movements
                if (scope_dx || scope_dy) {
                    final_pos_d = [];
                    // never move first (and last) point manually
                    final_pos_d.push({ dx: 0, dy: 0 });
                    for (let i = 1; i < edge.points.length - 1; i++) {
                        final_pos_d.push({ dx: scope_dx, dy: scope_dy });
                        if (edge_pos?.points) {
                            final_pos_d[i].dx += edge_pos.points[i].dx;
                            final_pos_d[i].dx += edge_pos.points[i].dy;
                        }
                    }
                    // never move last (and first) point manually
                    final_pos_d.push({ dx: 0, dy: 0 });
                } else if (edge_pos?.points) {
                    final_pos_d = edge_pos.points;
                }
                if (final_pos_d) {
                    // Move the element
                    if (this.graph)
                        this.canvas_manager?.translate_element(
                            edge, { x: 0, y: 0 },
                            { x: 0, y: 0 }, this.graph, this.sdfg_list,
                            this.state_parent_list, undefined, false, false,
                            final_pos_d
                        );
                }
            });
            return true;
        });
    }

    // Change translation and scale such that the chosen elements
    // (or entire graph if null) is in view
    public zoom_to_view(elements: any = null, animate: boolean = true): void {
        if (!elements || elements.length == 0)
            elements = this.graph?.nodes().map(x => this.graph?.node(x));

        const bb = boundingBox(elements);
        this.canvas_manager?.set_view(bb, animate);

        this.draw_async();
    }

    public collapse_all(): void {
        this.for_all_sdfg_elements((otype: SDFGElementType, odict: any, obj: any) => {
            if ('is_collapsed' in obj.attributes && !obj.type.endsWith('Exit'))
                obj.attributes.is_collapsed = true;
        });

        this.emit_event('collapse_state_changed', {
            collapsed: true,
            all: true,
        });

        this.relayout();
        this.draw_async();
    }

    public expand_all(): void {
        this.for_all_sdfg_elements((otype: SDFGElementType, odict: any, obj: any) => {
            if ('is_collapsed' in obj.attributes && !obj.type.endsWith('Exit'))
                obj.attributes.is_collapsed = false;
        });

        this.emit_event('collapse_state_changed', {
            collapsed: false,
            all: true,
        });

        this.relayout();
        this.draw_async();
    }

    public reset_positions(): void {
        this.for_all_sdfg_elements((otype: SDFGElementType, odict: any, obj: any) => {
            delete_positioning_info(obj);
        });

        this.emit_event('position_changed', {
            type: 'reset',
        });

        this.relayout();
        this.draw_async();
    }

    // Save functions
    public save(filename: string, contents: string | undefined): void {
        if (!contents)
            return;
        const link = document.createElement('a');
        link.setAttribute('download', filename);
        link.href = contents;
        document.body.appendChild(link);

        // wait for the link to be added to the document
        window.requestAnimationFrame(() => {
            const event = new MouseEvent('click');
            link.dispatchEvent(event);
            document.body.removeChild(link);
        });
    }

    public save_sdfg(): void {
        const name = this.sdfg.attributes.name;
        const contents = 'data:text/json;charset=utf-8,' + encodeURIComponent(stringify_sdfg(this.sdfg));
        this.save(name + '.sdfg', contents);
    }

    public save_as_png(): void {
        const name = this.sdfg.attributes.name;
        this.save(name + '.png', this.canvas?.toDataURL('image/png'));
    }

    public has_pdf(): boolean {
        try {
            blobStream;
            canvas2pdf.PdfContext;
            return true;
        } catch (e) {
            return false;
        }
    }

    public save_as_pdf(save_all = false): void {
        const stream = blobStream();

        // Compute document size
        const curx = this.canvas_manager?.mapPixelToCoordsX(0);
        const cury = this.canvas_manager?.mapPixelToCoordsY(0);
        let size;
        if (save_all) {
            // Get size of entire graph
            const elements: SDFGElement[] = [];
            this.graph?.nodes().forEach((n_id: string) => {
                const node = this.graph?.node(n_id);
                if (node)
                    elements.push(node);
            });
            const bb = boundingBox(elements);
            size = [bb.width, bb.height];
        } else {
            // Get size of current view
            const canvasw = this.canvas?.width;
            const canvash = this.canvas?.height;
            let endx = null;
            if (canvasw)
                endx = this.canvas_manager?.mapPixelToCoordsX(canvasw);
            let endy = null;
            if (canvash)
                endy = this.canvas_manager?.mapPixelToCoordsY(canvash);
            const curw = (endx ? endx : 0) - (curx ? curx : 0);
            const curh = (endy ? endy : 0) - (cury ? cury : 0);
            size = [curw, curh];
        }
        //

        const ctx = new canvas2pdf.PdfContext(stream, {
            size: size
        });
        const oldctx = this.ctx;
        this.ctx = ctx;
        (this.ctx as any).lod = !save_all;
        (this.ctx as any).pdf = true;
        // Center on saved region
        if (!save_all)
            this.ctx?.translate(-(curx ? curx : 0), -(cury ? cury : 0));

        this.draw_async();

        ctx.stream.on('finish', () => {
            const name = this.sdfg.attributes.name;
            this.save(name + '.pdf', ctx.stream.toBlobURL('application/pdf'));
            this.ctx = oldctx;
            this.draw_async();
        });
    }

    // Draw a debug grid on the canvas to indicate coordinates.
    public debug_draw_grid(
        curx: number, cury: number, endx: number, endy: number,
        grid_width: number = 100
    ): void {
        if (!this.ctx)
            return;

        const lim_x_min = Math.floor(curx / grid_width) * grid_width;
        const lim_x_max = Math.ceil(endx / grid_width) * grid_width;
        const lim_y_min = Math.floor(cury / grid_width) * grid_width;
        const lim_y_max = Math.ceil(endy / grid_width) * grid_width;
        for (let i = lim_x_min; i <= lim_x_max; i += grid_width) {
            this.ctx.moveTo(i, lim_y_min);
            this.ctx.lineTo(i, lim_y_max);
        }
        for (let i = lim_y_min; i <= lim_y_max; i += grid_width) {
            this.ctx.moveTo(lim_x_min, i);
            this.ctx.lineTo(lim_x_max, i);
        }
        this.ctx.strokeStyle = 'yellow';
        this.ctx.stroke();

        // Draw the zero-point.
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 10, 0, 2 * Math.PI, false);
        this.ctx.fillStyle = 'red';
        this.ctx.fill();
        this.ctx.strokeStyle = 'red';
        this.ctx.stroke();
    }

    // Render SDFG
    public draw(dt: number | null): void {
        if (!this.graph || !this.ctx)
            return;

        const ctx = this.ctx;
        const g = this.graph;
        const curx = this.canvas_manager?.mapPixelToCoordsX(0);
        const cury = this.canvas_manager?.mapPixelToCoordsY(0);
        const canvasw = this.canvas?.width;
        const canvash = this.canvas?.height;
        let endx = null;
        if (canvasw)
            endx = this.canvas_manager?.mapPixelToCoordsX(canvasw);
        let endy = null;
        if (canvash)
            endy = this.canvas_manager?.mapPixelToCoordsY(canvash);
        const curw = (endx ? endx : 0) - (curx ? curx : 0);
        const curh = (endy ? endy : 0) - (cury ? cury : 0);

        this.visible_rect = {
            x: curx ? curx : 0,
            y: cury ? cury : 0,
            w: curw,
            h: curh
        };

        this.on_pre_draw();

        draw_sdfg(this, ctx, g, this.mousepos);

        if (this.box_select_rect) {
            this.ctx.beginPath();
            const old_line_width = this.ctx.lineWidth;
            const new_line_width = this.canvas_manager?.points_per_pixel();
            if (new_line_width !== undefined)
                this.ctx.lineWidth = new_line_width;
            this.ctx.strokeStyle = 'grey';
            this.ctx.rect(
                this.box_select_rect.x_start, this.box_select_rect.y_start,
                this.box_select_rect.x_end - this.box_select_rect.x_start,
                this.box_select_rect.y_end - this.box_select_rect.y_start
            );
            this.ctx.stroke();
            this.ctx.lineWidth = old_line_width;
        }

        if (this.debug_draw) {
            this.debug_draw_grid(
                (curx ? curx : 0),
                (cury ? cury : 0),
                (endx ? endx : 0),
                (endy ? endy : 0),
                100
            );

            if (this.dbg_mouse_coords) {
                if (this.mousepos) {
                    this.dbg_mouse_coords.innerText =
                        'x: ' + Math.floor(this.mousepos.x) +
                        ' / y: ' + Math.floor(this.mousepos.y);
                } else {
                    this.dbg_mouse_coords.innerText = 'x: N/A / y: N/A';
                }
            }
        }

        this.on_post_draw();
    }

    public on_pre_draw(): void {
        return;
    }

    public on_post_draw(): void {
        if (this.overlay_manager !== null)
            this.overlay_manager.draw();

        try {
            (this.ctx as any).end();
        } catch (ex) {
            // TODO: make sure no error is thrown instead of catching and
            // silently ignoring it?
        }

        if (this.tooltip && this.realmousepos) {
            const br = this.canvas?.getBoundingClientRect();
            const pos = {
                x: this.realmousepos.x - (br ? br.x : 0),
                y: this.realmousepos.y - (br ? br.y : 0),
            };

            if (this.tooltip_container) {
                // Clear style and contents
                this.tooltip_container.style.top = '';
                this.tooltip_container.style.left = '';
                this.tooltip_container.innerHTML = '';
                this.tooltip_container.style.display = 'block';

                // Invoke custom container
                this.tooltip(this.tooltip_container);

                // Make visible near mouse pointer
                this.tooltip_container.style.top = pos.y + 'px';
                this.tooltip_container.style.left = (pos.x + 20) + 'px';
            }
        } else {
            if (this.tooltip_container)
                this.tooltip_container.style.display = 'none';
        }

        if (this.sdfg.error && this.graph) {
            const error = this.sdfg.error;

            let type = '';
            let state_id = -1;
            let el_id = -1;
            if (error.isedge_id !== undefined) {
                type = 'isedge';
                el_id = error.isedge_id;
            } else if (error.state_id !== undefined) {
                state_id = error.state_id;
                if (error.node_id !== undefined) {
                    type = 'node';
                    el_id = error.node_id;
                } else if (error.edge_id !== undefined) {
                    type = 'edge';
                    el_id = error.edge_id;
                } else {
                    type = 'state';
                }
            } else {
                return;
            }
            const offending_element = find_graph_element_by_uuid(
                this.graph, error.sdfg_id + '/' + state_id + '/' + el_id + '/-1'
            );
            if (offending_element) {
                this.zoom_to_view([offending_element.element]);

                if (this.error_popover_container) {
                    this.error_popover_container.style.display = 'block';
                    this.error_popover_container.style.bottom = '5%';
                    this.error_popover_container.style.left = '5%';
                }

                if (this.error_popover_text && error.message)
                    this.error_popover_text.innerText = error.message;
            }
        } else {
            if (this.error_popover_container)
                this.error_popover_container.style.display = 'none';
        }
    }

    public visible_elements(): {
        type: string,
        state_id: number,
        sdfg_id: number,
        id: number,
    }[] {
        if (!this.canvas_manager)
            return [];

        const curx = this.canvas_manager.mapPixelToCoordsX(0);
        const cury = this.canvas_manager.mapPixelToCoordsY(0);
        const canvasw = this.canvas?.width;
        const canvash = this.canvas?.height;
        let endx = null;
        if (canvasw)
            endx = this.canvas_manager.mapPixelToCoordsX(canvasw);
        let endy = null;
        if (canvash)
            endy = this.canvas_manager.mapPixelToCoordsY(canvash);
        const curw = (endx ? endx : 0) - curx;
        const curh = (endy ? endy : 0) - cury;
        const elements: any[] = [];
        this.do_for_intersected_elements(
            curx, cury, curw, curh,
            (type: any, e: any, obj: any) => {
                const state_id = e.state ? Number(e.state) : -1;
                let el_type = 'other';
                if (type === 'nodes')
                    el_type = 'node';
                else if (type === 'states')
                    el_type = 'state';
                else if (type === 'edges')
                    el_type = 'edge';
                else if (type === 'isedges')
                    el_type = 'isedge';
                else if (type === 'connectors')
                    el_type = 'connector';
                elements.push({
                    type: el_type,
                    sdfg_id: Number(e.sdfg_id),
                    state_id: state_id,
                    id: Number(e.id),
                });
            }
        );
        return elements;
    }

    public do_for_visible_elements(func: CallableFunction): void {
        if (!this.canvas_manager)
            return;

        const curx = this.canvas_manager.mapPixelToCoordsX(0);
        const cury = this.canvas_manager.mapPixelToCoordsY(0);
        const canvasw = this.canvas?.width;
        const canvash = this.canvas?.height;
        let endx = null;
        if (canvasw)
            endx = this.canvas_manager.mapPixelToCoordsX(canvasw);
        let endy = null;
        if (canvash)
            endy = this.canvas_manager.mapPixelToCoordsY(canvash);
        const curw = (endx ? endx : 0) - curx;
        const curh = (endy ? endy : 0) - cury;
        this.do_for_intersected_elements(curx, cury, curw, curh, func);
    }

    // Returns a dictionary of SDFG elements in a given rectangle. Used for
    // selection, rendering, localized transformations, etc.
    // The output is a dictionary of lists of dictionaries. The top-level keys
    // are:
    // states, nodes, connectors, edges, isedges (interstate edges).
    // For example:
    // {
    //  'states': [{sdfg: sdfg_name, state: 1}, ...],
    //  'nodes': [sdfg: sdfg_name, state: 1, node: 5],
    //  'edges': [],
    //  'isedges': [],
    //  'connectors': [],
    // }
    public elements_in_rect(x: number, y: number, w: number, h: number): any {
        const elements: any = {
            states: [], nodes: [], connectors: [],
            edges: [], isedges: []
        };
        this.do_for_intersected_elements(
            x, y, w, h, (type: string, e: any, obj: any) => {
                e.obj = obj;
                elements[type].push(e);
            }
        );
        return elements;
    }

    public do_for_intersected_elements(
        x: number, y: number, w: number, h: number, func: CallableFunction
    ): void {
        // Traverse nested SDFGs recursively
        function traverse_recursive(
            g: DagreSDFG | null, sdfg_name: string, sdfg_id: number
        ): void {
            g?.nodes().forEach((state_id: string) => {
                const state: dagre.Node<SDFGElement> = g.node(state_id);
                if (!state)
                    return;

                if (state.intersect(x, y, w, h)) {
                    // States
                    func(
                        'states',
                        {
                            sdfg: sdfg_name, sdfg_id: sdfg_id, id: state_id
                        },
                        state
                    );

                    if (state.data.state.attributes.is_collapsed)
                        return;

                    const ng = state.data.graph;
                    if (!ng)
                        return;
                    ng.nodes().forEach((node_id: string) => {
                        const node = ng.node(node_id);
                        if (node.intersect(x, y, w, h)) {
                            // Selected nodes
                            func(
                                'nodes',
                                {
                                    sdfg: sdfg_name, sdfg_id: sdfg_id,
                                    state: state_id, id: node_id
                                },
                                node
                            );

                            // If nested SDFG, traverse recursively
                            if (node.data.node.type === 'NestedSDFG')
                                traverse_recursive(
                                    node.data.graph,
                                    node.data.node.attributes.sdfg.attributes.name,
                                    node.data.node.attributes.sdfg.sdfg_list_id
                                );
                        }
                        // Connectors
                        node.in_connectors.forEach((c: Connector, i: number) => {
                            if (c.intersect(x, y, w, h))
                                func(
                                    'connectors',
                                    {
                                        sdfg: sdfg_name, sdfg_id: sdfg_id,
                                        state: state_id, node: node_id,
                                        connector: i, conntype: 'in'
                                    },
                                    c
                                );
                        });
                        node.out_connectors.forEach((c: Connector, i: number) => {
                            if (c.intersect(x, y, w, h))
                                func(
                                    'connectors',
                                    {
                                        sdfg: sdfg_name, sdfg_id: sdfg_id,
                                        state: state_id, node: node_id,
                                        connector: i, conntype: 'out'
                                    },
                                    c
                                );
                        });
                    });

                    // Selected edges
                    ng.edges().forEach((edge_id: number) => {
                        const edge = ng.edge(edge_id);
                        if (edge.intersect(x, y, w, h)) {
                            func(
                                'edges',
                                {
                                    sdfg: sdfg_name, sdfg_id: sdfg_id,
                                    state: state_id, id: edge.id
                                },
                                edge
                            );
                        }
                    });
                }
            });

            // Selected inter-state edges
            g?.edges().forEach(isedge_id => {
                const isedge = g.edge(isedge_id);
                if (isedge.intersect(x, y, w, h)) {
                    func(
                        'isedges',
                        {
                            sdfg: sdfg_name, sdfg_id: sdfg_id, id: isedge.id
                        },
                        isedge
                    );
                }
            });
        }

        // Start with top-level SDFG
        traverse_recursive(
            this.graph, this.sdfg.attributes.name,
            this.sdfg.sdfg_list_id
        );
    }

    public for_all_sdfg_elements(func: CallableFunction): void {
        // Traverse nested SDFGs recursively
        function traverse_recursive(sdfg: JsonSDFG) {
            sdfg.nodes.forEach((state: JsonSDFGState, state_id: number) => {
                // States
                func('states', { sdfg: sdfg, id: state_id }, state);

                state.nodes.forEach((node: JsonSDFGNode, node_id: number) => {
                    // Nodes
                    func(
                        'nodes',
                        {
                            sdfg: sdfg, state: state_id, id: node_id
                        },
                        node
                    );

                    // If nested SDFG, traverse recursively
                    if (node.type === 'NestedSDFG')
                        traverse_recursive(node.attributes.sdfg);
                });

                // Edges
                state.edges.forEach((edge: JsonSDFGEdge, edge_id: number) => {
                    func(
                        'edges',
                        {
                            sdfg: sdfg, state: state_id, id: edge_id
                        },
                        edge
                    );
                });
            });

            // Selected inter-state edges
            sdfg.edges.forEach((isedge: JsonSDFGEdge, isedge_id: number) => {
                func('isedges', { sdfg: sdfg, id: isedge_id }, isedge);
            });
        }

        // Start with top-level SDFG
        traverse_recursive(this.sdfg);
    }

    public for_all_elements(
        x: number, y: number, w: number, h: number, func: CallableFunction
    ): void {
        // Traverse nested SDFGs recursively
        function traverse_recursive(g: DagreSDFG | null, sdfg_name: string) {
            g?.nodes().forEach(state_id => {
                const state: State = g.node(state_id);
                if (!state)
                    return;

                // States
                func(
                    'states',
                    {
                        sdfg: sdfg_name, id: state_id, graph: g
                    },
                    state
                );

                if (state.data.state.attributes.is_collapsed)
                    return;

                const ng = state.data.graph;
                if (!ng)
                    return;
                ng.nodes().forEach((node_id: string) => {
                    const node = ng.node(node_id);
                    // Selected nodes
                    func(
                        'nodes',
                        {
                            sdfg: sdfg_name, state: state_id, id: node_id,
                            graph: ng
                        },
                        node
                    );

                    // If nested SDFG, traverse recursively
                    if (node.data.node.type === 'NestedSDFG')
                        traverse_recursive(
                            node.data.graph,
                            node.data.node.attributes.sdfg.attributes.name
                        );

                    // Connectors
                    node.in_connectors.forEach((c: Connector, i: number) => {
                        func('connectors', {
                            sdfg: sdfg_name, state: state_id, node: node_id,
                            connector: i, conntype: 'in', graph: ng
                        }, c
                        );
                    });
                    node.out_connectors.forEach((c: Connector, i: number) => {
                        func('connectors', {
                            sdfg: sdfg_name, state: state_id, node: node_id,
                            connector: i, conntype: 'out', graph: ng
                        }, c
                        );
                    });
                });

                // Selected edges
                ng.edges().forEach((edge_id: number) => {
                    const edge = ng.edge(edge_id);
                    func(
                        'edges',
                        {
                            sdfg: sdfg_name, state: state_id, id: edge.id,
                            graph: ng
                        },
                        edge
                    );
                });
            });

            // Selected inter-state edges
            g?.edges().forEach(isedge_id => {
                const isedge = g.edge(isedge_id);
                func(
                    'isedges',
                    {
                        sdfg: sdfg_name, id: isedge.id, graph: g
                    },
                    isedge
                );
            });
        }

        // Start with top-level SDFG
        traverse_recursive(this.graph, this.sdfg.attributes.name);
    }

    public get_nested_memlet_tree(edge: Edge): Set<Edge> {
        for (const tree of this.all_memlet_trees)
            if (tree.has(edge))
                return tree;
        return new Set<Edge>();
    }

    public find_elements_under_cursor(
        mouse_pos_x: number, mouse_pos_y: number
    ): any {
        // Find all elements under the cursor.
        const elements = this.elements_in_rect(mouse_pos_x, mouse_pos_y, 0, 0);
        const clicked_states = elements.states;
        const clicked_nodes = elements.nodes;
        const clicked_edges = elements.edges;
        const clicked_interstate_edges = elements.isedges;
        const clicked_connectors = elements.connectors;
        const total_elements =
            clicked_states.length + clicked_nodes.length +
            clicked_edges.length + clicked_interstate_edges.length +
            clicked_connectors.length;
        let foreground_elem = null, foreground_surface = -1;

        // Find the top-most element under the mouse cursor (i.e. the one with
        // the smallest dimensions).
        const categories = [
            clicked_states,
            clicked_interstate_edges,
            clicked_nodes,
            clicked_edges
        ];
        for (const category of categories) {
            for (let i = 0; i < category.length; i++) {
                const s = category[i].obj.width * category[i].obj.height;
                if (foreground_surface < 0 || s < foreground_surface) {
                    foreground_surface = s;
                    foreground_elem = category[i].obj;
                }
            }
        }

        return {
            total_elements,
            elements,
            foreground_elem,
        };
    }

    public clear_key_events(): void {
        this.mouse_mode = 'pan';
        this.update_toggle_buttons();
    }

    public on_key_event(event: KeyboardEvent): boolean {
        // Prevent handling of the event if the event is designed for something
        // other than the body, like an input element.
        if (event.target !== document.body)
            return false;

        if (this.ctrl_key_selection && !event.ctrlKey) {
            if (this.selectmode_btn?.onclick)
                (this.selectmode_btn as any)?.onclick(event, false);
        }

        if (this.shift_key_movement && !event.shiftKey) {
            if (this.movemode_btn?.onclick)
                (this.movemode_btn as any)?.onclick(event, false);
        }

        if (this.mouse_mode !== 'pan') {
            if (event.key === 'Escape' && !event.ctrlKey && !event.shiftKey) {
                if (this.panmode_btn?.onclick)
                    (this.panmode_btn as any)?.onclick(event);
            }
            return false;
        } else if (event.key === 'Escape') {
            if (this.selected_elements.length > 0) {
                this.selected_elements.forEach(el => {
                    el.selected = false;
                });
                this.deselect();
                this.draw_async();
            }
        } else if (event.key === 'Delete' && event.type === 'keyup') {
            // Sort in reversed order, so that deletion in sequence always retains original IDs
            this.selected_elements.sort((a, b) => (b.id - a.id));
            for (const e of this.selected_elements) {
                if (e instanceof Connector)
                    continue;
                else if (e instanceof Edge) {
                    if (e.parent_id == null)
                        e.sdfg.edges = e.sdfg.edges.filter((_, ind: number) => ind != e.id);
                    else {
                        const state: JsonSDFGState = e.sdfg.nodes[e.parent_id];
                        state.edges = state.edges.filter((_, ind: number) => ind != e.id);
                    }
                } else if (e instanceof State)
                    delete_sdfg_states(e.sdfg, [e.id]);
                else
                    delete_sdfg_nodes(e.sdfg, e.parent_id!, [e.id]);
            }
            this.deselect();
            this.set_sdfg(this.sdfg); // Reset and relayout
        }

        if (event.ctrlKey && !event.shiftKey) {
            if (this.selectmode_btn?.onclick)
                (this.selectmode_btn as any).onclick(event, true);
        }

        if (event.shiftKey && !event.ctrlKey) {
            if (this.movemode_btn?.onclick)
                (this.movemode_btn as any).onclick(event, true);
        }

        return true;
    }

    public on_mouse_event(
        event: any, comp_x_func: CallableFunction,
        comp_y_func: CallableFunction, evtype: string = 'other'
    ): boolean {
        if (!this.graph)
            return false;

        if (this.ctrl_key_selection || this.shift_key_movement)
            this.on_key_event(event);

        let dirty = false; // Whether to redraw at the end
        // Whether the set of visible or selected elements changed
        let element_focus_changed = false;
        // Whether the current multi-selection changed
        let multi_selection_changed = false;
        let selection_changed = false;

        if (evtype === 'mousedown' || evtype === 'touchstart') {
            this.drag_start = event;
        } else if (evtype === 'mouseup') {
            this.drag_start = null;
            this.last_dragged_element = null;
        } else if (evtype === 'touchend') {
            if (event.touches.length == 0)
                this.drag_start = null;
            else
                this.drag_start = event;
        } else if (evtype === 'mousemove') {
            // Calculate the change in mouse position in canvas coordinates
            const old_mousepos = this.mousepos;
            this.mousepos = {
                x: comp_x_func(event),
                y: comp_y_func(event)
            };
            this.realmousepos = { x: event.clientX, y: event.clientY };

            // Only accept the primary mouse button as dragging source
            if (this.drag_start && event.buttons & 1) {
                this.dragging = true;

                if (this.mouse_mode === 'move') {
                    if (this.last_dragged_element) {
                        if (this.canvas)
                            this.canvas.style.cursor = 'grabbing';
                        this.drag_start.cx = comp_x_func(this.drag_start);
                        this.drag_start.cy = comp_y_func(this.drag_start);
                        let elements_to_move = [this.last_dragged_element];
                        if (this.selected_elements.includes(
                            this.last_dragged_element
                        ) && this.selected_elements.length > 1) {
                            elements_to_move = this.selected_elements.filter(
                                el => {
                                    // Do not move connectors (individually)
                                    if (el instanceof Connector)
                                        return false;
                                    const list_id = el.sdfg.sdfg_list_id;

                                    // Do not move element individually if it is
                                    // moved together with a nested SDFG
                                    const nested_sdfg_parent =
                                        this.state_parent_list[list_id];
                                    if (nested_sdfg_parent &&
                                        this.selected_elements.includes(
                                            nested_sdfg_parent
                                        ))
                                        return false;

                                    // Do not move element individually if it is
                                    // moved together with its parent state
                                    const state_parent =
                                        this.sdfg_list[list_id].node(
                                            el.parent_id!.toString()
                                        );
                                    if (state_parent &&
                                        this.selected_elements.includes(
                                            state_parent
                                        ))
                                        return false;

                                    // Otherwise move individually
                                    return true;
                                }
                            );
                        }

                        const move_entire_edge = elements_to_move.length > 1;
                        for (const el of elements_to_move) {
                            if (old_mousepos)
                                this.canvas_manager?.translate_element(
                                    el, old_mousepos, this.mousepos,
                                    this.graph, this.sdfg_list,
                                    this.state_parent_list,
                                    this.drag_start,
                                    true,
                                    move_entire_edge
                                );
                        }

                        dirty = true;
                        this.draw_async();
                        return false;
                    } else {
                        const mouse_elements = this.find_elements_under_cursor(
                            this.mousepos.x, this.mousepos.y
                        );
                        if (mouse_elements.foreground_elem) {
                            this.last_dragged_element =
                                mouse_elements.foreground_elem;
                            if (this.canvas)
                                this.canvas.style.cursor = 'grabbing';
                            return false;
                        }
                        return true;
                    }
                } else if (this.mouse_mode === 'select') {
                    this.box_select_rect = {
                        x_start: comp_x_func(this.drag_start),
                        y_start: comp_y_func(this.drag_start),
                        x_end: this.mousepos.x,
                        y_end: this.mousepos.y,
                    };

                    // Mark for redraw
                    dirty = true;
                } else {
                    this.canvas_manager?.translate(
                        event.movementX, event.movementY
                    );

                    // Mark for redraw
                    dirty = true;
                }
            } else if (this.drag_start && event.buttons & 4) {
                // Pan the view with the middle mouse button
                this.dragging = true;
                this.canvas_manager?.translate(
                    event.movementX, event.movementY
                );
                dirty = true;
                element_focus_changed = true;
            } else {
                this.drag_start = null;
                this.last_dragged_element = null;
                if (event.buttons & 1 || event.buttons & 4)
                    return true; // Don't stop propagation
            }
        } else if (evtype === 'touchmove') {
            if (this.drag_start.touches.length != event.touches.length) {
                // Different number of touches, ignore and reset drag_start
                this.drag_start = event;
            } else if (event.touches.length == 1) { // Move/drag
                this.canvas_manager?.translate(
                    event.touches[0].clientX - this.drag_start.touches[0].clientX,
                    event.touches[0].clientY - this.drag_start.touches[0].clientY
                );
                this.drag_start = event;

                // Mark for redraw
                dirty = true;
                this.draw_async();
                return false;
            } else if (event.touches.length == 2) {
                // Find relative distance between two touches before and after.
                // Then, center and zoom to their midpoint.
                const touch1 = this.drag_start.touches[0];
                const touch2 = this.drag_start.touches[1];
                let x1 = touch1.clientX, x2 = touch2.clientX;
                let y1 = touch1.clientY, y2 = touch2.clientY;
                const oldCenter = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];
                const initialDistance = Math.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2
                );
                x1 = event.touches[0].clientX; x2 = event.touches[1].clientX;
                y1 = event.touches[0].clientY; y2 = event.touches[1].clientY;
                const currentDistance = Math.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2
                );
                const newCenter = [(x1 + x2) / 2.0, (y1 + y2) / 2.0];

                // First, translate according to movement of center point
                this.canvas_manager?.translate(
                    newCenter[0] - oldCenter[0], newCenter[1] - oldCenter[1]
                );
                // Then scale
                this.canvas_manager?.scale(
                    currentDistance / initialDistance, newCenter[0],
                    newCenter[1]
                );

                this.drag_start = event;

                // Mark for redraw
                dirty = true;
                this.draw_async();
                return false;
            }
        } else if (evtype === 'wheel') {
            // Get physical x,y coordinates (rather than canvas coordinates)
            const br = this.canvas?.getBoundingClientRect();
            const x = event.clientX - (br ? br.x : 0);
            const y = event.clientY - (br ? br.y : 0);
            this.canvas_manager?.scale(event.deltaY > 0 ? 0.9 : 1.1, x, y);
            dirty = true;
            element_focus_changed = true;
        }
        // End of mouse-move/touch-based events

        if (!this.mousepos)
            return true;

        // Find elements under cursor
        const elements_under_cursor = this.find_elements_under_cursor(
            this.mousepos.x, this.mousepos.y
        );
        const elements = elements_under_cursor.elements;
        const total_elements = elements_under_cursor.total_elements;
        const foreground_elem = elements_under_cursor.foreground_elem;

        if (this.mouse_mode == 'add') {
            const el = this.mouse_follow_element;
            if (check_valid_add_position(
                (this.add_type ? this.add_type : ''),
                foreground_elem, this.add_mode_lib, this.mousepos
            ))
                el.firstElementChild.setAttribute('stroke', 'green');
            else
                el.firstElementChild.setAttribute('stroke', 'red');

            el.style.left =
                (event.layerX - el.firstElementChild.clientWidth / 2) + 'px';
            el.style.top =
                (event.layerY - el.firstElementChild.clientHeight / 2) + 'px';
        }

        // Change mouse cursor accordingly
        if (this.canvas) {
            if (this.mouse_mode === 'select') {
                this.canvas.style.cursor = 'crosshair';
            } else if (total_elements > 0) {
                if (this.mouse_mode === 'move' && this.drag_start) {
                    this.canvas.style.cursor = 'grabbing';
                } else if (this.mouse_mode === 'move') {
                    this.canvas.style.cursor = 'grab';
                } else {
                    // Hovering over an element while not in any specific mode.
                    if ((foreground_elem.data.state &&
                        foreground_elem.data.state.attributes.is_collapsed) ||
                        (foreground_elem.data.node &&
                            foreground_elem.data.node.attributes.is_collapsed)) {
                        // This is a collapsed node or state, show with the
                        // cursor shape that this can be expanded.
                        this.canvas.style.cursor = 'alias';
                    } else {
                        this.canvas.style.cursor = 'pointer';
                    }
                }
            } else {
                this.canvas.style.cursor = 'auto';
            }
        }

        this.tooltip = null;

        // De-highlight all elements.
        this.do_for_visible_elements(
            (type: any, e: any, obj: any) => {
                obj.hovered = false;
                obj.highlighted = false;
            }
        );
        // Mark hovered and highlighted elements.
        this.do_for_visible_elements(
            (type: any, e: any, obj: any) => {
                const intersected = obj.intersect(this.mousepos!.x, this.mousepos!.y, 0, 0);

                // Highlight all edges of the memlet tree
                if (intersected && obj instanceof Edge &&
                    obj.parent_id != null) {
                    const tree = this.get_nested_memlet_tree(obj);
                    tree.forEach(te => {
                        if (te != obj && te !== undefined) {
                            te.highlighted = true;
                        }
                    });
                }

                // Highlight all access nodes with the same name in the same
                // nested sdfg
                if (intersected && obj instanceof AccessNode) {
                    traverse_sdfg_scopes(
                        this.sdfg_list[obj.sdfg.sdfg_list_id],
                        (node: any) => {
                            // If node is a state, then visit sub-scope
                            if (node instanceof State)
                                return true;
                            if (node instanceof AccessNode &&
                                node.data.node.label === obj.data.node.label)
                                node.highlighted = true;
                            // No need to visit sub-scope
                            return false;
                        }
                    );
                }

                // Highlight all access nodes with the same name as the hovered
                // connector in the nested sdfg
                if (intersected && obj instanceof Connector && e.graph) {
                    const nested_graph = e.graph.node(obj.parent_id).data.graph;
                    if (nested_graph) {
                        traverse_sdfg_scopes(nested_graph, (node: any) => {
                            // If node is a state, then visit sub-scope
                            if (node instanceof State) {
                                return true;
                            }
                            if (node instanceof AccessNode &&
                                node.data.node.label === obj.label()) {
                                node.highlighted = true;
                            }
                            // No need to visit sub-scope
                            return false;
                        });
                    }
                }

                if (intersected)
                    obj.hovered = true;
            }
        );

        // If adding an edge, mark/highlight the first/from element, if it has
        // already been selected.
        if (this.mouse_mode === 'add' && this.add_type === 'Edge' &&
            this.add_edge_start) {
            this.add_edge_start.highlighted = true;
        }

        if (evtype === 'mousemove') {
            // TODO: Draw only if elements have changed
            dirty = true;
        }

        if (evtype === 'dblclick') {
            const sdfg = (foreground_elem ? foreground_elem.sdfg : null);
            let sdfg_elem = null;
            if (foreground_elem instanceof State)
                sdfg_elem = foreground_elem.data.state;
            else if (foreground_elem instanceof SDFGNode) {
                sdfg_elem = foreground_elem.data.node;

                // If a scope exit node, use entry instead
                if (sdfg_elem.type.endsWith('Exit') &&
                    foreground_elem.parent_id !== null)
                    sdfg_elem = sdfg.nodes[foreground_elem.parent_id].nodes[
                        sdfg_elem.scope_entry
                    ];
            } else
                sdfg_elem = null;

            // Toggle collapsed state
            if (sdfg_elem && 'is_collapsed' in sdfg_elem.attributes) {
                sdfg_elem.attributes.is_collapsed =
                    !sdfg_elem.attributes.is_collapsed;

                this.emit_event('collapse_state_changed', null);

                // Re-layout SDFG
                this.relayout();
                dirty = true;
                element_focus_changed = true;
            }
        }

        let ends_drag = false;
        if (evtype === 'click') {
            if (this.dragging) {
                // This click ends a drag.
                this.dragging = false;
                ends_drag = true;

                element_focus_changed = true;

                if (this.box_select_rect) {
                    const elements_in_selection: any[] = [];
                    const start_x = Math.min(this.box_select_rect.x_start,
                        this.box_select_rect.x_end);
                    const end_x = Math.max(this.box_select_rect.x_start,
                        this.box_select_rect.x_end);
                    const start_y = Math.min(this.box_select_rect.y_start,
                        this.box_select_rect.y_end);
                    const end_y = Math.max(this.box_select_rect.y_start,
                        this.box_select_rect.y_end);
                    const w = end_x - start_x;
                    const h = end_y - start_y;
                    this.do_for_intersected_elements(start_x, start_y, w, h,
                        (type: any, e: any, obj: any) => {
                            if (obj.contained_in(start_x, start_y, w, h))
                                elements_in_selection.push(obj);
                        });
                    if (event.shiftKey && !this.ctrl_key_selection) {
                        elements_in_selection.forEach((el) => {
                            if (!this.selected_elements.includes(el))
                                this.selected_elements.push(el);
                        });
                    } else if (event.ctrlKey && !this.ctrl_key_selection) {
                        elements_in_selection.forEach((el) => {
                            if (this.selected_elements.includes(el)) {
                                this.selected_elements =
                                    this.selected_elements.filter((val) => {
                                        val.selected = false;
                                        return val !== el;
                                    });
                            }
                        });
                    } else {
                        this.selected_elements.forEach((el) => {
                            el.selected = false;
                        });
                        this.selected_elements = elements_in_selection;
                    }
                    this.box_select_rect = null;
                    dirty = true;
                    element_focus_changed = true;
                    multi_selection_changed = true;
                }

                if (this.mouse_mode === 'move')
                    this.emit_event('position_changed', {
                        type: 'manual_move'
                    });
            } else {
                if (this.mouse_mode === 'add') {
                    if (check_valid_add_position(
                        this.add_type, foreground_elem, this.add_mode_lib,
                        this.mousepos
                    )) {
                        if (this.add_type === 'Edge') {
                            if (this.add_edge_start) {
                                const start = this.add_edge_start;
                                this.add_edge_start = undefined;
                                this.emit_event(
                                    'add_graph_node',
                                    {
                                        type: this.add_type,
                                        parent: get_uuid_graph_element(
                                            foreground_elem
                                        ),
                                        edgeA: get_uuid_graph_element(start),
                                    }
                                );
                            } else {
                                this.add_edge_start = foreground_elem;
                                this.update_toggle_buttons();
                            }
                        } else if (this.add_type === 'LibraryNode') {
                            this.add_position = this.mousepos;
                            this.emit_event(
                                'add_graph_node',
                                {
                                    type:
                                        this.add_type + '|' + this.add_mode_lib,
                                    parent: get_uuid_graph_element(
                                        foreground_elem
                                    ),
                                    edgeA: null,
                                }
                            );
                        } else {
                            this.add_position = this.mousepos;
                            this.emit_event(
                                'add_graph_node',
                                {
                                    type: this.add_type ? this.add_type : '',
                                    parent: get_uuid_graph_element(
                                        foreground_elem
                                    ),
                                    edgeA: null,
                                }
                            );
                        }

                        if (!event.ctrlKey && !(this.add_type === 'Edge' &&
                            this.add_edge_start)) {
                            // Cancel add mode.
                            if (this.panmode_btn?.onclick)
                                this.panmode_btn.onclick(event);
                        }
                    }
                }

                if (foreground_elem) {
                    if (event.ctrlKey) {
                        // Ctrl + click on an object, add it, or remove it from
                        // the selection if it was previously in it.
                        if (this.selected_elements.includes(foreground_elem)) {
                            foreground_elem.selected = false;
                            this.selected_elements =
                                this.selected_elements.filter((el) => {
                                    return el !== foreground_elem;
                                });
                        } else {
                            this.selected_elements.push(foreground_elem);
                        }

                        // Indicate that the multi-selection changed.
                        multi_selection_changed = true;
                    } else if (event.shiftKey) {
                        // TODO: Implement shift-clicks for path selection.
                    } else {
                        // Clicked an element, select it and nothing else.
                        // If there was a multi-selection prior to this,
                        // indicate that it changed.
                        if (this.selected_elements.length > 1)
                            multi_selection_changed = true;

                        this.selected_elements.forEach((el) => {
                            el.selected = false;
                        });
                        this.selected_elements = [foreground_elem];
                        selection_changed = true;
                    }
                } else {
                    // Clicked nothing, clear the selection.

                    // If there was a multi-selection prior to this, indicate
                    // that it changed.
                    if (this.selected_elements.length > 1)
                        multi_selection_changed = true;

                    this.selected_elements.forEach((el) => {
                        el.selected = false;
                    });
                    this.selected_elements = [];
                    selection_changed = true;
                }
                dirty = true;
                element_focus_changed = true;
            }
        }
        this.selected_elements.forEach((el) => {
            el.selected = true;
        });

        if (evtype === 'contextmenu') {
            if (this.mouse_mode == 'move') {
                let elements_to_reset = [foreground_elem];
                if (this.selected_elements.includes(foreground_elem))
                    elements_to_reset = this.selected_elements;

                let element_moved = false;
                let relayout_necessary = false;
                for (const el of elements_to_reset) {
                    const position = get_positioning_info(el);
                    if (el && !(el instanceof Connector) && position) {
                        // Reset the position of the element (if it has been
                        // manually moved)
                        if (el instanceof Edge) {
                            if (!position.points)
                                continue;

                            const edge_el: Edge = el;
                            // Create inverted points to move it back
                            const new_points = new Array(
                                edge_el.get_points().length
                            );
                            for (
                                let j = 1;
                                j < edge_el.get_points().length - 1;
                                j++
                            ) {
                                new_points[j] = {
                                    dx: - position.points[j].dx,
                                    dy: - position.points[j].dy
                                };
                                // Reset the point movement
                                position.points[j].dx = 0;
                                position.points[j].dy = 0;
                            }

                            // Move it to original position
                            this.canvas_manager?.translate_element(
                                edge_el, { x: 0, y: 0 }, { x: 0, y: 0 },
                                this.graph, this.sdfg_list,
                                this.state_parent_list, undefined, false, false,
                                new_points
                            );

                            element_moved = true;
                        } else {
                            if (!position.dx && !position.dy)
                                continue;

                            // Calculate original position with the relative
                            // movement
                            const new_x = el.x - position.dx;
                            const new_y = el.y - position.dy;

                            position.dx = 0;
                            position.dy = 0;

                            // Move it to original position
                            this.canvas_manager?.translate_element(
                                el, { x: el.x, y: el.y },
                                { x: new_x, y: new_y }, this.graph,
                                this.sdfg_list, this.state_parent_list,
                                undefined, false, false, undefined
                            );

                            element_moved = true;
                        }

                        if (el instanceof EntryNode) {
                            // Also update scope position
                            position.scope_dx = 0;
                            position.scope_dy = 0;

                            if (!el.data.node.attributes.is_collapsed)
                                relayout_necessary = true;
                        }
                    }
                }

                if (relayout_necessary)
                    this.relayout();

                this.draw_async();

                if (element_moved)
                    this.emit_event('position_changed', {
                        type: 'manual_move'
                    });

            } else if (this.mouse_mode == 'add') {
                // Cancel add mode
                if (this.panmode_btn?.onclick)
                    this.panmode_btn?.onclick(event);
            }
        }

        const mouse_x = comp_x_func(event);
        const mouse_y = comp_y_func(event);
        if (this.external_mouse_handler) {
            const ext_mh_dirty = this.external_mouse_handler(
                evtype, event, { x: mouse_x, y: mouse_y }, elements,
                this, this.selected_elements, this.sdfv_instance
            );
            dirty = dirty || ext_mh_dirty;
        }

        if (this.overlay_manager !== null) {
            const ol_manager_dirty = this.overlay_manager.on_mouse_event(
                evtype,
                event,
                { x: mouse_x, y: mouse_y },
                elements,
                foreground_elem,
                ends_drag
            );
            dirty = dirty || ol_manager_dirty;
        }

        if (dirty)
            this.draw_async();

        if (element_focus_changed)
            this.emit_event(
                'renderer_selection_changed',
                {
                    multi_selection_changed: multi_selection_changed,
                }
            );

        if (selection_changed || multi_selection_changed)
            this.on_selection_changed();

        return false;
    }

    public get_inclusive_ranges(): boolean {
        return this.inclusive_ranges;
    }

    public get_canvas(): HTMLCanvasElement | null {
        return this.canvas;
    }

    public get_canvas_manager(): CanvasManager | null {
        return this.canvas_manager;
    }

    public get_context(): CanvasRenderingContext2D | null {
        return this.ctx;
    }

    public get_overlay_manager(): OverlayManager {
        return this.overlay_manager;
    }

    public get_visible_rect(): SimpleRect | null {
        return this.visible_rect;
    }

    public get_mouse_mode(): string {
        return this.mouse_mode;
    }

    public get_bgcolor(): string {
        return (this.bgcolor ? this.bgcolor : '');
    }

    public get_menu(): ContextMenu | null {
        return this.menu;
    }

    public get_sdfg(): JsonSDFG {
        return this.sdfg;
    }

    public get_graph(): DagreSDFG | null {
        return this.graph;
    }

    public get_in_vscode(): boolean {
        return this.in_vscode;
    }

    public get_mousepos(): Point2D | null {
        return this.mousepos;
    }

    public get_tooltip_container(): HTMLElement | null {
        return this.tooltip_container;
    }

    public get_selected_elements(): SDFGElement[] {
        return this.selected_elements;
    }

    public set_tooltip(tooltip_func: SDFVTooltipFunc): void {
        this.tooltip = tooltip_func;
    }

    public set_bgcolor(bgcolor: string): void {
        this.bgcolor = bgcolor;
    }

    public register_ext_event_handler(
        handler: (type: string, data: any) => any
    ): void {
        this.ext_event_handlers.push(handler);
    }

    public on_selection_changed(): void {
        if (this.selected_elements.length > 0 && this.filter_btn)
            this.filter_btn.className = 'button';
        else if (this.filter_btn)
            this.filter_btn.className = 'button hidden';
    }

    public deselect(): void {
        this.selected_elements.forEach((el) => {
            el.selected = false;
        });
        this.selected_elements = [];
        this.on_selection_changed();
    }

    public cutout_selection(): void {
        /*
        Rule set for creating a cutout subgraph:
          * Edges are selected according to the subgraph nodes - all edges between subgraph nodes are preserved.
          * In any element that contains other elements (state, nested SDFG, scopes), the full contents are used.
          * If more than one element is selected from different contexts (two nodes from two states), the parents
            will be preserved.
        */
        // Collect nodes and states
        const sdfgs: Set<number> = new Set<number>();
        const sdfg_list: { [key: string]: JsonSDFG } = {};
        const states: { [key: string]: Array<number> } = {};
        const nodes: { [key: string]: Array<number> } = {};
        for (const elem of this.selected_elements) {
            // Ignore edges and connectors
            if (elem instanceof Edge || elem instanceof Connector)
                continue;
            const sdfg_id = elem.sdfg.sdfg_list_id;
            sdfg_list[sdfg_id] = elem.sdfg;
            sdfgs.add(sdfg_id);
            let state_id: number = -1;
            if (elem.parent_id !== null) {
                const state_uid: string = JSON.stringify([sdfg_id, elem.parent_id]);
                if (state_uid in nodes)
                    nodes[state_uid].push(elem.id);
                else
                    nodes[state_uid] = [elem.id];
                state_id = elem.parent_id;
            } else {
                // Add all nodes from state
                const state_uid: string = JSON.stringify([sdfg_id, elem.id]);
                nodes[state_uid] = [...elem.data.state.nodes.keys()];
                state_id = elem.id;
            }
            // Register state
            if (sdfg_id in states)
                states[sdfg_id].push(state_id);
            else
                states[sdfg_id] = [state_id];
        }

        // Clear selection and redraw
        this.deselect();

        if (Object.keys(nodes).length === 0) {  // Nothing to cut out
            this.draw_async();
            return;
        }

        // Find root SDFG and root state (if possible)
        const root_sdfg_id = find_root_sdfg(sdfgs, this.sdfg_tree);
        if (root_sdfg_id !== null) {
            const root_sdfg = sdfg_list[root_sdfg_id];

            // For every participating state, filter out irrelevant nodes and memlets
            for (const nkey of Object.keys(nodes)) {
                const [sdfg_id, state_id] = JSON.parse(nkey);
                const sdfg = sdfg_list[sdfg_id];
                delete_sdfg_nodes(sdfg, state_id, nodes[nkey], true);
            }

            // For every participating SDFG, filter out irrelevant states and interstate edges
            for (const sdfg_id of Object.keys(states)) {
                const sdfg = sdfg_list[sdfg_id];
                delete_sdfg_states(sdfg, states[sdfg_id], true);
            }

            // Set root SDFG as the new SDFG
            this.set_sdfg(root_sdfg);
        }

    }
}


function calculateNodeSize(
    _sdfg_state: any, node: any, ctx: CanvasRenderingContext2D
): { width: number, height: number } {
    const labelsize = ctx.measureText(node.label).width;
    const inconnsize = 2 * SDFV.LINEHEIGHT * Object.keys(
        node.attributes.layout.in_connectors
    ).length - SDFV.LINEHEIGHT;
    const outconnsize = 2 * SDFV.LINEHEIGHT * Object.keys(
        node.attributes.layout.out_connectors
    ).length - SDFV.LINEHEIGHT;
    const maxwidth = Math.max(labelsize, inconnsize, outconnsize);
    let maxheight = 2 * SDFV.LINEHEIGHT;
    maxheight += 4 * SDFV.LINEHEIGHT;

    const size = { width: maxwidth, height: maxheight };

    // add something to the size based on the shape of the node
    if (node.type === 'AccessNode') {
        size.height -= 4 * SDFV.LINEHEIGHT;
        size.width += size.height;
    } else if (node.type.endsWith('Entry')) {
        size.width += 2.0 * size.height;
        size.height /= 1.75;
    } else if (node.type.endsWith('Exit')) {
        size.width += 2.0 * size.height;
        size.height /= 1.75;
    } else if (node.type === 'Tasklet') {
        size.width += 2.0 * (size.height / 3.0);
        size.height /= 1.75;
    } else if (node.type === 'LibraryNode') {
        size.width += 2.0 * (size.height / 3.0);
        size.height /= 1.75;
    } else if (node.type === 'Reduce') {
        size.height -= 4 * SDFV.LINEHEIGHT;
        size.width *= 2;
        size.height = size.width / 3.0;
    }

    return size;
}

// Layout SDFG elements (states, nodes, scopes, nested SDFGs)
function relayout_sdfg(
    ctx: any,
    sdfg: any,
    sdfg_list: SDFGListType,
    state_parent_list: any[],
    omit_access_nodes: boolean
): DagreSDFG {
    const STATE_MARGIN = 4 * SDFV.LINEHEIGHT;

    // Layout the SDFG as a dagre graph
    const g: DagreSDFG = new dagre.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel((u, v) => { return {}; });

    // layout each state to get its size
    sdfg.nodes.forEach((state: any) => {
        let stateinfo: any = {};

        stateinfo.label = state.id;
        let state_g = null;
        if (state.attributes.is_collapsed) {
            stateinfo.width = ctx.measureText(stateinfo.label).width;
            stateinfo.height = SDFV.LINEHEIGHT;
        } else {
            state_g = relayout_state(
                ctx, state, sdfg, sdfg_list,
                state_parent_list, omit_access_nodes
            );
            if (state_g)
                stateinfo = calculateBoundingBox(state_g);
        }
        stateinfo.width += 2 * STATE_MARGIN;
        stateinfo.height += 2 * STATE_MARGIN;
        g.setNode(state.id, new State({
            state: state,
            layout: stateinfo,
            graph: state_g
        }, state.id, sdfg));
    });

    sdfg.edges.forEach((edge: any, id: number) => {
        g.setEdge(edge.src, edge.dst, new Edge(edge.attributes.data, id, sdfg));
    });

    dagre.layout(g);

    // Annotate the sdfg with its layout info
    sdfg.nodes.forEach((state: any) => {
        const gnode = g.node(state.id);
        state.attributes.layout = {};
        state.attributes.layout.x = gnode.x;
        state.attributes.layout.y = gnode.y;
        state.attributes.layout.width = gnode.width;
        state.attributes.layout.height = gnode.height;
    });

    sdfg.edges.forEach((edge: any) => {
        const gedge = g.edge(edge.src, edge.dst);
        const bb = calculateEdgeBoundingBox(gedge);
        // Convert from top-left to center
        (bb as any).x += bb.width / 2.0;
        (bb as any).y += bb.height / 2.0;

        gedge.x = (bb as any).x;
        gedge.y = (bb as any).y;
        gedge.width = bb.width;
        gedge.height = bb.height;
        edge.attributes.layout = {};
        edge.attributes.layout.width = bb.width;
        edge.attributes.layout.height = bb.height;
        edge.attributes.layout.x = (bb as any).x;
        edge.attributes.layout.y = (bb as any).y;
        edge.attributes.layout.points = gedge.points;
    });

    // Offset node and edge locations to be in state margins
    sdfg.nodes.forEach((s: any, sid: any) => {
        if (s.attributes.is_collapsed)
            return;

        const state: any = g.node(sid);
        const topleft = state.topleft();
        offset_state(s, state, {
            x: topleft.x + STATE_MARGIN,
            y: topleft.y + STATE_MARGIN
        });
    });

    const bb = calculateBoundingBox(g);
    (g as any).width = bb.width;
    (g as any).height = bb.height;

    // Add SDFG to global store
    sdfg_list[sdfg.sdfg_list_id] = g;

    return g;
}

function relayout_state(
    ctx: CanvasRenderingContext2D, sdfg_state: JsonSDFGState,
    sdfg: JsonSDFG, sdfg_list: JsonSDFG[], state_parent_list: any[],
    omit_access_nodes: boolean
): DagreSDFG | null {
    // layout the state as a dagre graph
    const g: DagreSDFG = new dagre.graphlib.Graph({ multigraph: true });

    // Set layout options and a simpler algorithm for large graphs
    const layout_options: any = { ranksep: 30 };
    if (sdfg_state.nodes.length >= 1000)
        layout_options.ranker = 'longest-path';

    g.setGraph(layout_options);


    // Set an object for the graph label
    g.setDefaultEdgeLabel((u, v) => { return {}; });

    // Add nodes to the graph. The first argument is the node id. The
    // second is metadata about the node (label, width, height),
    // which will be updated by dagre.layout (will add x,y).

    // Process nodes hierarchically
    let toplevel_nodes = sdfg_state.scope_dict[-1];
    if (toplevel_nodes === undefined)
        toplevel_nodes = Object.keys(sdfg_state.nodes);
    const drawn_nodes: Set<string> = new Set();
    const hidden_nodes = new Map();

    function layout_node(node: any) {
        if (omit_access_nodes && node.type == 'AccessNode') {
            // add access node to hidden nodes; source and destinations will be
            // set later
            hidden_nodes.set(
                node.id.toString(), { node: node, src: null, dsts: [] }
            );
            return;
        }

        let nested_g = null;
        node.attributes.layout = {};

        // Set connectors prior to computing node size
        node.attributes.layout.in_connectors = node.attributes.in_connectors;
        if ('is_collapsed' in node.attributes && node.attributes.is_collapsed &&
            node.type !== 'NestedSDFG')
            node.attributes.layout.out_connectors = find_exit_for_entry(
                sdfg_state.nodes, node
            )?.attributes.out_connectors;
        else
            node.attributes.layout.out_connectors =
                node.attributes.out_connectors;

        const nodesize = calculateNodeSize(sdfg_state, node, ctx);
        node.attributes.layout.width = nodesize.width;
        node.attributes.layout.height = nodesize.height;
        node.attributes.layout.label = node.label;

        // Recursively lay out nested SDFGs
        if (node.type === 'NestedSDFG') {
            nested_g = relayout_sdfg(
                ctx, node.attributes.sdfg, sdfg_list, state_parent_list,
                omit_access_nodes
            );
            const sdfginfo = calculateBoundingBox(nested_g);
            node.attributes.layout.width = sdfginfo.width + 2 * SDFV.LINEHEIGHT;
            node.attributes.layout.height =
                sdfginfo.height + 2 * SDFV.LINEHEIGHT;
        }

        // Dynamically create node type
        const obj = new SDFGElements[node.type](
            { node: node, graph: nested_g }, node.id, sdfg, sdfg_state.id
        );

        // If it's a nested SDFG, we need to record the node as all of its
        // state's parent node
        if (node.type === 'NestedSDFG')
            state_parent_list[node.attributes.sdfg.sdfg_list_id] = obj;

        // Add input connectors
        let i = 0;
        let conns;
        if (Array.isArray(node.attributes.layout.in_connectors))
            conns = node.attributes.layout.in_connectors;
        else
            conns = Object.keys(node.attributes.layout.in_connectors);
        for (const cname of conns) {
            const conn = new Connector({ name: cname }, i, sdfg, node.id);
            obj.in_connectors.push(conn);
            i += 1;
        }

        // Add output connectors -- if collapsed, uses exit node connectors
        i = 0;
        if (Array.isArray(node.attributes.layout.out_connectors))
            conns = node.attributes.layout.out_connectors;
        else
            conns = Object.keys(node.attributes.layout.out_connectors);
        for (const cname of conns) {
            const conn = new Connector({ name: cname }, i, sdfg, node.id);
            obj.out_connectors.push(conn);
            i += 1;
        }

        g.setNode(node.id, obj);
        drawn_nodes.add(node.id.toString());

        // Recursively draw nodes
        if (node.id in sdfg_state.scope_dict) {
            if (node.attributes.is_collapsed)
                return;
            sdfg_state.scope_dict[node.id].forEach((nodeid: number) => {
                const node = sdfg_state.nodes[nodeid];
                layout_node(node);
            });
        }
    }


    toplevel_nodes.forEach((nodeid: number) => {
        const node = sdfg_state.nodes[nodeid];
        layout_node(node);
    });

    // add info to calculate shortcut edges
    function add_edge_info_if_hidden(edge: any) {
        const hidden_src = hidden_nodes.get(edge.src);
        const hidden_dst = hidden_nodes.get(edge.dst);

        if (hidden_src && hidden_dst) {
            // if we have edges from an AccessNode to an AccessNode then just
            // connect destinations
            hidden_src.dsts = hidden_dst.dsts;
            edge.attributes.data.attributes.shortcut = false;
        } else if (hidden_src) {
            // if edge starts at hidden node, then add it as destination
            hidden_src.dsts.push(edge);
            edge.attributes.data.attributes.shortcut = false;
            return true;
        } else if (hidden_dst) {
            // if edge ends at hidden node, then add it as source
            hidden_dst.src = edge;
            edge.attributes.data.attributes.shortcut = false;
            return true;
        }

        // if it is a shortcut edge, but we don't omit access nodes, then ignore
        // this edge
        if (!omit_access_nodes && edge.attributes.data.attributes.shortcut)
            return true;

        return false;
    }

    sdfg_state.edges.forEach((edge: any, id: any) => {
        if (add_edge_info_if_hidden(edge)) return;
        edge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!edge) return;
        const e = new Edge(edge.attributes.data, id, sdfg, sdfg_state.id);
        edge.attributes.data.edge = e;
        (e as any).src_connector = edge.src_connector;
        (e as any).dst_connector = edge.dst_connector;
        g.setEdge(edge.src, edge.dst, e, id);
    });

    hidden_nodes.forEach(hidden_node => {
        if (hidden_node.src) {
            hidden_node.dsts.forEach((e: any) => {
                // create shortcut edge with new destination
                const tmp_edge = e.attributes.data.edge;
                e.attributes.data.edge = null;
                const shortcut_e = deepCopy(e);
                e.attributes.data.edge = tmp_edge;
                shortcut_e.src = hidden_node.src.src;
                shortcut_e.src_connector = hidden_node.src.src_connector;
                shortcut_e.dst_connector = e.dst_connector;
                // attribute that only shortcut edges have; if it is explicitly
                // false, then edge is ignored in omit access node mode
                shortcut_e.attributes.data.attributes.shortcut = true;

                // draw the redirected edge
                const redirected_e = check_and_redirect_edge(
                    shortcut_e, drawn_nodes, sdfg_state
                );
                if (!redirected_e) return;

                // abort if shortcut edge already exists
                const edges = g.outEdges(redirected_e.src);
                if (edges) {
                    for (const oe of edges) {
                        if (oe.w == e.dst && oe.name &&
                            sdfg_state.edges[parseInt(oe.name)].dst_connector ==
                            e.dst_connector
                        ) {
                            return;
                        }
                    }
                }

                // add shortcut edge (redirection is not done in this list)
                sdfg_state.edges.push(shortcut_e);

                // add redirected shortcut edge to graph
                const edge_id = sdfg_state.edges.length - 1;
                const shortcut_edge = new Edge(
                    deepCopy(redirected_e.attributes.data), edge_id, sdfg,
                    sdfg_state.id
                );
                (shortcut_edge as any).src_connector =
                    redirected_e.src_connector;
                (shortcut_edge as any).dst_connector =
                    redirected_e.dst_connector;
                shortcut_edge.data.attributes.shortcut = true;

                g.setEdge(
                    redirected_e.src, redirected_e.dst, shortcut_edge,
                    edge_id.toString()
                );
            });
        }
    });

    dagre.layout(g);

    // Layout connectors and nested SDFGs
    sdfg_state.nodes.forEach((node: JsonSDFGNode, id: number) => {
        const gnode: any = g.node(id.toString());
        if (!gnode || (omit_access_nodes && gnode instanceof AccessNode)) {
            // ignore nodes that should not be drawn
            return;
        }
        const topleft = gnode.topleft();

        // Offset nested SDFG
        if (node.type === 'NestedSDFG') {

            offset_sdfg(node.attributes.sdfg, gnode.data.graph, {
                x: topleft.x + SDFV.LINEHEIGHT,
                y: topleft.y + SDFV.LINEHEIGHT
            });
        }
        // Connector management 
        const SPACING = SDFV.LINEHEIGHT;
        const iconn_length = (SDFV.LINEHEIGHT + SPACING) * Object.keys(
            node.attributes.layout.in_connectors
        ).length - SPACING;
        const oconn_length = (SDFV.LINEHEIGHT + SPACING) * Object.keys(
            node.attributes.layout.out_connectors
        ).length - SPACING;
        let iconn_x = gnode.x - iconn_length / 2.0 + SDFV.LINEHEIGHT / 2.0;
        let oconn_x = gnode.x - oconn_length / 2.0 + SDFV.LINEHEIGHT / 2.0;

        for (const c of gnode.in_connectors) {
            c.width = SDFV.LINEHEIGHT;
            c.height = SDFV.LINEHEIGHT;
            c.x = iconn_x;
            iconn_x += SDFV.LINEHEIGHT + SPACING;
            c.y = topleft.y;
        }
        for (const c of gnode.out_connectors) {
            c.width = SDFV.LINEHEIGHT;
            c.height = SDFV.LINEHEIGHT;
            c.x = oconn_x;
            oconn_x += SDFV.LINEHEIGHT + SPACING;
            c.y = topleft.y + gnode.height;
        }
    });

    sdfg_state.edges.forEach((edge: JsonSDFGEdge, id: number) => {
        const nedge = check_and_redirect_edge(edge, drawn_nodes, sdfg_state);
        if (!nedge) return;
        edge = nedge;
        const gedge = g.edge(edge.src, edge.dst, id.toString());
        if (!gedge || (omit_access_nodes &&
            gedge.data.attributes.shortcut === false
            || !omit_access_nodes && gedge.data.attributes.shortcut)) {
            // if access nodes omitted, don't draw non-shortcut edges and
            // vice versa
            return;
        }

        // Reposition first and last points according to connectors
        let src_conn = null, dst_conn = null;
        if (edge.src_connector) {
            const src_node: SDFGNode = g.node(edge.src);
            let cindex = -1;
            for (let i = 0; i < src_node.out_connectors.length; i++) {
                if (
                    src_node.out_connectors[i].data.name == edge.src_connector
                ) {
                    cindex = i;
                    break;
                }
            }
            if (cindex >= 0) {
                gedge.points[0].x = src_node.out_connectors[cindex].x;
                gedge.points[0].y = src_node.out_connectors[cindex].y;
                src_conn = src_node.out_connectors[cindex];
            }
        }
        if (edge.dst_connector) {
            const dst_node: SDFGNode = g.node(edge.dst);
            let cindex = -1;
            for (let i = 0; i < dst_node.in_connectors.length; i++) {
                if (dst_node.in_connectors[i].data.name == edge.dst_connector) {
                    cindex = i;
                    break;
                }
            }
            if (cindex >= 0) {
                gedge.points[gedge.points.length - 1].x =
                    dst_node.in_connectors[cindex].x;
                gedge.points[gedge.points.length - 1].y =
                    dst_node.in_connectors[cindex].y;
                dst_conn = dst_node.in_connectors[cindex];
            }
        }

        const n = gedge.points.length - 1;
        if (src_conn !== null)
            gedge.points[0] = intersectRect(src_conn, gedge.points[n]);
        if (dst_conn !== null)
            gedge.points[n] = intersectRect(dst_conn, gedge.points[0]);

        if (gedge.points.length == 3 && gedge.points[0].x == gedge.points[n].x)
            gedge.points = [gedge.points[0], gedge.points[n]];

        const bb = calculateEdgeBoundingBox(gedge);
        // Convert from top-left to center
        (bb as any).x += bb.width / 2.0;
        (bb as any).y += bb.height / 2.0;

        edge.width = bb.width;
        edge.height = bb.height;
        edge.x = (bb as any).x;
        edge.y = (bb as any).y;
        gedge.width = bb.width;
        gedge.height = bb.height;
        gedge.x = (bb as any).x;
        gedge.y = (bb as any).y;
    });

    return g;
}

