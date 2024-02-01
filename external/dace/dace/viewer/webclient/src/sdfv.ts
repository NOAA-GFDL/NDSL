// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { parse_sdfg, stringify_sdfg } from './utils/sdfg/json_serializer';
import { mean, median, string } from 'mathjs';
import { SDFGRenderer } from './renderer/renderer';
import { htmlSanitize } from './utils/sanitization';
import {
    Edge,
    SDFG,
    SDFGElement,
    SDFGNode,
    State,
    AccessNode,
    NestedSDFG,
} from './renderer/renderer_elements';
import {
    RuntimeMicroSecondsOverlay
} from './overlays/runtime_micro_seconds_overlay';
import {
    DagreSDFG,
    GenericSdfgOverlay,
    JsonSDFG,
    Point2D,
    sdfg_property_to_string,
    traverse_sdfg_scopes,
} from './index';
import $ from 'jquery';
import { OverlayManager } from './overlay_manager';

let fr: FileReader;
let file: File | null = null;
let instrumentation_file: File | null = null;

export class SDFV {

    public static LINEHEIGHT: number = 10;
    // Points-per-pixel threshold for drawing tasklet contents.
    public static TASKLET_LOD: number = 0.35;
    // Points-per-pixel threshold for simple version of map nodes (label only).
    public static SCOPE_LOD: number = 1.5;
    // Points-per-pixel threshold for not drawing memlets/interstate edges.
    public static EDGE_LOD: number = 8;
    // Points-per-pixel threshold for not drawing node shapes and labels.
    public static NODE_LOD: number = 60;
    // Pixel threshold for not drawing state contents.
    public static STATE_LOD: number = 50;

    public static DEFAULT_CANVAS_FONTSIZE: number = 10;
    public static DEFAULT_MAX_FONTSIZE: number = 50;
    public static DEFAULT_FAR_FONT_MULTIPLIER: number = 16;

    private renderer: SDFGRenderer | null = null;

    public constructor() {
        return;
    }

    public set_renderer(renderer: SDFGRenderer | null): void {
        this.renderer = renderer;
    }

    public get_renderer(): SDFGRenderer | null {
        return this.renderer;
    }

    public init_menu(): void {
        const right = document.getElementById('sidebar');
        const bar = document.getElementById('dragbar');

        const drag = (e: MouseEvent) => {
            if ((document as any).selection)
                (document as any).selection.empty();
            else
                window.getSelection()?.removeAllRanges();

            if (right)
                right.style.width = Math.max(
                    ((e.view ? e.view.innerWidth - e.pageX : 0)), 20
                ) + 'px';
        };

        if (bar) {
            bar.addEventListener('mousedown', () => {
                document.addEventListener('mousemove', drag);
                document.addEventListener('mouseup', () => {
                    document.removeEventListener('mousemove', drag);
                });
            });
        }
    }

    public sidebar_set_title(title: string): void {
        // Modify sidebar header
        const sidebar_header = document.getElementById('sidebar-header');
        if (sidebar_header)
            sidebar_header.innerText = title;
    }

    public sidebar_show(): void {
        // Open sidebar if closed
        const sidebar = document.getElementById('sidebar');
        if (sidebar)
            sidebar.style.display = 'flex';
    }

    public sidebar_get_contents(): HTMLElement | null {
        return document.getElementById('sidebar-contents');
    }

    public close_menu(): void {
        const sidebar_contents = this.sidebar_get_contents();
        if (sidebar_contents)
            sidebar_contents.innerHTML = '';
        const sidebar = document.getElementById('sidebar');
        if (sidebar)
            sidebar.style.display = 'none';
    }

    public outline(renderer: SDFGRenderer, sdfg: DagreSDFG): void {
        this.sidebar_set_title('SDFG Outline');

        const sidebar = this.sidebar_get_contents();
        if (!sidebar)
            return;

        sidebar.innerHTML = '';

        // Entire SDFG
        const d = document.createElement('div');
        d.className = 'context_menu_option';
        d.innerHTML = htmlSanitize`
            <i class="material-icons" style="font-size: inherit">
                filter_center_focus
            </i> SDFG ${renderer.get_sdfg().attributes.name}
        `;
        d.onclick = () => renderer.zoom_to_view();
        sidebar.appendChild(d);

        const stack: any[] = [sidebar];

        // Add elements to tree view in sidebar
        traverse_sdfg_scopes(sdfg, (node: SDFGNode, parent: DagreSDFG) => {
            // Skip exit nodes when scopes are known
            if (node.type().endsWith('Exit') &&
                node.data.node.scope_entry >= 0) {
                stack.push(null);
                return true;
            }

            // Create element
            const d = document.createElement('div');
            d.className = 'context_menu_option';
            let is_collapsed = node.attributes().is_collapsed;
            is_collapsed = (is_collapsed === undefined) ? false : is_collapsed;
            let node_type = node.type();

            // If a scope has children, remove the name "Entry" from the type
            if (node.type().endsWith('Entry') && node.parent_id && node.id) {
                const state = node.sdfg.nodes[node.parent_id];
                if (state.scope_dict[node.id] !== undefined) {
                    node_type = node_type.slice(0, -5);
                }
            }

            d.innerHTML = htmlSanitize`
                ${node_type} ${node.label()}${is_collapsed ? ' (collapsed)' : ''}
            `;
            d.onclick = (e) => {
                // Show node or entire scope
                const nodes_to_display = [node];
                if (node.type().endsWith('Entry') && node.parent_id &&
                    node.id) {
                    const state = node.sdfg.nodes[node.parent_id];
                    if (state.scope_dict[node.id] !== undefined) {
                        for (const subnode_id of state.scope_dict[node.id])
                            nodes_to_display.push(parent.node(subnode_id));
                    }
                }

                renderer.zoom_to_view(nodes_to_display);

                // Ensure that the innermost div is the one handling the event
                if (!e) {
                    if (window.event) {
                        window.event.cancelBubble = true;
                        window.event.stopPropagation();
                    }
                } else {
                    e.cancelBubble = true;
                    if (e.stopPropagation)
                        e.stopPropagation();
                }
            };
            stack.push(d);

            // If is collapsed, don't traverse further
            if (is_collapsed)
                return false;

        }, (_node: SDFGNode, _parent: DagreSDFG) => {
            // After scope ends, pop ourselves as the current element 
            // and add to parent
            const elem = stack.pop();
            if (elem)
                stack[stack.length - 1].appendChild(elem);
        });

        this.sidebar_show();
    }

    public fill_info(elem: SDFGElement): void {
        const contents = this.sidebar_get_contents();
        if (!contents)
            return;

        let html = '';
        if (elem instanceof Edge && elem.data.type === 'Memlet' &&
            elem.parent_id && elem.id) {
            const sdfg_edge = elem.sdfg.nodes[elem.parent_id].edges[elem.id];
            html += '<h4>Connectors: ' + sdfg_edge.src_connector + ' &rarr; ' +
                sdfg_edge.dst_connector + '</h4>';
        }
        html += '<hr />';

        for (const attr of Object.entries(elem.attributes())) {
            if (attr[0].startsWith('_meta_'))
                continue;

            switch (attr[0]) {
                case 'layout':
                case 'sdfg':
                case '_arrays':
                case 'position':
                    continue;
                default:
                    html += '<b>' + attr[0] + '</b>:&nbsp;&nbsp;';
                    html += sdfg_property_to_string(
                        attr[1], this.renderer?.view_settings()
                    ) + '<br />';
                    break;
            }
        }

        // If access node, add array information too
        if (elem instanceof AccessNode) {
            const sdfg_array = elem.sdfg.attributes._arrays[elem.attributes().data];
            html += '<br /><h4>' + sdfg_array.type + ' properties:</h4>';
            for (const attr of Object.entries(sdfg_array.attributes)) {
                if (attr[0] === 'layout' || attr[0] === 'sdfg' ||
                    attr[0].startsWith('_meta_'))
                    continue;
                html += '<b>' + attr[0] + '</b>:&nbsp;&nbsp;';
                html += sdfg_property_to_string(
                    attr[1], this.renderer?.view_settings()
                ) + '<br />';
            }
        }

        // If nested SDFG, add SDFG information too
        if (elem instanceof NestedSDFG) {
            const sdfg_sdfg = elem.attributes().sdfg;
            html += '<br /><h4>SDFG properties:</h4>';
            for (const attr of Object.entries(sdfg_sdfg.attributes)) {
                if (attr[0].startsWith('_meta_'))
                    continue;

                switch (attr[0]) {
                    case 'layout':
                    case 'sdfg':
                        continue;
                    default:
                        html += '<b>' + attr[0] + '</b>:&nbsp;&nbsp;';
                        html += sdfg_property_to_string(
                            attr[1], this.renderer?.view_settings()
                        ) + '<br />';
                        break;
                }
            }
        }

        contents.innerHTML = html;
    }

    public start_find_in_graph(): void {
        start_find_in_graph(this);
    }

}

function init_sdfv(
    sdfg: any,
    user_transform: DOMMatrix | null = null,
    debug_draw: boolean = false,
    existing_sdfv: SDFV | null = null
): SDFV {
    let sdfv: SDFV;
    if (existing_sdfv)
        sdfv = existing_sdfv;
    else
        sdfv = new SDFV();

    $('#sdfg-file-input').on('change', (e: any) => {
        if (e.target.files.length < 1)
            return;
        file = e.target.files[0];
        reload_file(sdfv);
    });
    $('#menuclose').on('click', () => sdfv.close_menu());
    $('#reload').on('click', () => {
        reload_file(sdfv);
    });
    $('#instrumentation-report-file-input').on('change', (e: any) => {
        if (e.target.files.length < 1)
            return;
        instrumentation_file = e.target.files[0];
        load_instrumentation_report(sdfv);
    });
    $('#outline').on('click', () => {
        const renderer = sdfv.get_renderer();
        if (renderer)
            setTimeout(() => {
                const graph = renderer.get_graph();
                if (graph)
                    sdfv.outline(renderer, graph);
            }, 1);
    });
    $('#search-btn').on('click', () => {
        const renderer = sdfv.get_renderer();
        if (renderer)
            setTimeout(() => {
                const graph = renderer.get_graph();
                const query = $('#search').val();
                if (graph && query)
                    find_in_graph(
                        sdfv, renderer, graph, query.toString(),
                        $('#search-case').is(':checked')
                    );
            }, 1);
    });
    $('#advsearch-btn').on('click', () => {
        const renderer = sdfv.get_renderer();
        if (renderer)
            setTimeout(() => {
                const graph = renderer.get_graph();
                const code = $('#advsearch').val();
                if (graph && code) {
                    const predicate = eval(code.toString());
                    find_in_graph_predicate(
                        sdfv, renderer, graph, predicate
                    );
                }
            }, 1);
    });
    $('#search').on('keydown', (e: any) => {
        if (e.key == 'Enter' || e.which == 13) {
            sdfv.start_find_in_graph();
            e.preventDefault();
        }
    });

    let mode_buttons = null;
    const pan_btn = document.getElementById('pan-btn');
    const move_btn = document.getElementById('move-btn');
    const select_btn = document.getElementById('select-btn');
    const add_btns = [];
    add_btns.push(document.getElementById('elem_map'));
    add_btns.push(document.getElementById('elem_consume'));
    add_btns.push(document.getElementById('elem_tasklet'));
    add_btns.push(document.getElementById('elem_nested_sdfg'));
    add_btns.push(document.getElementById('elem_access_node'));
    add_btns.push(document.getElementById('elem_stream'));
    add_btns.push(document.getElementById('elem_state'));
    if (pan_btn)
        mode_buttons = {
            pan: pan_btn,
            move: move_btn,
            select: select_btn,
            add_btns: add_btns,
        };

    if (sdfg !== null) {
        const container = document.getElementById('contents');
        if (container)
            sdfv.set_renderer(new SDFGRenderer(
                sdfv, sdfg, container, mouse_event, user_transform, debug_draw,
                null, mode_buttons
            ));
    }

    return sdfv;
}

function start_find_in_graph(sdfv: SDFV): void {
    const renderer = sdfv.get_renderer();
    if (renderer)
        setTimeout(() => {
            const graph = renderer.get_graph();
            const query = $('#search').val();
            if (graph && query)
                find_in_graph(
                    sdfv, renderer, graph, query.toString(),
                    $('#search-case').is(':checked')
                );
        }, 1);
}

function reload_file(sdfv: SDFV): void {
    if (!file)
        return;
    fr = new FileReader();
    fr.onload = () => {
        file_read_complete(sdfv);
    };
    fr.readAsText(file);
}

function file_read_complete(sdfv: SDFV): void {
    const result_string = fr.result;
    const container = document.getElementById('contents');
    if (result_string && container) {
        const sdfg = parse_sdfg(result_string.toString());
        sdfv.get_renderer()?.destroy();
        sdfv.set_renderer(new SDFGRenderer(sdfv, sdfg, container, mouse_event));
        sdfv.close_menu();
    }
}

function load_instrumentation_report(sdfv: SDFV): void {
    if (!instrumentation_file)
        return;
    fr = new FileReader();
    fr.onload = () => {
        load_instrumentation_report_callback(sdfv);
    };
    fr.readAsText(instrumentation_file);
}

function load_instrumentation_report_callback(sdfv: SDFV): void {
    let result_string = '';
    if (fr.result) {
        if (fr.result instanceof ArrayBuffer) {
            const decoder = new TextDecoder('utf-8');
            result_string = decoder.decode(new Uint8Array(fr.result));
        } else {
            result_string = fr.result;
        }
    }
    instrumentation_report_read_complete(sdfv, JSON.parse(result_string));
}

/**
 * Get the min/max values of an array.
 * This is more stable than Math.min/max for large arrays, since Math.min/max
 * is recursive and causes a too high stack-length with long arrays.
 */
function get_minmax(arr: number[]): [number, number] {
    let max = -Number.MAX_VALUE;
    let min = Number.MAX_VALUE;
    arr.forEach(val => {
        if (val > max)
            max = val;
        if (val < min)
            min = val;
    });
    return [min, max];
}

export function instrumentation_report_read_complete(
    sdfv: SDFV, report: any, renderer: SDFGRenderer | null = null
): void {
    const runtime_map: { [uuids: string]: number[] } = {};
    const summarized_map: { [uuids: string]: { [key: string]: number } } = {};

    if (!renderer)
        renderer = sdfv.get_renderer();

    if (report.traceEvents && renderer) {
        for (const event of report.traceEvents) {
            if (event.ph === 'X') {
                let uuid = event.args.sdfg_id + '/';
                if (event.args.state_id !== undefined) {
                    uuid += event.args.state_id + '/';
                    if (event.args.id !== undefined)
                        uuid += event.args.id + '/-1';
                    else
                        uuid += '-1/-1';
                } else {
                    uuid += '-1/-1/-1';
                }

                if (runtime_map[uuid] !== undefined)
                    runtime_map[uuid].push(event.dur);
                else
                    runtime_map[uuid] = [event.dur];
            }
        }

        for (const key in runtime_map) {
            const values = runtime_map[key];
            const minmax = get_minmax(values);
            const min = minmax[0];
            const max = minmax[1];
            const runtime_summary = {
                'min': min,
                'max': max,
                'mean': mean(values),
                'med': median(values),
                'count': values.length,
            };
            summarized_map[key] = runtime_summary;
        }

        const overlay_manager = renderer.get_overlay_manager();
        if (overlay_manager) {
            if (!overlay_manager.is_overlay_active(
                RuntimeMicroSecondsOverlay
            )) {
                overlay_manager.register_overlay(
                    RuntimeMicroSecondsOverlay
                );
            }
            const ol = overlay_manager.get_overlay(
                RuntimeMicroSecondsOverlay
            );
            if (ol && ol instanceof RuntimeMicroSecondsOverlay) {
                ol.set_runtime_map(summarized_map);
                ol.refresh();
            }
        }
    }
}

// https://stackoverflow.com/a/901144/6489142
function getParameterByName(name: string): string | null {
    const url = window.location.href;
    name = name.replace(/[\[\]]/g, '\\$&');
    const regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
}

function load_sdfg_from_url(sdfv: SDFV, url: string): void {
    const request = new XMLHttpRequest();
    request.responseType = 'text'; // Will be parsed as JSON by parse_sdfg
    request.onload = () => {
        if (request.status == 200) {
            const sdfg = parse_sdfg(request.response);
            sdfv.get_renderer()?.destroy();
            init_sdfv(sdfg);
        } else {
            alert('Failed to load SDFG from URL');
            init_sdfv(null);
        }
    };
    request.onerror = () => {
        alert('Failed to load SDFG from URL: ' + request.status);
        init_sdfv(null);
    };
    request.open(
        'GET', url + ((/\?/).test(url) ? '&' : '?') + (new Date()).getTime(),
        true
    );
    request.send();
}

function find_recursive(
    graph: DagreSDFG, predicate: CallableFunction, results: any[]
): void {
    for (const nodeid of graph.nodes()) {
        const node = graph.node(nodeid);
        if (predicate(graph, node))
            results.push(node);
        // Enter states or nested SDFGs recursively
        if (node.data.graph) {
            find_recursive(node.data.graph, predicate, results);

        }
    }
    for (const edgeid of graph.edges()) {
        const edge = graph.edge(edgeid);
        if (predicate(graph, edge))
            results.push(edge);
    }
}

export function find_in_graph_predicate(
    sdfv: SDFV, renderer: SDFGRenderer, sdfg: DagreSDFG, predicate: CallableFunction
): void {
    sdfv.sidebar_set_title('Search Results');

    const results: any[] = [];
    find_recursive(sdfg, predicate, results);

    // Zoom to bounding box of all results first
    if (results.length > 0)
        renderer.zoom_to_view(results);

    // Show clickable results in sidebar
    const sidebar = sdfv.sidebar_get_contents();
    if (sidebar) {
        sidebar.innerHTML = '';
        for (const result of results) {
            const d = document.createElement('div');
            d.className = 'context_menu_option';
            d.innerHTML = htmlSanitize`${result.type()} ${result.label()}`;
            d.onclick = () => { renderer.zoom_to_view([result]); };
            d.onmouseenter = () => {
                if (!result.highlighted) {
                    result.highlighted = true;
                    renderer.draw_async();
                }
            };
            d.onmouseleave = () => {
                if (result.highlighted) {
                    result.highlighted = false;
                    renderer.draw_async();
                }
            };
            sidebar.appendChild(d);
        }
    }

    sdfv.sidebar_show();
}

export function find_in_graph(
    sdfv: SDFV, renderer: SDFGRenderer, sdfg: DagreSDFG, query: string,
    case_sensitive: boolean = false
): void {
    if (!case_sensitive)
        query = query.toLowerCase();
    find_in_graph_predicate(sdfv, renderer, sdfg, (graph: DagreSDFG, element: SDFGElement) => {
        let label = element.label();
        if (!case_sensitive)
            label = label.toLowerCase();
        return label.indexOf(query) !== -1;
    });
    sdfv.sidebar_set_title('Search Results for "' + query + '"');
}

function recursive_find_graph(
    graph: DagreSDFG, sdfg_id: number
): DagreSDFG | undefined {
    let found = undefined;
    graph.nodes().forEach(n_id => {
        const n = graph.node(n_id);
        if (n && n.sdfg.sdfg_list_id === sdfg_id) {
            found = graph;
            return found;
        } else if (n && n.data.graph) {
            found = recursive_find_graph(n.data.graph, sdfg_id);
            if (found)
                return found;
        }
    });
    return found;
}

function find_state(graph: DagreSDFG, state_id: number): State | undefined {
    let state = undefined;
    graph.nodes().forEach(s_id => {
        if (Number(s_id) === state_id) {
            state = graph.node(s_id);
            return state;
        }
    });
    return state;
}

function find_node(state: State, node_id: number): SDFGNode | undefined {
    let node = undefined;
    state.data.graph.nodes().forEach((n_id: any) => {
        if (Number(n_id) === node_id) {
            node = state.data.graph.node(n_id);
            return node;
        }
    });
    return node;
}

function find_edge(state: State, edge_id: number): Edge | undefined {
    let edge = undefined;
    state.data.graph.edges().forEach((e_id: any) => {
        if (Number(e_id.name) === edge_id) {
            edge = state.data.graph.edge(e_id);
            return edge;
        }
    });
    return edge;
}

function find_graph_element(
    graph: DagreSDFG, type: string, sdfg_id: number, state_id: number = -1,
    el_id: number = -1
): SDFGElement | undefined {
    const requested_graph = recursive_find_graph(graph, sdfg_id);
    let state = undefined;
    let isedge = undefined;
    if (requested_graph) {
        switch (type) {
            case 'edge':
                state = find_state(requested_graph, state_id);
                if (state)
                    return find_edge(state, el_id);
                break;
            case 'state':
                return find_state(requested_graph, state_id);
            case 'node':
                state = find_state(requested_graph, state_id);
                if (state)
                    return find_node(state, el_id);
                break;
            case 'isedge':
                Object.values((requested_graph as any)._edgeLabels).forEach(
                    (ise: any) => {
                        if (ise.id === el_id) {
                            isedge = ise;
                            return isedge;
                        }
                    }
                );
                return isedge;
            default:
                return undefined;
        }
    }
    return undefined;
}

export function mouse_event(
    evtype: string,
    _event: Event,
    _mousepos: Point2D,
    _elements: any[],
    renderer: SDFGRenderer,
    selected_elements: SDFGElement[],
    sdfv: SDFV
): boolean {
    if (evtype === 'click' || evtype === 'dblclick') {
        const menu = renderer.get_menu();
        if (menu)
            menu.destroy();
        let element;
        if (selected_elements.length === 0)
            element = new SDFG(renderer.get_sdfg());
        else if (selected_elements.length === 1)
            element = selected_elements[0];
        else
            element = null;

        if (element !== null) {
            sdfv.sidebar_set_title(
                element.type() + ' ' + element.label()
            );
            sdfv.fill_info(element);
        } else {
            sdfv.close_menu();
            sdfv.sidebar_set_title('Multiple elements selected');
        }
        sdfv.sidebar_show();
    }
    return false;
}

$(() => {
    let sdfv = new SDFV();

    if (document.currentScript?.hasAttribute('data-sdfg-json')) {
        const sdfg_string =
            document.currentScript?.getAttribute('data-sdfg-json');
        if (sdfg_string)
            sdfv = init_sdfv(parse_sdfg(sdfg_string), null, false, sdfv);
    } else {
        const url = getParameterByName('url');
        if (url)
            load_sdfg_from_url(sdfv, url);
        else
            sdfv = init_sdfv(null, null, false, sdfv);
    }

    sdfv.init_menu();
});

// Define global exports outside of webpack
declare global {
    interface Window {
        // Extensible classes for rendering and overlays
        OverlayManager: typeof OverlayManager;
        GenericSdfgOverlay: typeof GenericSdfgOverlay;
        SDFGElement: typeof SDFGElement;

        // API classes
        SDFV: typeof SDFV;
        SDFGRenderer: typeof SDFGRenderer;

        // Exported functions
        parse_sdfg: (sdfg_json: string) => JsonSDFG;
        stringify_sdfg: (sdfg: JsonSDFG) => string;
        init_sdfv: (sdfg: JsonSDFG, user_transform?: DOMMatrix | null, debug_draw?: boolean, existing_sdfv?: SDFV | null) => SDFV;
    }
}

window.OverlayManager = OverlayManager;
window.GenericSdfgOverlay = GenericSdfgOverlay;
window.SDFGElement = SDFGElement;
window.SDFV = SDFV;
window.SDFGRenderer = SDFGRenderer;
window.parse_sdfg = parse_sdfg;
window.stringify_sdfg = stringify_sdfg;
window.init_sdfv = init_sdfv;
