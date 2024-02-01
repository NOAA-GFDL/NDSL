// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import {
    MapExit,
    NestedSDFG,
    SDFGElement,
    SDFGNode,
    State
} from '../renderer/renderer_elements';
import { Point2D } from '..';

export function equals<T>(a: T, b: T): boolean {
    return JSON.stringify(a) === JSON.stringify(b);
}

export function deepCopy<T>(obj: T): T {
    if (typeof obj !== 'object' || obj === null) return obj;
    if (Array.isArray(obj)) {
        return obj.map(o => deepCopy(o)) as any;
    } else {
        return Object.fromEntries(deepCopy([...Object.entries(obj)])) as any;
    }
}

/**
 * Create a DOM element with an optional given ID and class list.
 *
 * If a parent is provided, the element is automatically added as a child.
 *
 * @param {*} type      Element tag (div, span, etc.)
 * @param {*} id        Optional element id
 * @param {*} classList Optional array of class names
 * @param {*} parent    Optional parent element
 *
 * @returns             The created DOM element
 */
export function createElement<K extends keyof HTMLElementTagNameMap>(
    type: K,
    id = '',
    classList: string[] = [],
    parent: Node | undefined = undefined
): HTMLElementTagNameMap[K] {
    const element = document.createElement(type);
    if (id !== '')
        element.id = id;
    if (classList !== [])
        classList.forEach(class_name => {
            if (!element.classList.contains(class_name))
                element.classList.add(class_name);
        });
    if (parent)
        parent.appendChild(element);
    return element;
}

/**
 * Similar to Object.assign, but skips properties that already exist in `obj`.
 */
export function assignIfNotExists<T, E>(
    obj: T, other: E
): T & Omit<E, keyof T> {
    const o = obj as any;
    for (const [key, val] of Object.entries(other)) {
        if (!(key in obj)) o[key] = val;
    }
    return o;
}

// This function was taken from the now deprecated dagrejs library, see:
// https://github.com/dagrejs/dagre/blob/c8bb4a1b891fc50071e6fac7bd84658d31eb9d8a/lib/util.js#L96
/*
 * Finds where a line starting at point ({x, y}) would intersect a rectangle
 * ({x, y, width, height}) if it were pointing at the rectangle's center.
 */
export function intersectRect(
    rect: { x: number, y: number, height: number, width: number }, point: Point2D
): Point2D {
    const x = rect.x;
    const y = rect.y;

    // Rectangle intersection algorithm from:
    // http://math.stackexchange.com/questions/108113/find-edge-between-two-boxes
    const dx = point.x - x;
    const dy = point.y - y;
    let w = rect.width / 2;
    let h = rect.height / 2;

    if (!dx && !dy)
        throw new Error(
            'Not possible to find intersection inside of the rectangle'
        );

    let sx, sy;
    if (Math.abs(dy) * w > Math.abs(dx) * h) {
        // Intersection is top or bottom of rect.
        if (dy < 0)
            h = -h;
        sx = h * dx / dy;
        sy = h;
    } else {
        // Intersection is left or right of rect.
        if (dx < 0)
            w = -w;
        sx = w;
        sy = w * dy / dx;
    }

    return {
        x: x + sx,
        y: y + sy
    };
}

export function get_element_uuid(element: SDFGElement): string {
    const undefined_val = -1;
    if (element instanceof State) {
        return (
            element.sdfg.sdfg_list_id + '/' +
            element.id + '/' +
            undefined_val + '/' +
            undefined_val
        );
    } else if (element instanceof NestedSDFG) {
        const sdfg_id = element.data.node.attributes.sdfg.sdfg_list_id;
        return (
            sdfg_id + '/' +
            undefined_val + '/' +
            undefined_val + '/' +
            undefined_val
        );
    } else if (element instanceof MapExit) {
        // For MapExit nodes, we want to get the uuid of the corresponding
        // entry node instead.
        return (
            element.sdfg.sdfg_list_id + '/' +
            element.parent_id + '/' +
            element.data.node.scope_entry + '/' +
            undefined_val
        );
    } else if (element instanceof SDFGNode) {
        return (
            element.sdfg.sdfg_list_id + '/' +
            element.parent_id + '/' +
            element.id + '/' +
            undefined_val
        );
    }
    return (
        undefined_val + '/' +
        undefined_val + '/' +
        undefined_val + '/' +
        undefined_val
    );
}
