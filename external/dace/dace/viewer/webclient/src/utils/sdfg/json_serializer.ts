// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { Edge, JsonSDFG } from '../../index';

// Recursively parse SDFG, including nested SDFG nodes
export function parse_sdfg(sdfg_json: string): JsonSDFG {
    return JSON.parse(sdfg_json, reviver);
}

export function stringify_sdfg(sdfg: JsonSDFG): string {
    return JSON.stringify(sdfg, (name, val) => replacer(name, val, sdfg));
}

function reviver(name: string, val: unknown) {
    if (name == 'sdfg' && val && typeof val === 'string' && val[0] === '{') {
        return JSON.parse(val, reviver);
    }
    return val;
}

function isDict(v: unknown): v is Record<string, unknown> {
    return typeof v === 'object' && v !== null && !(v instanceof Array) &&
        !(v instanceof Date);
}

function replacer(name: string, val: unknown, orig_sdfg: unknown): unknown {
    if (name === 'edge' && val instanceof Edge) {  // Skip circular dependencies
        return undefined;
    }
    return val;
}
