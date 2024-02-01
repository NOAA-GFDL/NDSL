// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { simplify } from 'mathjs';
import { LogicalGroup } from '../../overlays/logical_group_overlay';

export function sdfg_range_elem_to_string(
    range: any,
    settings: any = null
): string {
    let preview = '';
    if (range.start == range.end && range.step == 1 && range.tile == 1)
        preview += sdfg_property_to_string(range.start, settings);
    else {
        if (settings && settings.inclusive_ranges) {
            preview += sdfg_property_to_string(range.start, settings) + '..' +
                sdfg_property_to_string(range.end, settings);
        } else {
            let endp1 = sdfg_property_to_string(range.end, settings) + ' + 1';
            try {
                endp1 = simplify(endp1).toString();
            } catch (e) { }
            preview += sdfg_property_to_string(range.start, settings) + ':' +
                endp1;
        }
        if (range.step != 1) {
            preview += ':' + sdfg_property_to_string(range.step, settings);
            if (range.tile != 1)
                preview += ':' + sdfg_property_to_string(range.tile, settings);
        } else if (range.tile != 1) {
            preview += '::' + sdfg_property_to_string(range.tile, settings);
        }
    }
    return preview;
}

export function sdfg_consume_elem_to_string(
    num_pes: number,
    settings: any = null
): string {
    let result = '0';
    if (settings && settings.inclusive_ranges)
        result += '..' + (num_pes - 1).toString();
    else
        result += ':' + num_pes.toString();
    return result;
}

// Includes various properties and returns their string representation
export function sdfg_property_to_string(
    prop: any,
    settings: any = null
): string {
    if (prop === null) return prop;
    if (typeof prop === 'boolean') {
        if (prop)
            return 'True';
        return 'False';
    } else if (prop.type === 'Indices' || prop.type === 'subsets.Indices') {
        const indices = prop.indices;
        let preview = '[';
        for (const index of indices) {
            preview += sdfg_property_to_string(index, settings) + ', ';
        }
        return preview.slice(0, -2) + ']';
    } else if (prop.type === 'Range' || prop.type === 'subsets.Range') {
        const ranges = prop.ranges;

        // Generate string from range
        let preview = '[';
        for (const range of ranges) {
            preview += sdfg_range_elem_to_string(range, settings) + ', ';
        }
        return preview.slice(0, -2) + ']';
    } else if (prop.type === 'LogicalGroup' && prop.color !== undefined &&
        prop.name !== undefined) {
        return '<span style="color: ' + prop.color + ';">' + prop.name + ' (' +
            prop.color + ' )</span>';
    } else if (prop.language !== undefined) {
        // Code
        if (prop.string_data !== '' && prop.string_data !== undefined &&
            prop.string_data !== null)
            return '<pre class="code"><code>' + prop.string_data.trim() +
                '</code></pre><div class="clearfix"></div>';
        return '';
    } else if (prop.approx !== undefined && prop.main !== undefined) {
        // SymExpr
        return prop.main;
    } else if (prop.constructor == Object) {
        // General dictionary
        return '<pre class="code"><code>' + JSON.stringify(prop, undefined, 4) +
            '</code></pre><div class="clearfix"></div>';
    } else if (prop.constructor == Array) {
        // General array
        let result = '[ ';
        let first = true;
        for (const subprop of prop) {
            if (!first)
                result += ', ';
            result += sdfg_property_to_string(subprop, settings);
            first = false;
        }
        return result + ' ]';
    } else {
        return prop;
    }
}

export function string_to_sdfg_typeclass(str: string): any {
    str.replace(/\s+/g, '');

    if (str === '' || str === 'null')
        return null;

    if (str.endsWith(')')) {
        if (str.startsWith('vector(')) {
            const argstring = str.substring(7, str.length - 1);
            if (argstring) {
                const splitidx = argstring.lastIndexOf(',');
                if (splitidx) {
                    const dtype = string_to_sdfg_typeclass(
                        argstring.substring(0, splitidx)
                    );
                    const count = argstring.substring(splitidx);
                    if (dtype && count)
                        return {
                            type: 'vector',
                            dtype: dtype,
                            elements: count,
                        };
                }
            }
        } else if (str.startsWith('pointer(')) {
            const argstring = str.substring(8, str.length - 1);
            if (argstring)
                return {
                    type: 'pointer',
                    dtype: string_to_sdfg_typeclass(argstring),
                };
        } else if (str.startsWith('opaque(')) {
            const argstring = str.substring(7, str.length - 1);
            if (argstring)
                return {
                    type: 'opaque',
                    name: argstring,
                };
        } else if (str.startsWith('callback(')) {
            const argstring = str.substring(9, str.length - 1);
            if (argstring) {
                const splitidx = argstring.lastIndexOf(',');
                if (splitidx) {
                    const cb_argstring = argstring.substring(0, splitidx);
                    if (cb_argstring.startsWith('[') &&
                        cb_argstring.endsWith(']')) {
                        const cb_args_raw = cb_argstring.substring(
                            1, cb_argstring.length - 1
                        ).split(',');
                        const ret_type = string_to_sdfg_typeclass(
                            argstring.substring(splitidx)
                        );

                        const cb_args: any[] = [];
                        if (cb_args_raw)
                            cb_args_raw.forEach(raw_arg => {
                                cb_args.push(string_to_sdfg_typeclass(raw_arg));
                            });

                        if (cb_args && ret_type)
                            return {
                                type: 'callback',
                                arguments: cb_args,
                                returntype: ret_type,
                            };
                    }
                }
            }
        }
    }
    return str;
}

export function sdfg_typeclass_to_string(typeclass: any): string {
    if (typeclass === undefined || typeclass === null)
        return 'null';

    if (typeclass.constructor === Object) {
        if (typeclass.type !== undefined) {
            switch (typeclass.type) {
                case 'vector':
                    if (typeclass.elements !== undefined &&
                        typeclass.dtype !== undefined)
                        return 'vector(' + sdfg_typeclass_to_string(
                            typeclass.dtype
                        ) + ', ' + typeclass.elements + ')';
                    break;
                case 'pointer':
                    if (typeclass.dtype !== undefined)
                        return 'pointer(' + sdfg_typeclass_to_string(
                            typeclass.dtype
                        ) + ')';
                    break;
                case 'opaque':
                    if (typeclass.name !== undefined)
                        return 'opaque(' + typeclass.name + ')';
                    break;
                case 'callback':
                    if (typeclass.arguments !== undefined) {
                        let str = 'callback([';
                        for (let i = 0; i < typeclass.arguments.length; i++) {
                            str += sdfg_typeclass_to_string(
                                typeclass.arguments[i]
                            );
                            if (i < typeclass.arguments.length - 1)
                                str += ', ';
                        }
                        str += '], ';
                        if (typeclass.returntype !== undefined)
                            str += sdfg_typeclass_to_string(
                                typeclass.returntype
                            );
                        else
                            str += 'None';
                        str += ')';
                        return str;
                    }
                    break;
            }
        }

        // This is an unknown typeclass, just show the entire JSON.
        return sdfg_property_to_string(typeclass);
    }

    // This typeclass already is a regular string.
    return typeclass;
}

