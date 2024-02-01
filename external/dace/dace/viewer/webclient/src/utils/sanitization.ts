// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

export function escapeHTML(s: string): string {
    const escapeCharacters: Map<string, string> = new Map([
        ['&', '&amp;'],
        ['<', '&lt;'],
        ['>', '&gt;'],
        ['"', '&quot;'],
        ["'", '&#039;'],
    ]);

    return `${s}`.replace(/[&<>"']/g, m => escapeCharacters.get(m)!);
}

export function htmlSanitize(
    strings: TemplateStringsArray, ...values: any[]
): string {
    return strings.length === 1 ? strings[0]
        : strings.reduce(
            (s, n, i) => `${s}${escapeHTML(String(values[i - 1]))}${n}`
        );
}
