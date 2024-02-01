// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import { htmlSanitize } from './sanitization';
import $ from 'jquery';

export class ContextMenu {

    private click_close_handlers: [string, (_x: any) => void][] = [];
    private options: any[] = [];
    private cmenu_elem: HTMLElement | null = null;

    public constructor(private html_content: string | null = null) {
        setTimeout(() => {
            this.click_close_handlers = [
                ['click', (_x: any) => {
                    this.destroy();
                }],
                ['contextmenu', (_x: any) => {
                    this.destroy();
                }]

            ];

            for (const x of this.click_close_handlers)
                window.addEventListener(...x);
        }, 30);
    }

    public width(): number | undefined {
        return this.cmenu_elem?.offsetWidth;
    }

    public visible(): boolean {
        return this.cmenu_elem != null;
    }

    public addOption(
        name: string,
        onselect: CallableFunction,
        onhover: CallableFunction | null = null
    ): void {
        this.options.push({
            name: name,
            func: onselect,
            onhover: onhover
        });
    }

    public addCheckableOption(
        name: string, checked: boolean,
        onselect: CallableFunction,
        onhover: CallableFunction | null = null
    ): void {
        this.options.push({
            name: name,
            checkbox: true,
            checked: checked,
            func: onselect,
            onhover: onhover
        });
    }

    public destroy(): void {
        if (!this.cmenu_elem)
            return;
        // Clear everything

        // Remove the context menu

        document.body.removeChild(this.cmenu_elem);

        for (const x of this.click_close_handlers)
            window.removeEventListener(...x);

        this.cmenu_elem = null;
    }

    public show(x: number, y: number): void {
        /*
            Shows the context menu originating at point (x,y)
        */

        const cmenu_div = document.createElement('div');
        cmenu_div.id = 'contextmenu';
        $(cmenu_div).css('left', x + 'px');
        $(cmenu_div).css('top', y + 'px');
        cmenu_div.classList.add('context_menu');


        if (this.html_content == null) {
            // Set default context menu

            for (const x of this.options) {

                const elem = document.createElement('div');
                elem.addEventListener('click', x.func);
                elem.classList.add('context_menu_option');

                if (x.checkbox) {
                    const markelem = document.createElement('span');
                    markelem.classList.add(
                        x.checked ? 'checkmark_checked' : 'checkmark'
                    );
                    elem.appendChild(markelem);
                    elem.innerHTML += htmlSanitize`${x.name}`;
                    elem.addEventListener('click', elem => {
                        x.checked = !x.checked;
                        x.func(elem, x.checked);
                    });
                } else {
                    elem.innerText = x.name;
                    elem.addEventListener('click', x.func);
                }
                cmenu_div.appendChild(elem);
            }
        } else {
            cmenu_div.innerHTML = this.html_content;
        }

        this.cmenu_elem = cmenu_div;
        document.body.appendChild(cmenu_div);
    }

    public get_cmenu_elem(): HTMLElement | null {
        return this.cmenu_elem;
    }

}
