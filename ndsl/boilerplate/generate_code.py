"""Generate boilerplate code with configurable executor at the bottom."""

from __future__ import annotations

import argparse
import textwrap

TEMPLATE = textwrap.dedent("""\
    from ndsl import NDSLRuntime
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.config import Backend
    from ndsl.constants import I_DIM, J_DIM, K_DIM
    from ndsl.dsl.gt4py import PARALLEL, computation, interval
    from ndsl.dsl.typing import FloatField


    def copy_stencil(input_field: FloatField, output_field: FloatField):
        \"""All stencil code should live in the global space.\"""

        with computation(PARALLEL), interval(...):
            output_field = input_field


    class {class_name}(NDSLRuntime):
        \"""All model code should be wrapped in NDSLRuntime object.\"""

        def __init__(self, stencil_factory):
            super().__init__(stencil_factory)

            self._copy = stencil_factory.from_dims_halo(
                func=copy_stencil,
                compute_dims=[I_DIM, J_DIM, K_DIM],
            )

        def __call__(self, input_field: FloatField, output_field: FloatField):
            self._copy(input_field, output_field)
    {main_section}
    """)

MAIN_SECTION = textwrap.dedent("""\

    def main() -> None:

        # We setup a single tile grid here
        stencil_factory, quantity_factory = get_factories_single_tile(
            nx=8,
            ny=8,
            nz=4,
            nhalo=1,
            backend=Backend({backend!r}),
        )

        {instance_name} = {class_name}(stencil_factory)

        qty_in = quantity_factory.full(dims=[I_DIM, J_DIM, K_DIM], units="n/a", value=42.42)
        qty_out = quantity_factory.zeros(dims=[I_DIM, J_DIM, K_DIM], units="n/a")

        {instance_name}(qty_in, qty_out)

        assert (qty_out.field[:] == qty_in.field[:]).all(), "{class_name} does a copy"


    if __name__ == "__main__":
        main()
    """)


def instance_name_from_class(class_name: str) -> str:
    if not class_name:
        raise ValueError("name must not be empty")
    return class_name[0].lower() + class_name[1:]


def build_script(class_name: str, backend: str, exec_main: bool) -> str:
    instance_name = instance_name_from_class(class_name)
    main_section = (
        MAIN_SECTION.format(
            backend=backend,
            class_name=class_name,
            instance_name=instance_name,
        )
        if exec_main
        else ""
    )
    return TEMPLATE.format(
        class_name=class_name,
        main_section=main_section,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the canonical NDSL model code."
    )
    parser.add_argument(
        "--name",
        default="MyCode",
        help="Class name to use for the generated Class and its instance.",
    )
    parser.add_argument(
        "--backend",
        default="st:python:cpu:IJK",
        help="Backend used when using --exec.",
    )
    parser.add_argument(
        "--exec",
        action="store_true",
        dest="exec_main",
        help='Write an "executor" main section at the bottom of the file.',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the generated script to a file. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_script(args.name, args.backend, args.exec_main)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
