import click

from ndsl.config import Backend
from ndsl.dsl.dace.utils import (
    kernel_theoretical_timing_from_path,
    memory_static_analysis_from_path,
)


# Count the memory from a given SDFG
ACTION_SDFG_MEMORY_STATIC_ANALYSIS = "sdfg_memory_static_analysis"
ACTION_SDFG_KERNEL_THEORETICAL_TIMING = "sdfg_kernel_theoretical_timing"


@click.command()
@click.argument(
    "action",
    required=True,
    type=click.Choice(
        [ACTION_SDFG_MEMORY_STATIC_ANALYSIS, ACTION_SDFG_KERNEL_THEORETICAL_TIMING]
    ),
)
@click.option(
    "--sdfg_path",
    required=True,
    type=click.STRING,
)
@click.option("--report_detail", is_flag=True, type=click.BOOL, default=False)
@click.option(
    "--hardware_bw_in_gb_s",
    required=False,
    type=click.FLOAT,
    default=0.0,
)
@click.option(
    "--output_format",
    required=False,
    type=click.STRING,
    default=None,
)
@click.option(
    "--backend",
    required=False,
    type=click.STRING,
    default="st:dace:gpu:KJI",
)
def command_line(
    action: str,
    sdfg_path: str,
    report_detail: bool,
    hardware_bw_in_gb_s: float | None,
    output_format: str | None,
    backend: str,
) -> None:
    """
    Run tooling.
    """
    if action == ACTION_SDFG_MEMORY_STATIC_ANALYSIS:
        print(memory_static_analysis_from_path(sdfg_path, detail_report=report_detail))
    elif action == ACTION_SDFG_KERNEL_THEORETICAL_TIMING:
        print(
            kernel_theoretical_timing_from_path(
                sdfg_path,
                backend=Backend(backend),
                hardware_bw_in_GB_s=(
                    None if hardware_bw_in_gb_s == 0 else hardware_bw_in_gb_s
                ),
                output_format=output_format,
            )
        )


if __name__ == "__main__":
    command_line()
