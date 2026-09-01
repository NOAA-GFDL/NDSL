import warnings

import dace.config
from gt4py.cartesian import config as gt_config

from ndsl import DaceConfig, DaCeOrchestration, ndsl_log
from ndsl.dsl.caches import get_cache_directory, get_cache_fullpath


def set_distributed_caches(config: DaceConfig, force_build: bool = False) -> None:
    """In Run mode, check required file then point current rank cache to source cache.

    Optional: force build irregardless of backend or orchestration mode.
    """

    warnings.warn(
        "Use DaceConfig._set_distributed_caches() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Execute specific initialization per orchestration state
    if not config.get_backend().is_orchestrated() and not force_build:
        return

    # Check that we have all the file we need to early out in case
    # of issues.
    orchestration_mode = config.get_orchestrate()
    if orchestration_mode == DaCeOrchestration.Run and not force_build:
        import os

        cache_directory = get_cache_fullpath(config.code_path)
        if not os.path.exists(cache_directory):
            raise RuntimeError(
                f"{orchestration_mode} error: Could not find caches for rank "
                f"{config.my_rank} at {cache_directory}"
            )

    # Set read/write caches to the target rank
    if config._do_compile:
        verb = "reading/writing"
    else:
        verb = "reading"

    gt_config.cache_settings["dir_name"] = get_cache_directory(config.code_path)

    # NOTE: In the (rare) case we orchestrate code _without_ any stencils, we need
    # to set the build folder. The other code is in FrozenStencil and deals with the
    # case of `dace` used in both orchestrated and not orchestrated.
    # A better build system would deal with this in BOTH cases.
    dace.config.Config.set(
        "default_build_folder",
        value="{gt_root}/{gt_cache}/dacecache".format(
            gt_root=gt_config.cache_settings["root_path"],
            gt_cache=gt_config.cache_settings["dir_name"],
        ),
    )

    ndsl_log.info(
        f"[{orchestration_mode}] Rank {config.my_rank} "
        f"{verb} cache {gt_config.cache_settings['dir_name']}"
    )
