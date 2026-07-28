"""
This module provides configuration for the global debugger via `get_debugger`

When loading, the configuration will be searched in the global environment variable
`NDSL_DEBUG_CONFIG`

Configuration is a yaml file of the shape
```yaml
stencils_or_class:
    - stencil_name
    - ClassName.orchestrated_method
    - ClassName.__call__
track_parameter_by_name:
    - name_of_variable
timestep_name: TopClassName
save_from_timestep: 3
save_all: False
dir_name: ./my/local/path
save_compute_domain_only: False
```

Functions:
    get_debugger: Retrieve the global debugger throughout the middleware, default to `None`
        if there is no configuration. Parameter "force_reload" will reload the configuration.
"""

import os

import yaml

from ndsl import ndsl_log
from ndsl.comm.mpi import MPIComm
from ndsl.debug.debugger import Debugger


def _set_debugger_from_config() -> Debugger | None:
    config = os.getenv("NDSL_DEBUG_CONFIG", "")
    if not os.path.exists(config):
        if config != "":
            ndsl_log.warning(
                f"NDSL_DEBUG_CONFIG set but path {config} does not exists."
            )
        else:
            return None
    with open(config) as file:
        config_dict = yaml.load(file.read(), Loader=yaml.SafeLoader)
    debugger = Debugger(rank=MPIComm().Get_rank(), **config_dict)
    ndsl_log.info("[NDSL Debugger] On")
    ndsl_log.info(f"[NDSL Debugger] Config:\n{config_dict}")
    return debugger


_ndsl_debugger = _set_debugger_from_config()
"""Global NDSL debugger, set to None if NDSL_DEBUG_CONFIG is unset"""


def get_debugger(force_reload: bool = False) -> Debugger | None:
    if force_reload:
        global _ndsl_debugger
        _ndsl_debugger = _set_debugger_from_config()
    return _ndsl_debugger
