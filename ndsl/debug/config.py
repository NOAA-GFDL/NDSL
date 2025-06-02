"""
This module provides configuration for the global debugger `ndsl_debugger`

When loading, the configuration will be searched in the global environment variable
`NDSL_DEBUG_CONFIG`

Configuration is a yaml file of the shape
```yaml
stencils_or_class:
  - copy_corners_x_nord
  - copy_corners_y_nord
  - DGridShallowWaterLagrangianDynamics.__call__
track_parameter_by_name:
  - fy
```

Global variable:
    ndsl_debugger: Debugger accessible throughout the middleware, default to `None`
        if there is no configuration
"""

import os

import yaml

from ndsl.comm.mpi import MPIComm
from ndsl.debug.debugger import Debugger
from ndsl.logging import ndsl_log


ndsl_debugger = None


def _set_debugger():
    config = os.getenv("NDSL_DEBUG_CONFIG", "")
    if not os.path.exists(config):
        if config != "":
            ndsl_log.warning(
                f"NDSL_DEBUG_CONFIG set but path {config} does not exists."
            )
        else:
            return
    with open(config) as file:
        config_dict = yaml.load(file.read(), Loader=yaml.SafeLoader)
    global ndsl_debugger
    ndsl_debugger = Debugger(rank=MPIComm().Get_rank(), **config_dict)
    ndsl_log.info("[NDSL Debugger] On")
    ndsl_log.debug(f"[NDSL Debugger] Config:\n{config_dict}")


_set_debugger()
