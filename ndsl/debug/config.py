import os
from ndsl.debug.debugger import Debugger
from ndsl.comm.mpi import MPIComm
import yaml


ndsl_debugger = None


def _set_debugger():
    config = os.getenv("NDSL_DEBUG_CONFIG", "")
    if not os.path.exists(config):
        return
    with open(config) as file:
        config_dict = yaml.load(file.read(), Loader=yaml.SafeLoader)
    global ndsl_debugger
    ndsl_debugger = Debugger(rank=MPIComm().Get_rank(), **config_dict)


_set_debugger()
