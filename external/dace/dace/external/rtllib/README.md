# rtllib

This repository contains multiple utilities for integrating RTL kernels into
higher level workflows, such as Xilinx Vitis.

# CMake files

# RTL cores

# Templates
The [templates](templates/) folder contains Python scripts for generating RTL
source files and TCL scripts.

## [control.py](templates/control.py)

This script generates an RTL controller, which handles the `control` interface required by RTL kernels in Xilinx Vitis. In addition to all of the required registers, the controller core will also provide all of the specified kernel parameters as ports from the RTL module. The resulting memory map will start by the required registers, followed by the scalar parameters, finally ending with the memory pointer parameters.

In order to generate a controller, a configuration file specifying the kernel name and its parameters is needed. The configuration file is a dictionary, with the following entries:

| Name | Type | Description |
| ---- | ---- | ----------- |
| `name` | `string` | The name of the RTL kernel. The module will be named `{name}_control`. |
| `params` | `dict` | Dictionary with two entries: `scalars` and `memory`. |
| `scalars` | `dict` | A dictionary, which specifies `'param_name': bit_width` of the scalar parameters. |
| `memory` | `dict` | A dictionary, which specifies `'param_name': bit_width` of the memory pointer parameters. Note: the bitwidth for Xilinx Vitis cores are always 64 bit. |

An example configuration file for the kernel `vadd(int *a, int *b, int *c, int size)`
```json
{
    'name': 'vadd',
    'params': {
        'scalars': {
            'size': 32
        },
        'memory': {
            'a': 64,
            'b': 64,
            'c': 64
        }
    }
}
```

- CMake files
  CMake configuration files for generating, compiling and linking higher level
  solutions.
- RTL cores
  Set of generic RTL cores, which are usefull in solutions.
- Templates
  Set of scripts, which can generate both RTL modules and TCL scripts for
  packaging RTL kernels.
