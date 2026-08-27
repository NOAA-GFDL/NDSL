# Getting started

NDSL is a modern domain-specific language (DSL) for portable, high-performance atmospheric modeling. It let's you build portable atmospheric modeling workflows across CPUs, GPUs, and emerging computing architectures from a single codebase.

NDSL is written in the [python programming language](https://www.python.org/) and user code will also be written in python. This guide will help you get up and running.

## Prerequisites

There are only two things we require you to have installed. First, we rely on [uv](https://docs.astral.sh/uv/) as package and project manager.

!!! note "Install uv"

    `uv` has excellent documentation. Please follow their [install guide](https://docs.astral.sh/uv/getting-started/installation/).

Second, we require you to have a compiler toolchain for C and C++ code. We support `gcc`, `clang` and the Intel compiler.

## Start a project

As mentioned, we are going to rely on [uv](https://docs.astral.sh/uv/) for project management:

```shell
$ uv init --python=3.12 example/

Initialized project `example` at `/tmp/example/`
```

Next, change into the newly created project directory and add `ndsl` as a dependency:

```shell
$ cd example/
$ uv add git+https://github.com/noaa-gfdl/ndsl.git --tag 2026.08.00 --extra openmpi

Using CPython 3.12.13 interpreter at: /usr/bin/python3.12
Creating virtual environment at: .venv
Resolved 92 packages in 283ms
[...]
```

Note how `uv add ...` not only added `ndsl`  to the dependencies, but also generated a virtual environment and install all dependencies in there. We are now ready to go!

!!! tip "Use your system's MPI"

    If you know you have MPI installed on your system, you can drop the `--extra openmpi` from the command above.

## Generate and run NDSL code

NDSL comes with a code generator to get you up and running in no time. In your project directory, run the following command

```shell
$ uv run ndsl-gencode --exec --output ndsl_example.py
```

to generate example NDSL code. Running then generated file is as easy as

```shell
$ uv run python ndsl_example.py

2026-08-21 15:00:59|INFO|rank 0|ndsl.logging:ndsl_example.py
2026-08-21 15:00:59|INFO|rank 0|ndsl.logging:Log level: info
2026-08-21 15:01:01|INFO|rank 0|ndsl.logging:Literal precision: 64
2026-08-21 15:01:01|INFO|rank 0|ndsl.logging:Constant selected: ConstantVersions.UFS
```

Congratulation, you just ran you first NDSL program. Have a look at `ndsl_example.py` and start experimenting with NDSL. 🚀

## Optional components

NDSL has the following extras:

- `demos`: extra dependencies to run [NDSL examples](https://github.com/NOAA-GFDL/NDSL/tree/develop/examples/NDSL)
- `openmpi`: install OpenMPI from python (do support GPU)
- `pyfms`: install [pyFMS](https://github.com/NOAA-GFDL/pyfms) to allow interaction with its diagnostics manager
- `serialbox`: install [serialbox](https://github.com/FlorianDeconinck/serialbox/) to support porting code from Fortran to NDSL

Workflows that port code from Fortran to NDSL might depend on [serialbox](https://github.com/FlorianDeconinck/serialbox/), e.g. the script `ndsl-serialbox_to_netcdf`. If you install the serialbox extra, you'll need the Boost library and development headers.

## Running on GPUs

To run on GPUs, we have some more expectations on your setup and you'll need some optional dependencies. In a nutshell

- You will need a graphics card (duh). NDSL supports NVIDIA and AMD cards.
- You will need a system MPI that is compiled with CUDA support.
- Install NDSL with either `--extra cuda12` or `--extra cuda13` extra. Use `--extra hmm-cuda12` if your card supports Heterogeneous
Memory Management (HMM).
- In your code (e.g. when you setup the factories), use a GPU-enabled backend, e.g. `backend = Backend.gpu()`.
