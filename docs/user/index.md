# Usage documentation

This part of the documentation is geared towards users of NDSL.

## Up and running

See our [quickstart guide](../quickstart.md) on how to get up and running.

## Configuration

NDSL tries to have sensible defaults. In cases you want tweak something, here are some pointers:

### Literal precision (float/int)

Unspecified integer and floating point literals (e.g. `42` and `3.1415`) default to 64-bit precision. This can be changed with the environment variable `NDSL_LITERAL_PRECISION`.

For mixed precision code, you can specify the "hard coded" precision with type hints and casts, e.g.

```python
with computation(PARALLEL), interval(...):
    # Either 32-bit or 64-bit depending on `NDSL_LITERAL_PRECISION`
    my_int = 42
    my_float = 3.1415

    # Always 32-bit
    my_int32: int32 = 42
    my_float32: float32 = 3.1415

    # Explicit 64-bit cast within otherwise unspecified calculation
    factor = 0.5 * float64(3.1415 + 2.71828)
```

### Full program optimizer

The behavior of the full program optimizer is controlled by `FV3_DACEMODE`. Valid values are:

`Python`

:   The default. Disables full program optimization and only accelerates stencil code.

`Build`

:   Build the program, then exit. This mode is only available for backends `dace:gpu` and `dace:cpu`.

`BuildAndRun`

:   Build the program, then run it immediately. This mode is only available for backends `dace:gpu` and `dace:cpu`.

`Run`

:   Load a pre-compiled program and run it. Fails if the pre-compiled program can not be found. This mode is only available for backends `dace:gpu` and `dace:cpu`.
