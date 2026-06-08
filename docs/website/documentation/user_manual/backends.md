# Backends

Throughout this user manual, we have been building our factories with some version of the following code:

``` py linenums="1"
domain = (5, 5, 3)
nhalo = 0
backend = "numpy"
stencil_factory, quantity_factory = get_factories_single_tile(
    domain[0],
    domain[1],
    domain[2],
    nhalo,
    backend,
)
```

but we have thus far glossed over what "backend" means. It is now time to address that term.

"Backend" refers to the underlying infrastructure that NDSL uses to construct and execute stencils.
Each backend approaches acceleration in a slightly different way, and by extention has different 
benefits and drawbacks. NDSL has a total of four backends:

- `dace:cpu`: stencils are compiled in C and has multiple optimization passes tailored to CPU execution.
longest compilation time, but best performance gain
- `dace:gpu`: stencils are compiled in C and has multiple optimization passes tailored to GPU execution.
longest compilation time, but best performance gain
- `numpy`: stencils are compiled in Python with minimal optimization, moderate compilation time, moderate
performance gain
- `debug`: code is executed as written in plain Python, no optimization, no performance gain 

A couple of notes from this list: `dace` provides the best performance gain, but may come with a significant
compilation time that makes debugging difficult. For this reason, we advise that you work in `numpy` when
testing smaller code (the performance difference between `numpy` and `dace` will be minimal at this scale)
and swap to `dace` for larger executions (e.g. running a model at scale). The `debug` backend should only be
used as a last resort. The removal of optimization features means that this is an ideal backend for
preliminary development and testing of new NDSL features, but execution with this backend is pitifully slow
compared to `numpy` and `dace` supported backends.