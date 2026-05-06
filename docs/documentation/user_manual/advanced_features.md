# Advanced Stencil Properties and Techniques

NDSL has a number of features which are designed to streamline development.

To help illustrate them, let's create a simple stencil where we convert temperature from Celsius
to Kelvin:


    ``` py linenums="1"
    from ndsl.dsl.gt4py import PARALLEL, computation, interval
    from ndsl import StencilFactory
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM
    from ndsl.dsl.typing import FloatField
    import random


    def celsius_to_kelvin(temperature_Celsius: FloatField, temperature_Kelvin: FloatField):
        with computation(PARALLEL), interval(...):
                # convert from Celsius to Kelvin
                temperature_Kelvin = temperature_Celsius + 273.15


    class Convert:
        def __init__(self, stencil_factory: StencilFactory):

            # construct the stencil
            self.constructed_copy_stencil = stencil_factory.from_dims_halo(
                func=celsius_to_kelvin,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
            )

        def __call__(self, in_quantity: FloatField, out_quantity: FloatField):

            # call the stencil
            self.constructed_copy_stencil(in_quantity, out_quantity)


    if __name__ == "__main__":

        # setup domain and generate factories
        domain = (5, 5, 3)
        nhalo = 0
        stencil_factory, quantity_factory = get_factories_single_tile(
            domain[0],
            domain[1],
            domain[2],
            nhalo,
        )

        # initialize the class
        convert = Convert(stencil_factory)

        # initialize quantities
        temperature_Celsius = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        for i in range(temperature_Celsius.field.shape[0]):
            for j in range(temperature_Celsius.field.shape[1]):
                for k in range(temperature_Celsius.field.shape[2]):
                    temperature_Celsius.field[i, j, k] = 20 + round(
                        random.uniform(-10, 10), 2
                    )

        temperature_Kelvin = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        temperature_Kelvin.field[:] = -999

        # call the class, perform the calculation
        convert(temperature_Celsius, temperature_Kelvin)
    ```

## Externals

NDSL has the ability build a stencil with immutable constants, called "externals", that can be
referenced only within that instance of the stencil. In this example, we can pass 273.15 to the
stencil as an external called `C_TO_K` and reference that throughout the stencil.

This ability is particularly useful as code gets more complicated, where constants may need to be
modified outside of the stencil (for instance, a tuning parameter which can vary from 0 to 1) and
where a constant is referenced numerous times throughout a stencil.

To build the stencil with an external, we just need to add a single argument to the build command:

``` py linenums="19"
        self.constructed_copy_stencil = stencil_factory.from_dims_halo(
            func=celsius_to_kelvin,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
            externals={
                "C_TO_K": 273.15,
            },
        )
```

And then to load in the external within the stencil:

``` py linenums="9"
def celsius_to_kelvin(temperature_Celsius: FloatField, temperature_Kelvin: FloatField):
    # read in the external
    from __externals__ import C_TO_K

    with computation(PARALLEL), interval(...):
            # convert from Celsius to Kelvin
            temperature_Kelvin = temperature_Celsius + C_TO_K
```

Note that these externals are not automatically passed down to any functions called with the
stencil. They must be passed in as an input.

Externals may be of type `float`, `int`, or `bool`, but must be scalar values.

NDSL has a a number of externals which are related to the compute domain of a stencil and are
always available (we will discuss compute domain more in a later section). These are:

- `i_start`
- `i_end`
- `j_start`
- `j_end`
- `k_start`
- `k_end`

## Restricting the Compute Domain

The compute domain is the subset of the model domain over which the stencil is being computed. By
default, the compute domain is the entire model domain, but NDSL has tools to introduce
restrictions. Thus far, we have used `from_dims_halo`, which always
includes the entire domain. There is another method - `from_origin_domain`, which allows more
flexibility (in fact, `from_dims_halo` calls this method with fixed arguments). Modifying our
example above:

``` py linenums="15"
class Convert:
    def __init__(self, stencil_factory: StencilFactory, domain):

        self.constructed_copy_stencil = stencil_factory.from_origin_domain(
            func=celcius_to_kelvin,
            origin=(0, 0, 0),
            domain=domain,
            externals={
                "C_TO_K": 273.15,
            },
        )
```

and now we need to pass in an additional argument to the class:

``` py linenums="43"
    convert = Convert(stencil_factory, domain)
```

The key arguments with `from_dims_halo` are `origin` (which specifies the starting point for
all computations) and `domain` (which specifies the end point). Any restrictions on `interval`
will count from these points. For example, if origin is set to `(0, 0, 1)`, and the stencil has
a computation using `with interval(1, None)`, the calculations will begin at the second K level.
A similar behavior occurs with `domain` and `interval` end points.

Note that restricting `domain` is the only way to restrict computation across the X/Y plane.
Additionally, a single stencil template can be used multiple times with different supporting
arguments to create multiple executable stencils.

Of course, using `from_dims_halo` requires information about the `domain` to be available, and a
previous guide stated that `domain` would not (and should not) be explicitly defined when
integrating code into a larger system. At that stage, information about the domain will be
automatically generated and stored in a variable called `grid` - but we will talk more about that
when we get there.

## Four Dimensional Fields and Data Dimensions

It is possible to create a four dimensional field in NDSL. This is constructed as a standard
three dimensional field plus a fourth "data dimension". This fourth dimension cannot be
parallelized, and therefore cannot be iterated over like the primary three dimensions.

The example below shows how to create a four dimensional field, where the fourth dimension
has size 36. 

```py

class example_4D_fields:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ) -> None:

        self.field_4D_quantity_factory = self.make_4D_quantity_factory(
            self.quantity_factory,
        )

        self.field4D = self.field_4D_quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, "extra_dim"], "n/a"
        )

    @staticmethod
    def make_4D_quantity_factory(
        ijk_quantity_factory: QuantityFactory,
    ):
        field_4D_quantity_factory = copy.deepcopy(ijk_quantity_factory)
        field_4D_quantity_factory.set_extra_dim_lengths(
            **{
                "extra_dim": 36,
            }
        )
        return field_4D_quantity_factory
```

These fields have a unique accessing method: `field4D[0, 0, 0][0]`. The first three indexes are
the standard relative indexing method, while the fourth is an *absolute* index corresponding to the
desired location along the additional axis that you want to reference.

Using a similar method as above, it is also possible to create data dimensions with one or two
of the primary three dimensions. Note that there must always be one primary dimension present.

## Typecasting

NDSL uses the traditional Python functions `int()` and `float()` to cast numbers to integers or
floating point precision, respectively. When using these casts, the precision is automatically
set to line up with the specified global precision. It is possible; however, to overwrite this and
cast explicitly to a 32 or 64 bit version with `i32()`/`i64()`/`f32()`/`f64()`

It is also possible to initiate a temporary field with a specific type. The correct nomenclature is:
`field: type = 0` where `type` can be any of `i32`/`i64`/`f32`/`f64`

## Current Index Information

NDSL stores information about the current K level in a variable called `THIS_K` as type `Int`.
This must be imported from `gt4py.cartesian.gtscript` prior to being used.

## Absolute K Indexing

NDSL does have a method for absolute indexing within a stencil, but it should be used with caution.
This can only be used at read (similar to offsets), and should only be used when absolutely
necessary (i.e. do not rely on this when relative offsetting is sufficient).

The proper nomenclature is `field.at(K=level)` where level is a `Int` type number, variable, or
expression which corresponds to a level present in the accessed field.

Similarly, absolute K-Indexing can be used on a four-dimensional field as follows: 
`field4D.at(K=level, ddim=[n])`, where `n` represents the index being accessed along the fourth
dimension.

## Summary

This guide introduced some more advanced features and exceptions to some of NDSL's rules. If used
as designed these features will have no impact on performance, but it is possible to misuse these
features in ways that will degrade performance. For that reason, it is recommended that these
features are used sparingly, to that they do not become a crutch that enables poor coding habits
and reduces the potency of the software as a whole.

In the next guide, we will introduce some more details about the inner workings of NDSL, and
in the process unlock more control over the acceleration process.
