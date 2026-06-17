<style>
/* re-enable the left side navigation bar for this page */
@media screen and (min-width: 76.1875em) {
  .md-sidebar--primary {
    display: block !important;
  }
}
</style>

# Writing Code in NDSL

NDSL finds power in its ability to accelerate and dynamically compile code. To do this,
NDSL creates an interface to two GT4Py constructs which are used to denote and contain accelerable
code: "stencils" and "functions". Conceptually, the two are similar, but their uses are different.

## Stencils

Stencils are the primary method of creating parallelizable code with NDSL, and indeed in NDSL
everything must begin with a stencil.

Within a 3-dimensional domain, NDSL evaluates computations in two parts. If we assume an (I, J, K)
coordinate system as a reference, NDSL separates computations in the horizontal (I, J) plane from the
vertical (Z) column.

In the horizontal plane, computations are **always** executed in parallel, which means that there is
no assumed calculation order within the plane. This concept is the foundation of of NDSL's
performance capabilities, and cannot be altered.

In the vertical column, computations are performed by an iteration policy that is declared
within the stencil. This is done to enable the implementation of more scientific patterns using
NDSL. We will discuss this in more detail shortly.

### Basic Stencil Example

To demonstrate how to implement a NDSL stencil and introduce the most important keywords, let's step through
an example of a stencil that copies the values of one field into another field.

First, we import several packages:

``` py linenums="1"
from ndsl import StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float, FloatField
import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import PARALLEL, computation, interval
```

Next, we define our stencil template:

``` py linenums="7"
def copy_stencil(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL):
        with interval(...):
            out_field = in_field
```

Note that there is no return statement here. Stencils modify fields in place **may not** contain
a return statement, and therefore must have all inputs *and* outputs passed into it at call.

This stencil template has a number of important features and keywords. Let's start with the inputs:
`in_field` and `out_field`. These are both declared to be type `FloatField`. This notation is used
in traditional Python, and may be familiar to you as optional "type hinting". For stencils
(and functions) these type hints are required, and the code will not execute if the supplied type
does not match the declared type.

Looking into the stencil code, we can see the two most important keywords in NDSL.

The statement `with computation(argument)` sets the iteration policy of the nested code. In this case,
we have declared `with computation(PARALLEL), signaling that *all three* dimensions can be executed in
parallel. The statement `with interval(argument)` sets the domain of the nested code. Once again, in this case,
we have declared `with interval(...)`, signaling that the computation should apply to all vertical levels
in the compute domain.

A stencil may have multiple sequential computation policies, and each computation policy may have multiple
intervals.

Now we set up our class:

``` py linenums="11"
class CopyData:
    def __init__(self, stencil_factory: StencilFactory):

        self.constructed_copy_stencil = stencil_factory.from_dims_halo(
            func=copy_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(self, in_quantity: FloatField, out_quantity: FloatField):
        self.constructed_copy_stencil(in_quantity, out_quantity)
```

Here is where we actually build our stencil (lines 14-17). Behind the scenes the system is
compiling code based on a number of considerations, most important of which is the target
architecture: CPU or GPU. In this example, we use the method `from_dims_halo`, which is the most
straightforward way to build a stencil. With this method, you will always have to specify `func` (to
determine what stencil is being built), and `compute_dims`. Stencils can also be built using
`from_origin_domain`, which requires manual entry of the `origin` and `domain` of the stencil. Behind the
scenes, `from_dims_halo` computes these and calls `from_origin_domain` automatically, so we recommend using
`from_dims_halo`, as it will suffice for the vast majority of situations.

Finally we can run the program:

``` py linenums="21"
if __name__ == "__main__":

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

    copy_data = CopyData(stencil_factory)

    in_quantity = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
    for i in range(in_quantity.field.shape[0]):
        for j in range(in_quantity.field.shape[1]):
            for k in range(in_quantity.field.shape[2]):
                in_quantity.field[i, j, k] = i * 100 + j * 10 + k

    out_quantity = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
    out_quantity.field[:] = -999

    copy_data(in_quantity, out_quantity)
```

To run the code, we first need to build the factories that will be used to construct the stencil and
quantities. In this small-scale example, we use the boilerplate function `get_factories_single_tile`,
supplying it the domain size, halo size, and backend. This function has limited capabilities, but is
sufficient for most small scale cases - testing code, debugging specific issues, etc. For larger projects
(such as a full Earth system model), it may be necessary to move away from the boilerplate code to get
more control over how these factoris are generated - but for now, just focus on using
`get_factories_single_tile`, as that will serve you well while you are learning the systems.

### Temporary Fields

NDSL has the ability to generate temporary quantities within a stencil. All temporary quantities
(defined as variables which are used within the stencil but not passed to the stencil at call) are
initialized as a field of dimensions equal to the full stencil domain. These fields can be used
just as any other field can be used, and are unavailable outside of the stencil.

### Offsets

Within a stencil, points are referenced relative to each other. For example, the statement
`field[0, 0, 1]` implies that you want an offset of positive one along the K axis (Z dimension
in our example). This offset occurs at each point in the domain independently, so you will always
be reading one "above" your current position.

Offsets can only occur at read; NDSL does not allow writing with an offset.
Additionally, it is not possible to write to a field when you read from it with an offset in the
same statement (e.g. `field = field[0, 0, 1]` is illegal).

With this knowledge, we can now create a stencil that copies data from the level above:

``` py linenums="1"
def copy_with_offset(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL):
        with interval(0, -1):
            out_field = in_field[0, 0, 1]
```

Note that the interval is restricted to prevent the computation from occurring on the top
level of `out_field`, as looking "up" from the top level would look into the extra point included
for interface calculations. Since this computation is not being performed on the interface (both
quantities are created using `Z_DIM`, not `Z_INTERFACE_DIM`), that row of data will be zero, and
accessing it here will have unintended consequences in subsequent calculations.

### Intervals and Iteration Policies

The statement `with interval()` controls the subset of the K axis over which the stencil is
executed, and is controlled using traditional Python indexing (e.g. `interval(0, 10)`,
`interval(1, -1)`). The argument `None` can be used to say "go to end of the domain" (e.g.
`interval(10, None)`). The argument `...` signals compute at all levels, equivalent to `0, None`.

NDSL has three possible iteration policies: `PARALLEL`, `FORWARD`, and `BACKWARD`.
As previously mentioned, choosing `PARALLEL` states that all three dimension will be executed in parallel.
In this state, each point in the compute domain is computed independently in the fastest possible order.
This means that fields are being written in random order, and therefore any operations which depend on data at
other points in a field which is being written (e.g. `out_field = field_computed_locally[0, 0, 1]`) cannot use
`PARALLEL`.

The `FORWARD` and `BACKWARD` options can be considered non-parallel options for the K axis. These
options require that all calculations for a particular K level are computed before moving on to the
next K level. This is **often significantly slower than `PARALLEL`**, but ensures that each kernel
has the correct information for the present level before moving on to the next one. `FORWARD` executes
from the first argument in the `with interval()` statement to the second (e.g.
`with computation(FORWARD), interval(0, 10)` begins at 0 and ends at 9), while `BACKWARD` does the
opposite (begins at 9, ends at 0).

`FORWARD` and `BACKWARD` are useful for more complex situations where data is being read with an
offset and written in the same stencil:

``` py linenums="1"
def offset_read_with_write(in_field: FloatField, out_field: FloatField):
    with computation(FORWARD):
        with interval(0, -1):
            out_field = in_field[0, 0, 1]
            in_field = in_field * 2
```

In this example, `in_field` is being read in but also modified. It is therefore necessary to use
`FORWARD` to ensure that there is not a situation where `in_field` is doubled at a level `n`
before it is read at level `n - 1`.

Using offsets often requires a careful consideration of iteration policy.

### Flow Control

A number of traditional Python flow control keywords can be used within NDSL stencils. These are:

- `if`
- `elif`
- `else`
- `while`

### Conditionals

NDSL allows conditionals to be used inside of a stencil in the same way they would be used
in traditional Python.

### Loops

It is possible to use a `while` loops within an NDSL stencil. All loops must have definite limits
(e.g. no `while True`), but these limits do not necessarily need to be hard coded. Other variables can be
used a bounds of the loop, creating a situation where the bound of the loop is potentially different at
each point within the domain.

## Functions

Functions in NDSL - much like traditional Python functions - serve as a way to store commonly used code so
that it can be referenced easily from multiple places. NDSL functions (hereafter referred to as "functions")
can only be used from within stencils, and often act as extentions of the stencil, performing repetitive or
particualrly detailed operations. Critically, however, functions have a additional set of rules which make
them different in both appearence and operation stencils which house them.

Functions:
- cannot be called outside of a stencil
- cannot contain the keywords `computation` or `interval` (they rely on the host stencil for this info)
- are "point operations" - the are executed independently at each point in the compute domain (see below)
- must have a single return statement, but can return multiple values
- are not tied to a single stencil, and may be reused multiple times in one or many stencils
- do not need to be independently constructed with a stencil factory (they are built with the stencil)

The concept of a "point operation" is particularly important, and warrants its own discussion. There are
two important ideas which come along with this concept: functions perform operation at each point in the
compute domain independently, in isolation from all other points; and (despite this) functions may take
scalar or fields as inputs. Functions take an entire field as an import (despite only operating on a
single point in any particular instance) to allow for offset reads of these inputs. It may be necessary
read an offset from a field within a stencil, and it would be overly combersome to require a series of scalar
inputs for each of these offsets.

Beyond their use for potentially reading offsets from the inputs, the concept of a field is effectively
banned within a stencil. All calculations are performed using scalars. There are no arrays, lists, fields,
etc. Furthermore, reading a offset (such as `field[0, 0, 1]`) will alwyas produce a scalar, which fits
this paradigm as the rest of the field is effectively discarded.

As a consequence of these rules, functions always return a scalar value. The *stencil* then takes this value
and writes it to the correct place in the field.

Below is an example of a NDSL function, called from within a stencil:

``` py linenums="1"
@gtscript.function
def add_five(in_field: FloatField):
    out_value = in_field + 5
    return out_value


def copy_stencil(in_field: FloatField, out_field: FloatField):
    with computation(PARALLEL):
        with interval(...):
            out_field = add_five(in_field)
```

It is worth reiterating that, while functions can only have a single return statment, they can return multiple
value. Such a case would take the following pattern:
``` py
    field_1, field_2 = my_function(input_1, input_2, input_3)
```

Note that functions with multiple returns
cannot be integrated into larger expressions - they must be written on their own line, and each output must
be recieved (or discarded with `_`).


### Builtin Functions

NDSL provides access to a number of builtin GT4Py functions:

- `abs`: absolute value
- `min`: minimum
- `max`: maximum
- `mod`: modulo
- `sin`: sine
- `cos`: cosine
- `tan`: tangent
- `sinh`: hyperbolic sine
- `cosh`: hyperbolic cosine
- `tanh`: hyperbolic tangent
- `asin`: arc sine
- `acos`: arc cosine
- `atan`: arc tangent
- `asinh`: inverse hyperbolic sine
- `acosh`: inverse hyperbolic cosine
- `atanh`: inverse hyperbolic tangent
- `sqrt`: square root
- `exp`: inverse hyperbolic sine
- `log`: natural log
- `log10`: base 10 log
- `gamma`: gamma function
- `cbrt`: cubic root
- `isfinite`: determine if number is finite
- `isinf`: determine if number is infinite
- `isnan`: determine if number is nan
- `floor`: round down to nearest integer
- `ceil`: round up to nearest integer
- `trunc`: truncate
- `round`: round to nearest integer
- `erf`: error function
- `erfc`: complementary error function
- `int32`: cast to a 32 bit integer
- `int64`: cast to a 64 bit integer
- `float32`: cast to a 32 bit float
- `float64`: cast to a 64 bit float

All of these functions are available via the module `ndsl.dsl.gt4py`

## Stencils vs Functions

Every block of NDSL-accelerable code begins and ends in a stencil. The stencil signals that
parallel computation is possible, defines the iteration policy and sets the interval. From this point
the following logic applies:

- It is not possible to call one stencil from within another, as that would create a situation where
there are two layers of parallelization. If you want to reuse code across multiple stencils, it
should be put into a function.

- For similar reasons, it is not possible to call a stencil from within a function.

- It is possible to call one function from within another function, and there is no limit on maximum depth.

## Summary

This guide introduced the basic principles of stencils and functions. We have discussed how and
when to use each, and highlighted features that make their use more flexible, and
discussed limitations which act as guardrails against unfavorable outcomes.

With this knowledge, it is possible to use NDSL and obtain much of the designed performance;
however, development may at times seem tedious and implementing more complicated patterns may
be challenging. Future sections will introduce more complex ideas and discuss quality of life
features which aim to alleviates these issues.
