# NDSL Data Types and Storage System

## Types

NDSL uses unique data types (`Float`/`Int`/`Bool`) which link to commonly used Python data
types (`float`/`int`/`bool`). Unlike python, however, with NDSL these will be either 32 or 64 bit, depending
on the desired execution environment. By default, NDSL executes in 64 bit precision. To control this manually,
ensure the environmental variable GT4PY_LITERAL_PRECISION is set to either 32 or 64 prior to execution.

## Fields

Unlike traditional Python, NDSL data types also implicitly carry information about the shape
of the object. The types `Float`/`Int`/`Bool` all denote scalar (dimensionless) objects.

All objects with one or more dimensions are considered "fields". NDSL assumes a cartesian
I/J/K coordinate system, where I and J are horizontal dimensions and K is the vertical
dimension, but allows these to be defined dynamically (more on this in the next section).

NDSL fields are typed according to the following: 
`[Type]Field[Axis][Precision]` - with `Type = [Int, Float, Bool]`; `Axis = [I, J, IJ, K]`;
`Precision = [32, 64]` (optional), if not specified then default to global precision.

Note that there is no `[Type]FieldIJK`; this is simply `[Type]Field`. These field types are only used within
NDSL-specific code ("stencils", described in the next section) to provide the system with expected shapes 
of arrays, facilitating many of the behind-the-scenes optimization processes.

Below is few examples of NDSL field types:

- `FloatFieldI`: one dimensional (I) field of type Float
- `IntFieldK`: one dimensional (K) field of type Int
- `FloatFieldIJ`: two dimensional (I, J) field of type Float
- `IntField`: three dimensional (I, J, K) field of type Int

NDSL has the ability to add a fourth dimension. This is a more complex process, and will be discussed
more in the future once we have a firm grasp of the basics.

## Data Storage

NDSL stores all field (>0 dimensions) data in a "quantity" object. Quantities allocate memory and assign
type automatically based on the dimensions and axis present. Scalars are not stored in quantities, and
therefore do not have access to the following functionality.

Below is an example of how to initialize an NDSL quantity with three dimensions:

``` py linenums="1"
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM

domain = (3, 2, 1)
nhalo = 0
stencil_factory, quantity_factory = get_factories_single_tile(
    domain[0], domain[1], domain[2], nhalo,
)

quantity_example = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a", dtype=Float)
```

For now, let's just worry about line 10. Here, we use the `quantity_factory` object (more about this
later) to initialize a quantity. The first argument, `[X_DIM, Y_DIM, Z_DIM]`, tells the system
which dimensions will be present. The size of each dimension is set in the `quantity_factory` object, and is
applied automatically. This size is immutable - to obtain a different size quantity, you must generate
another `quantity_factory`.

The second argument, `"n/a"` in this case, specifies the units of the data stored. This is only for
metadata and ease of use, and has no impact on calculations. The final argument `dtype` controls
the type of data stored in the quantity, and can be `Float`/`Int`/`Bool`.

Finally, we can print the quantity we have just created to check our work. Based on the code above,

``` py linenums="11"
print(quantity_example)
```

will return:

```none
Quantity(
    data=
[[[0. 0.]
  [0. 0.]
  [0. 0.]]

 [[0. 0.]
  [0. 0.]
  [0. 0.]]

 [[0. 0.]
  [0. 0.]
  [0. 0.]]

 [[0. 0.]
  [0. 0.]
  [0. 0.]]],
    dims=('x', 'y', 'z'),
    units=n/a,
    origin=(0, 0, 0),
    extent=(3, 2, 1)
)
```

Here you can see information about the dimensions present, size (extent), units,
and origin (a concept we will discuss later).

But wait, if our extent is (3, 2, 1), why is `data` size (4, 3, 2)?

### Center vs Interface Computations

This extra index along each dimension are for interface computations. In weather and climate
modeling, there are often situations where it is necessary to perform calculations on the interface
between grid points, rather than at the center of grid points - requiring one more point along that
particular axis. NDSL *always* creates quantities with one extra grid point on each dimension for this purpose;
however, this extra point is ignored by the system unless you explicitly state that it should be considered
by replacing (for example) `Z_DIM` with `Z_INTERFACE_DIM`. Specifying `Z_INTERFACE_DIM` does
*not* add an additional point along that dimension. It simple "enables" the point that is already there.

### Halo

NDSL quantities can be constructed with a halo on the X and Y dimensions. This is useful
when working with components such as the FV3 core, which uses a halo to facilitate data exchange
between faces of the cube. In the example above, the halo is set to zero (`nhalo=0`), but this can
be set to any value smaller than the smallest X/Y dimension.

### Accessing Data

There are two main methods for accessing data stored within a quantity:

- `quantity.field[:]`: returns the compute domain (excludes halos) of the quantity as a NumPy-like 
array. Note this will include the interface dimension point for axis specified to be operating on the
interface. Note that, since this is a NumPy-like 
array, it can be accessed using normal Python accessing rules, and much of the functionality of 
NumPy arrays is also available.

- `quantity.data[:]`: returns all data contained within the quantity, including the interface dimension point
(regardless of if it is including in the compute domain) and any halo present.

### Stencil/Quantity Factories

NDSL comes with a number of pre-designed factory functions (e.g. `get_factories_single_tile`).
These are designed to provide users with easy entry points to features while developing new code.

When working in a larger system the tasks of determining the domain and constructing factories
(lines 4-8 from the example above) are done automatically, and as such this code will not be needed.

Therefore, we recommend you don't worry about how `get_factories_single_tile` works for now. As we
move through more advanced guides we will introduce additional details and complexities, and those can be
reapplied to understand these functions, but knowledge of this function is not critical to using the software.

The `QuantityFactory` and `StencilFactory` types will be present in all your code, but once again you
need not worry about how they function. For now, just remember that a `QuantityFactory`
produces quantities and a `StencilFactory` produces stencils (something we will talk about in the
next guide), and as time progresses, we will introduce some of their more important methods.

## Summary

In this guide, we have discussed the unique NDSL data types and storage objects, focusing on how
to create, access, and print their contents.

Next, we will discuss how to write accelerable code with NDSL.
