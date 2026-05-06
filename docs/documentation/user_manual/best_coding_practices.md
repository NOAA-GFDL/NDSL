# Best Coding Practices

In this section, we provide some general guidelines for writing and structuring code in NDSL.
While these practices are not necessarily required, we strongly encourage users to adhere to them
as they are important for readability, maintainability, collaboration, and performance.

## Docstrings

When writing code using NDSL, it is important that your code is well-documented so that others can
easily understand the purpose of your code without having to dig into the implementation details.
We strongly encourage the use of docstrings to document your code.

Docstrings are strings written immediately after the definition of an NDSL function, stencil, or
class. They are enclosed in triple quotes (`"""` or `'''`) and can span multiple lines. A good
docstring should include information about what the function, stencil, or class does. It should also
include a summary of what the code does and a list of parameters, their type, and a description.
See below for an example of how to write docstrings for a GT4Py function.

```py
@gtscript.function
def sign(
    a: Float,
    b: Float,
):
    """
    Function that returns the magnitude of one argument and the sign of another.

    Inputs:
    a [Float]: Argument of which the magnitude is needed [unitless]
    b [Float]: Argument of which the sign is needed [unitless]

    Returns:
    result [Float]: The magnitude of a and sign of b [unitless]
    """

    if b >= 0.0:
        result = abs(a)
    else:
        result = -abs(a)

    return result
```

## Temporaries

To create and store temporary fields in NDSL, which are not explicitly defined within a stencil or
class, we strongly encourage the use of NDSL `dataclasses`. A `dataclass` is a Python class
that is used to store or hold data. While the use of `dataclasses` are not required, we strongly
encourage our users to store temporaries in a `dataclass` for cleanliness and readability purposes.

See below for an example of how to create a `dataclass` to hold temporary fields.

```py
from dataclasses import dataclass

from ndsl import Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM

@dataclass
class Temporaries:
    ssthl0: Quantity
    ssqt0: Quantity
    ssu0: Quantity
    ssv0: Quantity

    @classmethod
    def make(cls, quantity_factory: QuantityFactory):
        # FloatFields
        ssthl0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        ssqt0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        ssu0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        ssv0 = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        return cls(
            ssthl0,
            ssqt0,
            ssu0,
            ssv0,
        )
```

## General structure of NDSL repositories

Now that we have covered pretty much all of the topics needed to start learning and developing
code in NDSL, it's time to talk about how a repository containing NDSL code should be
structured.

In this example, we've created a mock-up repository which contains NDSL code to convert
temperature from Fahrenheit to Kelvin and then back to Fahrenheit. We've named our mock-up 
repository `tutorial`, which contains four Python scripts: `driver.py`, `stencils.py`, 
`constants.py`, and `temporaries.py`. 

Each script has a unique purpose. For example, `driver.py` contains code to create and call the
`class` that initializes the NDSL stencils.

```py
from ndsl import StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import FloatField
from pyMoist.tutorial.stencils import convert_F_to_K, convert_K_to_F
from pyMoist.tutorial.temporaries import Temporaries
import random

class Temperature_Conversion:
    def __init__(self, stencil_factory: StencilFactory):
        """
        Class to convert temperatures from Fahrenheit to Kelvin and then back to Fahrenheit

        Parameters:
        stencil_factory (StencilFactory): Factory for creating stencil computations
        """

        # Build stencils
        self.convert_F_to_K = stencil_factory.from_dims_halo(
            func=convert_F_to_K,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self.convert_K_to_F = stencil_factory.from_dims_halo(
            func=convert_K_to_F,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

        self.temporaries = Temporaries.make(quantity_factory)

    def __call__(
        self,
        temp_F: FloatField,
        temp_K: FloatField,
    ):

        self.convert_F_to_K(
            temp_F,
            temp_K,
        )

        self.convert_K_to_F(
            temp_K,
            self.temporaries.temp_C,
            temp_F,
        )


if __name__ == "__main__":

    # Setup domain and generate factories
    domain = (5, 5, 10)
    nhalo = 0
    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0],
        domain[1],
        domain[2],
        nhalo,
        backend="debug",
    )

    # Initialize quantities
    temp_F = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
    for i in range(temp_F.field.shape[0]):
        for j in range(temp_F.field.shape[1]):
            for k in range(temp_F.field.shape[2]):
                temp_F.field[i, j, k] = round(random.uniform(70, 90), 2)

    temp_K = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

    # Build stencil
    code = Temperature_Conversion(stencil_factory)

    # Check input data
    print(temp_F.field[0, 0, :])

    # Execute stencil
    code(temp_F, temp_K)

    # Check output data
    print(temp_K.field[0, 0, :])
    print(temp_F.field[0, 0, :])
```

`stencils.py` contains both NDSL functions and stencils that do the temperature conversion.

```py
from ndsl.dsl.gt4py import (
    computation,
    interval,
    PARALLEL,
)
import gt4py.cartesian.gtscript as gtscript
from ndsl.dsl.typing import FloatField, Float
import pyMoist.tutorial.constants as constants


@gtscript.function
def convert_F_to_C(
    t_F: Float,
):
    """
    Function to convert Fahrenheit to Celsius.

    Inputs:
    t_F (Float): Temperature in Fahrenheit (degrees)

    Returns:
    t_C (Float): Temperature in Celsius (degrees)
    """
    t_C = (t_F - 32) * (5 / 9)

    return t_C


@gtscript.function
def convert_C_to_F(
    t_C: Float,
):
    """
    Function to convert Celsius to Fahrenheit.

    Inputs:
    t_C (Float): Temperature in Celsius (degrees)

    Returns:
    t_F (Float): Temperature in Fahrenheit (degrees)
    """
    t_F = (t_C * (9 / 5)) + 32

    return t_F


def convert_F_to_K(
    temp_F: FloatField,
    temp_K: FloatField,
):
    """
    Stencil to convert temperature from Fahrenheit to Kelvin.

    Temperature is first converted to Celsius, then Celsius to Kelvin.

    Inputs:
    temp_F (FloatField): Temperature in degrees Fahrenheit

    Outputs:
    temp_K (FloatField): Temperature in Kelvin (unitless)
    """

    with computation(PARALLEL), interval(...):
        temp_C = convert_F_to_C(temp_F)
        temp_K = temp_C + constants.absolute_zero


def convert_K_to_F(
    temp_K: FloatField,
    temp_C: FloatField,
    temp_F: FloatField,
):
    """
    Stencil to convert temperature from Kelvin to Fahrenheit.

    Temperature is first converted to Celsius, then Celsius to Fahrenheit.

    Inputs:
    temp_K (FloatField): Temperature in Kelvin (unitless)

    Outputs:
    temp_F (FloatField): Temperature in degrees Fahrenheit
    """

    with computation(PARALLEL), interval(...):
        temp_C = temp_K - constants.absolute_zero
        temp_F = convert_C_to_F(temp_C)
```

`temporaries.py` contains a `dataclass` that holds any temporary fields used in the stencil
computations

```py
from dataclasses import dataclass

from ndsl import Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM


@dataclass
class Temporaries:
    temp_C: Quantity

    @classmethod
    def make(cls, quantity_factory: QuantityFactory):
        # FloatFields
        temp_C = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        return cls(
            temp_C,
        )
```

`constants.py` contains any constants needed in the functions and stencils

```py
from ndsl.dsl.typing import Float

absolute_zero = Float(273.15)
```

It is important to structure your NDSL repositories in a similar fashion not only for cleanliness
and readability, but for scalability and debugging purposes as well.
