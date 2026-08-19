from ndsl.dsl.gt4py import FORWARD, computation, interval
from ndsl.dsl.stencil import deprecated_stencil
from ndsl.dsl.typing import Bool, BoolFieldIJ, Float, FloatFieldIJ


def copy_2d(input: FloatFieldIJ, output: FloatFieldIJ) -> None:
    """
    Copy one field into another - 2D variant.

    Args:
        input: input field
        output: output field
    """
    with computation(FORWARD), interval(...):
        output = input


def add_2d(summand_1: FloatFieldIJ, summand_2: FloatFieldIJ, sum: FloatFieldIJ) -> None:
    """
    Add two inputs together, sum to a new field - 2D variant.

    Args:
        summand_1: input field
        summand_2: input field
        sum: output field
    """
    with computation(FORWARD), interval(0, 1):
        sum = summand_1 + summand_2


def add_to_self_2d(field: FloatFieldIJ, summand: FloatFieldIJ) -> None:
    """
    Add a summand to a field - 2D variant.

    Args:
        field: field to be modifid
        summand: modification to be made
    """
    with computation(FORWARD), interval(0, 1):
        field = field + summand


def subtract_2d(
    minuend: FloatFieldIJ, subtrahend: FloatFieldIJ, difference: FloatFieldIJ
) -> None:
    """
    Subtract subtrahend from minuend, output to a new field - 2D variant.

    Args:
        minuend: input field
        summand_2: input field
        difference: output field
    """
    with computation(FORWARD), interval(0, 1):
        difference = minuend - subtrahend


def subtract_from_self_2d(field: FloatFieldIJ, subtrahend: FloatFieldIJ) -> None:
    """
    Subtract a modification from a field - 2D variant.

    Args:
        field: field to be modified
        subtrahend: modification to be made
    """
    with computation(FORWARD), interval(0, 1):
        field = field - subtrahend


def multiply_2d(
    factor_1: FloatFieldIJ, factor_2: FloatFieldIJ, product: FloatFieldIJ
) -> None:
    """
    Multiply two inputs together, output to a new field - 2D variant.

    Args:
        factor_1: input field
        factor_2: input field
        product: output field
    """
    with computation(FORWARD), interval(0, 1):
        product = factor_1 * factor_2


def multiply_to_self_2d(field: FloatFieldIJ, factor: FloatFieldIJ) -> None:
    """
    Multiply a field by a factor - 2D variant.

    Args:
        field: field to be modifid
        factor: modification factor
    """
    with computation(FORWARD), interval(0, 1):
        field = field * factor


def divide_2d(
    dividend: FloatFieldIJ, divisor: FloatFieldIJ, quotient: FloatFieldIJ
) -> None:
    """
    Divide dividend by divisor, output to a new field - 2D variant.

    Args:
        dividend: input field
        divisor: input field
        quotient: output field
    """
    with computation(FORWARD), interval(0, 1):
        quotient = dividend / divisor


def divide_self_2d(field: FloatFieldIJ, divisor: FloatFieldIJ) -> None:
    """
    Divide dividend by divisor - 2D variant.

    Args:
        field: field to be modified
        divisor: modification factor
    """
    with computation(FORWARD), interval(0, 1):
        field = field / divisor


def set_value_2d(field: FloatFieldIJ, value: Float) -> None:
    """
    Sets every element of a field to a single value - 2D variant.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(FORWARD), interval(...):
        field = value


def set_boolean_value_2d(field: BoolFieldIJ, value: Bool) -> None:
    """
     Sets every element of a field to either True or False - 2D variant.

    Args:
        field: output field
        value: value of Bool type
    """
    with computation(FORWARD), interval(0, 1):
        field = value


#############################
# Deprecated stencils       #
#############################
# Will be removed and replaced with the above, properly named one
@deprecated_stencil
def divide_to_self_2d(field: FloatFieldIJ, divisor: FloatFieldIJ) -> None:
    """
    Muultiply a field by a factor - 2D variant.

    Args:
        field: field to be modifid
        divisor: modification factor
    """
    with computation(FORWARD), interval(0, 1):
        field = field / divisor


# Will be removed and replaced with the above, properly named one
@deprecated_stencil
def subtract_to_self_2d(field: FloatFieldIJ, subtrahend: FloatFieldIJ) -> None:
    """
    Subtract a modification from a field - 2D variant.

    Args:
        field: field to be modified
        subtrahend: modification to be made
    """
    with computation(FORWARD), interval(0, 1):
        field = field - subtrahend
