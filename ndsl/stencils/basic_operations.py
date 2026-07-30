import typing

from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, function, interval
from ndsl.dsl.stencil import deprecated_stencil
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    IntField,
    IntFieldIJ,
)


def copy(input: FloatField, output: FloatField) -> None:
    """
    Copy one field into another.

    Args:
        input: input field
        output: output field
    """
    with computation(PARALLEL), interval(...):
        output = input


def copy_2d(input: FloatFieldIJ, output: FloatFieldIJ) -> None:
    """
    Copy one field into another - 2D variant.

    Args:
        input: input field
        output: output field
    """
    with computation(FORWARD), interval(...):
        output = input


def add(summand_1: FloatField, summand_2: FloatField, sum: FloatField) -> None:
    """
    Add two inputs together, sum to a new field.

    Args:
        summand_1: input field
        summand_2: input field
        sum: output field
    """
    with computation(PARALLEL), interval(...):
        sum = summand_1 + summand_2


def add_to_self(field: FloatField, summand: FloatField) -> None:
    """
    Add a summand to a field.

    Args:
        field: field to be modifid
        summand: modification to be made
    """
    with computation(PARALLEL), interval(...):
        field = field + summand


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


def subtract(
    minuend: FloatField, subtrahend: FloatField, difference: FloatField
) -> None:
    """
    Subtract subtrahend from minuend, output to a new field.

    Args:
        minuend: input field
        subtrahend: input field
        difference: output field
    """
    with computation(PARALLEL), interval(...):
        difference = minuend - subtrahend


def subtract_to_self(field: FloatField, subtrahend: FloatField) -> None:
    """
    Subtract a subtrahend from a field.

    Args:
        field: field to be modifid
        subtrahend: modification to be made
    """
    with computation(PARALLEL), interval(...):
        field = field - subtrahend


def subtract_2d(
    minuend: FloatFieldIJ, subtrahend: FloatFieldIJ, difference: FloatFieldIJ
) -> None:
    """
    Subtract summand_2 from minuend, output to a new field - 2D variant.

    Args:
        minuend: input field
        summand_2: input field
        difference: output field
    """
    with computation(FORWARD), interval(0, 1):
        difference = minuend - subtrahend


def subtract_to_self_2d(field: FloatFieldIJ, subtrahend: FloatFieldIJ) -> None:
    """
    Subtract a modification from a field - 2D variant.

    Args:
        field: field to be modifid
        subtrahend: modification to be made
    """
    with computation(FORWARD), interval(0, 1):
        field = field - subtrahend


def multiply(factor_1: FloatField, factor_2: FloatField, product: FloatField) -> None:
    """
    Multiply two inputs together, output to a new field.

    Args:
        factor_1: input field
        factor_2: input field
        output: output field
    """
    with computation(PARALLEL), interval(...):
        product = factor_1 * factor_2


def multiply_to_self(field: FloatField, factor: FloatField) -> None:
    """
    Muultiply a field by a factor.

    Args:
        field: field to be modifid
        factor: modification factor
    """
    with computation(PARALLEL), interval(...):
        field = field * factor


def multiply_2d(
    factor_1: FloatFieldIJ, factor_2: FloatFieldIJ, product: FloatFieldIJ
) -> None:
    """
    Multiply two inputs together, output to a new field - 2D variant.

    Args:
        factor_1: input field
        factor_2: input field
        output: output field
    """
    with computation(FORWARD), interval(0, 1):
        product = factor_1 * factor_2


def multiply_to_self_2d(field: FloatFieldIJ, factor: FloatFieldIJ) -> None:
    """
    Muultiply a field by a factor - 2D variant.

    Args:
        field: field to be modifid
        factor: modification factor
    """
    with computation(FORWARD), interval(0, 1):
        field = field * factor


def divide(dividend: FloatField, divisor: FloatField, quotient: FloatField) -> None:
    """
    Divide dividend by divisor, output to a new field.

    Args:
        dividend: input field
        divisor: input field
        quotient: output field
    """
    with computation(PARALLEL), interval(...):
        quotient = dividend / divisor


def divide_to_self(field: FloatField, divisor: FloatField) -> None:
    """
    Muultiply a field by a factor - 2D variant.

    Args:
        field: field to be modifid
        divisor: modification factor
    """
    with computation(PARALLEL), interval(...):
        field = field / divisor


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


def divide_to_self_2d(field: FloatFieldIJ, divisor: FloatFieldIJ) -> None:
    """
    Muultiply a field by a factor - 2D variant.

    Args:
        field: field to be modifid
        divisor: modification factor
    """
    with computation(FORWARD), interval(0, 1):
        field = field / divisor


def set_value(field: FloatField, value: Float) -> None:
    """
    Sets every element of a field to a single value.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(PARALLEL), interval(...):
        field = value


def select_k(
    in_field: FloatField,
    out_field: FloatFieldIJ,
    k_mask: IntField,
    k_select: IntFieldIJ,
) -> None:
    """
    Saves a specific k-index of a 3D field to a new 2D array. The k-value can be
    different for each i,j point.

    Args:
        in_field: A 3D array to select from
        out_field: A 2D field to save values in
        k_mask: a field that lists each k-index
        k_select: the k-value to extract from in_field
    """
    # TODO: refactor this using THIS_K instead of a mask
    with computation(FORWARD), interval(...):
        if k_mask == k_select:
            out_field = in_field
