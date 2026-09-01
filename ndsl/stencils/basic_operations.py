from ndsl.dsl.gt4py import FORWARD, PARALLEL, K, computation, interval
from ndsl.dsl.typing import (
    Float,
    FloatField,
    FloatFieldIJ,
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
        field: field to be modified
        summand: modification to be made
    """
    with computation(PARALLEL), interval(...):
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


def subtract_from_self(field: FloatField, subtrahend: FloatField) -> None:
    """
    Subtract a subtrahend from a field.

    Args:
        field: field to be modified
        subtrahend: modification to be made
    """
    with computation(PARALLEL), interval(...):
        field = field - subtrahend


def multiply(factor_1: FloatField, factor_2: FloatField, product: FloatField) -> None:
    """
    Multiply two inputs together, output to a new field.

    Args:
        factor_1: input field
        factor_2: input field
        product: output field
    """
    with computation(PARALLEL), interval(...):
        product = factor_1 * factor_2


def multiply_to_self(field: FloatField, factor: FloatField) -> None:
    """
    Multiply a field by a factor.

    Args:
        field: field to be modified
        factor: modification factor
    """
    with computation(PARALLEL), interval(...):
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


def divide_self(field: FloatField, divisor: FloatField) -> None:
    """
    Divide a field by a divisor.

    Args:
        field: field to be modified
        divisor: modification factor
    """
    with computation(PARALLEL), interval(...):
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


def select_k_from_field(
    in_field: FloatField, out_field: FloatFieldIJ, k_select: IntFieldIJ
) -> None:
    """
    Saves a specific k-index of a 3D field to a new 2D array. The k-value can be
    different for each i,j point.

    Args:
        in_field: A 3D array to select from
        out_field: A 2D field to save values in
        k_select: the k-value to extract from in_field
    """
    with computation(FORWARD), interval(...):
        if K == k_select:
            out_field = in_field


def average_input(q_out: FloatField, adjustment: FloatField) -> None:
    """
    Averages every element of q_out with every element of the adjustment field,
    overwriting q_out.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = (q_out + adjustment) * 0.5
