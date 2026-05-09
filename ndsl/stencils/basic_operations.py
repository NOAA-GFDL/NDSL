import typing

from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation, function, interval
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


def add_2d(summand_1: FloatFieldIJ, summand_2: FloatFieldIJ, sum: FloatFieldIJ) -> None:
    """
    Add two inputs together, sum to a new field - 2D variant.

    Args:
        summand_1: input field
        summand_2: input field
        sum: output field
    """
    with computation(FORWARD), interval(...):
        sum = summand_1 + summand_2


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
    with computation(FORWARD), interval(...):
        difference = minuend - subtrahend


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
    with computation(FORWARD), interval(...):
        product = factor_1 * factor_2


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
    with computation(FORWARD), interval(...):
        quotient = dividend / divisor


def set_value(field: FloatField, value: Float) -> None:
    """
    Sets every element of a field to a single value.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(PARALLEL), interval(...):
        field = value


def set_value_2D(field: FloatFieldIJ, value: Float) -> None:
    """
    Sets every element of a field to a single value - 2D variant.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(FORWARD), interval(...):
        field = value


def set_bool_value_2D(field: BoolFieldIJ, value: Bool) -> None:
    """
    Sets every element of buffer to either True or False.

    Args:
        field: output field
        value: value of Bool type
    """
    with computation(FORWARD), interval(0, 1):
        field = value


def adjustmentfactor_stencil(adjustment: FloatFieldIJ, field: FloatField) -> None:
    """
    Multiplies a field by an adjustment factor, modifying the original field.

    Args:
        adjustment: adjustment factor
        field: field to be modified
    """
    with computation(PARALLEL), interval(...):
        field = field * adjustment


def adjust_divide_stencil(adjustment: FloatField, field: FloatField) -> None:
    """
    Divides a field by an adjustment factor, modifying the original field.

    Args:
        adjustment: adjustment factor
        field: field to be modified
    """
    with computation(PARALLEL), interval(...):
        field = field / adjustment


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


def average_in(
    q_out: FloatField,
    adjustment: FloatField,
) -> None:
    """
    Averages every element of q_out with every element of the adjustment field,
    overwriting q_out.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = (q_out + adjustment) * 0.5


@typing.no_type_check
@function
def sign(a, b):
    """
    Defines a_sign_b as the absolute value of a, and checks if b is positive or
    negative, assigning the analogous sign value to a_sign_b. a_sign_b is returned.

    Args:
        a: A number
        b: A number
    """
    a_sign_b = abs(a)
    return a_sign_b if b > 0 else -a_sign_b


@typing.no_type_check
@function
def dim(a, b):
    """
    Calculates a - b, camped to 0, i.e. max(a - b, 0).
    """
    return max(a - b, 0)
