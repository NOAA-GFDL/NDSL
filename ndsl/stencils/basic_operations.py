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
    with computation(FORWARD), interval(0, 1):
        output = input


def add(input_1: FloatField, input_2: FloatField, output: FloatField) -> None:
    """
    Add two inputs together, output to a new field.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(PARALLEL), interval(...):
        output = input_1 + input_2


def add_2d(input_1: FloatFieldIJ, input_2: FloatFieldIJ, output: FloatFieldIJ) -> None:
    """
    Add two inputs together, output to a new field - 2D variant.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(FORWARD), interval(0, 1):
        output = input_1 + input_2


def subtract(input_1: FloatField, input_2: FloatField, output: FloatField) -> None:
    """
    Subtract input_2 from input_1, output to a new field.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(PARALLEL), interval(...):
        output = input_1 - input_2


def subtract_2d(
    input_1: FloatFieldIJ, input_2: FloatFieldIJ, output: FloatFieldIJ
) -> None:
    """
    Subtract input_2 from input_1, output to a new field - 2D variant.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(FORWARD), interval(0, 1):
        output = input_1 - input_2


def multiply(input_1: FloatField, input_2: FloatField, output: FloatField) -> None:
    """
    Multiply two inputs together, output to a new field.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(PARALLEL), interval(...):
        output = input_1 * input_2


def multiply_2d(
    input_1: FloatFieldIJ, input_2: FloatFieldIJ, output: FloatFieldIJ
) -> None:
    """
    Multiply two inputs together, output to a new field - 2D variant.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(FORWARD), interval(0, 1):
        output = input_1 * input_2


def divide(input_1: FloatField, input_2: FloatField, output: FloatField) -> None:
    """
    Divide input_1 by input_2, output to a new field.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(PARALLEL), interval(...):
        output = input_1 / input_2


def divide_2d(
    input_1: FloatFieldIJ, input_2: FloatFieldIJ, output: FloatFieldIJ
) -> None:
    """
    Divide input_1 by input_2, output to a new field - 2D variant.

    Args:
        input_1: input field
        input_2: input field
        output: output field
    """
    with computation(FORWARD), interval(0, 1):
        output = input_1 / input_2


def set_value(field: FloatField, value: Float) -> None:
    """
    Sets every element of field a value.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(PARALLEL), interval(...):
        field = value


def set_value_2D(field: FloatFieldIJ, value: Float) -> None:
    """
    Sets every element of field a value - 2D variant.

    Args:
        field: output field
        value: value of Float type
    """
    with computation(FORWARD), interval(0, 1):
        field = value


def set_IJ_mask_value(mask: BoolFieldIJ, value: Bool) -> None:
    """
    Sets every element of buffer to the value specified by value argument.

    Args:
        mask: output field
        value: value of Bool type
    """
    with computation(FORWARD), interval(0, 1):
        mask = value


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
