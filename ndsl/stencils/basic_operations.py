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


def copy(q_in: FloatField, q_out: FloatField) -> None:
    """
    Copy q_in to q_out.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in


def copy_defn(q_in: FloatField, q_out: FloatField) -> None:
    """
    [DEPRECATED] Copy q_in to q_out.

    This stencil is deprecated, use `copy(q_in, q_out)` instead.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in


def adjustmentfactor_stencil(adjustment: FloatFieldIJ, q_out: FloatField) -> None:
    """
    Multiplies every element of q_out by every element of the adjustment field over the
    interval, replacing the elements of q_out by the result of the multiplication.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_out * adjustment


def adjustmentfactor_stencil_defn(adjustment: FloatFieldIJ, q_out: FloatField) -> None:
    """
    [DEPRECATED] Multiplies every element of q_out by every element of the adjustment
    field over the interval, replacing the elements of q_out by the result of the multiplication.

    This stencil is deprecated, use `adjustmentfactor_stencil(adjustment, q_out)` instead.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_out * adjustment


def set_value(q_out: FloatField, value: Float) -> None:
    """
    Sets every element of q_out to the value specified by value argument.

    Args:
        q_out: output field
        value: NDSL Float type
    """
    with computation(PARALLEL), interval(...):
        q_out = value


def set_value_defn(q_out: FloatField, value: Float) -> None:
    """
    [DEPRECATED] Sets every element of q_out to the value specified by value argument.

    This stencil is deprecated, use `set_value(q_out, value)` instead.

    Args:
        q_out: output field
        value: NDSL Float type
    """
    with computation(PARALLEL), interval(...):
        q_out = value


def set_value_2D(buffer: FloatFieldIJ, value: Float) -> None:
    """
    Sets every element of buffer to the value specified by value argument.

    Args:
        buffer: output field
        value: value of Float type
    """
    with computation(FORWARD), interval(0, 1):
        buffer = value


def set_IJ_mask_value(mask_out: BoolFieldIJ, value: Bool) -> None:
    """
    Sets every element of buffer to the value specified by value argument.

    Args:
        buffer: output field
        value: value of Bool type
    """
    with computation(FORWARD), interval(0, 1):
        mask_out = value


def adjust_divide_stencil(adjustment: FloatField, q_out: FloatField) -> None:
    """
    Divides every element of q_out by every element of the adjustment field over the
    interval, replacing the elements of q_out by the result of the division.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_out / adjustment


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


@function
def sign(a, b):  # type: ignore[no-untyped-def]
    """
    Defines a_sign_b as the absolute value of a, and checks if b is positive or
    negative, assigning the analogous sign value to a_sign_b. a_sign_b is returned.

    Args:
        a: A number
        b: A number
    """
    a_sign_b = abs(a)
    return a_sign_b if b > 0 else -a_sign_b


@function
def dim(a, b):  # type: ignore[no-untyped-def]
    """
    Calculates a - b, camped to 0, i.e. max(a - b, 0).
    """
    return max(a - b, 0)
