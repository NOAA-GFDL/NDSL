import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ


def copy_defn(q_in: FloatField, q_out: FloatField):
    """
    Copy q_in to q_out.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in


def adjustmentfactor_stencil_defn(adjustment: FloatFieldIJ, q_out: FloatField):
    """
    Multiplies every element of q_out
    by every element of the adjustment
    field over the interval, replacing
    the elements of q_out by the result
    of the multiplication.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_out * adjustment


def set_value_defn(q_out: FloatField, value: Float):
    """
    Sets every element of q_out to the
    value specified by value argument.

    Args:
        q_out: output field
        value: NDSL Float type
    """
    with computation(PARALLEL), interval(...):
        q_out = value


def adjust_divide_stencil(adjustment: FloatField, q_out: FloatField):
    """
    Divides every element of q_out
    by every element of the adjustment
    field over the interval, replacing
    the elements of q_out by the result
    of the multiplication.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_out / adjustment


def average_in(
    q_out: FloatField,
    adjustement: FloatField,
):
    """
    Averages every element of q_out
    with every element of the adjustment
    field, overwriting q_out.

    Args:
        adjustment: adjustment field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = (q_out + adjustement) * 0.5


@gtscript.function
def sign(a, b):
    """
    Defines asignb as the absolute value
    of a, and checks if b is positive
    or negative, assigning the analogus
    sign value to asignb. asignb is returned

    Args:
        a: A number
        b: A number
    """
    asignb = abs(a)
    if b > 0:
        asignb = asignb
    else:
        asignb = -asignb
    return asignb


@gtscript.function
def dim(a, b):
    """
    Performs a check on the difference
    between the values in arguments
    a and b. The variable diff is set
    to the difference between a and b
    when the difference is positive,
    otherwise it is set to zero. The
    function returns the diff variable.
    """
    diff = a - b if a - b > 0 else 0
    return diff
