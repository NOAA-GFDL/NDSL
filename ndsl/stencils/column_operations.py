import typing

from ndsl.dsl.gt4py import function


@typing.no_type_check
@function
def column_max(field, start_index, end_index):
    """
    Find the maximum value for a full or slice of a column.

    Args:
        field: data to be analyzed
        start_index: "bottom" index of slice, must be less than end_index
        end_index: "top" index of slice, must be greater than start_index

        Returns: [max value, index of max value]
    """
    max_index = start_index
    max_value = field.at(K=max_index)
    level = start_index
    while level <= end_index:
        value = field.at(K=level)
        if value > max_value:
            max_value = value
            max_index = level
        level += 1

    return max_value, max_index


@typing.no_type_check
@function
def column_max_ddim(field, ddim, start_index, end_index):
    """
    Find the maximum value for a full or slice of a column.

    Args:
        field: data to be analyzed
        start_index: "bottom" index of slice, must be less than end_index
        end_index: "top" index of slice, must be greater than start_index

        Returns: [max value, index of max value]
    """
    max_index = start_index
    max_value = field.at(K=max_index, ddim=[ddim])
    level = start_index
    while level <= end_index:
        value = field.at(K=level, ddim=[ddim])
        if value > max_value:
            max_value = value
            max_index = level
        level += 1

    return max_value, max_index


@typing.no_type_check
@function
def column_min(field, start_index, end_index):
    """
    Find the minimum value for a full or slice of a column.

    Args:
        field: data to be analyzed
        start_index: "bottom" index of slice, must be less than end_index
        end_index: "top" index of slice, must be greater than start_index

        Returns: [min value, index of min value]
    """
    min_index = start_index
    min_value = field.at(K=min_index)
    level = start_index
    while level <= end_index:
        value = field.at(K=level)
        if value < min_value:
            min_value = value
            min_index = level
        level += 1

    return min_value, min_index


@typing.no_type_check
@function
def column_min_ddim(field, ddim, start_index, end_index):
    """
    Find the minimum value for a full or slice of a column.

    Args:
        field: data to be analyzed
        start_index: "bottom" index of slice, must be less than end_index
        end_index: "top" index of slice, must be greater than start_index

        Returns: [min value, index of min value]
    """
    min_index = start_index
    min_value = field.at(K=min_index, ddim=[ddim])
    level = start_index
    while level <= end_index:
        value = field.at(K=level, ddim=[ddim])
        if value < min_value:
            min_value = value
            min_index = level
        level += 1

    return min_value, min_index
