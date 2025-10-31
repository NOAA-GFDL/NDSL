from ndsl.dsl.gt4py import function


@function
def column_max(field, start_index, end_index):  # type: ignore
    """
    Find the maximum value for a full or slice of a column.

    Args:
        field: data to be analyzed
        start_index: "bottom" index of slice, must be less than end_index
        end_index: "top" index of slice, must be greater than start_index

        Returns: [max value, index of max value]
    """
    max_index = 0
    level = start_index
    while level <= end_index:
        new = field.at(K=level)
        old = field.at(K=max_index)
        if new > old:
            max_index = level
        level += 1

    return field.at(K=max_index), max_index


@function
def column_min(field, start_index, end_index):  # type: ignore
    """
    Find the minimum value for a full or slice of a column.

    Args:
        field: data to be analyzed
        start_index: "bottom" index of slice, must be less than end_index
        end_index: "top" index of slice, must be greater than start_index

        Returns: [min value, index of min value]
    """
    min_index = 0
    level = start_index
    while level <= end_index:
        new = field.at(K=level)
        old = field.at(K=min_index)
        if new < old:
            min_index = level
        level += 1

    return field.at(K=min_index), min_index
