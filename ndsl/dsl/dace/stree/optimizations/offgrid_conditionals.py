from __future__ import annotations

from dace.sdfg.analysis.schedule_tree import treenodes as tn


class InlineOffgridConditionals(tn.ScheduleNodeTransformer):
    """Push offgrid conditional inside their cartesian block,
    duplicating the conditional if needed

    Turning:
    ```
        if a_flag == 0
            map i, j, k
                [ops...]
            map i, j, k
                [ops...]
    ```
    into
    ```
        map i,j, k
            if a_flag == 0
                [ops...]
        map i,j, k
            if a_flag == 0
                [ops...]
    ```
    """

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "InlineOffgridConditionals"


class ExtractOffgridConditionals(tn.ScheduleNodeTransformer):
    """Push offgrid conditional outside of their cartesian block

    Reverse transform from InlineOffgridConditionals
    """

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "ExtractOffgridConditionals"


class MergeConditionals(tn.ScheduleNodeTransformer):
    """Merge consecutive and equal conditionals

    Turning:
    ```
        if a_flag == 0
            map i, j, k
                [ops...]
        if a_flag == 0
            map i, j, k
                [ops...]
    ```
    into
    ```
        if a_flag == 0
            map i, j, k
                [ops...]
            map i, j, k
                [ops...]
    ```

    Outside of user code, vombination of ExtractOffgridConditionals,
    InlineOffgridConditionals and CartesianMapMerge can lead to this
    pattern.
    """

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return "MergeConditionals"
