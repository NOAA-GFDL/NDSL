from dace.sdfg.analysis.schedule_tree import treenodes as tn


class InlineVertical2DWrite(tn.ScheduleNodeTransformer):
    """Inline K index value for 2D write vertical while removing for loop.

    Transforming:
    ```
    for __k = 0; __k < 1; __k = __k + 1:
        map __j, __i:
            field[__i, __j] = tasklet(field_in[__i, __j, __k])
    ```

    Into
    ```
    map __j, __i:
        field[__i, __j] = tasklet(field_in[__i, __j, 0])
    ```
    """

    def __init__(self) -> None:
        super().__init__()
