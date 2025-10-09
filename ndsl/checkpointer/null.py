from ndsl.checkpointer.base import ArrayLike, Checkpointer, SavepointName


class NullCheckpointer(Checkpointer):
    def __call__(self, savepoint_name: SavepointName, **kwargs: ArrayLike) -> None:
        pass
