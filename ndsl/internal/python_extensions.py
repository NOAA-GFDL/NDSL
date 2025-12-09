from types import FrameType

from ndsl.dsl.ndsl_runtime import NDSLRuntime


def find_first_NDSLRuntime_caller(frame: FrameType | None) -> NDSLRuntime | None:
    """Inspect the given stack and return the first NDSLRuntime object encountered"""
    if frame is None:
        return frame

    search_frame: FrameType | None = frame

    # Search for a NDSLRuntime frame
    while search_frame:
        if "self" in search_frame.f_locals and issubclass(
            type(search_frame.f_locals["self"]), NDSLRuntime
        ):
            return search_frame.f_locals["self"]
        search_frame = search_frame.f_back

    return None


def find_all_NDSLRuntime_callers(frame: FrameType | None) -> list[NDSLRuntime]:
    """Inspect the given stack and return the NDSLRuntime objects
    in order of hits.
    """
    if frame is None:
        return []

    search_frame: FrameType | None = frame

    runtimes = []

    # Search for a NDSLRuntime frame
    while search_frame:
        if "self" in search_frame.f_locals and issubclass(
            type(search_frame.f_locals["self"]), NDSLRuntime
        ):
            runtimes.append(search_frame.f_locals["self"])
        search_frame = search_frame.f_back

    return runtimes
