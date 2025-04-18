from typing import Any, Callable
from ndsl.debug.config import ndsl_debugger
import inspect
from functools import wraps


def instrument(func) -> Callable:
    @wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any):
        if ndsl_debugger is None:
            return func(self, *args, **kwargs)
        savename = func.__qualname__
        params = inspect.signature(func).parameters
        data_as_dict = {}

        # Positional
        positional_count = 0
        for name, param in params.items():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if positional_count == 0:  # self
                    positional_count += 1
                    continue
                if positional_count < len(args) + 1:
                    data_as_dict[name] = args[positional_count - 1]
                    positional_count += 1
        # Keyword arguments
        for name, value in kwargs.items():
            if name in params:
                data_as_dict[name] = value
        ndsl_debugger.save_as_dataset(data_as_dict, func.__qualname__, is_in=True)
        ndsl_debugger.track_data(data_as_dict, func.__qualname__, is_in=True)
        r = func(self, *args, **kwargs)
        ndsl_debugger.save_as_dataset(data_as_dict, func.__qualname__, is_in=False)
        ndsl_debugger.track_data(data_as_dict, func.__qualname__, is_in=False)
        ndsl_debugger.increment_call_count(savename)
        return r

    return wrapper
