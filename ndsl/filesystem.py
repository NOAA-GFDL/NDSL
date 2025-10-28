import warnings

import fsspec


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    """Return the fsspec filesystem required to handle a given path."""
    warnings.warn(
        "Usage of `get_fs()` is discouraged if favor `os.path` and `pathlib` "
        "modules. The function will be removed in the next version of NDSL.",
        DeprecationWarning,
        stacklevel=2,
    )
    fs, _, _ = fsspec.get_fs_token_paths(path)
    return fs


def is_file(filename):
    warnings.warn(
        "Usage of `is_file()` is discouraged if favor of plain `os.path.isfile()`. "
        "The function will be removed in the next version of NDSL.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_fs(filename).isfile(filename)


def open(filename, *args, **kwargs):
    warnings.warn(
        "Usage of `open()` is discouraged if favor the python built-in file "
        "open context manager. The function will be removed in the next version "
        "of NDSL.",
        DeprecationWarning,
        stacklevel=2,
    )
    fs = get_fs(filename)
    return fs.open(filename, *args, **kwargs)
