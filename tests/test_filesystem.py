import pytest

import ndsl.filesystem as fs


def test_is_file_is_deprecated() -> None:
    with pytest.deprecated_call():
        fs.is_file("path/to/my_file.txt")


def test_open_is_deprecated() -> None:
    with pytest.deprecated_call():
        with fs.open("README.md", "r"):
            pass


def test_get_fs_is_deprecated() -> None:
    with pytest.deprecated_call():
        fs.get_fs("path/to/my/file.txt")
