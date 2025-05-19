from ndsl.dsl.gt4py import computation


def test_wrapper() -> None:
    """Tests importing the gt4py wrapper defined in ndsl/dsl/gt4py/__init__.py

    We don't need to import everything that's defined there, but we should import
    the file once in the NDSL testsuite such that we get a test failure in case we
    import anything that isn't available in mainline gt4py (yet)."""
    assert computation is not None
