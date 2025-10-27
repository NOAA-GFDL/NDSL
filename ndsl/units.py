import warnings


def ensure_equal_units(units1: str, units2: str) -> None:
    warnings.warn(
        "`ensure_equal_units` is unused and usage is discouraged. The function "
        "will be removed in the next version of NDSL.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not units_are_equal(units1, units2):
        raise UnitsError(f"incompatible units {units1} and {units2}")


def units_are_equal(units1: str, units2: str) -> bool:
    warnings.warn(
        "`units_are_equal` is unused and usage is discouraged. The function will "
        "be removed in the next version of NDSL.",
        DeprecationWarning,
        stacklevel=2,
    )
    return units1.strip() == units2.strip()


class UnitsError(Exception):
    def __init__(self, *args) -> None:
        warnings.warn(
            "`UnitsError` is unused and usage is discouraged. The class will be "
            "removed in the next version of NDSL.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args)
