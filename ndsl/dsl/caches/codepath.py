import enum


class FV3CodePath(enum.Enum):
    """
    Enum listing all possible code paths on a cube sphere.

    For any layout the cube sphere has up to 9 different code paths depending on
    the positioning of the rank on the tile and which of the edge/corner cases
    it has to handle, as well as the possibility for all boundary computations in
    the 1x1 layout case.

    Since the framework inlines code to optimize, we _cannot_ pre-suppose which code
    being kept and/or ejected. This enum serves as the ground truth to map rank to
    the proper generated code.
    """

    All = "FV3_A"
    "All boundary computations, e.g. 1x1 layout."
    BottomLeft = "FV3_BL"
    "Bottom left corner."
    Left = "FV3_L"
    "Left edge."
    TopLeft = "FV3_TL"
    "Top left corner."
    Top = "FV3_T"
    "Top edge."
    TopRight = "FV3_TR"
    "Top right corner."
    Right = "FV3_R"
    "Right edge."
    BottomRight = "FV3_BR"
    "Bottom right corner."
    Bottom = "FV3_B"
    "Bottom edge."
    Center = "FV3_C"
    "Center tile with boundaries, e.g. in a 3x3 layout."

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return self.value
