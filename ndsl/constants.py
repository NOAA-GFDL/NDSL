import os
import warnings
from enum import Enum
from typing import Literal

import numpy as np

from ndsl.dsl.typing import Float
from ndsl.logging import ndsl_log


# The FV3GFS model ships with two sets of constants, one used in the UFS physics
# package and the other used for the Dycore. Their difference are small but significant
# In addition the GSFC's GEOS model as its own variables
class ConstantVersions(Enum):
    GFDL = "GFDL"  # NOAA's FV3 dynamical core constants (original port)
    UFS = "UFS"  # Constant as defined in NOAA's UFS
    GEOS = "GEOS"  # Constant as defined in GEOS v11.4.2


def _get_constant_version(
    default: Literal["GFDL", "UFS", "GEOS"] = "UFS",
) -> Literal["GFDL", "UFS", "GEOS"]:
    constants_as_str = os.getenv("NDSL_CONSTANTS", default)
    expected: list[Literal["GFDL", "UFS", "GEOS"]] = ["GFDL", "UFS", "GEOS"]

    if constants_as_str not in expected:
        raise RuntimeError(
            f"Constants '{constants_as_str}' is not implemented, abort. Valid values are {expected}."
        )

    return constants_as_str  # type: ignore


CONST_VERSION = ConstantVersions[_get_constant_version()]
ndsl_log.info(f"Constant selected: {CONST_VERSION}")

#####################
# Common constants
#####################

I_DIM = "i"
I_INTERFACE_DIM = "i_interface"
J_DIM = "j"
J_INTERFACE_DIM = "j_interface"
K_DIM = "k"
K_INTERFACE_DIM = "k_interface"
K_SOIL_DIM = "k_soil"

_DEPRECATED_NAMES = {
    "X_DIM": I_DIM,
    "Y_DIM": J_DIM,
    "Z_DIM": K_DIM,
    "X_INTERFACE_DIM": I_INTERFACE_DIM,
    "Y_INTERFACE_DIM": J_INTERFACE_DIM,
    "Z_INTERFACE_DIM": K_INTERFACE_DIM,
    "Z_SOIL_DIM": K_SOIL_DIM,
}


class DeprecatedAxis:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        warnings.warn(
            f"Constant {self.name} is deprecated, please use I/J/K equivalent",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_NAMES[self.name]

    def __repr__(self) -> str:
        warnings.warn(
            f"Constant {self.name} is deprecated, please use I/J/K equivalent",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_NAMES[self.name]


X_DIM = DeprecatedAxis("X_DIM")
X_INTERFACE_DIM = DeprecatedAxis("X_INTERFACE_DIM")
Y_DIM = DeprecatedAxis("Y_DIM")
Y_INTERFACE_DIM = DeprecatedAxis("Y_INTERFACE_DIM")
Z_DIM = DeprecatedAxis("Z_DIM")
Z_INTERFACE_DIM = DeprecatedAxis("Z_INTERFACE_DIM")
Z_SOIL_DIM = DeprecatedAxis("Z_SOIL_DIM")

I_DIMS = (I_DIM, I_INTERFACE_DIM)
J_DIMS = (J_DIM, J_INTERFACE_DIM)
K_DIMS = (K_DIM, K_INTERFACE_DIM)
HORIZONTAL_DIMS = I_DIMS + J_DIMS
INTERFACE_DIMS = (I_INTERFACE_DIM, J_INTERFACE_DIM, K_INTERFACE_DIM)
SPATIAL_DIMS = I_DIMS + J_DIMS + K_DIMS

ROOT_RANK = 0
TILE_DIM = "tile"

WEST = 0
EAST = 1
NORTH = 2
SOUTH = 3
NORTHWEST = 4
NORTHEAST = 5
SOUTHWEST = 6
SOUTHEAST = 7
INTERIOR = 8
EDGE_BOUNDARY_TYPES = (NORTH, SOUTH, WEST, EAST)
CORNER_BOUNDARY_TYPES = (NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST)
BOUNDARY_TYPES = EDGE_BOUNDARY_TYPES + CORNER_BOUNDARY_TYPES
N_HALO_DEFAULT = 3

#######################
# Tracers configuration
#######################

# nq is actually given by ncnst - pnats, where those are given in atmosphere.F90 by:
# ncnst = Atm(mytile)%ncnst
# pnats = Atm(mytile)%flagstruct%pnats
# here we hard-coded it because 8 is the only supported value, refactor this later!
if CONST_VERSION == ConstantVersions.GEOS:
    # 'qlcd' is exchanged in GEOS
    NQ = 9
elif CONST_VERSION == ConstantVersions.UFS or CONST_VERSION == ConstantVersions.GFDL:
    NQ = 8
else:
    raise RuntimeError("Constant selector failed, bad code.")

#####################
# Physical constants
#####################
if CONST_VERSION == ConstantVersions.GEOS:
    RADIUS = Float(6.371e6)
    """Radius of the Earth [m]"""
    PI_8 = np.float64(3.14159265358979323846)
    PI = Float(PI_8)
    OMEGA = Float(2.0) * PI / Float(86164.0)
    """Rotation of the earth"""
    GRAV = Float(9.80665)
    """Acceleration due to gravity [m/s^2].04"""
    RGRAV = Float(1.0) / GRAV
    """Inverse of gravitational acceleration"""
    RDGAS = Float(8314.47) / Float(28.965)
    """Gas constant for dry air [J/kg/deg] ~287.04"""
    RVGAS = Float(8314.47) / Float(18.015)
    """Gas constant for water vapor [J/kg/deg]"""
    HLV = Float(2.4665e6)
    """Latent heat of evaporation [J/kg]"""
    HLF = Float(3.3370e5)
    """Latent heat of fusion [J/kg]  ~3.34e5"""
    KAPPA = RDGAS / (Float(3.5) * RDGAS)
    """Specific heat capacity of dry air at"""
    CP_AIR = RDGAS / KAPPA
    TFREEZE = Float(273.16)
    """Freezing temperature of fresh water [K]"""
    SAT_ADJUST_THRESHOLD = Float(1.0e-6)
    DZ_MIN = Float(6.0)
elif CONST_VERSION == ConstantVersions.UFS:
    RADIUS = Float(6.3712e6)
    """Radius of the Earth [m]"""
    PI = Float(3.1415926535897931)
    OMEGA = Float(7.2921e-5)
    """Rotation of the earth"""
    GRAV = Float(9.80665)
    """Acceleration due to gravity [m/s^2].04"""
    RGRAV = Float(1.0 / GRAV)
    """Inverse of gravitational acceleration"""
    RDGAS = Float(287.05)
    """Gas constant for dry air [J/kg/deg] ~287.04"""
    RVGAS = Float(461.50)
    """Gas constant for water vapor [J/kg/deg]"""
    HLV = Float(2.5e6)
    """Latent heat of evaporation [J/kg]"""
    HLF = Float(3.3358e5)
    """Latent heat of fusion [J/kg]  ~3.34e5"""
    CP_AIR = Float(1004.6)
    KAPPA = RDGAS / CP_AIR
    """Specific heat capacity of dry air at"""
    TFREEZE = Float(273.15)
    """Freezing temperature of fresh water [K]"""
    SAT_ADJUST_THRESHOLD = Float(1.0e-8)
    DZ_MIN = Float(2.0)
elif CONST_VERSION == ConstantVersions.GFDL:
    RADIUS = Float(6371.0e3)
    """Radius of the Earth [m] #6371.0e3"""
    PI = Float(3.14159265358979323846)
    """3.14159265358979323846"""
    OMEGA = Float(7.292e-5)
    """Rotation of the earth  # 7.292e-5"""
    GRAV = Float(9.80)
    """Acceleration due to gravity [m/s^2].04"""
    RGRAV = Float(1.0) / GRAV
    """Inverse of gravitational acceleration"""
    RDGAS = Float(287.04)
    """Gas constant for dry air [J/kg/deg] ~287.04"""
    RVGAS = Float(461.50)
    """Gas constant for water vapor [J/kg/deg]"""
    HLV = Float(2.500e6)
    """Latent heat of evaporation [J/kg]"""
    HLF = Float(3.34e5)
    """Latent heat of fusion [J/kg]  ~3.34e5"""
    KAPPA = Float(2.0) / Float(7.0)
    CP_AIR = RDGAS / KAPPA
    """Specific heat capacity of dry air at"""
    TFREEZE = Float(273.16)
    """Freezing temperature of fresh water [K]"""
    SAT_ADJUST_THRESHOLD = Float(1.0e-8)
    DZ_MIN = Float(2.0)
else:
    raise RuntimeError("Constant selector failed, bad code.")

SECONDS_PER_DAY = Float(86400.0)
SBC = Float(5.670400e-8)
"""Stefan-Boltzmann constant (W/m^2/K^4)"""
RHO_H2O = Float(1000.0)
"""Density of water in kg/m^3"""
CV_AIR = CP_AIR - RDGAS
"""Heat capacity of dry air at constant volume"""
RDG = -RDGAS / GRAV
K1K = RDGAS / CV_AIR
CNST_0P20 = np.float64(0.2)
CV_VAP = Float(3.0) * RVGAS
"""Heat capacity of water vapor at constant volume"""
ZVIR = RVGAS / RDGAS - Float(1)
"""con_fvirt in Fortran physics"""
C_ICE = Float(1972.0)
"""Heat capacity of ice at -15 degrees Celsius"""
C_ICE_0 = Float(2106.0)
"""Heat capacity of ice at 0 degrees Celsius"""
C_LIQ = Float(4.1855e3)
"""Heat capacity of water at 15 degrees Celsius"""
CP_VAP = Float(4.0) * RVGAS
"""Heat capacity of water vapor at constant pressure"""
TICE = Float(273.16)
"""Freezing temperature"""
DC_ICE = C_LIQ - C_ICE
"""Isobaric heating / cooling"""
DC_VAP = CP_VAP - C_LIQ
"""Isobaric heating / cooling"""
D2ICE = DC_VAP + DC_ICE
"""Isobaric heating / cooling"""
LI0 = HLF - DC_ICE * TICE
EPS = RDGAS / RVGAS
EPSM1 = EPS - Float(1.0)
LV0 = HLV - DC_VAP * TICE
"""3.13905782e6, evaporation latent heat coefficient at 0 degrees Kelvin"""
LI00 = HLF - DC_ICE * TICE
"""-2.7105966e5, fusion latent heat coefficient at 0 degrees Kelvin"""
LI2 = LV0 + LI00
"""2.86799816e6, sublimation latent heat coefficient at 0 degrees Kelvin"""
E00 = Float(611.21)
"""Saturation vapor pressure at 0 degrees Celsius (Pa)"""
PSAT = Float(610.78)
"""Saturation vapor pressure at H2O 3pt (Pa)"""
T_WFR = TICE - Float(40.0)
"""homogeneous freezing temperature"""
TICE0 = Float(2.7315e2)
""" Temp at 0C"""
T_MIN = Float(178.0)
"""Minimum temperature to freeze-dry all water vapor"""
T_SAT_MIN = TICE - Float(160.0)
"""Minimum temperature used in saturation calculations"""
LAT2 = np.power((HLV + HLF), 2, dtype=Float)
"""Used in bigg mechanism"""
TTP = Float(2.7316e2)
"""Temperature of H2O triple point"""
