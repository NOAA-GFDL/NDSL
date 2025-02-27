import os
from enum import Enum

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


CONST_VERSION_AS_STR = os.environ.get("PACE_CONSTANTS", "UFS")

try:
    CONST_VERSION = ConstantVersions[CONST_VERSION_AS_STR]
    ndsl_log.info(f"Constant selected: {CONST_VERSION}")
except KeyError as e:
    raise RuntimeError(f"Constants {CONST_VERSION_AS_STR} is not implemented, abort.")

#####################
# Common constants
#####################

ROOT_RANK = 0
X_DIM = "x"
X_INTERFACE_DIM = "x_interface"
Y_DIM = "y"
Y_INTERFACE_DIM = "y_interface"
Z_DIM = "z"
Z_INTERFACE_DIM = "z_interface"
Z_SOIL_DIM = "z_soil"
TILE_DIM = "tile"
X_DIMS = (X_DIM, X_INTERFACE_DIM)
Y_DIMS = (Y_DIM, Y_INTERFACE_DIM)
Z_DIMS = (Z_DIM, Z_INTERFACE_DIM)
HORIZONTAL_DIMS = X_DIMS + Y_DIMS
INTERFACE_DIMS = (X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM)
SPATIAL_DIMS = X_DIMS + Y_DIMS + Z_DIMS

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
    RDGAS = Float(8314.47) / Float(
        28.965
    )
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
else:
    raise RuntimeError("Constant selector failed, bad code.")

SECONDS_PER_DAY = Float(86400.0)
DZ_MIN = Float(2.0)
CV_AIR = CP_AIR - RDGAS
"""Heat capacity of dry air at constant volume"""
RDG = -RDGAS / GRAV
CNST_0P20 = Float(0.2)
K1K = RDGAS / CV_AIR
CNST_0P20 = Float(0.2)
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
LV0 = (
    HLV - DC_VAP * TICE
)
"""3.13905782e6, evaporation latent heat coefficient at 0 degrees Kelvin"""
LI00 = (
    HLF - DC_ICE * TICE
)
"""-2.7105966e5, fusion latent heat coefficient at 0 degrees Kelvin"""
LI2 = (
    LV0 + LI00
)
"""2.86799816e6, sublimation latent heat coefficient at 0 degrees Kelvin"""
E00 = Float(611.21)
"""Saturation vapor pressure at 0 degrees Celsius (Pa)"""
PSAT = Float(610.78)
"""Saturation vapor pressure at H2O 3pt (Pa)"""
T_WFR = TICE - Float(40.0)
"""homogeneous freezing temperature"""
TICE0 = TICE - Float(0.01)
T_MIN = Float(178.0)
"""Minimum temperature to freeze-dry all water vapor"""
T_SAT_MIN = TICE - Float(160.0)
LAT2 = np.power((HLV + HLF), 2, dtype=Float)  
"""Used in bigg mechanism"""
TTP = 2.7316e2
"""Temperature of H2O triple point"""
