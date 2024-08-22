import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import xarray as xr


ETA_0 = 0.252
SURFACE_PRESSURE = 1.0e5  # units of (Pa), from Table VI of DCMIP2016


@dataclass
class HybridPressureCoefficients:
    """
    Attributes:
     - ks: The number of pure-pressure layers at the top of the model
        Also the level where model transitions from pure pressure to
        hybrid pressure levels
     - ptop: The pressure at the top of the atmosphere
     - ak: The additive coefficient in the pressure calculation
     - bk: The multiplicative coefficient in the pressure calculation
    """

    ks: int
    ptop: int
    ak: np.ndarray
    bk: np.ndarray


def _load_ak_bk_from_file(eta_file: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(eta_file):
        raise ValueError("eta file does not exist")

    # read file into ak, bk arrays
    data = xr.open_dataset(eta_file)
    ak = data["ak"].values
    bk = data["bk"].values

    return ak, bk


def set_hybrid_pressure_coefficients(
    km: int,
    eta_file: str,
    ak_data: Optional[np.ndarray] = None,
    bk_data: Optional[np.ndarray] = None,
) -> HybridPressureCoefficients:
    """
    Sets the coefficients describing the hybrid pressure coordinates.

    The pressure of each k-level is calculated as Pk = ak + (bk * Ps)
    where Ps is the surface pressure. Values are currently stored in
    lookup tables.

    Args:
        km: The number of vertical levels in the model

    Returns:
        a HybridPressureCoefficients dataclass
    """
    if ak_data is None or bk_data is None:
        ak, bk = _load_ak_bk_from_file(eta_file)
    else:
        ak, bk = ak_data, bk_data

    # check size of ak and bk array is km+1
    if ak.size - 1 != km:
        raise ValueError(f"size of ak array {ak.size} is not equal to km+1={km+1}")
    if bk.size - 1 != km:
        raise ValueError(f"size of bk array {ak.size} is not equal to km+1={km+1}")

    # check that the eta values computed from ak and bk are monotonically increasing
    eta, etav = _check_eta(ak, bk)

    if not np.all(eta[:-1] <= eta[1:]):
        raise ValueError("ETA values are not monotonically increasing")
    if not np.all(etav[:-1] <= etav[1:]):
        raise ValueError("ETAV values are not monotonically increasing")

    if 0.0 in bk:
        ks = 0 if km == 91 else np.where(bk == 0)[0][-1]
        ptop = ak[0]
    else:
        raise ValueError("bk must contain at least one 0.")

    return HybridPressureCoefficients(ks, ptop, ak, bk)


def vertical_coordinate(eta_value) -> np.ndarray:
    """
    Equation (1) JRMS2006
    computes eta_v, the auxiliary variable vertical coordinate
    """
    return (eta_value - ETA_0) * math.pi * 0.5


def compute_eta(ak, bk) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equation (1) JRMS2006
    eta is the vertical coordinate and eta_v is an auxiliary vertical coordinate
    """
    eta = 0.5 * ((ak[:-1] + ak[1:]) / SURFACE_PRESSURE + bk[:-1] + bk[1:])
    eta_v = vertical_coordinate(eta)
    return eta, eta_v


def _check_eta(ak, bk):
    return compute_eta(ak, bk)
