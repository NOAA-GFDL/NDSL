# Data for hybrid pressure calculations at each vertical level

NDSL requires the coefficients necessary for calculation of the pressure at each k-level to be supplied during a run. The equation for calculating these pressures takes the form:

$$\\P_k = a_k + b_k * P_s\\$$

where $P_k$ (also $\eta$) is the pressure at the k-level, $a_k$ and $b_k$, the needed coefficients, and $P_s$ the surface level pressure. These coefficients must be supplied in a NetCDF file format, and in a monotonically increasing format.

# Testing of $\eta$ calculation coefficients data file input

Current unit tests check that an input file will set the values correctly, check that a clear error is thrown when no file is supplied, and that the data contained within is monotonically increasing. To run these tests contained in [tests/grid/test_eta.py](../../../tests/grid/test_eta.py) after a successful installation of NDSL, run:

```shell
pytest tests/grid/test_eta.py
```

from the top level of the repository. Sample data files for these tests are contained in the directory [tests/data/eta](../../../tests/data/eta/)
