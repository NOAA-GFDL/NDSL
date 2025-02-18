Fortran Interoperability
=============

Alongside NDSL there are Fortran based methods that are currently leveraged by the physics and dynamics packages from which GEOS, pace, pySHiELD, and pyFV3 are ported, that handle aspects such as domain generation and data communication.

Packages are currently in development to introduce interfaces which will enable the use of these methods within a Python environment.

One of the ways this is possible is through the use of the ISO_C_BINDING module in Fortan, enabling Fortran-C interoperability, and the ctypes package in Python.

Fortran-C interoperable objects are compiled into a shared object library, and then access to these objects is possible after loading the library into a Python module via ctypes.

The ctypes package contains methods for converting Python objects into C-like objects for use by the Fortran-C source methods.

The `pyFMS <https://github.com/fmalatino/pyFMS>` package is under development and will contains methods from the `Flexible Modeling System (FMS) <https://github.com/NOAA-GFDL/FMS>`, which are made accesible by the `cFMS <https://github.com/mlee03/cFMS>` C-interface to FMS package, by the methods described above.

The methods included in pyFMS have been selected based on the needs of pace, pySHiELD, and pyFV3, but is designed to be independent of these packages.
