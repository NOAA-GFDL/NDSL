<style>
/* re-enable the left side navigation bar for this page */
@media screen and (min-width: 76.1875em) {
  .md-sidebar--primary {
    display: block !important;
  }
}
</style>

# Validating our NDSL port of GEOS

## Fine-grain numerical validation: Translate test

Using Serialbox to output test data and the translate test infrastructure of `ndsl` we can make numerical regression test, known as Translate test.

This documentation is based on the work done at AI2 to do the original port FV3 dycore, which is still available:

- [fv3gfs-fortran](https://github.com/ai2cm/fv3gfs-fortran) is the serialbox annotated source Fortran
- [pyFV3](https://github.com/NOAA-GFDL/pyFV3) is the ported code which also contains the [translate test](https://github.com/NOAA-GFDL/PyFV3/tree/develop/tests/savepoint)

### Example `FillQZero` from `moist` physics

We are going to port the following simple subroutine.

```Fortran
  subroutine FILLQ2ZERO1( Q, MASS, FILLQ  )
    real, dimension(:,:,:),   intent(inout)  :: Q
    real, dimension(:,:,:),   intent(in)     :: MASS
    real, dimension(:,:),     intent(  out)  :: FILLQ
    integer                                  :: IM,JM,LM
    integer                                  :: I,J,K,L
    real                                     :: TPW, NEGTPW
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Fills in negative q values in a mass conserving way.
    ! Conservation of TPW was checked.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    IM = SIZE( Q, 1 )
    JM = SIZE( Q, 2 )
    LM = SIZE( Q, 3 )
    do j=1,JM
       do i=1,IM
          TPW = SUM( Q(i,j,:)*MASS(i,j,:) )
          NEGTPW = 0.
          do l=1,LM
             if ( Q(i,j,l) < 0.0 ) then
                NEGTPW   = NEGTPW + ( Q(i,j,l)*MASS( i,j,l ) )
                Q(i,j,l) = 0.0
             endif
          enddo
          do l=1,LM
             if ( Q(i,j,l) >= 0.0 ) then
                Q(i,j,l) = Q(i,j,l)*( 1.0+NEGTPW/(TPW-NEGTPW) )
             endif
          enddo
          FILLQ(i,j) = -NEGTPW
       end do
    end do
  end subroutine FILLQ2ZERO1
```

#### Generate data using `serialbox`

1. First we need to annotate the Fortran on a relevant call with serialbox.

⚠️ Some serializer code has been omitted for simplification ⚠️

```Fortran
!$ser savepoint FILLQ2ZERO-In
!$ser data q=RAD_QV
!$ser data mass=MASS
!$ser data fillq=TMP2D
call FILLQ2ZERO(RAD_QV, MASS, TMP2D)
!$ser savepoint FILLQ2ZERO-Out
```

It's easier to keep a `*.F90.ser` version of the file with serialization, because the system will generate the `F90` that needs to be compiled.

Turn the annotated code to proper fortran using

```bash
python /PATH_TO_SERIALBOX/src/serialbox-python/pp_ser/pp_ser.py ./microphysics.F90.ser >| ./microphysics.F90
```

This will dump two `netcdf` when running a simulation:

- FILLQZERO-In.nc4
- FILLQZERO-Out.nc4

For the translate test the naming convention is that the `-In` and `-Out` exists, and the string before will be the name of the test.

2. Turn the raw Fortran to NETCDF

[A script existing](https://github.com/ai2cm/fv3gfs-fortran/blob/master/tests/serialized_test_data_generation/serialbox_to_netcdf.py) in the original [fv3gfs-fortran](https://github.com/ai2cm/fv3gfs-fortran) source used for the original port.

#### Testing the ported code

In `ndsl` we structure the code using classes that wraps all relevant stencils and in-between code from a scientific standpoint. `FillQZero` would be part of the `microphysics` code, for simplicity we will write it as a standalone class.

!!! Note

    In `ndsl`, because we code generate and optimize across code, your code structure does _not_ impact the performance and should be as _readable_ as possible.

⚠️ Some `ndsl` code has been omitted for simplification ⚠️

Ported code looks as follows

```Python
from ndsl import StencilFactory, QuantityFactory
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, IntFieldIJ

def fill_q_zero_stencil(q:FloatField, mass: FloatField, fillq: FloatFieldIJ, tpw:FloatFieldIJ)
    with computation(PARALLEL), interval(...):
        neg_tpw = 0.
        if q < 0:
            neg_tpw = neg_tpw + q*mass
            q = 0
        if q >= 0
            q = q * (1+neg_tpw/(tpw_negtpw))
        fillq = -neg_tpw



class FillQZero:
    def __init__(stencil_factory: StencilFactory, quantity_factory: QuantityFactory):
        self._tpw = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        self._fillq = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        self._fill_q_zero = stencil_factory.from_dims_halo(
            fill_q_zero_stencil,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

    def __call__(
        q: FloatField,
        mass: FloatField,
    )
        # This is not a stencil code, so we do it outside
        self._tpw = np.sum(q, 3)
        # Call stencil code
        self._fill_q_zero(
            q=q,
            mass=mass,
            fillq=self._fillq,
            tpw=self._tpw
        )
```

1. Make a Translate test

Write a `translate_fill_q_zero.py`

```python
from ndsl import Namelist, StencilFactory
from fill_q_zero import FillQZero
from ndsl.testing.translate import TranslateFortranData2Py

class TranslateFillQZero(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.compute_func = FillQZero(
            self.stencil_factory,
            self.grid.quantity_factory,
        )

    def compute_from_storage(self, inputs):
        self.compute_func(**inputs)
        return inputs

```

2. Run

The above uses the `pytest` system under the hood, therefore the command is using pytest.

```bash
pytest \
    -v -s --data_path=/PATH/TO/DATA_DIR \
    --backend=numpy --which_modules=FillQZero \
    --threshold_overrides_file=/PATH/TO/threshold.yaml \
    /PATH/TO/TEST_FILE_DIR/

```

## Scientific Validation: plotting

The diagnostics output of GEOS is driven by the `HISTORY.rc` files. The output is a well formed Netcdf4 file with relevant comments and units. Those are the basis for the large GEOS products.

### `tcn` package

#### `tcb.plots`

Automatic plotting system with default to match GEOS needs. See `tcn-plots` command.

### Discover Jupyter and `tcn` package

In order to quickly check results we have developed a `tcn.plots` and made it available as part of a Jupyter notebook on Discover. That way, plotting results can be done on the HPC directly, removing the need for downloading the data.

**Basic howto**

- Get IT access (talk to Florian)
- On the Hub use the `smt-tcn` conda kernel
- Code:

```python
import plotly.io as pio
pio.renderers.default = 'iframe' #required for display

from tcn.plots.geos.plot_via_plotly import plot
import xarray as xr

d = xr.open_mfdataset("/path/to/dump.nc4")
var = "U"

f = plot(dataset=d, variable=var, write=False)
f.show()
```

As of 24.01 the `tcn` package version is the unreleased develop version of `fdeconinck`
