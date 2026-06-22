# Notes on porting FORTRAN code

This part of the documentation includes notes about porting FORTRAN code to NDSL.

## General Concepts

Since we are not trying to do model developing but rather replicate an existing model, the main philosophy is to replicate model behavior as precisely as possible.
Since weather and climate models can take diverging paths based on very small input differences, as described in [\[1\]][1], a bitwise reproducible code is impossible to achieve.
There were attempts at solving this problem like shown in [\[2\]][2] or [\[3\]][3] but all of those require heavy modification to the original code.
In our case, the switch from the original FORTRAN environment to a C++ environment can already contribute to these small errors showing up and therefore a 1:1 validation on a large scale is impossible.
This effect gets further enhanced by computation on GPUs.
Lastly the mixing of precisions found in various models is often done slightly unmethodical and can further complicate the understand of what precision is required where.

Since large scale validation is therefore close to impossible, we are trying to get reproducible results (within a margin) on smaller sub-components of the model.
When porting code, we therefore try to break down larger components into logical, numerically coherent substructures that can be tested and validated individually.
This breakdown serves two main purposes:

1. Give us confidence, that the ported code behaves as intended.
2. Allow us to monitor if or how performance optimization down the road changes the numerical results of our model components.

## Porting Guidelines

Since GT4Py has certain restrictions on what can be in the same stencil and what needs to be in separate stencils, there is no absolute 1:1 mapping that can or should be applied.

The best practices we found are:

1. A numerically self-contained module should always live in a single class.
2. If possible, try to isolate individual numerical motifs into functions.

### Example

To illustrate best practices, we show a stripped version of the the nonhydrostatic vertical solver on the C-grid (Also know as the Riemann Solver):

#### Main definition

```python
class NonhydrostaticVerticalSolverCGrid:
    def __init__(self, ...):
        # Definition of the (potentially multiple) stencils to call
        self._precompute_stencil = stencil_factory.from_origin_domain(
            precompute,
            origin=origin,
            domain=domain,
        )
        self._compute_sim1_solve = stencil_factory.from_origin_domain(
            sim1_solver,
            origin=origin,
            domain=domain,
        )
        # Definition of temporary variables share across two stencils
        # that are not used outside the module
        self._pfac = FloatFieldIJ()
        ...
    def __call__(self, cappa: FloatField, delpc: FloatField):
        self._precompute_stencil(cappa, _pfac)
        self._compute_sim1_solve(_pfac, delpc)
```

#### Stencil Definitions

```python
#constants definition
c1 = Float(-2.0) / Float(14.0)
c2 = Float(11.0) / Float(14.0)
c3 = Float(5.0) / Float(14.0)

#function for numerical standalone motif
@gtscript.function
def vol_conserv_cubic_interp_func_y(v):
    return c1 * v[0, -2, 0] + c2 * v[0, -1, 0] + c3 * v

def precompute(cappa: FloatField, _pfac: FloatFieldIJ):
    # small computation directly in the stencil
    with computation(PARALLEL), interval(...):
        # a variable used only in one stencil can be defined here
        tmpvar = cappa[1,0,0] + 1
    with computation(PARALLEL), interval(0, 1):
        _pfac = tmpvar[0,0,1]

def sim1_solver(cappa: FloatField, _pfac: FloatFieldIJ):
    with computation(PARALLEL), interval(...):
        cappa = vol_conserv_cubic_interp_func_y(cappa) + _pfac
```

[1]: <https://www.climate.gov/news-features/blogs/enso/butterflies-rounding-errors-and-chaos-climate-models> "Chaos in climate models"
[2]: <https://pasc17.org/fileadmin/user_upload/pasc17/program/post125s2.pdf> "Reproducible Climate Simulations"
[3]: <http://htor.inf.ethz.ch/sec/bitrep-ipdps.pdf> "Bit reproducible HPC applications"
