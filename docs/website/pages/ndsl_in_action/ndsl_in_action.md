# NDSL in Action

Our NDSL port of GEOS has been evaluated through numerical verification, scientific validation, and large-scale performance benchmarking to ensure physical fidelity, computational efficiency, and portability across modern hardware architectures.

---

<div class="app-grid" markdown>
<div class="section-card" markdown>

## Numerical Validation

Numerical checks were performed to verify agreement between NDSL-generated code and the original Fortran.

- Serialization of Fortran reference state
- Translate tests used to evaluate DSL vs Fortran
- Differences were checked against numerical thresholds
</div>

<div class="section-card" markdown>

## Scientific Validation

NDSL-enabled GEOS simulations were compared against reference simulations to validate physical consistency.

- Single Column Model (SCM) simulations
- GEOS-FP simulations
- Aquaplanet simulations
</div>

<div class="section-card" markdown>

## Performance Benchmarking

Model runs were conducted on both CPUs and GPUs to evaluate performance portability and computational efficiency.

- CPU vs GPU acceleration
- Throughput improvements (days/day)
- Scaling efficiency with increased grid resolution
</div>
</div>
