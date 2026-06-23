# DaCe

[DaCe](https://spcldace.readthedocs.io/en/latest/index.htm) is is the full-program optimization framework used in NDSL. DaCe is short for Data-Centric Parallel Programming and developed at ETH's scalable parallel computing lab (SPCL).

In NDSL, DaCe powers the [performance backends](https://geos-esm.github.io/SMT-Nebulae/technical/backend/dace-bridge/) of [GT4Py](./gt4py.md). In particular, in NDSL's orchestration feature we will encode [macro-level optimizations](https://geos-esm.github.io/SMT-Nebulae/technical/backend/ADRs/stree/) like loop re-ordering and stencil fusing using DaCe.
