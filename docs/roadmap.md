# Roadmap

We are building NDSL in the open. On this page we'll start to share our public roadmap. This allows you to track our progress and see what's coming in the future. Head over to the [community section](./community.md) for discussions, questions, collaboration opportunities, and development updates.

## 2026.09.00

- :rocket: Cache-friendly merger debug
- :rocket: CPU serial execution merger
- :rocket: Scalarization and refinement of local memory for better cache access
- :gear: use `uv` for dependency management

## 2026.08.00

- :sparkles: Support for (integer) enum types in stencil code
- :sparkles: Support for `gcc-16`
- :rocket: Handle off-grid conditionals when merging cartesian axis
- :rocket: Avoid allocation of temporaries inside loops
- :beetle: Register data dimension fields with NDSL `Int` type
- :construction: Guard usage of `Local`s in non-orchestrated code paths
- :snake: Drop support for python 3.11
