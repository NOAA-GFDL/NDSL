
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>NDSL API Docs</title>
</head>

<body>

<div class="dsl-page">

  <!-- HERO -->
  <div class="page-hero">

    <h1>API Documentation</h1>

    <p>
      Explore the core NDSL APIs for stencil computation,
      data structures, runtime orchestration,
      and scalable backend execution across CPUs and GPUs.
    </p>

    <div class="tag-row">
      <div class="badge">DSL</div>
      <div class="badge">Runtime</div>
      <div class="badge">Backends</div>
      <div class="badge">Diagnostics</div>
      <div class="badge">Testing</div>
      <div class="badge">Performance</div>
    </div>

  </div>

  <!-- API GRID -->
  <div class="two-col">

    <div class="section-card">
      <h2>Core DSL</h2>
      <p>
        APIs for stencil configuration, data types, storage,
        and GT4Py utilities.
      </p>
      <a class="btn" href="./docstrings/dsl/gt4py_utils.md">Explore →</a>
    </div>

    <div class="section-card">
      <h2>Grid & Dimensions</h2>
      <p>
        Structured grid indexing, halo regions,
        dimensional metadata, and quantity management.
      </p>
      <a class="btn" href="./docstrings/grid/eta.md">Explore →</a>
    </div>

    <div class="section-card">
      <h2>Performance & Diagnostics</h2>
      <p>
        Profiling, monitoring, debugging, and scalable execution analysis.
      </p>
      <a class="btn" href="./docstrings/performance/collector.md">Explore →</a>
    </div>

    <div class="section-card">
      <h2>Testing</h2>
      <p>
        APIs and helpful tools for testing NDSL against the reference Fortran.
      </p>
      <a class="btn" href="./docstrings/testing/comparison.md">Explore →</a>
    </div>

  </div>

</div>

</body>
</html>
