
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />

<title>NDSL API Docs</title>

<style>

.api-page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0.5rem 1.5rem;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  color: #111827;
}

/* HERO */

.api-hero {
  margin-bottom: 2rem;
}

.api-hero h1 {
  font-size: 38px;
  font-weight: 650;
  color: #185FA5;
  margin: 0 0 0.75rem 0;
  letter-spacing: -0.03em;
  line-height: 1.05;
}

.api-hero p {
  max-width: 800px;
  font-size: 16px;
  line-height: 1.7;
  color: #6b7280;
  margin: 0;
}

/* TAGS */

.api-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin: 1.5rem 0 2rem 0;
}

.api-pill {
  background: #EAF3FC;
  color: #185FA5;
  border-radius: 999px;
  padding: 0.45rem 0.9rem;
  font-size: 13px;
  font-weight: 500;
}

/* GRID */

.api-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 2rem;
}

.api-card {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 1.2rem;
  background: #ffffff;
  transition: 0.2s ease;
}

.api-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.06);
  border-color: #185FA5;
}

.api-card h2 {
  font-size: 18px;
  color: #185FA5;
  margin: 0 0 0.4rem 0;
}

.api-card p {
  font-size: 14px;
  line-height: 1.6;
  color: #6b7280;
  margin: 0 0 1rem 0;
}

/* BUTTON */

.api-btn {
  display: inline-block;
  padding: 0.55rem 1rem;
  background: #185FA5;
  color: #ffffff !important;
  border-radius: 10px;
  text-decoration: none;
  font-size: 13px;
  font-weight: 500;
  transition: background 0.2s ease;
}

.api-btn:hover {
  background: #0C447C;
}

/* CODE */

.code-card {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  overflow: hidden;
  background: #ffffff;
}

.code-header {
  padding: 0.8rem 1rem;
  border-bottom: 1px solid #e5e7eb;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9ca3af;
}

.code-block {
  margin: 0;
  padding: 1.2rem;

  background: #f8fafc;

  font-family:
    "SFMono-Regular",
    Consolas,
    "Liberation Mono",
    Menlo,
    monospace;

  font-size: 13px;
  line-height: 1.7;

  overflow-x: auto;
  color: #111827;
}

/* FOOTER CTA */

.api-footer {
  margin-top: 2.5rem;
  padding: 1.5rem;
  border-radius: 18px;
  background: linear-gradient(135deg, #f0f7ff, #ffffff);
  text-align: center;
  box-shadow: 0 10px 25px rgba(24,95,165,0.08);
}

.api-footer h2 {
  margin: 0 0 0.5rem 0;
  color: #185FA5;
  font-size: 24px;
}

.api-footer p {
  max-width: 700px;
  margin: 0 auto 1rem auto;
  color: #6b7280;
  font-size: 14px;
  line-height: 1.6;
}

/* MOBILE */

@media (max-width: 768px) {

  .api-grid {
    grid-template-columns: 1fr;
  }

  .api-hero h1 {
    font-size: 34px;
  }

}

</style>
</head>

<body>

<div class="api-page">

  <!-- HERO -->

  <div class="api-hero">

    <h1>API Documentation</h1>

    <p>
      Explore the core NDSL APIs for stencil computation,
      data structures, runtime orchestration,
      and scalable backend execution across CPUs and GPUs.
    </p>

    <div class="api-tags">
      <div class="api-pill">DSL</div>
      <div class="api-pill">Runtime</div>
      <div class="api-pill">Backends</div>
      <div class="api-pill">Diagnostics</div>
      <div class="api-pill">Testing</div>
      <div class="api-pill">Performance</div>
    </div>

  </div>

  <!-- API GRID -->

  <div class="api-grid">

    <div class="api-card">
      <h2>Core DSL</h2>
      <p>
        APIs for stencil configuration, data types, storage,
        and GT4Py utilities.
      </p>

      <a class="api-btn" href="./docstrings/dsl/gt4py_utils.md">
        Explore →
      </a>
    </div>

    <div class="api-card">
      <h2>Grid & Dimensions</h2>
      <p>
        Structured grid indexing, halo regions,
        dimensional metadata, and quantity management.
      </p>

      <a class="api-btn" href="./docstrings/grid/eta.md">
        Explore →
      </a>
    </div>

    <div class="api-card">
      <h2>Performance & Diagnostics</h2>
      <p>
        Profiling, monitoring, debugging, and scalable execution analysis.
      </p>

      <a class="api-btn" href="./docstrings/performance/collector.md">
        Explore →
      </a>
    </div>

    <div class="api-card">
      <h2>Testing</h2>
      <p>
        APIs and helpful tools for testing NDSL against the reference Fortran.
      </p>

      <a class="api-btn" href="./docstrings/testing/comparison.md">
        Explore →
      </a>
    </div>

  </div>

  <!-- CODE EXAMPLE -->

  <!-- <div class="code-card">

    <div class="code-header">
      Example Stencil Expression
    </div>

<pre class="code-block"><code># Pressure-gradient force
with computation(PARALLEL), interval(...):

    w_new = w - dt / rho * (
        p[0, 0, 1] - p[0, 0, -1]
    ) / (2.0 * dz)
</code></pre>

  </div> -->

  <!-- CTA -->

</div>

</body>
</html>