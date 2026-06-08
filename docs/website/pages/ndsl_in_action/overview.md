
<style>
/* Hide the right-side table of contents */

.md-sidebar--secondary {
  display: none !important;
}

/* Expand content slightly */

.md-content__inner {
  max-width: 1000px;
}
</style>

<div style="
  padding: 1.75rem 1.75rem;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff 0%, #ffffff 100%);
  border: 1px solid #e5e7eb;
  margin-bottom: 1.5rem;
">

<h1 style="
  margin: 0;
  font-size: 38px;
  font-weight: 600;
  color: #185FA5;
  letter-spacing: -0.03em;
  line-height: 1.1;
">
  NDSL in Action
</h1>

<p style="
  margin-top: 0.7rem;
  max-width: 820px;
  font-size: 14px;
  color: #6b7280;
  line-height: 1.6;
">
  Our NDSL port of GEOS has been evaluated through numerical verification,
  scientific validation, and large-scale performance benchmarking
  to ensure physical fidelity, computational efficiency,
  and portability across modern hardware architectures.
</p>

</div>

<!-- TOP CARDS -->

<div style="
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap:1rem;
  margin-bottom:2rem;
">

<!-- NUMERICAL VALIDATION -->

<div style="
  border:1px solid #e5e7eb;
  border-radius:16px;
  padding:1.35rem;
  background:white;
">

<div style="
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background:#E6F1FB;
  color:#185FA5;
  font-size:11px;
  font-weight:600;
  margin-bottom:0.75rem;
">
  Verification
</div>

<h2 style="
  margin:0 0 0.5rem 0;
  color:#185FA5;
  font-size:22px;
  font-weight:600;
">
  Numerical Validation
</h2>

<p style="
  font-size:14px;
  color:#6b7280;
  line-height:1.6;
">
  Numerical checks were performed
  to verify agreement between NDSL-generated code
  and the original Fortran.
</p>

<ul style="
  color:#4b5563;
  font-size:14px;
  line-height:1.8;
  padding-left:1.1rem;
">
  <li>Serialization of Fortran reference state</li>
  <li>Translate tests used to evaluate DSL vs Fortran</li>
  <li>Differences were checked against numerical thresholds</li>
</ul>

</div>

<!-- SCIENTIFIC VALIDATION -->

<div style="
  border:1px solid #e5e7eb;
  border-radius:16px;
  padding:1.35rem;
  background:white;
">

<div style="
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background:#E6F1FB;
  color:#185FA5;
  font-size:11px;
  font-weight:600;
  margin-bottom:0.75rem;
">
  Validation
</div>

<h2 style="
  margin:0 0 0.5rem 0;
  color:#185FA5;
  font-size:22px;
  font-weight:600;
">
  Scientific Validation
</h2>

<p style="
  font-size:14px;
  color:#6b7280;
  line-height:1.6;
">
  NDSL-enabled GEOS simulations were compared against
  reference simulations to validate physical consistency.
</p>

<ul style="
  color:#4b5563;
  font-size:14px;
  line-height:1.8;
  padding-left:1.1rem;
">
  <li>Single Column Model (SCM) simulations</li>
  <li>GEOS-FP simulations</li>
  <li>Aquaplanet simulations</li>
</ul>

</div>

<!-- PERFORMANCE BENCHMARKING -->

<div style="
  border:1px solid #e5e7eb;
  border-radius:16px;
  padding:1.35rem;
  background:white;
">

<div style="
  display:inline-block;
  padding:4px 10px;
  border-radius:999px;
  background:#E6F1FB;
  color:#185FA5;
  font-size:11px;
  font-weight:600;
  margin-bottom:0.75rem;
">
  Benchmarking
</div>

<h2 style="
  margin:0 0 0.5rem 0;
  color:#185FA5;
  font-size:22px;
  font-weight:600;
">
  Performance Benchmarking
</h2>

<p style="
  font-size:14px;
  color:#6b7280;
  line-height:1.6;
">
  Model runs were conducted on both
  CPUs and GPUs to evaluate performance portability
  and computational efficiency.
</p>

<ul style="
  color:#4b5563;
  font-size:14px;
  line-height:1.8;
  padding-left:1.1rem;
">
  <li>CPU vs GPU acceleration</li>
  <li>Throughput improvements (days/day)</li>
  <li>Scaling efficiency with increased grid resolution</li>
</ul>

</div>

</div>

<!-- FEATURED RESULTS SECTION -->

<div style="
  border:1px solid #e5e7eb;
  border-radius:18px;
  padding:1.25rem;
  background:white;
">

<div>


<h2 style="
  margin:0;
  color:#185FA5;
  font-size:28px;
  font-weight:600;
">
  NDSL Highlights
</h2>

</div>

<!-- CAROUSEL -->

<div style="
  position:relative;
  margin-top:1rem;
">

<div class="results-carousel" style="
  display:flex;
  overflow-x:auto;
  scroll-snap-type:x mandatory;
  gap:1rem;
  scroll-behavior:smooth;
  padding-bottom:0.4rem;
">

  <!-- SLIDE 1 -->

  <div style="
    min-width:100%;
    scroll-snap-align:start;
  ">

    <img src="../img/scm_moist_arm_97jun_hovmoller_gpu_72.png"
         alt="Scientific Validation"
         style="
           width:100%;
           height:320px;
           object-fit:contain;
           background:white;
           border-radius:14px;
           border:1px solid #e5e7eb;
         ">

    <div style="margin-top:0.6rem;">

      <h3 style="
        margin:0 0 0.3rem 0;
        color:#185FA5;
        font-size:18px;
        font-weight:600;
      ">
        Scientific Validation
      </h3>

      <p style="
        margin:0;
        font-size:13px;
        color:#6b7280;
        line-height:1.5;
      ">
        Comparison of SCM GEOS simulations with NDSL-enabled moist physics against
        the reference Fortran.
      </p>

    </div>

  </div>

  <!-- SLIDE 2 -->

  <div style="
    min-width:100%;
    scroll-snap-align:start;
  ">

    <img src="../images/results_2.png"
         alt="Performance Benchmark"
         style="
           width:100%;
           height:320px;
           object-fit:contain;
           background:white;
           border-radius:14px;
           border:1px solid #e5e7eb;
         ">

    <div style="margin-top:0.6rem;">

      <h3 style="
        margin:0 0 0.3rem 0;
        color:#185FA5;
        font-size:18px;
        font-weight:600;
      ">
        GPU Performance Scaling
      </h3>

      <p style="
        margin:0;
        font-size:13px;
        color:#6b7280;
        line-height:1.5;
      ">
        Benchmarking across CPU and GPU architectures,
        highlighting performance portability and acceleration
        achieved with NDSL.
      </p>

    </div>

  </div>

  <!-- SLIDE 3 -->

  <div style="
    min-width:100%;
    scroll-snap-align:start;
  ">

    <img src="../img/results_3.png"
         alt="Numerical Validation"
         style="
           width:100%;
           height:320px;
           object-fit:contain;
           background:white;
           border-radius:14px;
           border:1px solid #e5e7eb;
         ">

    <div style="margin-top:0.6rem;">

      <h3 style="
        margin:0 0 0.3rem 0;
        color:#185FA5;
        font-size:18px;
        font-weight:600;
      ">
        Numerical Consistency
      </h3>

      <p style="
        margin:0;
        font-size:13px;
        color:#6b7280;
        line-height:1.5;
      ">
        Show some kind of figure here from our numerical validation?
      </p>

    </div>

  </div>

</div>

</div>

<style>
.results-carousel::-webkit-scrollbar {
  height: 8px;
}

.results-carousel::-webkit-scrollbar-track {
  background: #f3f4f6;
  border-radius: 999px;
}

.results-carousel::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 999px;
}

.results-carousel::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}

@media (max-width: 900px) {
  div[style*="grid-template-columns: repeat(3, 1fr)"] {
    grid-template-columns: 1fr !important;
  }
}
</style>