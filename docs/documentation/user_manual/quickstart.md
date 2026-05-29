<div class="getting-started-page">

  <!-- HERO -->

  <div class="hero-section">

    <h1>Getting Started</h1>

    <p>
      Install NDSL and start building portable atmospheric modeling
      workflows across CPUs and GPUs in just a few minutes.
    </p>

  </div>

  <!-- REQUIREMENTS -->

  <section>

    <div class="section-label">Prep</div>

    <h2>Requirements</h2>

    <p>
      Before installing NDSL, make sure your environment includes:
    </p>

    <ul>
      <li>Python <code>3.11</code></li>
      <li>GNU compiler toolchain (<code>gcc</code> / <code>gfortran</code>)</li>
    </ul>

    <div class="callout">
      We strongly recommend using either a virtual environment
      or Conda environment for installation.
    </div>

  </section>

  <!-- CLONE -->

  <section>

    <div class="section-label">Step 1</div>

    <h2>Clone the Repository</h2>

    <p>
      NDSL uses Git submodules for dependencies including GT4Py and DaCe,
      so be sure to clone recursively.
    </p>

<pre class="code-block"><code>git clone --recurse-submodules git@github.com:NOAA-GFDL/NDSL.git

cd NDSL/
</code></pre>

    <div class="note">
      <strong>Why clone the repository?</strong><br>
      NDSL is currently not available on PyPI,
      so installation requires cloning the source repository.
    </div>

  </section>

  <!-- VENV -->

  <section>

    <div class="section-label">Step 2</div>

    <h2>Create a Virtual Environment</h2>

    <p>
      Create and activate a clean Python environment.
    </p>

<pre class="code-block"><code>python -m venv .venv

source .venv/bin/activate
</code></pre>

  </section>

  <!-- MPI -->

  <section>

    <div class="section-label">Optional</div>

    <h2>Install MPI</h2>

    <p>
      If your system does not already provide MPI,
      you can install OpenMPI using pip.
    </p>

<pre class="code-block"><code>pip install openmpi
</code></pre>

  </section>

  <!-- INSTALL -->

  <section>

    <div class="section-label">Step 3</div>

    <h2>Install NDSL</h2>

    <p>
      Install NDSL along with demo dependencies.
    </p>

<pre class="code-block"><code>pip install .[demos]
</code></pre>

  </section>

  <!-- EXAMPLES -->

  <section>

    <div class="section-label">Next Steps</div>

    <h2>Run the Examples</h2>

    <p>
      Launch the notebooks located in:
    </p>

<pre class="code-block"><code>examples/NDSL
</code></pre>

    <p>
      Start experimenting with NDSL 🚀
    </p>

  </section>

  <!-- COMPILER -->

  <div class="compiler-section">

    <h2>Supported Compilers</h2>

    <div class="warning">
      <strong>GNU Compiler Required</strong><br><br>

      NDSL currently supports the GNU compiler toolchain only.

      Using <code>clang</code> may result in undefined OpenMP flag errors.

      <br><br>

      For macOS users,
      <code>gcc-14</code> installed through Homebrew
      is known to work successfully.
    </div>

  </div>

</div>

<style>

.getting-started-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0.5rem 1rem 2rem 1rem;

  font-family:
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;

  color: #111827;
}

/* HERO */

.hero-section {
  margin-bottom: 1.5rem;
}

.hero-section h1 {
  margin: 0 0 0.4rem 0;

  font-size: 40px;
  font-weight: 650;

  color: #185FA5;

  letter-spacing: -0.03em;
  line-height: 1.05;
}

.hero-section p {
  max-width: 720px;

  font-size: 15px;
  line-height: 1.6;

  color: #6b7280;
}

/* SECTIONS */

section {
  margin-bottom: 1rem;

  padding: 1rem 1.1rem;

  border: 1px solid #e5e7eb;
  border-radius: 16px;

  background: #ffffff;

  transition: 0.2s ease;
}

section:hover {
  border-color: #185FA5;

  box-shadow:
    0 6px 18px rgba(0,0,0,0.03);
}

.section-label {
  margin-bottom: 0.2rem;

  font-size: 11px;
  font-weight: 600;

  color: #185FA5;

  text-transform: uppercase;
  letter-spacing: 0.06em;
}

/* HEADERS */

section h2,
.compiler-section h2 {
  margin: 0 0 0.45rem 0;

  font-size: 24px;
  font-weight: 600;

  color: #185FA5;

  letter-spacing: -0.02em;
}

/* TEXT */

section p,
section li,
.compiler-section {
  font-size: 14px;
  line-height: 1.6;
  color: #4b5563;
}

ul {
  margin: 0.4rem 0;
  padding-left: 1.2rem;
}

/* CODE */

.code-block {
  margin: 0.7rem 0;

  padding: 0.8rem 1rem;

  border-radius: 12px;

  background: #f8fafc;
  border: 1px solid #e5e7eb;

  overflow-x: auto;

  font-size: 12px;
  line-height: 1.5;

  font-family:
    "SFMono-Regular",
    Consolas,
    "Liberation Mono",
    Menlo,
    monospace;
}

/* CALLOUTS */

.callout,
.note,
.warning {
  margin-top: 0.7rem;

  padding: 0.8rem 0.9rem;

  border-radius: 12px;

  line-height: 1.6;
  font-size: 13px;
}

.callout {
  background: #f8fafc;
  color: #4b5563;
}

.note {
  background: #f5f9ff;
  color: #335c85;
}

.warning {
  background: #fff7ed;
  color: #9a3412;
}

/* COMPILER SECTION */

.compiler-section {
  margin-top: 1.5rem;
}

/* INLINE CODE */

code {
  font-family:
    "SFMono-Regular",
    Consolas,
    monospace;

  font-size: 0.94em;
}

</style>