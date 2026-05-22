# 
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>NDSL Platform</title>

<style>
  .dsl-page {
    max-width: 1100px;
    margin: 0 auto;
    padding: 1rem 1.5rem;
    box-sizing: border-box;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }

  .section-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.25rem;
    background: #fff;
    margin-bottom: 0.9rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
  }

  .section-card:hover {
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    transform: translateY(-1px);
  }

  h2 {
  font-size: 22px;
  font-weight: 600;
  color: #185FA5;
  margin: 0 0 0.2rem 0;
  line-height: 1.2;
}

  h3 {
    margin: 0 0 0.2rem 0;
    color: #185FA5;
    line-height: 1.2;
  }

  p {
    font-size: 14px;
    color: #6b7280;
    line-height: 1.5;
    margin: 0.15rem 0 0.6rem 0;
  }

  .btn {
  display: inline-block;
  padding: 8px 16px;
  background: #185FA5;
  color: #fff !important;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 500;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  text-decoration: none !important;
  border: none;
  transition: background 0.2s ease;
}

  .btn:hover {
    background: #0C447C;
    text-decoration: none !important;
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 0.9rem;
  }

  .code-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    background: #f9fafb;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .code-header {
    padding: 0.6rem 1rem;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    border-bottom: 1px solid #e5e7eb;
  }

  .tab-bar {
    display: flex;
    border-bottom: 1px solid #e5e7eb;
  }

  .tab-btn {
    flex: 1;
    padding: 7px;
    font-size: 12px;
    background: none;
    border: none;
    cursor: pointer;
    color: #6b7280;
  }

  .tab-btn.active {
    color: #185FA5;
    border-bottom: 2px solid #185FA5;
    font-weight: 500;
  }

  .tab-panel {
    display: none;
    padding: 0.9rem;
  }

  .tab-panel.active {
    display: block;
  }

  .code-block {
    font-family: "SFMono-Regular",
    Consolas,
    "Liberation Mono",
    Menlo,
    monospace;
    font-size: 12px;
    white-space: pre;
    overflow-x: auto;
    color: #111827;
  }

  .apps-section h2 {
    margin: 1rem 0 0.5rem 0;
  }

  .app-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .app-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1rem;
    background: #fff;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: 0.2s;
  }

  .app-card:hover {
    border-color: #185FA5;
  }

  .badge {
  display: inline-block;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 999px;
  background: #E6F1FB;
  color: #185FA5;
  margin-bottom: 0rem;
  }

  @media (max-width: 768px) {
    .two-col, .app-grid {
      grid-template-columns: 1fr;
    }
  }

  .cta-card {
    margin-top: 1.5rem;
    padding: 1.5rem;
    text-align: center;
    border: none;
    border-radius: 14px;
    background: linear-gradient(135deg, #f0f7ff, #ffffff);
    box-shadow: 0 10px 25px rgba(24,95,165,0.10);
  }
</style>
</head>

<body>

<div class="dsl-page">
<!-- PAGE HEADER -->

<div style="margin-bottom: 1.2rem;">

  <h1 style="
    font-size: 40px;
    font-weight: 600;
    color: #185FA5;
    margin: 0 0 0.45rem 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
  ">
    NASA-NOAA Domain Specific Language
  </h1>

  <p style="
    font-size: 15px;
    color: #6b7280;
    line-height: 1.6;
    margin: 0;
    max-width: 760px;
  ">
    A modern domain-specific language for portable, high-performance atmospheric modeling across CPUs, GPUs, 
    and emerging computing architectures.
  </p>

</div>
    

  <div class="section-card">
    <span class="badge">Quickstart</span>
    <h2>Install NDSL</h2>
    <p>Install the latest version of NDSL with a few commands and start writing code.</p>
    <a class="btn" href="./documentation/user_manual/quickstart.md">NDSL v2026.03.00 →</a>
  </div>

  <div class="two-col">

    <div class="section-card">
      <span class="badge">Learn</span>
      <h2>Getting Started</h2>
      <p>New to NDSL? Walk through our beginner-friendly user manual to learn the core syntax and patterns.</p>
      <a class="btn" href="./documentation/user_manual/">User Manual →</a>
    </div>

    <div class="section-card">
      <span class="badge">Connect</span>
      <h2>Community</h2>
      <p>Connect with developers, researchers, and contributors working on NDSL and atmospheric modeling tools.</p>
      <a class="btn" href="./community.md">Join Community →</a>
    </div>

  </div>

  <div class="code-card">
    <div class="code-header">Fortran to NDSL</div>

    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab(event,'fortran')">Fortran</button>
      <button class="tab-btn" onclick="switchTab(event,'ndsl')">NDSL</button>
      <button class="tab-btn" onclick="switchTab(event,'gen')">Generated</button>
    </div>

    <div id="fortran" class="tab-panel active">
      <div class="code-block">subroutine calculate_cape_cin(virtual_temp_environment, virtual_temp_parcel, pressure_interface, cape, cin, source_level, level_free_convection, equilibrium_level, ni, nj, nk)

    integer, intent(in)  :: ni, nj, nk
    real,    intent(in)  :: virtual_temp_environment(ni, nj, nk)
    real,    intent(in)  :: virtual_temp_parcel(ni, nj, nk)
    real,    intent(in)  :: pressure_interface(ni, nj, nk+1)
    real,    intent(out) :: cape(ni, nj)
    real,    intent(out) :: cin(ni, nj)
    integer, intent(in)  :: source_level(ni, nj)
    integer, intent(in)  :: level_free_convection(ni, nj)
    integer, intent(in)  :: equilibrium_level(ni, nj)

    integer :: i, j, k

    do j = 1, nj
        do i = 1, ni
            if (source_level(i,j) == -1) then
                cape(i,j) = FILL_VALUE
                cin(i,j)  = FILL_VALUE
            else
                cape(i,j) = 0.0
                cin(i,j)  = 0.0
            end if

            if (source_level(i,j) /= -1) then
                do k = 1, nk
                    if (k >= source_level(i,j) .and. k < level_free_convection(i,j)) then
                        cin(i,j) = cin(i,j) + (Rd * (virtual_temp_parcel(i,j,k) - virtual_temp_environment(i,j,k)) * log(pressure_interface(i,j,k) / pressure_interface(i,j,k+1)))
                    end if

                    if (k >= level_free_convection(i,j) .and. k <= equilibrium_level(i,j)) then
                        cape(i,j) = cape(i,j) + (Rd * (virtual_temp_parcel(i,j,k) - virtual_temp_environment(i,j,k)) * log(pressure_interface(i,j,k) / pressure_interface(i,j,k+1)))
                    end if
                end do
            end if
        end do
    end do

end subroutine calculate_cape_cin </div>
    </div>

    <div id="ndsl" class="tab-panel">
      <div class="code-block">def calculate_cape_cin(
    virtual_temp_environment: FloatField,
    virtual_temp_parcel: FloatField,
    pressure_interface: FloatField,
    cape: FloatFieldIJ,
    cin: FloatFieldIJ,
    source_level: IntFieldIJ,
    level_free_convection: IntFieldIJ,
    equilibrium_level: IntFieldIJ,
):
    """Compute CAPE and CIN for a parcel originating at source_level.

    A source_level of -1 indicates no convection is occuring at this grid point, in which case the computation is skipped and CAPE/CIN are filled with FILL_VALUE.

    Some requirements:
        level_free_convection must be less than (lower than)
        equilibrium_level
        both level_free_convection and equilibrium_level must be larger than (higher than) source_level
        pressure_interface must have one more point in the vertical dimension than all other 3D non-interface fields

    Args:
        virtual_temp_environment (FloatField): virtual temperature of the environment
        virtual_temp_parcel (FloatField): virtual temperature of the parcel
        pressure_interface (FloatField): pressure at the grid interface
        cape (FloatFieldIJ): convective available potential energy
        cin (FloatFieldIJ): convective inhibition
        level_free_convection (IntFieldIJ): level of free convection for a parcel originating at source level
        equilibrium_level (IntFieldIJ): equilibrium level for a parcel originating at source level
    """
    with computation(FORWARD), interval(0, 1):
        cape = 0.0
        cin = 0.0

        if source_level == -1:
            # no convection, use fill value
            cape = FILL_VALUE
            cin = FILL_VALUE

    with computation(FORWARD), interval(...):
        # check if convection is enabled for the current grid point
        if source_level != -1:
            if K >= source_level and K < level_free_convection:
                cin = cin + (Rd * (virtual_temp_parcel - virtual_temp_environment) * (log(pressure_interface / pressure_interface[0, 0, 1])))

            if K >= level_free_convection and K <= equilibrium_level:
                cape = cape + (Rd * (virtual_temp_parcel - virtual_temp_environment) * (log(pressure_interface / pressure_interface[0, 0, 1])))

</div>
    </div>

    <div id="gen" class="tab-panel">
      <div class="code-block">Generated code here...</div>
    </div>

  </div>

  <div class="apps-section">
    <h2>Applications</h2>

    <div class="app-grid">
      <div class="app-card">
        <h3>GEOS</h3>
        <p>Accelerating global climate model simulations with NDSL.</p>
        <a class="btn" href="./ndsl_in_action/overview.md">Explore →</a>
      </div>

      <div class="app-card">
        <h3>NOAA</h3>
        <p>Supporting operational atmospheric modeling systems.</p>
        <a class="btn" href="#">Explore →</a>
      </div>

      <div class="app-card">
        <h3>Pace</h3>
        <p>Next-generation scalable atmospheric modeling.</p>
        <a class="btn" href="https://www.gfdl.noaa.gov/wp-content/uploads/2025/01/2025ReviewQ1-2_PaceDSLModeling.pdf">Explore →</a>
      </div>
    </div>
  </div>

  <div class="cta-card">
    <h2 style="margin-bottom:0.5rem; font-size:20px; font-weight:600; color:#185FA5;">
      Want to Learn More?
    </h2>
    <p style="max-width:800px; margin:0 auto; font-size:14px; color:#4b5563; line-height:1.5;">
      Do you want to learn more about NDSL, its capabilities, and decide whether the platform is suited for your needs? Talk to an NDSL expert.
    </p>
    <a class="btn" href="https://mm.smce.nasa.gov/astg/channels/smt" style="margin-top:1rem;">
      <svg xmlns="http://www.w3.org/2000/svg"
       width="15"
       height="15"
       viewBox="0 0 24 24"
       fill="none"
       stroke="currentColor"
       stroke-width="2"
       stroke-linecap="round"
       stroke-linejoin="round">
      <path d="M22 2L11 13"></path>
      <path d="M22 2L15 22L11 13L2 9L22 2Z"></path>
      </svg>
      Contact Us →
    </a>
  </div>

</div>

<script>
function switchTab(event, tabId) {
  const parent = event.target.closest('.code-card');
  parent.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
  parent.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  parent.querySelector('#' + tabId).classList.add('active');
  event.target.classList.add('active');
}
</script>

</body>
</html>

