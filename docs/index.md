# The NASA-NOAA Domain Specific Language (NDSL)
!!! info "Quick Links"
    - [User Manual](./documentation/ndsl_introduction.md)
    - [Community](community.md)
    - [NDSL in Action](./ndsl_in_action/overview.md)

The NASA–NOAA Domain-Specific Language (NDSL) is a modern framework for developing high-performance Earth system model components with a focus on portability, readability, and computational efficiency. NDSL enables scientists and developers to write expressive, maintainable code that can target multiple hardware architectures, including CPUs and GPUs, while preserving scientific integrity and performance.

NDSL is designed to bridge the gap between atmospheric science and modern software engineering practices. By abstracting backend-specific implementation details, NDSL allows users to focus on scientific development while leveraging scalable and portable computational infrastructure underneath.

This documentation serves as both an introduction for new users and a reference for active developers working within the NDSL ecosystem.

<h2 align="center">
Want to learn more about NDSL, its capabilities, and decide whether it is suited for your needs? 
</h2>
<div style="display: flex; justify-content: center;">

<a class="md-button md-button--primary"
   href="https://mm.smce.nasa.gov/astg/channels/smt"
   style="
      border-radius: 999px;
      padding: 0.72em 1.45em;
      font-size: 0.97rem;
   ">

  <svg xmlns="http://www.w3.org/2000/svg"
       width="17"
       height="17"
       viewBox="0 0 24 24"
       fill="none"
       stroke="currentColor"
       stroke-width="2"
       stroke-linecap="round"
       stroke-linejoin="round"
       style="vertical-align: middle; margin-right: 7px;">

    <path d="M22 2 11 13"/>
    <path d="M22 2 15 22 11 13 2 9 22 2z"/>

  </svg>

  Contact us
</a>

</div>





<!-- ## NDSL Highlights

<div class="carousel">

  <div class="slides fade">
    <img src="./img/scm_moist_bomex_hovmoller_gpu_72.png">
  </div>

  <div class="slides fade">
    <img src="./img/scm_moist_armtwp_ice_timeseries_gpu_72.png">
  </div>

</div>

<script>
let slideIndex = 0;
showSlides();

function showSlides() {
  let i;
  let slides = document.getElementsByClassName("slides");

  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }

  slideIndex++;

  if (slideIndex > slides.length) {
    slideIndex = 1;
  }

  slides[slideIndex-1].style.display = "block";

  setTimeout(showSlides, 3000);
}
</script> -->


!!! example ""

    === "Fortran"

        ``` fortran
        real function compute_alpha(del_CIN,ke)
        ! ------------------------------------------------ !
        ! Subroutine to compute proportionality factor for !
        ! implicit CIN calculation.                        !   
        ! ------------------------------------------------ !
          real   :: del_CIN, ke
          real*8 :: del_CIN8, ke8
          real*8 :: x0, x1

          integer  :: iteration

          x0 = 0._r8
          del_CIN8 = del_CIN
          ke8 = ke
          do iteration = 1, 10
              x1 = x0 - (exp(-x0*ke8*del_CIN8) - x0)/(-ke8*del_CIN8*exp(-x0*ke8*del_CIN8) - 1.)
              x0 = x1
          end do
          compute_alpha = x0

          return

        end function compute_alpha 
        ```

    === "NDSL"

        ``` python

        @gtfunction
        def compute_alpha(
            del_CIN: Float,
            ke: Float,
        ):
          """
          Subroutine to compute proportionality factor for
          implicit CIN calculation.

          Arguments:
              del_CIN [Float]: Difference between initial and final CIN calculations [J/kg]
              ke [Float]: Evaporative efficiency [?]

          Returns:
              compute_alpha [Float]: Proportionality factor for CIN calculation [unitless]

          reference Fortran: uwshcu.F90: function compute_alpha
          """
          x0: float64 = float64(0.0)
          del_CIN8_f64: float64 = float64(del_CIN)
          ke8_f64: float64 = ke
          iteration = 0
          while iteration < 10:
              x1 = x0 - (exp(-x0 * ke8_f64 * del_CIN8_f64) - x0) / (
                  -ke8_f64 * del_CIN8_f64 * exp(-x0 * ke8_f64 * del_CIN8_f64) - 1.0
              )
              x0 = x1
              iteration += 1

          compute_alpha = float32(x0)

          return compute_alpha
        ```

    === "Generated code"

        ``` markdown
        Add gen code here
        ```