# The NASA-NOAA Domain Specific Language (NDSL)
!!! note "Quick Links"
    - [User Guide](./documentation/ndsl_introduction.md)
    - [Community](community.md)
    - [NDSL in Action](./ndsl_in_action/GEOS-FP_results/summary.md)

The NASA–NOAA Domain-Specific Language (NDSL) is a modern framework for developing high-performance Earth system model components with a focus on portability, readability, and computational efficiency. NDSL enables scientists and developers to write expressive, maintainable code that can target multiple hardware architectures, including CPUs and GPUs, while preserving scientific integrity and performance.

NDSL is designed to bridge the gap between atmospheric science and modern software engineering practices. By abstracting backend-specific implementation details, NDSL allows users to focus on scientific development while leveraging scalable and portable computational infrastructure underneath.

NDSL brings together [GT4Py](https://github.com/GridTools/gt4py/) and [DaCe](https://github.com/spcl/dace/), two libraries developed for high-performance and portability. On top of these pillars, NDSL deploys a series of optimized APIs for common operations (e.g., halo exchange or domain decomposition) and tools to port existing models.

This documentation serves as both an introduction for new users and a reference for active developers working within the NDSL ecosystem.

### Next: [Getting Started](./quickstart.md)!


## NDSL Highlights

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
</script>