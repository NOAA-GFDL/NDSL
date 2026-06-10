# NDSL at NASA

NASA has used NDSL to accelerate targeted portions of the Goddard Earth Observing System (GEOS). To allow for optimal performance while ensuring numerical accuracy of
all new code, this process has been completed in two distinct steps: **numerical validation** and **benchmarking**.

##### Numerical Validation

The first step, porting the Fortran code to NDSL, was completed with two mandates: ensure numerical accuracy is maintained, while eliminating or rewriting unused,
redundant, or poorly organized code. In most cases, these dual mandates were fulfilled in their entirety; however, on occasion small errors (generally no more than
100 ULP, or roughly six orders of magnitude less than the observed value) on a small fraction (<5%) of grid points were considered acceptable.

Numerical validation has been performed by comparing the results of two GEOS runs: one using the original Fortran and another using the integrated NDSL component.
For the purposes of this work, numerical validation has been performed on both single column model (SCM) runs and general circulation model (GCM) runs. To avoid
conflating errors associated with different hardware types with true porting errors, numerical validation is performed with all optimization turned off - for both
the Fortran and NDSL.

KATRINA PUT YOUR PICTURES FOR SCM AND GCM HERE

i don't think we need a whole lot more words just pictures people can flip through that show

a. the component tested
b. the resolution (or gcm/scm, idk)

i feel like less is more here, so maybe we don't even show all the components. just a few. I don't forsee people having interest in flipping through 20 plots or something

##### Benchmarking

@FLORIAN ADD SPECIFIC INSTRUCTIONS ABOUT YOUR BENCHARKING STUFF

To ensure optimal performance, all benchmarking uses the best available optimization schemes.