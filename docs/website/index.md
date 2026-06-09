<style>
  /* disable the nav bar only on the homepage */
  /* this stype block must remain on this page so that other pages are not effected */
  .md-sidebar--primary {
  display: none !important;
  }
</style>

<div markdown>

<div style="margin-bottom: 1.2rem;" markdown>

# NASA-NOAA Domain Specific Language { .page-title }

A modern domain-specific language for portable, high-performance atmospheric modeling across CPUs, GPUs, and emerging computing architectures.

</div>

<div class="section-card" markdown>

## Install NDSL

Install the latest version of NDSL with a few commands and start writing code.

[NDSL v2026.03.00 →](./pages/install_ndsl.md){ .pretty-internal-link }
</div>

<div class="two-col" markdown>
<div class="section-card" markdown>

## Getting Started

New to NDSL? Walk through our beginner-friendly user manual to learn the core syntax and patterns.

[User Manual →](./pages/user_manual/ndsl_introduction.md){ .pretty-internal-link }
</div>

<div class="section-card" markdown>

## Community

Connect with developers, researchers, and contributors working on NDSL and atmospheric modeling tools.

[Join Community →](./community.md){ .pretty-internal-link }
</div>
</div>

<div class="code-card" markdown>

=== "Fortran"

    ```fortran
    subroutine calculate_cape_cin(virtual_temp_environment, virtual_temp_parcel, pressure_interface, cape, cin, source_level, level_free_convection, equilibrium_level, ni, nj, nk)

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

    end subroutine calculate_cape_cin
    ```

=== "NDSL"

    ```python
    def calculate_cape_cin(
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

        A source_level of -1 indicates no convection is occuring at this grid
        point, in which case the computation is skipped and CAPE/CIN are filled
        with FILL_VALUE.

        Some requirements:
            level_free_convection must be less than (lower than) equilibrium_level
            both level_free_convection and equilibrium_level must be larger than
            (higher than) source_level
            pressure_interface must have one more point in the vertical dimension
            than all other 3D non-interface fields

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
    ```

=== "Generated"

    Generated code here...

</div>

<div class="section-title" markdown>

## Applications

<div class="three-col" markdown>
<div class="section-card" markdown>

### GEOS

Accelerating global climate model simulations with NDSL.

[Explore →](./ndsl_in_action/overview.md){ .pretty-internal-link }
</div>

<div class="section-card" markdown>

### NOAA

Supporting operational atmospheric modeling systems.

</div>

<div class="section-card" markdown>

### Pace

Next-generation scalable atmospheric modeling.

[Explore →](https://www.gfdl.noaa.gov/wp-content/uploads/2025/01/2025ReviewQ1-2_PaceDSLModeling.pdf){ .pretty-internal-link }
</div>
</div>
</div>

<div class="section-card centered-text" markdown>

## Want to Learn More?

Do you want to learn more about NDSL, its capabilities, and decide whether the platform is suited for your needs? Talk to an NDSL expert.

[✉ Contact Us →](https://mm.smce.nasa.gov/astg/channels/smt){ .pretty-internal-link }
</div>

</div>
