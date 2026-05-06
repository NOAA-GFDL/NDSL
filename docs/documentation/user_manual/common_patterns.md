# Common Patterns

Now that we have introduced the core features of NDSL and highlighted features which enable more
complex coding structures, it is prudent to provide examples of how NDSL is implemented.

Below are a plethora of patterns which may be useful when developing code for weather and
climate models. These examples have been generalized, but should be broadly applicable to a
variety of situations. Each example has a "source" Fortran program, and a "ported" NDSL
implementation.

## Writing to a `Z_INTERFACE_DIM` Variable

If working with `Z_INTERFACE_DIM` quantities, one will inevitably need to write to the "extra"
point. The easiest way to achieve this is by declaring your `compute_dims` to operate on
the `Z_INTERFACE_DIM` instead of the `Z_DIM`, and then restrict your interval and offset `Z_DIM`
stencil reads accordingly:

Example "Fortran code":
    ``` fortran linenums="1"
    program compute_height
        implicit none

        ! setup domain
        integer, parameter :: X_DIM = 10
        integer, parameter :: Y_DIM = 10
        integer, parameter :: Z_DIM = 10

        ! allocate scalars
        integer :: i, j, k, p_crit
        real :: seed, total

        ! allocate arrays
        real, dimension(X_DIM, Y_DIM, Z_DIM) :: d_height
        real, dimension(X_DIM, Y_DIM, Z_DIM+1) :: height

        ! initalize data
        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 1, Z_DIM
                    call RANDOM_NUMBER(seed)
                    ! generate a random change in height for each layer in the range 125:175
                    d_height(i, j, k) = 125 + FLOOR(51*seed)
                enddo
            enddo
        enddo

        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 2, Z_DIM+1
                    height(i, j, k) = height(i, j, k-1) + d_height(i, j, k-1)
                enddo
            enddo
        enddo

        PRINT *, height(1, 1, :)

    end program compute_height
    ```

Example "NDSL Code"

    ``` py linenums="1"
    from ndsl.dsl.gt4py import (
        computation,
        interval,
        PARALLEL,
    )
    from ndsl import StencilFactory
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
    from ndsl.dsl.typing import FloatField
    import random


    def compute_height(
        height: FloatField,
        d_height: FloatField,
    ):

        with computation(PARALLEL), interval(1, None):
            # d_height is offset down one because this field operates on the Z_DIM,
            # while the stencil is computing on the Z_INTERFACE_DIM
            height = height[0, 0, -1] + d_height[0, 0, -1]


    class Code:
        def __init__(self, stencil_factory: StencilFactory):

            # build stencil
            self.constructed_stencil = stencil_factory.from_dims_halo(
                func=compute_height,
                compute_dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM],
            )

        def __call__(
            self,
            height: FloatField,
            d_height: FloatField,
        ):
            # execute stencil
            self.constructed_stencil(
                height,
                d_height,
            )


    if __name__ == "__main__":

        # setup domain and generate factories
        domain = (10, 10, 10)
        nhalo = 0
        stencil_factory, quantity_factory = get_factories_single_tile(
            domain[0],
            domain[1],
            domain[2],
            nhalo,
            backend="dace:cpu",
        )

        # initialize data
        height = quantity_factory.zeros([X_DIM, Y_DIM, Z_INTERFACE_DIM], "n/a")

        d_height = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        for i in range(d_height.field.shape[0]):
            for j in range(d_height.field.shape[1]):
                for k in range(d_height.field.shape[2]):
                    d_height.field[i, j, k] = round(random.uniform(125, 175))

        # build stencil
        code = Code(stencil_factory)

        # execute stencil
        code(height, d_height)

        print(height.field[0, 0, :])
    ```

## K-Dimension Dependent Computations

In weather and climate modeling, it is often necessary to identify a specific level, then perform
one or more operations based on that level (e.g. identify LCL, compute convective parameters).

Example "Fortran Code":

    ``` fortran linenums="1"
    program average_below_level
        implicit none
    
        ! setup domain
        integer, parameter :: X_DIM = 10
        integer, parameter :: Y_DIM = 10
        integer, parameter :: Z_DIM = 10

        ! allocate scalars
        integer :: i, j, k, p_crit
        real :: seed, total

        ! allocate arrays
        integer, dimension(X_DIM, Y_DIM, Z_DIM) :: temperature, pressure
        integer, dimension(X_DIM, Y_DIM) :: desired_level
        real, dimension(X_DIM, Y_DIM) :: average_temperature

        ! initalize data
        p_crit = 500
        do k = 1, Z_DIM
            pressure(:, :, k) = 1000 - (k-1) * 100 ! pressure decreases by 100mb per level
            do j = 1, Y_DIM
                do i = 1, X_DIM
                    call RANDOM_NUMBER(seed)
                    ! generate a random temperature in the range 20:30
                    temperature(i, j, k) = 20 + FLOOR(11*seed)
                enddo
            enddo
        enddo


        ! determine k level where pressure meets a critical value
        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 1, Z_DIM
                    if (pressure(i, j, k) < p_crit) then
                        desired_level(i, j) = k
                        ! in this simplified example desired level is the same everywhere, but one can
                        ! imagine a case where this is spatially variable (e.g. determine LCL or LFC)
                        exit
                    endif
                enddo
            enddo
        enddo
        
        ! calculate average based on desired_level
        do i = 1, X_DIM
            do j = 1, Y_DIM
                total = 0
                do k = 1, desired_level(i, j)
                    total = total + temperature(i, j, k)
                enddo
                average_temperature(i, j) = total/desired_level(i, j)
            enddo
        enddo

        PRINT *, average_temperature

    end program average_below_level
    ```

Example "NDSL Code"

    ``` py linenums="1"
    from ndsl.dsl.gt4py import (
        FORWARD,
        computation,
        interval,
    )
    from gt4py.cartesian.gtscript import THIS_K
    from ndsl import StencilFactory
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM
    from ndsl.dsl.typing import FloatField, FloatFieldIJ, IntFieldIJ, Int, Float
    import random


    def average_layers(
        temperature: FloatField,
        pressure: FloatField,
        average_temperature: FloatFieldIJ,
        desired_level: IntFieldIJ,
    ):
        from __externals__ import p_crit, k_end

        with computation(FORWARD), interval(...):
            # determine critical level
            if pressure > p_crit:
                desired_level = THIS_K

        with computation(FORWARD), interval(...):
            # sum all temperatures below critical level
            if THIS_K <= desired_level:
                average_temperature = average_temperature + temperature
            # calculate the average on the final level
            if THIS_K == k_end:
                average_temperature = average_temperature / desired_level


    class Code:
        def __init__(self, stencil_factory: StencilFactory):

            self.constructed_stencil = stencil_factory.from_dims_halo(
                func=average_layers,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
                externals={
                    "p_crit": 500,
                },
            )

        def __call__(
            self,
            temperature: FloatField,
            pressure: FloatField,
            average_temperature: FloatFieldIJ,
            desired_level: IntFieldIJ,
        ):
            self.constructed_stencil(
                temperature,
                pressure,
                average_temperature,
                desired_level,
            )


    if __name__ == "__main__":

        # setup domain and generate factories
        domain = (10, 10, 10)
        nhalo = 0
        stencil_factory, quantity_factory = get_factories_single_tile(
            domain[0],
            domain[1],
            domain[2],
            nhalo,
            backend="dace:cpu",
        )

        # initialize data
        temperature = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        for i in range(temperature.field.shape[0]):
            for j in range(temperature.field.shape[1]):
                for k in range(temperature.field.shape[2]):
                    temperature.field[i, j, k] = round(random.uniform(20, 30))

        pressure = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        for k in range(pressure.field.shape[2]):
            pressure.field[:, :, k] = 1000 - k * 100

        average_temperature = quantity_factory.zeros([X_DIM, Y_DIM], "n/a")
        desired_level = quantity_factory.zeros([X_DIM, Y_DIM], "n/a", dtype=Int)

        # build stencil
        code = Code(stencil_factory)

        # execute stencil
        code(temperature, pressure, average_temperature, desired_level)

        print(average_temperature.field)
    ```

## Global Tables

There may be situations where a table is needed for referencing throughout the execution of a
component/model. It is highly unlikely that these tables will conform to the grid system used by
the rest of the model in any way. Generating these tables, therefore, requires a somewhat
unconventional use of systems. The following example is an adaptation of code used to generate one
of the saturation vapor pressure tables in the GEOS model, and displays the flexibility of
NDSL systems:

Example "Fortran Code"

    ``` fortran linenums="1"
    program make_table

        implicit none

        integer, parameter :: size = 1000 ! dimension not related to the size of the encompassing model

        real :: constant_one, constant_two
        real, dimension(size) :: table

        integer :: l

        constant_one = 10
        constant_two = 2

        ! construct a table based on temperature input

        do l = 1, size
            table(l) = log(0.1 * (l-1) / constant_one) + constant_two
        enddo

        print *, table

    end program make_table
    ```

Example "NDSL Code"

    ``` py linenums="1"
    from ndsl.dsl.gt4py import (
        computation,
        interval,
        log,
        PARALLEL,
        GlobalTable,
    )
    from ndsl import StencilFactory, QuantityFactory
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM
    from ndsl.dsl.typing import FloatField, Float
    from gt4py.cartesian.gtscript import THIS_K


    # NOTE Ideally, these next two statements should be done elsewhere (perhaps
    # in a constants and types file, respectively) and imported
    # declare table size
    table_size = 1000
    # define table type
    GlobalTable_local_type = GlobalTable[(Float, int(table_size))]


    def _compute_table(table: FloatField):
        from __externals__ import constant_one, constant_two

        with computation(PARALLEL), interval(1, None):
            table = log(0.1 * THIS_K / constant_one) + constant_two


    def _use_table(out_field: FloatField, table: GlobalTable_local_type):
        with computation(PARALLEL), interval(...):
            out_field = 2 * table.A[150]


    class ConstructTable:
        def __init__(self, stencil_factory: StencilFactory, quantity_factory: QuantityFactory):

            # build stencil
            self.compute_table = stencil_factory.from_dims_halo(
                func=_compute_table,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
                externals={
                    "constant_one": 10,
                    "constant_two": 2,
                },
            )

            self._table = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        def __call__(self):
            # execute stencil
            self.compute_table(
                self._table,
            )

            # return a GlobalTable object

        @property
        def table(self):
            return self._table.field[0, 0, :]


    if __name__ == "__main__":
        # consider the following model domain and factories are presnet
        domain = (10, 10, 10)
        nhalo = 0
        stencil_factory, quantity_factory = get_factories_single_tile(
            domain[0],
            domain[1],
            domain[2],
            nhalo,
            backend="dace:cpu",
        )

        # we must create a new set for this table calculation, because we require an off-grid dimension
        domain_table = (1, 1, table_size)
        nhalo_table = 0
        stencil_factory_table, quantity_factory_table = get_factories_single_tile(
            domain_table[0],
            domain_table[1],
            domain_table[2],
            nhalo_table,
            backend="debug",
        )

        # initialize data
        temperature = quantity_factory_table.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        for l in range(temperature.field.shape[2]):
            temperature.field[:, :, l] = 10 + 0.01 * (l + 1)

        # build stencil
        construct_table = ConstructTable(stencil_factory_table, quantity_factory_table)

        # execute stencil
        construct_table()

        # construct stencil that will use the table
        use_table = stencil_factory.from_dims_halo(
            func=_use_table,
            compute_dims=[X_DIM, Y_DIM, Z_DIM],
        )

        # # initalize array for calculation using table
        out_field = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        use_table(out_field, construct_table.table)

        print(construct_table.table)
        print("Done 🚀")
    ```

These tables must be subsequently referenced as a local variant of the `GlobalTable` type,
informing the system that the object features a single off-grid axis, and may not conform
to the larger model grid specifications:

Example "NDSL Code"

    This type is defined using the following pattern:
    ``` py
    GlobalTable_local_type = GlobalTable[(<data type (Float/Int/Bool)>, int(table_size))]
    ```

    And accurately typed and referenced within the stencil:
    ``` py
    def stencil(data: FloatField, table: GlobalTable_local_type):
        with computation(PARALLEL), interval(...):
            data = table.A[desired_index]
    ```

## Optional Inputs

In Fortran - and in traditional Python - it is possible to have optional inputs to a function. NDSL does not
support optional field inputs (a quantity with one or more dimensions) but it is possible to create 
optional scalar inputs.

For a field, the best way to create an "optional" input in NDSL is to create a situation where an input
(which is always suppled) has the option of being used - a calculation tied to some boolean, perhaps.
Furthermore, the field supplied need not be relevant. If the option will never be used, a dummy
field can be passed to satisfy the API of the function and simply never touched (or touched, with the result
discarded upon completion of the function).

For scalar inputs, the notation is quite similar to that of traditional Python. Note that NDSL requires
that all scalars - if not suppled - have a default value, and that default value must be of the declared type
(see line 19 of the example below):

Example "NDSL Code"

    ``` py linenums="1"

    from ndsl.dsl.gt4py import (
        computation,
        interval,
        PARALLEL,
    )
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM
    from ndsl.dsl.typing import FloatField, Float
    from ndsl import StencilFactory, orchestrate
    import numpy as np

    domain = (2, 2, 5)

    stencil_factory, quantity_factory = get_factories_single_tile(
        domain[0], domain[1], domain[2], 0, backend="numpy"
    )


    def the_stencil(in_field: FloatField, out_field: FloatField, factor: Float = 2.0):
        with computation(PARALLEL), interval(...):
            out_field = in_field * factor


    class Code:
        def __init__(self, stencil_factory: StencilFactory):
            orchestrate(obj=self, config=stencil_factory.config.dace_config)
            self._the_stencil = stencil_factory.from_dims_halo(
                func=the_stencil,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
            )

        def __call__(self, in_field: FloatField, out_field: FloatField):
            self._the_stencil(in_field, out_field)


    if __name__ == "__main__":
        in_arr = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM, Z_DIM],
            units="inputs",
        )
        sz = domain[0] * domain[1] * domain[2]
        in_arr.view[:] = np.arange(sz, dtype=Float).reshape(domain)

        out_arr = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="outputs")

        code = Code(stencil_factory)
        code(in_arr, out_arr)

        print("Done 🚀")
    ```

## Exit Statements and Internal Masks

Often there are times where it is necessary to control the execution of specific code paths
dynamically within code execution (e.g. computing precipitation if there is sufficient liquid
water present). In Fortran, this can be done easily with an `exit` statement. In NDSL, the Python equivalent
`break` statement is strictly forbidden, as there is no way to stop a computation early. Instead, the correct
way to control execution of different coordinates on the X/Y plane is by using two dimensional boolean fields,
which act as masks to turn on/off chunks of code on a per-point basis.

Example "Fortran Code"

    ``` fortran linenums="1"
    program conditional_calculation
        implicit none

        ! setup domain
        integer, parameter :: X_DIM = 5
        integer, parameter :: Y_DIM = 5
        integer, parameter :: Z_DIM = 3

        ! allocate scalars
        integer :: i, j, k
        real :: seed, critical_value

        ! allocate arrays
        integer, dimension(X_DIM, Y_DIM, Z_DIM) :: in_data, out_data
        logical, dimension(X_DIM, Y_DIM) :: mask

        critical_value = 5
        ! initalize data
        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 1, Z_DIM
                    call RANDOM_NUMBER(seed)
                    ! generate a random number between 0 and 10
                    in_data(i, j, k) = nint(10*seed)
                enddo
            enddo
        enddo

        
        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 1, Z_DIM
                    ! if data is above a critical value anywhere in the column
                    ! perform the subsequenc calculation on the entire column
                    if (in_data(i, j, k) > critical_value) then
                        mask(i, j) = .true.
                        exit
                    endif
                enddo
                if (mask(i,j) .eqv. .true.) then
                    out_data(i,j,:) = 10*in_data
                else
                    out_data(i,j,:) = -999
                endif
            enddo
        enddo

        PRINT *, "Input data: ", in_data
        PRINT *, "Mask data: ", mask
        PRINT *, "Output data: ", out_data

    end program conditional_calculation
    ```

Example "NDSL Code"

    ``` py linenums="1"
    from ndsl.dsl.gt4py import PARALLEL, FORWARD, computation, interval, exp, function
    from ndsl import StencilFactory
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM
    from ndsl.dsl.typing import FloatField, BoolFieldIJ, Bool
    import random


    def check_value(data: FloatField, mask: BoolFieldIJ):
        from __externals__ import critical_value

        with computation(FORWARD), interval(...):
            # if the data surpasses a critical value anywhere in the column, set the mask to true
            if data > critical_value:
                mask = True


    def computation_stencil(data_in: FloatField, data_out: FloatField, mask: BoolFieldIJ):
        with computation(PARALLEL), interval(...):
            if mask == True:
                data_out = 2 * exp(data_in)
            else:
                data_out = -999


    class DoSomeMath:
        def __init__(self, stencil_factory: StencilFactory):

            self.check_value = stencil_factory.from_dims_halo(
                func=check_value,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
                externals={
                    "critical_value": 5,
                },
            )

            self.computation_stencil = stencil_factory.from_dims_halo(
                func=computation_stencil,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
            )

        def __call__(self, in_field: FloatField, out_field: FloatField, mask_field: BoolFieldIJ):
            self.check_value(in_field, mask_field)
            self.computation_stencil(in_field, out_field, mask_field)


    if __name__ == "__main__":

        domain = (5, 5, 3)
        nhalo = 0
        stencil_factory, quantity_factory = get_factories_single_tile(
            domain[0],
            domain[1],
            domain[2],
            nhalo,
        )

        do_some_math = DoSomeMath(stencil_factory)

        in_field = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        for i in range(in_field.field.shape[0]):
            for j in range(in_field.field.shape[1]):
                for k in range(in_field.field.shape[2]):
                    in_field.field[i, j, k] = round(random.uniform(0, 10), 2)

        out_field = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        mask_field = quantity_factory.zeros([X_DIM, Y_DIM], "n/a", dtype=Bool)

        do_some_math(in_field, out_field, mask_field)

        print(f"Input data: {in_field.field}")
        print(f"Mask data: {mask_field.field}")
        print(f"Output data: {out_field.field}")
    ```

## Nested K Loops

Occasionally there may be situations when a nested K loop is required (such as
accumulating precipitation in a column). NDSL does not have the ability to nest computation/interval
statements; however, such a calculation can be performed using a `while` loop and clever indexing within
a single computation/iteration statement:

Below is an example of a nested K loop from the lagrangian_contributions stencil. It is not 
currently possible in NDSL to nest a `with computation(PARALLEL)` within a 
`with computation(PARALLEL)`, however a `while`loop can be used to create a nested K loop 
(lines x-x).

Example "Fortran Code"

    ```fortran linenums="1"
    program nested_k_loop
        implicit none

        ! setup domain
        integer, parameter :: X_DIM = 5
        integer, parameter :: Y_DIM = 5
        integer, parameter :: Z_DIM = 10

        ! allocate scalars
        integer :: i, j, k, kk
        real :: seed

        ! allocate arrays
        integer, dimension(X_DIM, Y_DIM, Z_DIM) :: in_data, out_data

        ! initalize data
        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 1, Z_DIM
                    call RANDOM_NUMBER(seed)
                    ! generate a random value between 0 and 10
                    in_data(i, j, k) = nint(10*seed)
                enddo
            enddo
        enddo

        out_data(:,:,:) = 0
        do i = 1, X_DIM
            do j = 1, Y_DIM
                do k = 1, Z_DIM
                    do kk = 1, k
                        ! sum all values at and below the current point
                        out_data(i,j,k) = out_data(i,j,k) + in_data(i,j,kk)
                    enddo
                enddo
            enddo
        enddo

        PRINT *, "Input data: ", in_data(1,1,:)
        PRINT *, "Output data: ", out_data(1,1,:)

    end program nested_k_loop
    ```

Example "NDSL Code"

    ``` py linenums="1"
    from ndsl.dsl.gt4py import (
        computation,
        interval,
        PARALLEL,
    )
    from gt4py.cartesian.gtscript import THIS_K
    from ndsl import StencilFactory
    from ndsl.boilerplate import get_factories_single_tile
    from ndsl.constants import X_DIM, Y_DIM, Z_DIM
    from ndsl.dsl.typing import FloatField
    import random


    def nested_k_calculation(
        in_data: FloatField,
        out_data: FloatField,
    ):

        with computation(PARALLEL), interval(...):
            nested_k_index = 0
            while nested_k_index <= THIS_K:
                out_data = out_data + in_data.at(K=nested_k_index)
                nested_k_index += 1


    class Code:
        def __init__(self, stencil_factory: StencilFactory):

            # build stencil
            self.constructed_stencil = stencil_factory.from_dims_halo(
                func=nested_k_calculation,
                compute_dims=[X_DIM, Y_DIM, Z_DIM],
            )

        def __call__(
            self,
            in_data: FloatField,
            out_data: FloatField,
        ):
            # execute stencil
            self.constructed_stencil(
                in_data,
                out_data,
            )


    if __name__ == "__main__":

        # setup domain and generate factories
        domain = (5, 5, 10)
        nhalo = 0
        stencil_factory, quantity_factory = get_factories_single_tile(
            domain[0],
            domain[1],
            domain[2],
            nhalo,
            backend="debug",
        )

        # initialize data
        out_data = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")

        in_data = quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], "n/a")
        for i in range(in_data.field.shape[0]):
            for j in range(in_data.field.shape[1]):
                for k in range(in_data.field.shape[2]):
                    in_data.field[i, j, k] = round(random.uniform(0, 10))

        # build stencil
        code = Code(stencil_factory)

        # execute stencil
        code(in_data, out_data)

        print(in_data.field[0, 0, :])
        print(out_data.field[0, 0, :])
    ```

Nested K loops provide an excellent example of how thinking outside the box - and possessing a willingness
to re-approach traditional coding practices - allows for a much wider application of the software and reveals
that the rules put in place by NDSL are not necessarily as restrictive as they may seem.

## Goto Statements

In Fortran, the `goto` construct allows the user to jump to another portion of the code. NDSL is
incapable of "jumping" from one portion of code to another; stencils are always executed linearly
and completely. Since `goto` statement are tremendously flexile, there is no "standard" way of translating
a piece of code with a `goto` statement into NDSL. Indeed, the presence of `goto` statements often signals
a non-parallelizable code structure, which requires refactoring to be implemented in NDSL. During this
process, however, the aforementioned patterns - particularly the use of masks - may be extremely useful.
