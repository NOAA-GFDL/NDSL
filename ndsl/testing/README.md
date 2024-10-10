# Translate test

## Summary

NDSL exposes a "Translate" test system which allows the automatic numerical regression test against a pre-defined sets of NetCDFs.

To write a translate test, derive from `TranslateFortranData2Py`. The system works by matching name of Translate class and data, e.g.:

- if `TranslateNAME` is the name of the translate class
- then the name of the data should be `NAME-In.nc` for the inputs and `NAME-Out.nc` for outputs that'll be check.

The test runs via the `pytest` harness and can be triggered with the `pytest` commands.

Options ares:

- --insert-assert-print:         Print statements that would be substituted for insert_assert(), instead of writing to files
- --insert-assert-fail:          Fail tests which include one or more insert_assert() calls
- --backend=BACKEND:             Backend to execute the test with, can only be one.
- --which_modules=WHICH_MODULES: Whitelist of modules to run. Only the part after Translate, e.g. in TranslateXYZ it'd be XYZ
- --skip_modules=SKIP_MODULES:   Blacklist of modules to not run. Only the part after Translate, e.g. in TranslateXYZ it'd be XYZ
- --which_rank=WHICH_RANK:       Restrict test to a single rank
- --data_path=DATA_PATH:         Path of Netcdf input and outputs. Naming pattern needs to be XYZ-In and XYZ-Out for a test class named TranslateXYZ
- --threshold_overrides_file=THRESHOLD_OVERRIDES_FILE: Path to a yaml overriding the default error threshold for a custom value.
- --print_failures:              Print the failures detail. Default to True.
- --failure_stride=FAILURE_STRIDE: How many indices of failures to print from worst to best. Default to 1.
- --grid=GRID:                   Grid loading mode. "file" looks for "Grid-Info.nc", "compute" does the same but recomputes MetricTerms, "default" creates a simple grid with no metrics terms. Default to "file".
- --topology=TOPOLOGY            Topology of the grid. "cubed-sphere" means a 6-faced grid, "doubly-periodic" means a 1 tile grid. Default to "cubed-sphere".
- --multimodal_metric:           Use the multi-modal float metric. Default to False.

More options of `pytest` are available when doing `pytest --help`.

## Metrics

There is three state of a test in `pytest`: FAIL, PASS and XFAIL (expected fail). To clear the PASS status, the output data contained in `NAME-Out.nc` is compared to the computed data via the `TranslateNAME` test. Because this system was developped to port Fortran numerics to many targets (mostly C, but also Python, and CPU/GPU), we can't rely on bit-to-bit comparison and have been developping a couple of metrics.

### Legacy metric

The legacy metric was used throughout the developement of the dynamical core and microphysics scheme at 64-bit precision. It tries to solve differences over big and small amplitutde values with a single formula that goes as follows: $\abs{computed-reference}/reference$ where `reference` has been purged of 0.
NaN values are considered no-pass.
To pass the metric has to be lower than `1e-14`, any value lower than `1e-18` will be considered pass by default. The pass threshold can be overriden (see below).

### Multi-modal metric

Moving to mixed precision code, the legacy metric didn't give enough flexibility to account for 32-bit precision errors that could accumulate. Another metric was built with the intent of breaking the one-fit-all concept and giving back flexibility. The metric is a combination of three differences:

- _Absolute Difference_ ($`|computed-reference|<threshold`$): the absolute difference between the reference value and the computed value. Good for small amplitude, decays to direct comparison in higher amplitude. Default threshold is `1e-13` for 64-bit, `1e-10` at 32-bit.
- _Relative Difference_ ($`|computed-reference|<threshold*\max{|computed|, |reference|}`$): the difference relative to the maximum value. This can be seen at a % of error. Good for high amplitutde value, decay to direct comparison at smaller amplitude. Default is `0.0001%`
- _ULP Difference_ ($`|computed-reference|/\spacing{\max{|computed|, |reference|}}<=threshold`$): Unit of Least Precision (ULP) can be shortly describe as a way to quantify the space between two describable floating points. This is useful to measure differences that are in the "noise" of the machine representation. Default threshold is 1, meaning the two values are virtually indistiguinshible.

## Results

A summary of all results is dumped in a `.translate-errors` directory. When failing, this folder carries also data (netcdf) and logs helpful to explore the errors.

## Overrides

### Threshold overrides

`--threshold_overrides_file` takes in a yaml file with error thresholds specified for specific backend and platform configuration. Currently, two types of error overrides are allowed: maximum error and near zero.

For maximum error, a blanket `max_error` is specified to override the parent classes relative error threshold.

For near zero override, `ignore_near_zero_errors` is specified to allow some fields to pass with higher relative error if the absolute error is very small. Additionally, it is also possible to define a global near zero value for all remaining fields not specified in `ignore_near_zero_errors`. This is done by specifying `all_other_near_zero=<value>`.

Override yaml file should have one of the following formats:

### One near zero value for all variables

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   near_zero: <value>
   ignore_near_zero_errors:
    - <var1>
    - <var2>
    - ...
```

### Variable specific near zero value

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   ignore_near_zero_errors:
    <var1>:<value1>
    <var2>:<value2>
    ...
```

### [optional] Global near zero value for remaining fields

```Stencil_name:
 - backend: <backend>
   max_error: <value>
   ignore_near_zero_errors:
    <var1>:<value1>
    <var2>:<value2>
   all_other_near_zero:<global_value>
    ...
```

where fields other than `var1` and `var2` will use `global_value`.

### Multimodal overrides

```Stencil_name:
 - backend: <backend>
   multimodal:
    absolute_eps: <value>
    relative_fraction: <value>
    ulp_threshold: <value>
```
