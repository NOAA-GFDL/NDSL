Translate name needs to be the same as data
- ideally you would need to touch nothing
- for side-cases we have special treatment
    normally you only need an `__init__`

- you can create an overwrite file to manually set the threshold in you data directory, see [image1.png](image1.png)
- create a configuration (either from namelists) or write constants at run-time

- the compute_func will be called automatically in the test. If your names in the netcdf are matching the kwargs of your function directly, no further action required, see image 2

- if you need to rename it from the netcdf, you can use ["serialname"] - see image 3
- same for scalar inputs with parameters  - image 4
- out_vars can be modified from in-vars when something should not be here - ALSO image 4


- compute:
    setup input
    compute_func is called from compute_from_storage
    slice outputs

slight adaptations to every step should be doable, see image 5