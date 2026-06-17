# Build & Run GEOS on Discover

## Deploying SMT stack via modules

On `Discover`, we have a pre-built stack that is accessible via `module load`.

- `module use -a <Path to SMT-Nebulae Repo>/sw_stack/discover/sles15/modulefiles` adds the various stack versions to your module list
- `ml SMTStack/YYYY.MM.PP` to load the targeted version of the stack

For development, a module `SMTStack/YYYY.MM.PP-no-venv` is available that doesn't pull `ndsl` or any other python packages.

Versions of all the software in the stack are present in `/discover/nobackup/projects/geosongpu/sw_sles15/HISTORY.md`

## Build & Run GEOS

- Git clone the root of GEOS then using `mepo`, you clone the rest of the components which pull on the `components.yaml` at the root of `GEOSgcm`
    - There are two sources for a component: the default and the `develop` source
    - `v11.5.2` is our baseline GEOS version (or tag), but we have `dsl/develop` branch where needed
    - We do not use `develop` for `GEOSgcm_App` or `cmake` since those have been setup for OpenACC but are not up-to-date for `v11.5.2`

```bash
git clone -b dsl/develop git@github.com:GEOS-ESM/GEOSgcm.git geos
cd geos
mepo clone --partial blobless
mepo develop env GEOSgcm GEOSgcm_GridComp FVdycoreCubed_GridComp pyFV3
```

- Use CMake to set up the GEOS makefile that uses the SMT stack.
- Then `make install`. Grab a coffee (or 2)
    - We override the compiler with their `mpi` counterpart to make sure `libmpi` is pulled properly. CMake should deal with that, but there's some failure floating around.

Single script to execute both cmake & make to install GEOS:

```bash
module use -a SW_STACK/modulefiles
ml SMTStack/2024.04.00

mkdir -p build
cd build
export TMP=GEOS_DIR/geos/build/tmp
export TMPDIR=$TMP
export TEMP=$TMP
mkdir $TMP
echo $TMP

export FC=mpif90
export CC=mpicc
export CXX=mpic++

cmake .. -DBASEDIR=$BASEDIR/Linux \
         -DCMAKE_Fortran_COMPILER=mpif90 \
         -DBUILD_PYFV3_INTERFACE=ON \
         -DCMAKE_INSTALL_PREFIX=../install \
         -DPython3_EXECUTABLE=`which python3`

make -j48 install
```

- Matt T. prepared us some GEOS-FP like data. Copy them from `/discover/nobackup/projects/geosongpu/geos_data/geos-fp/stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-cX-L137` where the `X` in `cX` is set as the resolution from {24, 180, 720}
- We need to set a soft link to the `GEOSgcm.x` executable from the GEOS `install/bin` directory, e.g.

```bash
cd <PATH_TO>/stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-cX-L137
ln -s <PATH_TO_geos>/install/bin/GEOSgcm.x GEOSgcm.x
```

- We then need to edit the `gcm_run.j` to make it local to our experience directory and have the job parameters match our account and the GPU partition.  The following assumes that we're running c24 GEOS-FP experiment.

*gcm_run.j*

```diff
#SBATCH --time=0:30:00
- #SBATCH --nodes=1 --ntasks-per-node=126
+ #SBATCH --nodes=1 --ntasks-per-node=48
- #SBATCH --job-name=stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-c24-L137_RUN
+ #SBATCH --job-name=NDSL-geos-fp
- #SBATCH --constraint=mil
+ #SBATCH --constraint=rome
- #SBATCH --account=g0620
+ #SBATCH --account=j1013
- #SBATCH --partition=preops
+ #SBATCH --partition=gpu_a100
- #SBATCH --qos=benchmark
+ #SBATCH --qos=4n_a100
- #SBATCH --mail-type=ALL
```

```diff
setenv SITE             NCCS
- setenv GEOSDIR          /gpfsm/dnb05/projects/p50/Models/GEOSgcm-v11.5.2-GNU-SLES15/GEOSgcm/install-Release
+ setenv GEOSDIR          PATH_TO_GEOS/install/
- setenv GEOSBIN          /gpfsm/dnb05/projects/p50/Models/GEOSgcm-v11.5.2-GNU-SLES15/GEOSgcm/install-Release/bin
+ setenv GEOSBIN          PATH_TO_GEOS/install/bin
- setenv GEOSETC          /gpfsm/dnb05/projects/p50/Models/GEOSgcm-v11.5.2-GNU-SLES15/GEOSgcm/install-Release/etc
+ setenv GEOSETC          PATH_TO_GEOS/install/etc
- setenv GEOSUTIL         /gpfsm/dnb05/projects/p50/Models/GEOSgcm-v11.5.2-GNU-SLES15/GEOSgcm/install-Release
+ setenv GEOSUTIL         PATH_TO_GEOS/install/
```

```diff
setenv  EXPID   stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-c24-L137
- setenv  EXPDIR  /discover/nobackup/mathomp4/Experiments/RunsForFlorian/stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-c24-L137
+ setenv  EXPDIR  PATH_TO_EXP_DIR/stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-c24-L137
- setenv  HOMDIR  /discover/nobackup/mathomp4/Experiments/RunsForFlorian/stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-c24-L137
+ setenv  HOMDIR  PATH_TO_EXP_DIR/stock-v11.5.2-1day-GNU-NH-GDATA-RRTMGP-GFDL-GF2020-OPS-c24-L137
```

- The pipeline of GEOSgcm calculates the numbers of process required based on the `NX/NY` given in `AGCM.rc`. Previously we modified the sbatch parameters to be `--ntasks-per-node=48` to fit within a Rome node. Therefore, we need to adapt the `AGCM.rc`

*AGCM.rc*

```diff
# Atmospheric Model Configuration Parameters
# ------------------------------------------
- NX: 4
+ NX: 2
- NY: 24
+ NY: 12
```

- Running the simulation on the compute node is now as simple as a `sbatch gcm_run.j`. The environment will be loaded via `PATH_TO_GEOS/@env/g5_modules`. Logged will be dumped in `slurm-XXXXX.out`

- To run the `pyFV3` version, modification to the `AGCM.rc` and `gcm_run.j`. This will run the `numpy` backend.

*AGCM.rc*

```diff
###########################################################
# dynamics options
# ----------------------------------------
DYCORE: FV3
 FV3_CONFIG: HWT
+ RUN_PYFV3: 1
AdvCore_Advection: 0
```

*gcm_run.j*

```diff
setenv RUN_CMD "$GEOSBIN/esma_mpirun -np "

setenv GCMVER `cat $GEOSETC/.AGCM_VERSION`
echo   VERSION: $GCMVER

+setenv FV3_DACEMODE           Python
+setenv GEOS_PYFV3_BACKEND     numpy
+setenv PACE_CONSTANTS         GEOS
+setenv PACE_FLOAT_PRECISION   32
+setenv PYTHONOPTIMIZE         1
+setenv PACE_LOGLEVEL          Debug

+setenv FVCOMP_DIR PATH_TO_GEOS/src/Components/@GEOSgcm_GridComp/GEOSagcm_GridComp/GEOSsuperdyn_GridComp/@FVdycoreCubed_GridComp/
+setenv PYTHONPATH $FVCOMP_DIR/python/interface:$FVCOMP_DIR/python/@pyFV3


#######################################################################
#             Experiment Specific Environment Variables
######################################################################
```
