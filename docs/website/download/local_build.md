# Build & Run GEOS on desktop

## Requirements

* `tcsh` for GEOS workflow
* A `gcc/g++/gfortran` compiler (gcc-12 is our current workhorse in Linux.  macOS has been tested using gcc-14 via Homebrew)
    * As of macOS Sequoia 15.5, `clang/clang++` can also be used in place of `gcc/g++`.
* If using macOS, [Homebrew](https://brew.sh/) is needed to install certain packages

```bash
# *** If building on Linux ***
sudo apt install g++ gcc gfortran
# If you have multiple installation, you might have to set the preferred
# compiler using the combo:
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12
# sudo update-alternatives --config gcc

# *** If building on macOS, use Homebrew to install gcc-14 to access gfortran***
brew install gcc@14
```

* A `cuda` toolkit if GPU backends will be used. (nvhpc/23.5+ or appropriate for your hardware)
* Use `lmod` to manipulate the stack with module like on HPCs: <https://lmod.readthedocs.io/en/latest/030_installing.html>

```bash
sudo apt install -y lua5.3 lua-bit32 lua-posix lua-posix-dev liblua5.3-0 liblua5.3-dev tcl tcl-dev tcl8.6 tcl8.6-dev libtcl8.6
wget https://github.com/TACC/Lmod/archive/refs/tags/8.7.37.tar.gz
tar -xvf 8.7.37.tar.gz
cd Lmod-8.7.37/
sudo ./configure --prefix=/opt/apps
sudo make install
sudo ln -s /opt/apps/lmod/lmod/init/profile        /etc/profile.d/z00_lmod.sh
sudo ln -s /opt/apps/lmod/lmod/init/cshrc          /etc/profile.d/z00_lmod.csh
sudo ln -s /opt/apps/lmod/lmod/init/profile.fish   /etc/fish/conf.d/z00_lmod.fish

# On macOS, lmod can be installed via Homebrew without the above commands
brew install lmod
```

* For macOS, install the following additional packages via Homebrew
    * `brew install automake autoconf libtool texinfo m4 cmake wget boost`
    * **IMPORTANT** : Examine `brew info libtool` and `brew info m4` to access `libtool` and `m4` easily on macOS

## Build our software stack

* Copy code from SMT-Nebulae from the `main` branch on the `sw_stack/discover/sles15`
* Change all `/discover` path to local
* In `build_0_on-node.sh` remove the `--with-gdrcopy=` line for UCX (you don't have it installed or/and it doesn't matter outside of HPCs)
* Then run the pipeline

```bash
./download.sh
./build_0_on-node.sh
./build_1_on-login.sh
```

The stack will take some time to install, especially the `baselibs` part of it. To check that baselibs has been built and installed, you can run

```bash
./verify_baselibs.sh
```

which should output

```none
-------+---------+---------+--------------
Config | Install |  Check  |   Package
-------+---------+---------+--------------
  ok   |   ok    |   --    | jpeg
  ok   |   ok    |   --    | zlib
  ok   |   ok    |   --    | szlib
  ok   |   ok    |   --    | hdf5
  ok   |   ok    |   --    | netcdf
  ok   |   ok    |   --    | netcdf-fortran
  ok   |   ok    |   --    | udunits2
  ok   |   ok    |   --    | fortran_udunits2
  ok   |   ok    |   --    | esmf
  ok   |   ok    |   --    | GFE
-------+---------+---------+--------------
```

### macOS Notes on building Software stack

* Use the `SMT-Nebulae` branch called `feature/functional_mac_build` that contains build scripts specific to macOS.

* Adjust `basics.sh` as follows

```diff
-source /Users/ckropiew/SMT-Nebulae/sw_stack/local/modulefiles/SMTStack/2024.08.LOCAL.sh
+source <YOUR PATH TO SMT-Nebulae>/sw_stack/local/modulefiles/SMTStack/2024.08.LOCAL.sh

-export DSLSW_BASELIBS_VER=7.27.0
+export DSLSW_BASELIBS_VER=8.14.0

-export DSLSW_BOOST_VER=1.76.0
-export DSLSW_BOOST_VER_STR=1_76_0 
+export DSLSW_BOOST_VER=1.88.0
+export DSLSW_BOOST_VER_STR=1_88_0

-export DSLSW_BASE=/Users/ckropiew/SMT-Nebulae/sw_stack/local/src/build
+export DSLSW_BASE=<YOUR PATH TO SMT-Nebulae>/sw_stack/local/src/build
```

* Adjust `2024.08.LOCAL.sh` as follows

```diff
-install_dir="/Users/ckropiew/GEOS_dependencies/install"
-ser_pkgdir="/Users/ckropiew/GEOS_dependencies/install/serialbox"
+install_dir="<YOUR PATH TO SMT-Nebulae>/sw_stack/local/src/install"
+ser_pkgdir="<YOUR PATH TO SMT-Nebulae>/sw_stack/local/src/install/serialbox"

-boost_pkgdir="$homebrew_install_dir/boost/1.86.0_1"
+boost_pkgdir="$homebrew_install_dir/boost/1.88.0"

-baselibs_pkgdir="$install_dir/baselibs-7.27.0/install/"
+baselibs_pkgdir="$install_dir/baselibs-8.14.0/install/"

-export PYTHONPATH="$ser_pkgdir/python":$PYTHONPATH
+#export PYTHONPATH="$ser_pkgdir/python":$PYTHONPATH

-py_pkgdir="/Library/Frameworks/Python.framework/Versions/3.11"
-export PATH="$py_pkgdir/bin":$PATH
-export LD_LIBRARY_PATH="$py_pkgdir/lib":$LD_LIBRARY_PATH
-export LD_LIBRARY_PATH="$py_pkgdir/lib64":$LD_LIBRARY_PATH
+py_pkgdir="<YOUR PATH TO miniconda>/envs/venv"
+#export PATH="$py_pkgdir/bin":$PATH
+#export LD_LIBRARY_PATH="$py_pkgdir/lib":$LD_LIBRARY_PATH
+#export LD_LIBRARY_PATH="$py_pkgdir/lib64":$LD_LIBRARY_PATH

# *** If clang/clang++ is the C/C++ compiler, make the changes below
-export CC=/opt/homebrew/opt/gcc@14/bin/gcc-14
-export CXX=/opt/homebrew/opt/gcc@14/bin/g++-14
+export CC=clang
+export CXX=clang++
```

* Install miniconda for access to Python
    * In the terminal, retrieve the installation file for miniconda via `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh`
    * Run `minconda.sh` to set up miniconda
    * Create a virtual Python v3.11.7 environment via miniconda.  The command below will create a virtual environment called `venv`.
        * `conda create -n venv python=3.11.7`
    * Whenever you want to activate the environment, run `conda activate venv`

* If you are using `clang` and `clang++` in place of `gcc` and `g++` for compiling, modify the setup of Baselibs in `build_baselibs.sh` as follows.

```diff
make -j ESMF_COMM=openmpi \
-    ESMF_COMPILER=gfortran \
+    ESMF_COMPILER=gfortranclang \
```

* Install OpenMPI by running the `build_ompi.sh` script
* Install serialbox by running the `build_serialbox.sh` script
* Install Baselibs by running the `build_baselibs.sh` script
* Install NDSL by running the `build_ndsl.sh` script

* Check that Baselibs is installed properly by running `./verify_baselibs.sh` and see if you get the output shown above when running `./verify_baselibs.sh`.

⚠️ `cffi`, `venv` and embedded Python interpreter ⚠️
Due to a non standard approach of non-Posix or Posix-adjacent (Darwin, Windows...) file pathing, the `venv` can be misdetected when using an embedded Python interpreter via CPython. The CPython crew is aware of the issue but noe fix is ready. To test build the following:

```c++
#include <stdio.h>
#include <Python.h>

int main(void)
{
    Py_InitializeEx(0);
    PyObject *f = PySys_GetObject((char *)"stderr");
    PyFile_WriteString("\nsys.path: ", f);
    PyFile_WriteObject(PySys_GetObject((char *)"path"), f, 0);
    PyFile_WriteString("\n\n", f);
    return 0;
}
```

with (replace the path to include & lib of Python)

```bash
gcc -o pex \
    -I/home/fgdeconi/.pyenv/versions/3.11.9/include/python3.11/ \
    run_python.c \
    -L/home/fgdeconi/.pyenv/versions/3.11.9/lib -lpython3.11
```

Then run `./pex`. This will print the python `sys.path` which _should_ have your virtual environment `sites-packages` in there. If it doesn't then it's the bug.

Our workaround is to use `conda` which seems to use a different approach, more sandboxy, which goes around the issue.

**macOS Note**: You may have to set the `DYLD_LIBRARY_PATH` environment variable to the `lib` path related to your conda Python environment to run the `pex` binary.

Some reference: <https://github.com/PyO3/pyo3/issues/1741>

### Notes on building Software stack with Linux Kernel v6.8+

Make the following adjustments before running the above pipeline...

* Adjust `basics.sh` as follows

```diff
-export DSLSW_OMPI_MAJOR_VER=4.1
-export DSLSW_OMPI_VER=${DSLSW_OMPI_MAJOR_VER}.6
+export DSLSW_OMPI_MAJOR_VER=5.0
+export DSLSW_OMPI_VER=${DSLSW_OMPI_MAJOR_VER}.2

-export DSLSW_BASELIBS_VER=7.17.1
+export DSLSW_BASELIBS_VER=7.23.0

-export DSLSW_BOOST_VER=1.76.0
-export DSLSW_BOOST_VER_STR=1_76_0
+export DSLSW_BOOST_VER=1.86.0
+export DSLSW_BOOST_VER_STR=1_86_0

+#CUDA_DIR=/usr/local/other/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/

-export FC=gfortran
+export FC=gfortran-12
-export CC=gcc
+export CC=gcc-12
-export CXX=g++
+export CXX=g++-12
```

* Modify the command to configure OpenMPI in `build_0_on-node.sh`

```diff
-./configure --prefix=$DSLSW_INSTALL_DIR/ompi \
-            --disable-libxml2 \
-            --disable-wrapper-rpath \
-            --disable-wrapper-runpath \
-            --with-pmix \
-            --with-cuda=$CUDA_DIR \
-            --with-cuda-libdir=$CUDA_DIR/lib64/stubs \
-            --with-ucx=$DSLSW_INSTALL_DIR/ucx \
-            --with-slurm \
-            --enable-mpi1-compatibility

+./configure --disable-wrapper-rpath --disable-wrapper-runpath \
+    CC=gcc-12 CXX=g++-12 FC=gfortran-12 \
+           --with-hwloc=internal --with-libevent=internal --with-pmix=internal --prefix=$DSLSW_INSTALL_DIR/ompi
```

As of September 6th, 2024, there is a fix that needs to be applied to ESMF file `ESMCI_Time.C` to resolve a GEOS runtime issue.

```diff
diff --git a/src/Infrastructure/TimeMgr/src/ESMCI_Time.C b/src/Infrastructure/TimeMgr/src/ESMCI_Time.C
index 78c21bce04..ab40a39379 100644
--- a/src/Infrastructure/TimeMgr/src/ESMCI_Time.C
+++ b/src/Infrastructure/TimeMgr/src/ESMCI_Time.C
@@ -1479,7 +1479,10 @@ namespace ESMCI{
      if (sN != 0) {
         sprintf(timeString, "%s%02d:%lld/%lld", timeString, s, sN, sD);
       } else { // no fractional seconds, just append integer seconds
-        sprintf(timeString, "%s%02d", timeString, s);
+       char *tmpstr;
+       tmpstr = strdup(timeString);
+        sprintf(timeString, "%s%02d", tmpstr, s);
+       free(tmpstr);
```

## Build GEOS

* Get and setup `mepo` which is pre-installed on `Discover`
    * `mepo` stands for "Multiple rEPOsitory": a Python tool held by Matt T in the SI team that handles the GEOS multi-repository strategy.  It can be installed using `pip`.

```bash
pip install mepo
```

* Git clone the root of GEOS then using `mepo`, you clone the rest of the components which pull on the `components.yaml` at the root of `GEOSgcm`
    * There is two sources for a component the default and the `develop` source
    * `v11.5.2` is our baseline, but we have `dsl/develop` branch where needed
    * We do not use `develop` for `GEOSgcm_App` or `cmake` since those have been setup for OpenACC but are not up-to-date for v11.5.2

```bash
git clone -b dsl/develop git@github.com:GEOS-ESM/GEOSgcm.git geos
cd geos
mepo clone --partial blobless
mepo develop env GEOSgcm GEOSgcm_GridComp FVdycoreCubed_GridComp pyFV3
```

* CMake GEOS, using the stack, our custom build `baselibs` and turning on the interface to `pyFV3` for the dynamical core
* Then `make install`. Grab a coffee (or 2)
    * We override the compiler with their `mpi` counterpart to make sure `libmpi` is pulled properly. CMake should deal with that, but there's some failure floating around.

```bash
module use -a SW_STACK/modulefiles
ml SMTStack/2024.04.00
export BASEDIR=SW_STACK/src/2024.04.00/install/baselibs-7.17.1/install/x86_64-pc-linux-gnu
# *** Note: Above BASEDIR path will be different for macOS ***

mkdir -f build
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
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_INSTALL_PREFIX=../install \
         -DPython3_EXECUTABLE=`which python3`

# *** To add Serialbox serialization to GEOS, add the following two flags to the above cmake command
#     1) -DBUILD_SERIALBOX_SER=ON 
#     2) -DSERIALBOX_ROOT=<Path to Serialbox Installation>

# *** If MKL isn't found by cmake, add the following flag to the above cmake command: -DMKL_INCLUDE_DIR=<MKL Include Path>

make -j48 install
```

Note: There may be an error that occurs when building GEOS on macOS that refers to `./src/Shared/@GMAO_Shared/LANL_Shared/CICE4/bld/makdep.c`.  This error points out that the `main` routine in `makdep.c` is missing a type.  Simply put `int` in front of `main` to resolve this problem (ex: `int main`).

**macOS Note**: To build GEOS using `clang`/`clang++`, there are additional OpenMP files needed that macOS does not have.  Here is how to get those files and set up the environment variables needed for building GEOS using `clang`/`clang++` on macOS.

* Use Homebrew to get the `libomp` package : `brew install libomp`
* Set up the following environment variables
    * `export OPENMP_CPPFLAGS="-I/opt/homebrew/opt/libomp/include`
    * `export OPENMP_LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp -Xpreprocessor -fopenmp`
    * `export GT4PY_EXTRA_COMPILE_OPT_FLAGS="-fbracket-depth=512`

⚠️ MKL Dependency⚠️

Some code in GEOS requires MKL. It's unclear which (apart from a random number generator in Radiation) and it seems to be an optional dependency. The `cmake` process will look for it. If you want to install it follow steps for the standalone [OneAPI MKL installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

* Get data (curtesy of Matt T.) in it's most compact form: TinyBC
    * WARNING: TinyBC is Matt way to move around a small example and/or develop on desktop, it's fragile and we shall treat it as a dev tool, offer help when needed.
    * _Do not rename it_

```bash
wget https://portal.nccs.nasa.gov/datashare/astg/smt/geos-fp/restarts/TinyBCs-GitV10.2024Apr04.tar.gz
tar -xvf TinyBCs-GitV10.2024Apr04.tar.gz
rm TinyBCs-GitV10.2024Apr04.tar.gz
cd TinyBCs-GitV10

```

* Make experiment, using TinyBC. First we will make our live easier by aliasing the relevant script. In your `.bashrc` (remember to re-bash)

```bash
alias tinybc_create_exp=PATH_TO/TinyBCs-GitV10/scripts/create_expt.py
alias tinybc_makeoneday=PATH_TO/TinyBCs-GitV10/scripts/makeoneday.bash
```

* Then we can make experiment for c24

```bash
cd PATH_TO_GEOS/install/bin
tinybc_create_exp --horz c24 --vert 72 --nonhydro --nooserver --gocart C --moist GFDL --link --expdir /PATH_TO_EXPERIMENT/ TBC_C24_L72
cd /PATH_TO_EXPERIMENT/TBC_C24_L72
tinybc_makeoneday noext # Temporary: we deactivate ExtData to go around a crash
```

⚠️ ⚠️ ⚠️
Bash might not work, if you see errors try to run under `tcsh`.
Prefer a PATH_TO_EXPERIMENT outside of GEOS to not mix experiment data & code.
⚠️ ⚠️ ⚠️

* The simulation is ready to run a 24h simulation (with the original Fortran) with

```bash
module load SMTStack/2024.04.04
./gcm_run.j
```

* To run the `pyFV3` version, modification to the `AGCM.rc` and `gcm_run.j` needs to be done _after_ the `tinybc_makeoneday`. This runs the `numpy` backend.

_AGCM.rc_

```diff
###########################################################
# dynamics options
# ----------------------------------------
DYCORE: FV3
 FV3_CONFIG: HWT
+ RUN_PYFV3: 1
AdvCore_Advection: 0
```

_gcm_run.j_

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

Remove the `source $GEOSBIN/g5_modules` to make sure your local environment is left clean.
