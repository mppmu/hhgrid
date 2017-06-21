[![Build Status](https://travis-ci.org/mppmu/hhgrid.svg?branch=master)](https://travis-ci.org/mppmu/hhgrid)

# hhgrid

gghh 2-loop virtual amplitude interfered with 1-loop amplitude.

Returns V_fin using the conventions specified in Eq(2.5) of https://arxiv.org/pdf/1703.09252.pdf

As required by `POWHEG` the born factor of alpha_s^2 is included in the grid. We evaluate alpha_s at a scale of `m_HH/2` using `LHAPDF` with the `PDF4LHC15_nlo_100` set.

Physical parameters are set to:
* `G_F = 1.1663787e-5 GeV^(-2)` (Fermi constant)
* `m_T = 173 GeV` (top-quark mass)
* `m_H = 125 GeV` (Higgs boson mass)

## Installation

Pre-requisites:
* `python` (tested for 2.7.13)
* `numpy` (tested for 1.11.1)
* `scipy` (tested for 0.17.1)

Assuming you have `python` you can check for the availability of `numpy` and `scipy` with the commands:
```shell
python -c "import numpy"
python -c "import scipy"
```
If no output is given from these commands then the packages are available.
 
To use the Python/C API you need to build the `hhgrid` library. 
 To do this run the usual `autotools` commands (replacing `<build-directory>` with the path you would like to install the library to):
```shell
autoreconf -i
./configure --prefix=<build-directory>
make
make check
make install
```
If you do not have a fortran compiler installed and do not want to use the grid from a fortran program, pass the `--disable-fortran` flag to the configure script.
 We strongly advise to run the `make check` command. This will call the grid for a few selected points and validate that the correct result is produced. 
 
You should now add the directory containing the `hhgrid-config` script to your `PATH` and the `python` data directory to your `PYTHONPATH`. This may be accomplished by adding the following lines to your `.bash_profile` or `.bashrc` (replacing `<build-directory>` with the path you installed the library to):
```shell
export PATH="<build-directory>/bin":$PATH
export PYTHONPATH="<build-directory>/share/hhgrid":$PYTHONPATH
```

## Python/C API Usage

Here we demonstrate how to access the grid with one of the example programs `c_example.c`, `cpp_example.cpp` or `f90_example.f90` in the `examples` directory,
this will check that everything is set up correctly and show the steps required to call the grid from your own program.

The `hhgrid-config` script helps to obtain the correct flags for compiling and linking the program.

If you would like to use the C interface you can compile the example program `examples/c_example.c`:
```shell
cc examples/c_example.c -o c_example.x `hhgrid-config --cflags --libs`
```

If you would like to use the C++ interface you can compile the test program `examples/cpp_example.cpp`:
```shell
c++ examples/cpp_example.cpp -o cpp_example.x `hhgrid-config --cflags --libs`
```

If you would like to use the C++ interface you can compile the test program `examples/f90_example.f90`:
```shell
gfortran examples/f90_example.f90 -o f90_example.x `hhgrid-config --cflags --libs`
```

To launch the C, C++ and Fortran test program enter:
```shell
./c_example.x
./cpp_example.x
./f90_example.x
```
The example programs will call the grid for a single phase-space point. They demonstate as simply as possible how to use the library within your own code 

## Using the grid with your own code
 
The `hhgrid` library exposes the following functions:

* ```void python_initialize(void)```
  A thin wrapper around the Python C API function `Py_Initialize()`, used at the beginning of your program to start the `python` interpreter.
* ```void python_decref(PyObject* grid)```
  A thin wrapper around the Python C API function `Py_DECREF(grid)` used to decrement the `python` interpreter reference count by 1 (required for garbage collection).
* ```void python_finalize(void)```
  A thin wrapper around the Python C API function `Py_Finalize()` used at the end of your program to terminate the `python` interpreter.
* ```void python_printinfo(void)```
  Prints the python paths and versions being used by the library (useful for debugging).
* ```PyObject* grid_initialize(const char* grid_name)```
  Searches `PYTHONPATH` for a grid with a file name matching the input `grid_name`, if found it initialises this grid else terminates your program.
* ```double grid_virt(PyObject* grid, double s, double t)```
  Uses the `grid` object to evaluate the phase-space point specified by the Mandelstam invariants `s`,`t`.
 
Usually, the easiest way to understand how to use the grid with your own code is to look at the relevant example in the `examples` directory.

Broadly the following steps should be taken to use the grid:
* include the `hhgrid.h` as in the examples (C or C++).
* Call `python_initialize()`, `python_printinfo()` once at the beginning of your program.
* Construct a grid object using `PyObject* pGrid = grid_initialize(grid_name);` at the beginning of your program. 
* Call `grid_virt(pGrid, s, t)` for each phase-space point you wish to evaluate, here `pGrid` is an instance of the grid and `s`,`t` are the Mandelstam invariants.
* Call `python_decref(pGrid)` for each instance of the grid and `python_finalize()` once at the end of your program.

When linking your code against `hhgrid` we advise to use the `hhgrid-config --cflags --libs` command to ensure that your program is compiled with the correct `python` headers and libraries.

## Grid files and validation points
 
The grid files contain 4 columns `beta_s`, `cos(theta)`, `V_fin` and `V_fin Error`.
 
Here:
* `beta_s = \sqrt{1-4*mH^2/s}`
* `cos(theta) = (t-u)/(s*beta_s)`, where `u= 2 mH^2 - s -t`
 
For ease of validation we give here a few values from the `Virt_EFT.grid` along with the corresponding Mandelstam `s` and `t` and the value that is returned by `grid.py` (note: since `grid.py` internally bins results the output is not perfect for each phase space point).
 
| `beta_s` | `cos(theta)` | `V_fin` | `V_fin Error` | `s` | `t` | `grid.py result` |
| --- | --- | --- | --- | --- | --- | --- |
| `0.670215461783` | `0.942415327461`   | `3.14977281395e-05` | `1e-20` | `113469.` | `-5274.78` | `3.186491270e-05` |
| `0.749658272418` | `0.358719756797`   | `7.18370167972e-05` | `1e-20` | `142690.` | `-36534.1` | `7.230060450e-05` |
| `0.774816230102` | `0.667225193755`   | `9.55249958151e-05` | `1e-20` | `156383.` | `-22143.3` | `9.579402805e-05` | 
| `0.961960758291` | `0.828468885357`   | `0.00450166545088`  | `1e-20` | `837448.` | `-69395.`  | `4.524576122e-03` |
| `0.831112919562` | `0.00835036268349` | `0.000195523077174` | `1e-20` | `202101.` | `-84724.2` | `1.954913649e-04` |
 
For validation the corresponding Born HTL cross-section values are:
 
| `s` | `t` | `B` |
| --- | --- | --- |
| `113469.` | `-5274.78` | `9.75634751746e-07` |
| `142690.` | `-36534.1` | `2.19481322735e-06` |
| `156383.` | `-22143.3` | `2.90841407195e-06` |
| `837448.` | `-69395.`  | `0.000134710295203` |
| `202101.` | `-84724.2` | `5.9137806539e-06`  |


## Reference Guide

### `examples/grid_interactive.py`

A convenience script that can be used to interactively evaluate grid points (primarily for debugging).
 
Arguments:
* `--grid` a string used to specify which grid to load. 
 
# Internal Note

* `creategrid.py`, `events.cdf`, `Virt_full.grid`, `Virt_EFT.grid` taken from commit `c43eea9` of `POWHEG-BOX-V2/ggHH` repo.
* rewrote `creategrid.py` as class

