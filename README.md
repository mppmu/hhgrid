# hhgrid

ggHH 2-loop virtual amplitude interfered with 1-loop amplitude.

Returns V_fin using the conventions specified in Eq(2.5) of https://arxiv.org/pdf/1703.09252.pdf

## Installation

Pre-requisites:
* `python` (tested for 2.7.13)
* `numpy` (tested for 1.11.1)
* `scipy` (tested for 0.17.1)

Assuming you have `python` you can check for the availability of `numpy` and `scipy` with the commands:
```shell
python -c "import numpy"
python -c "import sympy"
```

If you would like to use the C++ interface you can compile the test program `cpp/testgrid.cpp`:
```shell
c++ -std=c++11 cpp/testgrid.cpp -o cpp_testgrid.x
```

If you would like to use the Fortran interface you can compile the test program `fortran/testgrid.f90`:
```shell
gfortran fortran/testgrid.f90 -o fortran_testgrid.x
```

## Usage

Here we demonstrate how to access the grid with one of the test programs `cpp_testgrid.x` or `fortran_testgrid.x`,
this will check that everything is set up correctly and show the steps required to call the grid from your own program.

### Step 1

To start the grid server program type the command:
```shell
python grid.py --seed=0 --grid='Virt_EFT.grid' --verbose
```
This will load the virtual amplitude for the NLO HEFT result. 
This grid can be used to numerically compare your own implementation of the HEFT result against the grid result. 
The `Virt_EFT.grid` result was produced using precisely the same conventions as the full result grid.

After a minute or so you will see the output:
```shell
== Ready to evaluate function ==
Input Pipe:  pyInputPipe-0000
Output Pipe:  pyOutputPipe-0000
Ready Signal:  pyReadySignal-0000
```

The file `pyReadySignal-0000` should now exist on disk. 
The server is now ready to evaluate the virtual amplitude using the grid.

### Step 2

Once the file `pyReadySignal-0000` is present on the disk the grid is ready to communicate via the FIFOs `pyInputPipe-0000` and `pyOutputPipe-0000`, which it creates.

Open a second terminal (be sure to leave the first terminal open and running `grid.py`) and type either:
```shell
./cpp_testgrid.x
```
or
```shell
./fortran_testgrid.x
```
to launch the C++ or Fortran test program.

The test program will call the grid for a few selected points and validate that the correct result is produced. 

## Using the grid with your own code

To interface the grid from your own code you will need to copy the functions `grid_virt(seed,s,t)` and `kill_python(seed)` from `cpp/testgrid.cpp` or `fortran/testgrid.f90`. After this, follow the same steps as above but rather than launching the test programs launch your own code which calls `grid_virt(seed,s,t)`.

We provide a bash shell script `demo.sh` which demonstrates how to automatically launch the python grid server, wait for it to initialise, then launch a program which uses the grid.

## Reference Guide

### `grid.py`

A server which initialises a python grid and responds to calls to `grid_virt`.

Arguments:
* `--seed` an integer used to specify which FIFOs should be created and responded to. 
* `--grid` a string used to specify which grid to load. 
* `--verbose` a flag used to indicate if detailed output should be printed to `stdout`.

Currently we distribute the grids:
* `Virt_EFT.grid` the NLO QCD result in the HEFT.
* `Virt_full.grid` the NLO QCD result in the Standard Model.

### `grid_virt(int seed, double s, double t)`

Returns the value of the virtual matrix element interfered with the born with the conventions specified in Eq(2.5) of https://arxiv.org/pdf/1703.09252.pdf.

Parameters:
* `seed` an integer specifying the seed used to call a `grid.py` server. The `grid.py` server is not threadsafe and should only be called from one thread, thus it is usually necessary to launch different copies of `grid.py` each with a different `seed` then have each thread call `grid_virt(seed,s,t)` with a `seed` unique to the thread.
* `s` the Mandelstam s=(p1+p2)^2.
* `t` the Mandelstam t=(p1-p3)^2.

### `kill_python(int seed)`

Instructs the `grid.py` server to terminate cleanly.

Parameters:
* `seed` an integer specifying the seed used to call a `grid.py` server.

## Internal Note

* `creategrid.py`, `events.cdf`, `Virt_full.grid`, `Virt_EFT.grid` taken from commit `c43eea9` of `POWHEG-BOX-V2/ggHH` repo.
* rewrote `creategrid.py` as class
