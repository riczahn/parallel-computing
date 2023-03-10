# Finite difference approximation of the 1D shallow water equations using MPI
In this assignment a serial implementation of the Finite Difference Method (FDM) for solving the 1D shallow water equations was taken and transformed into a distributed version using the Message Passing Interface (MPI).

## Run
### Setup
`make setup`

Creates folders `data`, `plots` and `video`.
- `data`: contains output from the simulation
- `plots`: contains output from plotting
- `video`: contains output from video generation

Compiles the code for comparing solutions.

**Compile**

`make compile`

**Run**

`mpirun -np [number of MPI processes] [--oversubscribe] ./solution -n [grid_size] -i [max_iteration] -s [snapshot_frequency]`

**!** MPI will complain that there are "not enough slots available" if you try to run with more processes than there are available processors. Passing the `--oversubscribe` option to `mpirun` will circumvent this.

**Example**  

```
make compile
mpirun -np 4 ./solution -n 256 -i 5000 -s 40
```

**Compile and Run**

You can also execute both of the above commands together with default values with `make run`.

## Visualize
### Plots
`./plot_solution.sh -n [grid_size]`

Plots the program output using [gnuplot](http://gnuplot.sourceforge.net).

Alternatively, you can compile, run, and plot the solution with default values with `make plot` .

**Example**

`./plot_solution.sh -n 256`

### Video
`make show`

Compiles, runs, and plots the parallel solution with default values and creates a video using [ffmpeg](https://ffmpeg.org).

## Check
`make check`

Compiles and runs the parallel solution with default values and compares the output data to reference data.

## Options
Option | Description | Restrictions | Default value
:------------ | :------------ | :------------ | :------------
**-n** | Number of grid points in one spatial dimension | > 0 | 256
**-i** | Number of iterations | > 0 | 5000
**-s** | Number of iterations between each time the grid state is saved to file | > 0 | 40
**-np**| Number of processes (MPI option) | > 0 | 4

## Installing dependencies
**OpenMPI**

Linux/Ubuntu:

```
sudo apt update
sudo apt install -y openmpi-bin openmpi-doc libopenmpi-dev
```

MacOSX:

```
brew update
brew install open-mpi
```

**gnuplot**

Linux/Ubuntu:

```
sudo apt update
sudo apt install gnuplot
```

MacOSX:

```
brew update
brew install gnuplot
```

**ffmpeg**

Linux/Ubuntu:

```
sudo apt update
sudo apt install ffmpeg
```

MacOSX:

```
brew update
brew install ffmpeg
```
