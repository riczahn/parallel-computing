# Finite difference approximation of the 2D shallow water equations using CUDA
In this assignment a serial implementation of the Finite Difference Method (FDM) for solving the 1D shallow water equations was taken and transformed into a distributed version using the CUDA API for NVIDIA GPUs.

## Run
### Setup
`make setup`

Creates folders `data`, `plots` and `video`.
- `data`: contains output from the simulation
- `plots`: contains output from plotting
- `video`: contains output from video generation

**Compile**

`make compile`

**Run**

`./solution -n [grid_size] -i [max_iteration] -s [snapshot_frequency]`

**Example**

```
make solution
./solution -n 256 -i 5000 -s 40
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

## Installing dependencies
**CUDA**

Linux/Ubuntu:

An in-depth overview of how to install CUDA on Linux can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

MacOSX:

NVIDIA no longer supports development on macOS. Please opt for using
the Snotra cluster or another operating system. The TDT4200 staff will not
be able to offer support for compiling and running CUDA applications on MacOSX.

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
