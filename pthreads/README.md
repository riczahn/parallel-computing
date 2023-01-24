# Finite difference approximation of the 2D shallow water equations using OpenMP
This is the result of a former assignment of the course TDT4200 at the Norwegian University of Science and Technology. In this assignment a serial implementation of the Finite Difference Method (FDM) for solving the 2D shallow water equations was taken and transformed into a parallel version using OpenMP. 

## Run
### Setup
`make setup`

Creates folders `data`, `plots` and `video`.
- `data`: contains output from the simulation
- `plots`: contains output from plotting
- `video`: contains output from video generation

**Compile**
 
`make compile`

**Compile and Run**

You can also execute both of the above commands together with default values with `make run`.

## Visualize
### Plots
`./plot_solution.sh -n [grid_size]`

Plots the program output using [gnuplot](http://gnuplot.sourceforge.net).

Alternatively, you can compile, run, and plot the solution with default values with `make plot`.

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
**OpenMP**
Your compiler should already include support for OpenMP by default.

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
