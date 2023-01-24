#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <cooperative_groups.h>

#include "../inc/argument_utils.h"

namespace cg = cooperative_groups;

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size = 10.0,
    gravity = 9.81,
    density = 997.0;

real_t
    *h_mass_0 = NULL,
    *h_mass_1 = NULL,
    *d_mass_0 = NULL,
    *d_mass_1 = NULL,

    *h_mass_velocity_x_0 = NULL,
    *h_mass_velocity_x_1 = NULL,
    *d_mass_velocity_x_0 = NULL,
    *d_mass_velocity_x_1 = NULL,

    *h_mass_velocity_y_0 = NULL,
    *h_mass_velocity_y_1 = NULL,
    *d_mass_velocity_y_0 = NULL,
    *d_mass_velocity_y_1 = NULL,

    *h_mass_velocity = NULL,
    *d_mass_velocity = NULL,

    *h_velocity_x = NULL,
    *d_velocity_x = NULL,
    *h_velocity_y = NULL,
    *d_velocity_y = NULL,

    *h_acceleration_x = NULL,
    *d_acceleration_x = NULL,
    *h_acceleration_y = NULL,
    *d_acceleration_y = NULL,
    dx,
    dt;

#define PN(y,x)         mass_0[(y)*(N+2)+(x)]
#define PN_next(y,x)    mass_1[(y)*(N+2)+(x)]
#define PNU(y,x)        mass_velocity_x_0[(y)*(N+2)+(x)]
#define PNU_next(y,x)   mass_velocity_x_1[(y)*(N+2)+(x)]
#define PNV(y,x)        mass_velocity_y_0[(y)*(N+2)+(x)]
#define PNV_next(y,x)   mass_velocity_y_1[(y)*(N+2)+(x)]
#define PNUV(y,x)       mass_velocity[(y)*(N+2)+(x)]
#define U(y,x)          velocity_x[(y)*(N+2)+(x)]
#define V(y,x)          velocity_y[(y)*(N+2)+(x)]
#define DU(y,x)         acceleration_x[(y)*(N+2)+(x)]
#define DV(y,x)         acceleration_y[(y)*(N+2)+(x)]


#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void time_step(
    real_t *velocity_x,
    real_t *velocity_y,
    real_t *acceleration_x,
    real_t *acceleration_y,
	real_t *mass_velocity_x_0,
    real_t *mass_velocity_x_1,
    real_t *mass_velocity_y_0,
    real_t *mass_velocity_y_1,
	real_t *mass_velocity,
    real_t *mass_0,
    real_t *mass_1,
    int_t N,
    real_t dx,
    real_t dt
);

__device__ void time_step_1(
    real_t *velocity_x,
    real_t *mass_velocity_x_0,
    real_t *mass_0,
    real_t *velocity_y,
    real_t *mass_velocity_y_0,
    real_t *acceleration_x,
    real_t *acceleration_y,
    real_t *mass_velocity,
    int_t N
);

__device__ void time_step_2(
    real_t *velocity_x,
    real_t *velocity_y,
    real_t *acceleration_x,
    real_t *acceleration_y,
	real_t *mass_velocity_x_0,
    real_t *mass_velocity_x_1,
    real_t *mass_velocity_y_0,
    real_t *mass_velocity_y_1,
	real_t *mass_velocity,
    real_t *mass_0,
    real_t *mass_1,
    int_t N,
    real_t dx,
    real_t dt
);

// TODO: Rewrite boundary_condition as a device function.
__device__ void boundary_condition(real_t *domain_variable, int sign, int_t N);
void domain_init(void);
void domain_save(int_t iteration);
void domain_finalize(void);

// Pthreads threaded domain save function
void *domain_save_threaded(void *iter);

void swap(real_t** t1, real_t** t2)
{
    real_t* tmp;
	tmp = *t1;
	*t1 = *t2;
	*t2 = tmp;
}


int main (int argc, char **argv)
{

    OPTIONS *options = parse_args(argc, argv);
    if(!options)
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    dim3 threadBlockDims;
    dim3 gridDims;

    if((N+2) <= 32) {
        threadBlockDims = {(unsigned) N+2, (unsigned) N+2, 1};
        gridDims = {1, 1, 1};
    } else {
        threadBlockDims = {32, 32, 1};
        unsigned int size = ceil(((N+2)/32.0));

        gridDims = {size, size, 1};
    }

    printf("N: %ld, SIZE: %d\n", N, gridDims.x);

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        /*
        // Launch the cooperative kernel
        void *kernelArgs[] = {
            d_velocity_x,
            d_velocity_y,
			d_acceleration_x,
            d_acceleration_y,
			d_mass_velocity_x_0,
            d_mass_velocity_x_1,
			d_mass_velocity_y_0,
            d_mass_velocity_y_1,
			d_mass_velocity,
            d_mass_0,
            d_mass_1,
            &N,
            &dx,
            &dt
        };

        cudaError_t error = cudaLaunchCooperativeKernel((void*)time_step, gridDims, threadBlockDims, kernelArgs);
        // printf("N: %d\n", error);
        */
        
        // launch the kernel
        time_step<<<gridDims, threadBlockDims>>>(
            d_velocity_x,
            d_velocity_y,
			d_acceleration_x,
            d_acceleration_y,
			d_mass_velocity_x_0,
            d_mass_velocity_x_1,
			d_mass_velocity_y_0,
            d_mass_velocity_y_1,
			d_mass_velocity,
            d_mass_0,
            d_mass_1,
            N,
            dx,
            dt
        );
        

        if(iteration % snapshot_frequency == 0)
        {
            printf(
                "Iteration %ld of %ld, (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            // copy the masses from the device to host prior to domain_save
            cudaMemcpy(h_mass_0, d_mass_0, (N+2) * (N+2) * sizeof(real_t), cudaMemcpyDeviceToHost);

            domain_save(iteration);
        }

        // swap device buffer pointers between iterations
        swap(&d_mass_0, &d_mass_1);
        swap(&d_mass_velocity_x_0, &d_mass_velocity_x_1);
        swap(&d_mass_velocity_y_0, &d_mass_velocity_y_1);
    }

    domain_finalize();

    exit(EXIT_SUCCESS);
}

// performs a full time step
__global__ void time_step(
    real_t *velocity_x,
    real_t *velocity_y,
    real_t *acceleration_x,
    real_t *acceleration_y,
	real_t *mass_velocity_x_0,
    real_t *mass_velocity_x_1,
    real_t *mass_velocity_y_0,
    real_t *mass_velocity_y_1,
	real_t *mass_velocity,
    real_t *mass_0,
    real_t *mass_1,
    int_t N,
    real_t dx,
    real_t dt
)
{
    // would be used if cooperative kernel would work :<
    cg::grid_group grid = cg::this_grid();

    time_step_1(
        velocity_x,
        mass_velocity_x_0,
        mass_0,
        velocity_y,
        mass_velocity_y_0,
        acceleration_x,
        acceleration_y,
        mass_velocity,
        N
    );

    // sync the device before continuing with time_step_2

    __syncthreads();
    // grid.sync();

    time_step_2(
        velocity_x,
        velocity_y,
        acceleration_x,
        acceleration_y,
        mass_velocity_x_0,
        mass_velocity_x_1,
        mass_velocity_y_0,
        mass_velocity_y_1,
        mass_velocity,
        mass_0,
        mass_1,
        N,
        dx,
        dt
    );
}

// performs the update for the accelerations
__device__ void time_step_1(
    real_t *velocity_x,
    real_t *mass_velocity_x_0,
    real_t *mass_0,
    real_t *velocity_y,
    real_t *mass_velocity_y_0,
    real_t *acceleration_x,
    real_t *acceleration_y,
    real_t *mass_velocity,
    int_t N
)
{
    boundary_condition(mass_0, 1, N);
    boundary_condition(mass_velocity_x_0, -1, N);
    boundary_condition(mass_velocity_y_0, -1, N);

    // calculate the thread id
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    // guard to not access data out of bound
    if(!(x > N || y > N || x < 1 || y < 1)) {
        U(y,x) = PNU(y,x) / PN(y,x);
        V(y,x) = PNV(y,x) / PN(y,x);

        PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
    }

    if(!(x > N+1 || y > N+1 || x < 0 || y < 0)) {
        DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
    }
}

// performs the rest of the time step
__device__ void time_step_2(
    real_t *velocity_x,
    real_t *velocity_y,
    real_t *acceleration_x,
    real_t *acceleration_y,
	real_t *mass_velocity_x_0,
    real_t *mass_velocity_x_1,
    real_t *mass_velocity_y_0,
    real_t *mass_velocity_y_1,
	real_t *mass_velocity,
    real_t *mass_0,
    real_t *mass_1,
    int_t N,
    real_t dx,
    real_t dt
)
{
    // calculate the thread id
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(!(x > N || y > N || x < 1 || y < 1)) {
        PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                        ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                      + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
        );

        PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                        ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                      + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
        );

        PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                        ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                      + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
        );
    }
}

__device__ void boundary_condition(real_t *domain_variable, int sign, int_t N)
{
    // calculate the thread id
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    #define VAR(y,x) domain_variable[(y)*(N+2)+(x)]
    if(x == 0) {
        if(y == 0) {
            // corner (0,0)
            VAR(   0, 0   ) = sign*VAR(   2, 2   );
            return;
        }

        if(y == N+1) {
            // corner (N+1, 0)
            VAR( N+1, 0   ) = sign*VAR( N-1, 2   );
            return;
        }

        // first column from the left
        VAR(   y, 0   ) = sign*VAR(   y, 2   );
    }

    if(x == N+1) {
        if(y == 0) {
            // corner (0, N+1)
            VAR(   0, N+1 ) = sign*VAR(   2, N-1 );
            return;
        }

        if(y == N+1) {
            // corner (N+1, N+1)
            VAR( N+1, N+1 ) = sign*VAR( N-1, N-1 );
            return;
        }

        // last column on the right
        VAR(   y, N+1 ) = sign*VAR(   y, N-1 );
    }

    if(y == 0) {
        // first row
        VAR(   0, x   ) = sign*VAR(   2, x   );
    }

    if(y == N+1) {
        // last row
        VAR( N+1, x   ) = sign*VAR( N-1, x   );
    }

    #undef VAR
}


void domain_init(void)
{
    int elements = (N+2)*(N+2);

    // Allocate memory for the host buffers
    h_mass_0 = (real_t *) calloc(elements, sizeof(real_t));
    h_mass_1 = (real_t *) calloc(elements, sizeof(real_t));

    h_mass_velocity_x_0 = (real_t *) calloc(elements, sizeof(real_t));
    h_mass_velocity_x_1 = (real_t *) calloc(elements, sizeof(real_t));
    h_mass_velocity_y_0 = (real_t *) calloc(elements, sizeof(real_t));
    h_mass_velocity_y_1 = (real_t *) calloc(elements, sizeof(real_t));

    h_mass_velocity = (real_t *) calloc(elements, sizeof(real_t));

    h_velocity_x = (real_t *) calloc(elements, sizeof(real_t));
    h_velocity_y = (real_t *) calloc(elements, sizeof(real_t));
    h_acceleration_x = (real_t *) calloc(elements, sizeof(real_t));
    h_acceleration_y = (real_t *) calloc(elements, sizeof(real_t));

    // Allocate memory for the device buffers
    cudaMalloc(&d_mass_0, elements * sizeof(real_t));
    cudaMalloc(&d_mass_1, elements * sizeof(real_t));

    cudaMalloc(&d_mass_velocity_x_0, elements * sizeof(real_t));
    cudaMalloc(&d_mass_velocity_x_1, elements * sizeof(real_t));
    cudaMalloc(&d_mass_velocity_y_0, elements * sizeof(real_t));
    cudaMalloc(&d_mass_velocity_y_1, elements * sizeof(real_t));

    cudaMalloc(&d_mass_velocity, elements * sizeof(real_t));

    cudaMalloc(&d_velocity_x, elements * sizeof(real_t));
    cudaMalloc(&d_velocity_y, elements * sizeof(real_t));
    cudaMalloc(&d_acceleration_x, elements * sizeof(real_t));
    cudaMalloc(&d_acceleration_y, elements * sizeof(real_t));

    for(int_t y=1; y<=N; y++)
    {
        for(int_t x=1; x<=N; x++)
        {
	    h_mass_0[y*(N+2) + x] = 1e-3;
	    h_mass_velocity_x_0[y*(N+2) + x] = 0.0;
	    h_mass_velocity_y_0[y*(N+2) + x] = 0.0;

            real_t cx = x-N/2;
            real_t cy = y-N/2;
            if(sqrt(cx*cx + cy*cy) < N/20.0)
            {
                h_mass_0[y*(N+2) + x] -= 5e-4*exp (
                    - 4*pow(cx, 2.0) / (real_t)(N)
                    - 4*pow(cy, 2.0) / (real_t)(N)
                );
            }

            h_mass_0[y*(N+2) + x] *= density;
        }
    }

    dx = domain_size / (real_t) N;
    dt = 5e-2;

    // copy data over to device buffer
    cudaMemcpy(d_mass_0, h_mass_0, elements * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_velocity_x_0, h_mass_velocity_x_0, elements * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_velocity_y_0, h_mass_velocity_y_0, elements * sizeof(real_t), cudaMemcpyHostToDevice);

    // also copy the zeros over
    cudaMemcpy(d_mass_1, h_mass_1, elements * sizeof(real_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mass_velocity_x_1, h_mass_velocity_x_1, elements * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass_velocity_y_1, h_mass_velocity_y_1, elements * sizeof(real_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mass_velocity, h_mass_velocity, elements * sizeof(real_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_velocity_x, h_velocity_x, elements * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_y, h_velocity_y, elements * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceleration_x, h_acceleration_x, elements * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceleration_y, h_acceleration_y, elements * sizeof(real_t), cudaMemcpyHostToDevice);
}


void domain_save(int_t iteration)
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset(filename, 0, 256*sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    FILE *out = fopen(filename, "wb");
    if(!out)
    {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(1);
    }
    //fwrite ( mass[0], (N+2)*(N+2), sizeof(real_t), out );
    for(int_t y = 1; y <= N; y++)
    {
        fwrite(&h_mass_0[y*(N+2)+1], N, sizeof(real_t), out);
    }
    fclose(out);
}

void domain_finalize(void)
{
    free(h_mass_0);
    free(h_mass_1);
    free(h_mass_velocity_x_0);
    free(h_mass_velocity_x_1);
    free(h_mass_velocity_y_0);
    free(h_mass_velocity_y_1);
    free(h_mass_velocity);
    free(h_velocity_x);
    free(h_velocity_y);
    free(h_acceleration_x);
    free(h_acceleration_y);

    // Free device arrays
    cudaFree(d_mass_0);
    cudaFree(d_mass_1);
    cudaFree(d_mass_velocity_x_0);
    cudaFree(d_mass_velocity_x_1);
    cudaFree(d_mass_velocity_y_0);
    cudaFree(d_mass_velocity_y_1);
    cudaFree(d_mass_velocity);
    cudaFree(d_velocity_x);
    cudaFree(d_velocity_y);
    cudaFree(d_acceleration_x);
    cudaFree(d_acceleration_y);
}
