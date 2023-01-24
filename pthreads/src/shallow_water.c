#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#include "../inc/argument_utils.h"


typedef int64_t int_t;
typedef double real_t;

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

pthread_barrier_t barrier;

int HOLY_THREAD_FOR_SINGLE_OPERATIONS = 1;
int NUMBER_OF_THREADS = 4;

struct thread_args {
    int thread_num;
};

struct timeval
    t_start,
    t_stop;
double
    t_total;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size = 10.0,
    gravity = 9.81,
    density = 997.0;

real_t
    *mass[2] = { NULL, NULL},
    *mass_velocity_x[2] = { NULL, NULL },
    *mass_velocity_y[2] = { NULL, NULL },
    *mass_velocity = NULL,
    *velocity_x = NULL,
    *velocity_y = NULL,
    *acceleration_x = NULL,
    *acceleration_y = NULL,
    dx,
    dt;

#define PN(y,x)         mass[0][(y)*(N+2)+(x)]
#define PN_next(y,x)    mass[1][(y)*(N+2)+(x)]
#define PNU(y,x)        mass_velocity_x[0][(y)*(N+2)+(x)]
#define PNU_next(y,x)   mass_velocity_x[1][(y)*(N+2)+(x)]
#define PNV(y,x)        mass_velocity_y[0][(y)*(N+2)+(x)]
#define PNV_next(y,x)   mass_velocity_y[1][(y)*(N+2)+(x)]
#define PNUV(y,x)       mass_velocity[(y)*(N+2)+(x)]
#define U(y,x)          velocity_x[(y)*(N+2)+(x)]
#define V(y,x)          velocity_y[(y)*(N+2)+(x)]
#define DU(y,x)         acceleration_x[(y)*(N+2)+(x)]
#define DV(y,x)         acceleration_y[(y)*(N+2)+(x)]

void time_step ( int thread_number );
void boundary_condition ( real_t *domain_variable, int sign );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );
void swap( real_t** t1, real_t** t2);

void* pthread_func(void* input)
{
    // parse thread arguments
    struct thread_args* args = (struct thread_args*) input;
    int thread_num = args->thread_num;

    // iterate
    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        if (thread_num == HOLY_THREAD_FOR_SINGLE_OPERATIONS) {
            boundary_condition ( mass[0], 1 );
            boundary_condition ( mass_velocity_x[0], -1 );
            boundary_condition ( mass_velocity_y[0], -1 );
        }

        time_step(thread_num);

        if (thread_num == HOLY_THREAD_FOR_SINGLE_OPERATIONS && iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld, (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );
            domain_save ( iteration );
        }

        pthread_barrier_wait (&barrier);

        if (thread_num == HOLY_THREAD_FOR_SINGLE_OPERATIONS) {
            swap ( &mass[0], &mass[1] );
            swap ( &mass_velocity_x[0], &mass_velocity_x[1] );
            swap ( &mass_velocity_y[0], &mass_velocity_y[1] );
        }
    }
    return 0;
}

void swap ( real_t** t1, real_t** t2 )
{
    real_t* tmp;
	tmp = *t1;
	*t1 = *t2;
	*t2 = tmp;
}


int main ( int argc, char **argv )
{

    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    gettimeofday ( &t_start, NULL );

    // allocate memory for threads and thread arguments
    pthread_t *threads = malloc(sizeof(pthread_t) * NUMBER_OF_THREADS);
    struct thread_args *args = malloc(sizeof(struct thread_args) * NUMBER_OF_THREADS);

    // initialise barrier
    pthread_barrier_init (&barrier, NULL, NUMBER_OF_THREADS);

    // create threads
    for (int i = 1; i <= NUMBER_OF_THREADS; i++){
        args[i] = (struct thread_args) {.thread_num = i};
        pthread_create(&(threads[i]), NULL, &pthread_func, &(args[i]));
    }
    
    // join threads
    for (int i = 1; i <= NUMBER_OF_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    gettimeofday ( &t_stop, NULL );
    t_total = WALLTIME(t_stop) - WALLTIME(t_start);
    printf ( "%.2lf seconds total runtime\n", t_total );

    domain_finalize();

    exit ( EXIT_SUCCESS );
}


void time_step ( int thread_number )
{
    // divide the grid horizontally by the number of threads
    int grid_size = N/NUMBER_OF_THREADS;
    // thread zero doesn't work so we exclude it
    int offset = (thread_number - 1) * grid_size;

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1 + offset; x<=offset + grid_size; x++ )
        {
            U(y,x) = PNU(y,x) / PN(y,x);
            V(y,x) = PNV(y,x) / PN(y,x);
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1 + offset; x<=offset + grid_size; x++ )
        {
            PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
        }

    for ( int_t y=0; y<=N+1; y++ )
        for ( int_t x=0 + offset; x<=offset + grid_size + 1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1 + offset; x<=offset + grid_size; x++ )
        {
            PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                            ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                          + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1 + offset; x<=offset + grid_size; x++ )
        {
            PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                            ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                          + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1 + offset; x<=offset + grid_size; x++ )
        {
            PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                           ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                         + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
            );
        }
}


void boundary_condition ( real_t *domain_variable, int sign )
{
    #define VAR(y,x) domain_variable[(y)*(N+2)+(x)]
    VAR(   0, 0   ) = sign*VAR(   2, 2   );
    VAR( N+1, 0   ) = sign*VAR( N-1, 2   );
    VAR(   0, N+1 ) = sign*VAR(   2, N-1 );
    VAR( N+1, N+1 ) = sign*VAR( N-1, N-1 );

    for ( int_t y=1; y<=N; y++ ) VAR(   y, 0   ) = sign*VAR(   y, 2   );
    for ( int_t y=1; y<=N; y++ ) VAR(   y, N+1 ) = sign*VAR(   y, N-1 );
    for ( int_t x=1; x<=N; x++ ) VAR(   0, x   ) = sign*VAR(   2, x   );
    for ( int_t x=1; x<=N; x++ ) VAR( N+1, x   ) = sign*VAR( N-1, x   );
    #undef VAR
}


void domain_init ( void )
{
    mass[0] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass[1] = calloc ( (N+2)*(N+2), sizeof(real_t) );

    mass_velocity_x[0] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass_velocity_x[1] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass_velocity_y[0] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass_velocity_y[1] = calloc ( (N+2)*(N+2), sizeof(real_t) );

    mass_velocity = calloc ( (N+2)*(N+2), sizeof(real_t) );

    velocity_x = calloc ( (N+2)*(N+2), sizeof(real_t) );
    velocity_y = calloc ( (N+2)*(N+2), sizeof(real_t) );
    acceleration_x = calloc ( (N+2)*(N+2), sizeof(real_t) );
    acceleration_y = calloc ( (N+2)*(N+2), sizeof(real_t) );

    for ( int_t y=1; y<=N; y++ )
    {
        for ( int_t x=1; x<=N; x++ )
        {
            PN(y,x) = 1e-3;
            PNU(y,x) = 0.0;
            PNV(y,x) = 0.0;

            real_t cx = x-N/2;
            real_t cy = y-N/2;
            if ( sqrt ( cx*cx + cy*cy ) < N/20.0 )
            {
                PN(y,x) -= 5e-4*exp (
                    - 4*pow( cx, 2.0 ) / (real_t)(N)
                    - 4*pow( cy, 2.0 ) / (real_t)(N)
                );
            }

            PN(y,x) *= density;
        }
    }

    dx = domain_size / (real_t) N;
    dt = 5e-2;
}


void domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( !out )
    {
        fprintf( stderr, "Failed to open file %s\n", filename );
        exit(1);
    }
    for ( int_t y = 1; y <= N; y++ )
    {
        fwrite ( &mass[0][y*(N+2)+1], N, sizeof(real_t), out );
    }
    fclose ( out );
}


void domain_finalize ( void )
{
    free ( mass[0] );
    free ( mass[1] );
    free ( mass_velocity_x[0] );
    free ( mass_velocity_x[1] );
    free ( mass_velocity_y[0] );
    free ( mass_velocity_y[1] );
    free ( mass_velocity );
    free ( velocity_x );
    free ( velocity_y );
    free ( acceleration_x );
    free ( acceleration_y );
}