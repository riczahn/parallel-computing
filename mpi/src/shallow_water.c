#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

#include "../inc/argument_utils.h"

typedef int64_t int_t;
typedef double real_t;

int 
    rank,
    number_of_processes;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size = 10.0,
    gravity = 9.81,
    density = 997.0;

real_t
    *mass[2] = { NULL, NULL },
    *mass_velocity_x[2] = { NULL, NULL },
    *velocity_x = NULL,
    *acceleration_x = NULL,
    dx,
    dt;

#define PN(x)        mass[0][(x)]
#define PN_next(x)   mass[1][(x)]
#define PNU(x)       mass_velocity_x[0][(x)]
#define PNU_next(x)  mass_velocity_x[1][(x)]
#define U(x)         velocity_x[(x)]
#define DU(x)        acceleration_x[(x)]

void time_step ( void );
void boundary_condition( real_t *domain_variable, int sign );
void domain_init ( void );
void domain_save ( int_t iteration );
void exchange_border_values( void );
void domain_finalize ( void );


void swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


int main ( int argc, char **argv )
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

    // Create MPI datatype for options struct
    MPI_Datatype options_datatype;
    MPI_Type_contiguous(3, MPI_LONG, &options_datatype);
    MPI_Type_commit(&options_datatype);

    // Parse arguments in the rank 0 process and broadcast to other processes
    OPTIONS *options;
    if (rank == 0) {
        options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }
    } else {
        options = malloc(sizeof(OPTIONS));
    }

    MPI_Bcast(options, 1, options_datatype, 0, MPI_COMM_WORLD);
    MPI_Type_free(&options_datatype);

    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;
    
    printf("Options received on process %d! N: %d, max_iteration: %d, snapshot_frequency: %d\n", rank, (int)N, (int)max_iteration, (int)snapshot_frequency);
    MPI_Barrier(MPI_COMM_WORLD);

    domain_init();

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        exchange_border_values();

        // Boundary conditions
        boundary_condition(mass[0], 1);
        boundary_condition(mass_velocity_x[0], -1);

        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );
            domain_save ( iteration );
        }

        swap( &mass[0], &mass[1] );
        swap( &mass_velocity_x[0], &mass_velocity_x[1] );
    }

    domain_finalize();
    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}


void time_step ( void )
{
    int size_of_sub_grid = N / number_of_processes;

    for ( int_t x=0; x<=size_of_sub_grid+1; x++ )
    {
        DU(x) = PN(x) * U(x) * U(x)
                + 0.5 * gravity * PN(x) * PN(x) / density;
    }

    for ( int_t x=1; x<=size_of_sub_grid; x++ )
    {
        PNU_next(x) = 0.5*( PNU(x+1) + PNU(x-1) ) - dt*(
                      ( DU(x+1) - DU(x-1) ) / (2*dx)
        );
    }

    for ( int_t x=1; x<=size_of_sub_grid; x++ )
    {
        PN_next(x) = 0.5*( PN(x+1) + PN(x-1) ) - dt*(
                       ( PNU(x+1) - PNU(x-1) ) / (2*dx)
        );
    }

    for ( int_t x=1; x<=size_of_sub_grid; x++ )
    {
        U(x) = PNU_next(x) / PN_next(x);
    }
}


void boundary_condition ( real_t *domain_variable, int sign )
{
    if (rank == 0) {
        #define VAR(x) domain_variable[(x)]
        VAR(   0 ) = sign*VAR( 2   );
        #undef VAR
        return;
    }

    int rank_of_last_process = number_of_processes - 1;
    if (rank == rank_of_last_process) {
        int size_of_sub_grid = N / number_of_processes;
        #define VAR(x) domain_variable[(x)]
        VAR( size_of_sub_grid+1 ) = sign*VAR( size_of_sub_grid-1 );
        #undef VAR
    }
}


void domain_init ( void )
{
    // Allocate space for each process' sub-grid
    int size_of_sub_grid = N / number_of_processes;
    int global_start_index = rank * size_of_sub_grid;

    mass[0] = calloc ( (size_of_sub_grid+2), sizeof(real_t) );
    mass[1] = calloc ( (size_of_sub_grid+2),  sizeof(real_t) );

    mass_velocity_x[0] = calloc ( (size_of_sub_grid+2), sizeof(real_t) );
    mass_velocity_x[1] = calloc ( (size_of_sub_grid+2),  sizeof(real_t) );

    velocity_x = calloc ( (size_of_sub_grid+2), sizeof(real_t) );
    acceleration_x = calloc ( (size_of_sub_grid+2), sizeof(real_t) );

    // Data initialization
    for ( int_t x=1; x<=size_of_sub_grid; x++ )
    {
        int_t global_x = global_start_index + x;
        PN(x) = 1e-3;
        PNU(x) = 0.0;

        real_t c = global_x-N/2;
        if ( sqrt ( c*c ) < N/20.0 )
        {
            PN(x) -= 5e-4*exp (
                    - 4*pow( c, 2.0 ) / (real_t)(N)
            );
        }

        PN(x) *= density;
    }

    dx = domain_size / (real_t) N;
    dt = 0.1*dx;
}


void domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    MPI_File out;
    MPI_File_open (
        MPI_COMM_WORLD,
        filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &out
    );

    int_t size_of_sub_grid = N / number_of_processes;
    MPI_Offset offset = rank * sizeof(real_t) * size_of_sub_grid;
    
    MPI_File_write_at_all(out, offset,  &mass[0][1], size_of_sub_grid, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&out);
}

void exchange_border_values( void ) {
    int rank_of_eastern_neighbour = (rank + 1) % number_of_processes;
    int rank_of_western_neighbour = (rank - 1);
    if (rank_of_western_neighbour == -1) {
        rank_of_western_neighbour = number_of_processes - 1;
    }

    int index_for_first_element = 1;
    int index_for_last_element = N / number_of_processes;

    real_t western_border = mass[0][index_for_first_element];
    real_t eastern_border = mass[0][index_for_last_element];

    // send eastern border to eastern neighbour
    MPI_Send(&eastern_border, 1, MPI_DOUBLE, rank_of_eastern_neighbour, 0, MPI_COMM_WORLD);

    // receive western border from western neighbour
    real_t received;
    MPI_Recv(&received, 1, MPI_DOUBLE, rank_of_western_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mass[0][index_for_first_element - 1] = received;

    // send western border to western neighbour
    MPI_Send(&western_border, 1, MPI_DOUBLE, rank_of_western_neighbour, 2, MPI_COMM_WORLD);

    // receive eastern border from eastern neighbour
    MPI_Recv(&received, 1, MPI_DOUBLE, rank_of_eastern_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mass[0][index_for_last_element + 1] = received;

    // mass velocity
    real_t western_border_mass_velocity = mass_velocity_x[0][index_for_first_element];
    real_t eastern_border_mass_velocity = mass_velocity_x[0][index_for_last_element];

    // send eastern border to eastern neighbour
    MPI_Send(&eastern_border_mass_velocity, 1, MPI_DOUBLE, rank_of_eastern_neighbour, 0, MPI_COMM_WORLD);

    // receive western border from western neighbour
    real_t received_mass_velocity;
    MPI_Recv(&received_mass_velocity, 1, MPI_DOUBLE, rank_of_western_neighbour, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mass_velocity_x[0][index_for_first_element - 1] = received_mass_velocity;

    // send western border to western neighbour
    MPI_Send(&western_border_mass_velocity, 1, MPI_DOUBLE, rank_of_western_neighbour, 2, MPI_COMM_WORLD);

    // receive eastern border from eastern neighbour
    MPI_Recv(&received, 1, MPI_DOUBLE, rank_of_eastern_neighbour, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    mass_velocity_x[0][index_for_last_element + 1] = received;
}

void
domain_finalize ( void )
{
    free ( mass[0] );
    free ( mass[1] );
    free ( mass_velocity_x[0] );
    free ( mass_velocity_x[1] );
    free ( velocity_x );
    free ( acceleration_x );
}

