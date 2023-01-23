#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include "stdbool.h"

#include "../inc/argument_utils.h"

#include <mpi.h>

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

MPI_Comm
    cart;
MPI_Datatype
    grid,
    subgrid;

int
    rank,
    comm_size,
    local_rows,
    local_cols;

int dims[2] = {0, 0};
int period[2] = {1, 1};

#define MPI_RANK_ROOT  ( rank == 0 )

struct timeval
    t_start,
    t_stop;
double
    t_total;

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size= 10.0,
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

#define PN(y,x)         mass[0][(y)*(local_cols+2)+(x)]
#define PN_next(y,x)    mass[1][(y)*(local_cols+2)+(x)]
#define PNU(y,x)        mass_velocity_x[0][(y)*(local_cols+2)+(x)]
#define PNU_next(y,x)   mass_velocity_x[1][(y)*(local_cols+2)+(x)]
#define PNV(y,x)        mass_velocity_y[0][(y)*(local_cols+2)+(x)]
#define PNV_next(y,x)   mass_velocity_y[1][(y)*(local_cols+2)+(x)]
#define PNUV(y,x)       mass_velocity[(y)*(local_cols+2)+(x)]
#define U(y,x)          velocity_x[(y)*(local_cols+2)+(x)]
#define V(y,x)          velocity_y[(y)*(local_cols+2)+(x)]
#define DU(y,x)         acceleration_x[(y)*(local_cols+2)+(x)]
#define DV(y,x)         acceleration_y[(y)*(local_cols+2)+(x)]

void time_step ( void );
void boundary_condition ( real_t *domain_variable, int sign );
void create_types ( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );
void communicate_borders( void );
void communicate_mass_borders( void );
void communicate_mass_velocity_borders( void );
void exchange_vertical_borders( real_t *domain_variable );
void exchange_horizontal_borders( real_t *domain_variable );
void exchange_northern_borders( real_t *domain_variable );
void exchange_southern_borders( real_t *domain_variable );
void exchange_eastern_borders( real_t *domain_variable );
void exchange_western_borders( real_t *domain_variable );
void do_shift (int direction, int displacement, real_t *send_buff, real_t *recv_buff, int message_size);


void
swap ( real_t** t1, real_t** t2 )
{
    real_t* tmp;
	tmp = *t1;
	*t1 = *t2;
	*t2 = tmp;
}


int
main ( int argc, char **argv )
{
    MPI_Init ( &argc, &argv );
    MPI_Comm_size ( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

    // create cartesian communicator with auto-inferred dimension
    MPI_Dims_create(comm_size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &cart);

    if ( MPI_RANK_ROOT )
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
    }

    MPI_Bcast ( &N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );

    domain_init();

    create_types();

    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration<=max_iteration; iteration++ )
    {
        communicate_borders();

        boundary_condition ( mass[0], 1 );
        boundary_condition ( mass_velocity_x[0], -1 );
        boundary_condition ( mass_velocity_y[0], -1 );

        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            if ( MPI_RANK_ROOT )
            {
                printf (
                    "Iteration %ld of %ld, (%.2lf%% complete)\n",
                    iteration,
                    max_iteration,
                    100.0 * (real_t) iteration / (real_t) max_iteration
                );
            }

            domain_save ( iteration );
        }

        swap ( &mass[0], &mass[1] );
        swap ( &mass_velocity_x[0], &mass_velocity_x[1] );
        swap ( &mass_velocity_y[0], &mass_velocity_y[1] );
    }

    domain_finalize();

    gettimeofday ( &t_stop, NULL );
    t_total = WALLTIME(t_stop) - WALLTIME(t_start);

    if ( MPI_RANK_ROOT )
        printf ( "%.2lf seconds total runtime\n", t_total );

    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    // iterate over subgrid and update values
    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            U(y,x) = PNU(y,x) / PN(y,x);
            V(y,x) = PNV(y,x) / PN(y,x);
        }


    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
        }

    for ( int_t y=0; y<=local_rows+1; y++ )
        for ( int_t x=0; x<=local_cols+1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                            ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                          + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                            ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                          + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                           ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                         + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
            );
        }
}


void
boundary_condition ( real_t *domain_variable, int sign )
{
    // get coordinates of process' subgrid
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);

    // MPI uses row-major ordering -> coords[0] is indicating which row
    bool is_x_zero = coords[1] == 0;
    bool is_x_max = coords[1] == dims[0] - 1;
    bool is_y_zero = coords[0] == 0;
    bool is_y_max = coords[0] == dims[1] - 1;

    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    if (is_x_zero) {
        if (is_y_zero) {
            // lower left corner
            VAR(   0, 0   ) = sign*VAR(   2, 2   );

            for ( int_t x=1; x<=local_cols; x++ ) VAR(   0, x   ) = sign*VAR(   2, x   );
        } else {
            if (is_y_max) {
                // upper left corner
                VAR( local_rows+1, 0   ) = sign*VAR( local_rows-1, 2   );

                for ( int_t x=1; x<=local_cols; x++ ) VAR( local_rows+1, x   ) = sign*VAR( local_rows-1, x   );
            }
        }

        for ( int_t y=1; y<=local_rows; y++ ) VAR(   y, 0   ) = sign*VAR(   y, 2   );
    } else {
        if (is_x_max) {
            if (is_y_zero) {
                // lower right corner
                VAR(   0, local_cols+1 ) = sign*VAR(   2, local_cols-1 );

                for ( int_t x=1; x<=local_cols; x++ ) VAR(   0, x   ) = sign*VAR(   2, x   );
            } else {
                if (is_y_max) {
                    // upper right corner
                    VAR( local_rows+1, local_cols+1 ) = sign*VAR( local_rows-1, local_cols-1 );
                    
                    for ( int_t x=1; x<=local_cols; x++ ) VAR( local_rows+1, x   ) = sign*VAR( local_rows-1, x   );
                }
            }
        }

        for ( int_t y=1; y<=local_rows; y++ ) VAR(   y, local_cols+1 ) = sign*VAR(   y, local_cols-1 );
    }

    #undef VAR
}


void
create_types ( void )
{
    int cart_rank, cart_offset[2];
    MPI_Comm_rank ( cart, &cart_rank );
    MPI_Cart_coords ( cart, cart_rank, 2, cart_offset);

    MPI_Type_create_subarray ( 2, (int[2]) { local_rows+2, local_cols+2 }, (int[2]) { local_rows, local_cols }, (int[2]) {1,1}, MPI_ORDER_C, MPI_DOUBLE, &subgrid );
    MPI_Type_create_subarray ( 2, (int[2]) {N, N} , (int[2]) { local_rows, local_cols }, (int[2]) { cart_offset[0] * local_rows, cart_offset[1] * local_cols}, MPI_ORDER_C, MPI_DOUBLE, &grid );

    MPI_Type_commit ( &subgrid );
    MPI_Type_commit ( &grid ) ;
}


void
domain_init ( void )
{
    // divide total size by size of dimension to obtain subgrid size
    local_rows  = N/dims[0];
    local_cols  = N/dims[1];

    int_t local_size = (local_rows + 2) * (local_cols + 2);

    mass[0] = calloc ( local_size, sizeof(real_t) );
    mass[1] = calloc ( local_size, sizeof(real_t) );

    mass_velocity_x[0] = calloc ( local_size, sizeof(real_t) );
    mass_velocity_x[1] = calloc ( local_size, sizeof(real_t) );
    mass_velocity_y[0] = calloc ( local_size, sizeof(real_t) );
    mass_velocity_y[1] = calloc ( local_size, sizeof(real_t) );

    mass_velocity = calloc ( local_size, sizeof(real_t) );

    velocity_x = calloc ( local_size, sizeof(real_t) );
    velocity_y = calloc ( local_size, sizeof(real_t) );

    acceleration_x = calloc ( local_size, sizeof(real_t) );
    acceleration_y = calloc ( local_size, sizeof(real_t) );

    // get coordinates of process' subgrid
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);

    // offset = position of process' subgrid * size of subgrid
    // needs to be done for x and y axis
    int_t local_x_offset = coords[1] * local_cols;
    int_t local_y_offset = coords[0] * local_rows;

    for ( int_t y=1; y<=local_rows; y++ )
    {
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PN(y,x) = 1e-3;
            PNU(y,x) = 0.0;
            PNV(y,x) = 0.0;

            real_t cx = (local_x_offset + x) - N/2;
            real_t cy = (local_y_offset + y) - N/2;

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


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    MPI_File out;
    MPI_File_open ( cart, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out );

    MPI_File_set_view ( out, 0, MPI_DOUBLE, grid, "native", MPI_INFO_NULL );
    MPI_File_write_all ( out, mass[0], 1, subgrid, MPI_STATUS_IGNORE );

    MPI_File_close ( &out );
}


void
communicate_borders( void ) {
    /*
    Grid Example:
        ?   ?   ?   ?   ?
        ?   1   2   3   ?
        ?   4   5   6   ?
        ?   7   8   9   ?
        ?   ?   ?   ?   ?
    
    Grid is saved as:
        [?, ?, ?, ?, ?,      ?, 1, 2, 3, ?,      ?, 4, 5, 6, ?,      ?, 7, 8, 9, ?,     ?, ?, ?, ?, ?]
    */

    // first exchange borders for mass then for mass velocity
    communicate_mass_borders();
    communicate_mass_velocity_borders();
}

void
communicate_mass_borders( void ) {
    // first exchange vertical borders, then horizontally
    exchange_vertical_borders(mass[0]);
    exchange_horizontal_borders(mass[0]);
}

void
communicate_mass_velocity_borders( void ) {
    // exchange mass_velocity for the x-axis first vertically, then horizontally
    exchange_vertical_borders(mass_velocity_x[0]);
    exchange_horizontal_borders(mass_velocity_x[0]);

    // exchange mass_velocity for the y-axis first vertically, then horizontally
    exchange_vertical_borders(mass_velocity_y[0]);
    exchange_horizontal_borders(mass_velocity_y[0]);

    // exchange mass_velocity first vertically, then horizontally
    exchange_vertical_borders(mass_velocity);
    exchange_horizontal_borders(mass_velocity);
}

void
exchange_vertical_borders( real_t *domain_variable ) {
    // "send down" then "send up"
    exchange_southern_borders(domain_variable);
    exchange_northern_borders(domain_variable);
}

void
exchange_horizontal_borders( real_t *domain_variable ) {
    // "send right" then "send left"
    exchange_eastern_borders(domain_variable);
    exchange_western_borders(domain_variable);
}

void
exchange_northern_borders( real_t *domain_variable ) {
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    // get first row of sub-grid
    real_t first_row[local_cols+2];
    for (int x = 0; x < local_cols+2; x++) {
        first_row[x] = VAR(1, x);
    }

    // direction: 0 = vertical; displacement: 1 = upwards shift
    real_t receive_buffer[local_cols+2];
    do_shift(0, 1, first_row, receive_buffer, local_cols+2);

    // save received message in row for southern border values
    for (int x = 0; x < local_cols+2; x++) {
        VAR(local_rows+1, x) = receive_buffer[x];
    }

    #undef VAR
}

void
exchange_southern_borders( real_t *domain_variable ) {
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    // get last row of sub-grid
    real_t last_row[local_cols+2];
    for (int x = 0; x < local_cols+2; x++) {
        last_row[x] = VAR(local_rows, x);
    }

    // direction: 0 = vertical; displacement: -1 = downwards shift
    real_t receive_buffer[local_cols+2];
    do_shift(0, 1, last_row, receive_buffer, local_cols+2);
    
    // save received message in row for northern border values
    for (int x = 0; x < local_cols+2; x++) {
        VAR(0, x) = receive_buffer[x];
    }

    #undef VAR
}

void
exchange_eastern_borders( real_t *domain_variable ) {
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    // get last column of sub-grid
    real_t last_column[local_rows+2];
    for (int y = 0; y < local_rows+2; y++) {
        last_column[y] = VAR(y+1, -2);
    }

    // direction: 1 = horizontal; displacement: 1 = eastern shift
    real_t receive_buffer[local_cols+2];
    do_shift(1, 1, last_column, receive_buffer, local_cols+2);

    // save received message in column for western border values
    for (int y = 0; y < local_rows+2; y++) {
        VAR(y, 0) = receive_buffer[y];
    }

    #undef VAR
}

void
exchange_western_borders( real_t *domain_variable ) {
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    // get first column of sub-grid
    real_t first_column[local_rows+2];
    for (int y = 0; y < local_rows+2; y++) {
        first_column[y] = VAR(y, 1);
    }

    // direction: 1 = horizontal; displacement: -1 = western shift
    real_t receive_buffer[local_cols+2];
    do_shift(1, -1, first_column, receive_buffer, local_cols+2);

    // save received message in column for eastern border values
    for (int y = 0; y < local_rows+2; y++) {
        VAR(y+1, -1) = receive_buffer[y];
    }

    #undef VAR
}

void
do_shift (int direction, int displacement, real_t *send_buff, real_t *recv_buff, int message_size) {
    // get source & destination process rank
    // source rank is the one that sends us their data
    // destination rank is the one that receives our data
    int source, dest;
    MPI_Cart_shift(cart, direction, displacement, &source, &dest);

    // send message
    MPI_Sendrecv(send_buff, message_size, MPI_DOUBLE, dest, 0, recv_buff, message_size, MPI_DOUBLE, source, 0, cart, MPI_STATUS_IGNORE);
}

void
domain_finalize ( void )
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
