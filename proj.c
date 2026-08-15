#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define DIRECTIONS 6

// Adjustable grid size
int X_NODES = 300;    // Number of nodes along the long direction
int Y_NODES = 100;    // Number of nodes along the short direction
int STEPS = 100;      // Number of simulation steps (increased to 100)

// Directions based on hexagonal grid angles (0, 60, 120, 180, 240, 300 degrees)
const double dx[DIRECTIONS] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
const double dy[DIRECTIONS] = {0.0, 0.86602540378, 0.86602540378, 0.0, -0.86602540378, -0.86602540378};

int ***grid;
int **obstacle;
FILE *data_file;
int rank, size, rows_per_process, rows_start, rows_end;

// Function prototypes
void allocate_memory();
void free_memory();
void initialize_grid();
void initialize_obstacle();
void log_grid_state(int step);
void collision();
void streaming();
void apply_boundary_conditions();
void exchange_boundaries();
void compute_density_and_velocity();
void block_average_density_velocity();
void debug_state();

// Function definitions
void allocate_memory() {
    grid = (int ***)malloc(X_NODES * sizeof(int **));
    obstacle = (int **)malloc(X_NODES * sizeof(int *));
    if (!grid || !obstacle) {
        fprintf(stderr, "Memory allocation failed for grid or obstacle\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int x = 0; x < X_NODES; x++) {
        grid[x] = (int **)malloc(Y_NODES * sizeof(int *));
        obstacle[x] = (int *)calloc(Y_NODES, sizeof(int));
        if (!grid[x] || !obstacle[x]) {
            fprintf(stderr, "Memory allocation failed at grid[%d] or obstacle[%d]\n", x, x);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int y = 0; y < Y_NODES; y++) {
            grid[x][y] = (int *)calloc(DIRECTIONS, sizeof(int));
            if (!grid[x][y]) {
                fprintf(stderr, "Memory allocation failed at grid[%d][%d]\n", x, y);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }
}

void free_memory() {
    if (grid) {
        for (int x = 0; x < X_NODES; x++) {
            if (grid[x]) {
                for (int y = 0; y < Y_NODES; y++) {
                    if (grid[x][y]) {
                        free(grid[x][y]); // Free each cell
                        grid[x][y] = NULL; // Avoid dangling pointers
                    }
                }
                free(grid[x]); // Free each row
                grid[x] = NULL;
            }
            if (obstacle[x]) {
                free(obstacle[x]); // Free obstacle row
                obstacle[x] = NULL;
            }
        }
        free(grid); // Free grid
        free(obstacle); // Free obstacle
        grid = NULL;
        obstacle = NULL;
    }
}

void initialize_grid() {
    srand(time(NULL));  // Seed the random number generator
    double population_density = 0.1; // Set a population density (probability of a particle in each cell)

    // Loop through each grid point and decide randomly if a particle should be present
    for (int x = rows_start; x < rows_end; x++) {
        for (int y = 0; y < Y_NODES; y++) {
            if (obstacle[x][y] == 0) {  // No particles inside the obstacle
                if ((rand() / (double)RAND_MAX) < population_density) {
                    int random_direction = rand() % DIRECTIONS;
                    grid[x][y][random_direction] = 1;
                }
            }
        }
    }
}

void initialize_obstacle() {
    int cx = 8, cy = 4, radius = 1;
    for (int x = cx - radius; x <= cx + radius; x++) {
        for (int y = cy - radius; y <= cy + radius; y++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius) {
                if (x >= 0 && x < X_NODES && y >= 0 && y < Y_NODES) {
                    obstacle[x][y] = 1;  // Mark obstacle
                }
            }
        }
    }
}

void log_grid_state(int step) {
    char filename[50];
    sprintf(filename, "simulation_output_%d.txt", size);

    if (rank == 0) {
        data_file = fopen(filename, "a");
        if (!data_file) {
            fprintf(stderr, "Could not open output file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(data_file, "Step: %d\n", step);
        for (int x = rows_start; x < rows_end; x++) {
            for (int y = 0; y < Y_NODES; y++) {
                for (int d = 0; d < DIRECTIONS; d++) {
                    if (grid[x][y][d]) {
                        fprintf(data_file, "%d %d %d\n", x, y, d);
                    }
                }
            }
        }
        fprintf(data_file, "EndStep\n");
        fclose(data_file);
    }
}

void collision() {
    for (int x = rows_start; x < rows_end; x++) {
        for (int y = 0; y < Y_NODES; y++) {
            if (obstacle[x][y]) continue;

            int state = 0;
            for (int d = 0; d < DIRECTIONS; d++) {
                state |= grid[x][y][d] << d;
            }

            if (state == 0b000011 || state == 0b110000) {
                state = (rand() % 2) ? 0b110000 : 0b000011;
            } else if (__builtin_popcount(state) == 3 && (state & 0b010101) == 0b010101) {
                state = (~state) & 0b111111;
            } else if (__builtin_popcount(state) == 4 && (state & 0b000011) == 0) {
                state ^= 0b000011;
            } else if (__builtin_popcount(state) == 3) {
                state ^= 0b010010;
            }

            for (int d = 0; d < DIRECTIONS; d++) {
                grid[x][y][d] = (state >> d) & 1;
            }
        }
    }
}

void streaming() {
    int ***new_grid = (int ***)malloc(X_NODES * sizeof(int **));
    if (!new_grid) {
        fprintf(stderr, "Memory allocation failed for new_grid\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int x = 0; x < X_NODES; x++) {
        new_grid[x] = (int **)malloc(Y_NODES * sizeof(int *));
        for (int y = 0; y < Y_NODES; y++) {
            new_grid[x][y] = (int *)calloc(DIRECTIONS, sizeof(int));
        }
    }

    for (int x = rows_start; x < rows_end; x++) {
        for (int y = 0; y < Y_NODES; y++) {
            for (int d = 0; d < DIRECTIONS; d++) {
                if (grid[x][y][d]) {
                    int nx = (x + (int)round(dx[d]) + X_NODES) % X_NODES;
                    int ny = (y + (int)round(dy[d]) + Y_NODES) % Y_NODES;

                    if (obstacle[nx][ny]) {
                        new_grid[x][y][(d + 3) % DIRECTIONS] = 1;
                    } else {
                        new_grid[nx][ny][d] = 1;
                    }
                }
            }
        }
    }

    for (int x = rows_start; x < rows_end; x++) {
        for (int y = 0; y < Y_NODES; y++) {
            for (int d = 0; d < DIRECTIONS; d++) {
                grid[x][y][d] = new_grid[x][y][d];
            }
            free(new_grid[x][y]);
        }
        free(new_grid[x]);
    }
    free(new_grid);
}

void apply_boundary_conditions() {
    for (int y = 0; y < Y_NODES; y++) {
        grid[0][y][0] = 1;
        grid[X_NODES - 1][y][0] = 0;
    }

    for (int x = 0; x < X_NODES; x++) {
        for (int d = 0; d < DIRECTIONS; d++) {
            if (grid[x][Y_NODES - 1][d]) {
                grid[x][Y_NODES - 1][(d + 3) % DIRECTIONS] = grid[x][Y_NODES - 1][d];
                grid[x][Y_NODES - 1][d] = 0;
            }
            if (grid[x][0][d]) {
                grid[x][0][(d + 3) % DIRECTIONS] = grid[x][0][d];
                grid[x][0][d] = 0;
            }
        }
    }
}

void exchange_boundaries() {
    MPI_Status status;

    if (rank > 0) {
        MPI_Sendrecv(&grid[rows_start][0][0], Y_NODES * DIRECTIONS, MPI_INT, rank - 1, 0,
                     &grid[rows_start - 1][0][0], Y_NODES * DIRECTIONS, MPI_INT, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
    }

    if (rank < size - 1) {
        MPI_Sendrecv(&grid[rows_end - 1][0][0], Y_NODES * DIRECTIONS, MPI_INT, rank + 1, 0,
                     &grid[rows_end][0][0], Y_NODES * DIRECTIONS, MPI_INT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
    }
}

void compute_density_and_velocity() {
    char filename[50];
    sprintf(filename, "density_velocity_%d.txt", size);

    if (rank == 0) {
        data_file = fopen(filename, "a");
        if (!data_file) {
            fprintf(stderr, "Could not open output file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        fprintf(data_file, "Density and Velocity:\n");
        for (int x = rows_start; x < rows_end; x++) {
            for (int y = 0; y < Y_NODES; y++) {
                int density = 0;
                double vx = 0, vy = 0;
                for (int d = 0; d < DIRECTIONS; d++) {
                    density += grid[x][y][d];
                    vx += grid[x][y][d] * dx[d];
                    vy += grid[x][y][d] * dy[d];
                }
                fprintf(data_file, "%d %d %d %.2f %.2f\n", x, y, density, vx, vy);
            }
        }
        fclose(data_file);
    }
}

void block_average_density_velocity() {
    // Implement the block averaging logic
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows_per_process = X_NODES / size;
    rows_start = rank * rows_per_process;
    rows_end = (rank == size - 1) ? X_NODES : (rank + 1) * rows_per_process;

    allocate_memory();
    initialize_obstacle();
    initialize_grid();

    for (int step = 0; step < STEPS; step++) {
        collision();
        streaming();
        apply_boundary_conditions();
        exchange_boundaries();

        log_grid_state(step);
        compute_density_and_velocity();
    }

    free_memory();
    MPI_Finalize();
    return 0;
}

