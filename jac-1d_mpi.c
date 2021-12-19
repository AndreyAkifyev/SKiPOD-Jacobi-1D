#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

double bench_t_start, bench_t_end;

int nProcs, id, block, left, right;

MPI_Status status;

static void init_array (int n, float B[n]) {
    int i;
    for (i = 0; i < n; i++) {
        B[i] = ((float) i+ 2) / n;
    }
}

static void print_array(int n, float A[n]) {
    int i;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "A");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) 
            fprintf(stderr, "\n");
        fprintf(stderr, "%0.2f ", A[i]);
    }
    fprintf(stderr, "\nend   dump: %s\n", "A");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void calc(int n, float B[n]) {
    int i;
    for (i = left; i < right; i++) {
        B[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
}

static void kernel_jacobi_1d(int steps, int n, float B[n]) {
    int t;
    MPI_Status status;
    block = (n - 2) / nProcs;
    if (id == nProcs - 1) {
        left = id * block + 1;
        right = n - 1;
    } else {
        left = id * block + 1;
        right = (id + 1) * block + 1;
    }
    for (t = 0; t < steps * 2; t++) {
        calc(n, B);
        if (t < steps * 2 - 1) {
            if (id == 0) {
                MPI_Send(&(B[right- 1]),  1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&(B[right]),     1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD, &status);
            } else if (id == nProcs - 1) {
                MPI_Recv(&(B[left - 1]),  1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD, &status); 
                MPI_Send(&(B[left]),      1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD);
            } else {
                MPI_Recv(&(B[left - 1]),  1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&(B[right - 1]), 1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&(B[right]),     1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&(B[left]),      1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } else {
            MPI_Gather(&(B[id * block + 1]), block, MPI_FLOAT, &(B[1]), block, MPI_FLOAT, nProcs - 1, MPI_COMM_WORLD);
        } 
    }
}


int main(int argc, char** argv) {
    int n_arr[] = {40, 150, 700, 1500, 3000, 15000, 75000, 150000};
    int steps_arr[] = {30, 100, 500, 1000, 2000, 10000, 50000, 100000};
    int n, steps;
    double tm1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int i;
    for (int i = 0; i < 8; i++) {
        n = n_arr[i];
        steps = steps_arr[i];
  
        float (*B)[n]; 
        B = (float(*)[n]) malloc ((n) * sizeof(float));

        init_array(n, *B);
        if (id == nProcs - 1) {
            printf("%d, %d\n", n, steps);
            tm1 = MPI_Wtime();
        }
        kernel_jacobi_1d(steps, n, *B);
        if (id == nProcs - 1) {
            printf("Time: %lf\n", MPI_Wtime() - tm1);
            print_array(n, *B);
        }
    
        free((void*)B);
    }
    MPI_Finalize();
    return 0;
}