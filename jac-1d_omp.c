#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

static void init_array (int n, float A[n], float B[n]) {
	int i;
	#pragma omp parallel for private(i)
	for (i = 0; i < n; i++) {
		A[i] = ((float) i+ 2) / n;
		B[i] = ((float) i+ 3) / n;
	}
}

static void print_array(int n, float A[n]) {
	int i
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

static void kernel_jacobi_1d(int tsteps, int n, float A[n], float B[n]) {
	int t, i;
	double tm1 = omp_get_wtime();
	#pragma omp parallel private(i, t) {
	for (t = 0; t < tsteps; t++) {
		#pragma omp for
		for (i = 1; i < n - 1; i++)
			B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
		#pragma omp for
		for (i = 1; i < n - 1; i++) 
			A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
		}
	}
	printf("%lf\n", omp_get_wtime() - tm1);
}


int main(int argc, char** argv) {
	int tmp_n, tmp_tsteps;
	if (argv[1] == 1) {
		printf("1_DATASET:\n");
		tmp_tsteps = 30;
		tmp_n = 40;
	} else if (argv[1] == 2) {
		printf("2_DATASET:\n");
		tmp_tsteps = 100;
		tmp_n = 150;
	} else if (argv[1] == 3) {
		printf("3_DATASET:\n");
		tmp_tsteps = 500;
		tmp_n = 700;
	} else if (argv[1] == 4) {
		printf("4_DATASET:\n");
		tmp_tsteps = 1000;
		tmp_n = 1500;
	} else if (argv[1] == 5) {
		printf("5_DATASET:\n");
		tmp_tsteps = 2000;
		tmp_n = 3000;
	} else if (argv[1] == 6) {
		printf("6_DATASET:\n");
		tmp_tsteps = 10000;
		tmp_n = 15000;
	} else if (argv[1] == 7) {
		printf("7_DATASET:\n");
		tmp_tsteps = 50000;
		tmp_n = 75000;
	} else if (argv[1] == 8) {
		printf("8_DATASET:\n");
		tmp_tsteps = 100000;
		tmp_n = 150000;
	} else {
		printf("DEFAULT:\n");
		tmp_tsteps = 10;
		tmp_n = 5;
	}

	int n = tmp_n;
	int tsteps = tmp_tsteps;
	float (*A)[n]; A = (float(*)[n])malloc ((n) * sizeof(float));
	float (*B)[n]; B = (float(*)[n])malloc ((n) * sizeof(float));

	init_array (n, *A, *B);

	kernel_jacobi_1d(tsteps, n, *A, *B);

	free((void*)A);
	free((void*)B);

	return 0;
}
