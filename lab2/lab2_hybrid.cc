#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    unsigned long long r = atoll(argv[1]);
    unsigned long long rr = r * r;
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
    int rank, size;
    unsigned long long sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#pragma omp parallel for reduction(+ : pixels)
    for (unsigned long long x = rank; x < r; x += size) {
        pixels += ceil(sqrtl(rr - x * x));
    }
    pixels %= k;
    MPI_Reduce(&pixels, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%llu\n", (4 * sum) % k);
    }

    MPI_Finalize();
    return 0;
}
