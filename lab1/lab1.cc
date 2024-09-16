#include <math.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long local_sum = 0, global_sum = 0;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long threshold = ceil(sqrtl(r * r / 2));
    unsigned long long tmp = 0;

    for (unsigned long long x = rank; x < r; x += size) {
        tmp = ceil(sqrtl((r + x) * (r - x)));
        if (tmp <= threshold) break;
        local_sum += tmp - threshold;
    }
    local_sum %= k;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
        printf("%llu\n",
               (4 * (global_sum * 2 + (threshold * threshold) % k) % k) % k);
    MPI_Finalize();
}
