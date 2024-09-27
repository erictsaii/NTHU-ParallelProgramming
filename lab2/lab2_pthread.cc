#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct arg {
    unsigned long long tid;
    unsigned long long r;
    unsigned long long rr;
    unsigned long long k;
    unsigned long long ncpus;
    unsigned long long threshold;
};

void *cal(void *threads_args) {
    unsigned long long *local_sum = (unsigned long long *)malloc(sizeof(unsigned long long));
    *local_sum = 0;

    struct arg *args = (struct arg *)threads_args;
    unsigned long long tid = args->tid;
    unsigned long long r = args->r;
    unsigned long long rr = args->rr;
    unsigned long long k = args->k;
    unsigned long long ncpus = args->ncpus;
    unsigned long long threshold = args->threshold;
    unsigned long long tmp = 0;

    for (unsigned long long x = tid; x < r; x += ncpus) {
        tmp = ceil(sqrtl(rr - x * x));
        if (tmp <= threshold) break;
        *local_sum += tmp - threshold;
    }
    *local_sum %= k;

    pthread_exit((void *)local_sum);
}

int main(int argc, char **argv) {
    unsigned long long r = atoll(argv[1]);
    unsigned long long rr = r * r;
    unsigned long long k = atoll(argv[2]);
    unsigned long long threshold = ceil(sqrtl(rr / 2));

    // get ncpus
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    unsigned long long ncpus = CPU_COUNT(&cpuset);

    // pthreads
    pthread_t threads[ncpus];
    struct arg threads_args[ncpus];
    unsigned long long global_sum = 0;

    for (unsigned long long i = 0; i < ncpus; ++i) {
        threads_args[i].tid = i;
        threads_args[i].r = r;
        threads_args[i].rr = rr;
        threads_args[i].k = k;
        threads_args[i].ncpus = ncpus;
        threads_args[i].threshold = threshold;
        pthread_create(&threads[i], NULL, cal, (void *)&threads_args[i]);
    }

    for (unsigned long long i = 0; i < ncpus; ++i) {
        unsigned long long *local_sum = NULL;
        pthread_join(threads[i], (void **)&local_sum);
        global_sum += *local_sum;
        global_sum %= k;
    }

    printf("%llu\n", (4 * ((global_sum * 2 + (threshold * threshold) % k) % k)) % k);
    return 0;
}
