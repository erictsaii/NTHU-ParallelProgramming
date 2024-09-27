#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char** argv) {
    unsigned long long r = atoll(argv[1]);
    unsigned long long rr = r * r;
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
#pragma omp parallel for reduction(+ : pixels)
    for (unsigned long long x = 0; x < r; x++) {
        unsigned long long y = ceil(sqrtl(rr - x * x));
        pixels += y;
    }
    pixels %= k;
    printf("%llu\n", (4 * pixels) % k);
}
