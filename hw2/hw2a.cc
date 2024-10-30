#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <emmintrin.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

int iters;
double left;
double right;
double lower;
double upper;
double y0_offset;
double x0_offest;
int width;
int height;
int cpu_cnt;
pthread_mutex_t mutex;
int cur_width;
int* image;
void write_png(const char* filename, int iters, int width, int height, const int* buffer);

void* mandelbrot_col(void* threadid) {
    while (true) {
        int w;
        bool last_one = false;
        pthread_mutex_lock(&mutex);
        w = cur_width;
        cur_width += 2;
        pthread_mutex_unlock(&mutex);
        if (w == width - 1) last_one = true;
        if (w >= width) break;

        // initialize
        __m128d x0 = _mm_setzero_pd();
        __m128d y0 = _mm_setzero_pd();
        __m128d x = _mm_setzero_pd();
        __m128d y = _mm_setzero_pd();
        __m128d xy = _mm_setzero_pd();
        __m128d xx = _mm_setzero_pd();
        __m128d yy = _mm_setzero_pd();
        __m128d length_squared = _mm_setzero_pd();
        x0[0] = w * x0_offest + left;
        x0[1] = (w + 1) * x0_offest + left;
        int image_cnt = 0;
        int repeats[2];

        for (int i = 0; i < height; ++i) {
            // initialize
            y0[0] = y0[1] = i * y0_offset + lower;
            repeats[0] = repeats[1] = 0;
            x[0] = x[1] = y[0] = y[1] = xy[0] = xy[1] = xx[0] = xx[1] = yy[0] = yy[1] = 0;
            length_squared[0] = length_squared[1] = 0;
            // calculate
            while (length_squared[0] < 4 && length_squared[1] < 4 && repeats[0] < iters) {
                xy = _mm_mul_pd(x, y);
                y = _mm_add_pd(_mm_add_pd(xy, xy), y0);
                x = _mm_add_pd(_mm_sub_pd(xx, yy), x0);
                xx = _mm_mul_pd(x, x);
                yy = _mm_mul_pd(y, y);
                length_squared = _mm_add_pd(xx, yy);
                ++repeats[0];
                ++repeats[1];
            }
            // maybe someone hasn't finished yet
            int z = (last_one) ? 1 : 2;
            for (int i = 0; i < z; ++i) {
                while (length_squared[i] < 4 && repeats[i] < iters) {
                    xy = _mm_mul_pd(x, y);
                    y = _mm_add_pd(_mm_add_pd(xy, xy), y0);
                    x = _mm_add_pd(_mm_sub_pd(xx, yy), x0);
                    xx = _mm_mul_pd(x, x);
                    yy = _mm_mul_pd(y, y);
                    length_squared = _mm_add_pd(xx, yy);
                    ++repeats[i];
                }
            }
            image[w + image_cnt] = repeats[0];
            if (!last_one) image[w + 1 + image_cnt] = repeats[1];
            image_cnt += width;
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    cpu_cnt = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /*initialize*/
    pthread_t threads[cpu_cnt];
    cur_width = 0;
    y0_offset = ((upper - lower) / height);
    x0_offest = ((right - left) / width);

    /*start mandelbrot*/
    for (int i = 0; i < cpu_cnt; ++i) {
        pthread_create(&threads[i], NULL, mandelbrot_col, NULL);
    }
    for (int i = 0; i < cpu_cnt; ++i) {
        pthread_join(threads[i], NULL);
    }

    write_png(filename, iters, width, height, image);
    free(image);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p & 15) * 16;
                } else {
                    color[0] = (p & 15) * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
