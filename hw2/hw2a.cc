#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <emmintrin.h>
#include <immintrin.h>
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
double x0_offset;
int width;
int height;
int cpu_cnt;
pthread_mutex_t mutex;
int cur_width;
int* image;
void write_png(const char* filename, int iters, int width, int height, const int* buffer);

void* mandelbrot_col(void* threadid) {
    while (true) {
        int w = 0;
        bool last_one = false;
        pthread_mutex_lock(&mutex);
        w = cur_width;
        cur_width += 8;
        pthread_mutex_unlock(&mutex);
        if (w >= width) break;
        if (w + 8 >= width - 1) last_one = true;  // 有餘數，且這個thread是最後一個
        int z = (last_one) ? width - w : 8;
        __mmask8 mask = (1 << z) - 1;

        // initialize
        // __m512d x0 = _mm512_setzero_pd();
        __m512d y0 = _mm512_setzero_pd();
        __m512d x = _mm512_setzero_pd();
        __m512d y = _mm512_setzero_pd();
        __m512d xy = _mm512_setzero_pd();
        __m512d xx = _mm512_setzero_pd();
        __m512d yy = _mm512_setzero_pd();
        __m512d length_squared = _mm512_setzero_pd();
        __m512d two = _mm512_set1_pd(2.0);
        __m256i one = _mm256_set1_epi32(1);
        __m512d four = _mm512_set1_pd(4.0);

        // __m512d w_vec = _mm512_set_pd(w, w + 1.0, w + 2.0, w + 3.0, w + 4.0, w + 5.0, w + 6.0, w + 7.0);
        __m512d w_vec = _mm512_setzero_pd();
        for (int i = 0; i < 8; ++i) {
            w_vec[i] = w + i;
        }
        __m512d x0_offset_vec = _mm512_set1_pd(x0_offset);
        __m512d left_vec = _mm512_set1_pd(left);
        __m512d w_mul_x0 = _mm512_mul_pd(w_vec, x0_offset_vec);

        __m512d x0 = _mm512_add_pd(w_mul_x0, left_vec);

        // for (int i = 0; i < 8; ++i) {
        //     x0[i] = (w + i) * x0_offset + left;
        // }

        for (int i = 0; i < height; ++i) {
            // initialize
            y0 = _mm512_set1_pd(i * y0_offset + lower);
            x = _mm512_setzero_pd();
            y = _mm512_setzero_pd();
            xy = _mm512_setzero_pd();
            xx = _mm512_setzero_pd();
            yy = _mm512_setzero_pd();
            length_squared = _mm512_setzero_pd();
            __m256i repeats = _mm256_setzero_si256();

            // calculate

            int flags[8];
            int complete_num = 0;
            memset(flags, 0, sizeof(flags));

            for (int j = 0; j < iters; j++) {
                xx = _mm512_mul_pd(x, x);
                yy = _mm512_mul_pd(y, y);
                length_squared = _mm512_fmadd_pd(x, x, yy);

                __mmask8 finished_mask = _mm512_cmplt_pd_mask(length_squared, four);
                finished_mask &= mask;
                if (finished_mask == 0) break;
                repeats = _mm256_mask_add_epi32(repeats, finished_mask, repeats, one);

                xy = _mm512_mul_pd(x, y);
                y = _mm512_fmadd_pd(xy, two, y0);
                x = _mm512_add_pd(_mm512_sub_pd(xx, yy), x0);
            }
            _mm256_mask_storeu_epi32((__m256*)&image[w + width * i], mask, repeats);
        }
    }
    // pthread_exit(NULL);
    return NULL;
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
    x0_offset = ((right - left) / width);

    /*start mandelbrot*/
    for (int i = 0; i < cpu_cnt; ++i) {
        pthread_create(&threads[i], NULL, mandelbrot_col, NULL);
    }

    for (int i = 0; i < cpu_cnt; ++i) {
        pthread_join(threads[i], NULL);
    }

    write_png(filename, iters, width, height, image);
    // free(image);
    return 0;
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
    png_set_filter(png_ptr, 0, PNG_FILTER_NONE);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);
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
                    color[1] = color[2] = (p & 15) << 4;
                } else {
                    color[0] = (p & 15) << 4;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    // free(row);
    png_write_end(png_ptr, NULL);
    // png_destroy_write_struct(&png_ptr, &info_ptr);
    // fclose(fp);
}
