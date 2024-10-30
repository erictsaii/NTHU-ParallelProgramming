#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <emmintrin.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
int cur_height;
int cur_width;
int* image;
void write_png(const char* filename, int iters, int width, int height, const int* buffer);

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    cpu_cnt = CPU_COUNT(&cpu_set);

    /* initialize */
    MPI_Init(&argc, &argv);
    // double total_start = MPI_Wtime();
    int rank, size;
    MPI_File output_file;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    int malloc_size = width * height * sizeof(int);
    image = (int*)malloc(malloc_size);
    memset(image, 0, malloc_size);
    int* final_image;
    if (rank == 0) {
        final_image = (int*)malloc(malloc_size);
        memset(final_image, 0, malloc_size);
    }

    assert(image);

    y0_offset = ((upper - lower) / height);
    x0_offest = ((right - left) / width);

/* mandelbrot set */
#pragma omp parallel for schedule(dynamic, 1)
    for (int j = rank; j < width; j += size) {
        for (int i = 0; i < height - 1; i += 2) {
            __m128d x0 = _mm_setzero_pd();
            __m128d y0 = _mm_setzero_pd();
            __m128d x = _mm_setzero_pd();
            __m128d y = _mm_setzero_pd();
            __m128d xy = _mm_setzero_pd();
            __m128d xx = _mm_setzero_pd();
            __m128d yy = _mm_setzero_pd();
            __m128d length_squared = _mm_setzero_pd();
            int repeats[2];

            x0[0] = x0[1] = j * x0_offest + left;
            y0[0] = i * y0_offset + lower;
            y0[1] = (i + 1) * y0_offset + lower;
            x[0] = x[1] = y[0] = y[1] = xy[0] = xy[1] = xx[0] = xx[1] = yy[0] = yy[1] = 0;
            length_squared[0] = length_squared[1] = 0;
            repeats[0] = repeats[1] = 0;

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
            for (int k = 0; k < 2; ++k) {
                while (length_squared[k] < 4 && repeats[k] < iters) {
                    xy = _mm_mul_pd(x, y);
                    y = _mm_add_pd(_mm_add_pd(xy, xy), y0);
                    x = _mm_add_pd(_mm_sub_pd(xx, yy), x0);
                    xx = _mm_mul_pd(x, x);
                    yy = _mm_mul_pd(y, y);
                    length_squared = _mm_add_pd(xx, yy);
                    ++repeats[k];
                }
            }
            int image_offset = j + i * width;
            image[image_offset] = repeats[0];
            image[image_offset + width] = repeats[1];
            // handle the last one when height is odd
            if (i == height - 3) {
                x0[0] = j * x0_offest + left;
                y0[0] = (i + 2) * y0_offset + lower;
                x[0] = x[1] = y[0] = y[1] = xy[0] = xy[1] = xx[0] = xx[1] = yy[0] = yy[1] = 0;
                length_squared[0] = 0;
                repeats[0] = 0;
                while (length_squared[0] < 4 && repeats[0] < iters) {
                    xy = _mm_mul_pd(x, y);
                    y = _mm_add_pd(_mm_add_pd(xy, xy), y0);
                    x = _mm_add_pd(_mm_sub_pd(xx, yy), x0);
                    xx = _mm_mul_pd(x, x);
                    yy = _mm_mul_pd(y, y);
                    length_squared = _mm_add_pd(xx, yy);
                    ++repeats[0];
                }
                // image[j + (i+2)*width] = repeats[0];
                image[image_offset + width + width] = repeats[0];
            }
            // update image_cnt
            // image_cnt += width+width;
        }
    }

    // double communicate_start = MPI_Wtime();
    MPI_Reduce(image, final_image, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // double communicate_time = MPI_Wtime() - communicate_start;

    /* draw and cleanup */
    if (rank == 0) {
        // double io_start = MPI_Wtime();
        write_png(filename, iters, width, height, final_image);
        // double io_time = MPI_Wtime() - io_start;
        // printf("IO time: %lf\n", io_time);
        // printf("communication time: %lf\n", communicate_time);
        // printf("computing time: %lf\n", MPI_Wtime() - total_start - io_time - communicate_time);
        free(final_image);
    }
    // printf("process %d: %lf s\n", rank, MPI_Wtime() - total_start);
    free(image);
    MPI_Finalize();
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
