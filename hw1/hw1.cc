#include <mpi.h>

#include <boost/sort/spreadsort/float_sort.hpp>

void mergeSortedArrays(float arr1[], int size1, float arr2[], int size2, float mergedArray[], int mergedSize) {
    int i = 0, j = 0, k = 0;
    while (i < size1 && j < size2 && k < mergedSize) {
        if (arr1[i] < arr2[j]) {
            mergedArray[k++] = arr1[i++];
        } else {
            mergedArray[k++] = arr2[j++];
        }
    }
    while (i < size1 && k < mergedSize) {
        mergedArray[k++] = arr1[i++];
    }
    while (j < size2 && k < mergedSize) {
        mergedArray[k++] = arr2[j++];
    }
}
void mergeSortedArrays2(float arr1[], int size1, float arr2[], int size2, float mergedArray[], int mergedSize) {
    int i = size1 - 1, j = size2 - 1, k = mergedSize - 1;
    while (i >= 0 && j >= 0 && k >= 0) {
        if (arr1[i] > arr2[j]) {
            mergedArray[k--] = arr1[i--];
        } else {
            mergedArray[k--] = arr2[j--];
        }
    }
    while (i >= 0 && k >= 0) {
        mergedArray[k--] = arr1[i--];
    }
    while (j >= 0 && k >= 0) {
        mergedArray[k--] = arr2[j--];
    }
}

int main(int argc, char **argv) {
    // ====================initialization====================
    MPI_Init(&argc, &argv);
    int rank, size;
    int total_n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    MPI_File input_file, output_file;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int t_divided_s = total_n / size;
    int t_mod_s = total_n % size;
    int local_n = t_divided_s;
    if (rank == size - 1) local_n += t_mod_s;
    float *data = (float *)malloc(sizeof(float) * (local_n));
    float *tmp_data = (float *)malloc(sizeof(float) * (local_n));
    float *merge_data = (float *)malloc(sizeof(float) * (local_n + size));

    // ===================read file===================
    if (local_n / 2 == 0) {
        MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, sizeof(float) * rank * t_divided_s, data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

        // MPI_File_close(&input_file);
        boost::sort::spreadsort::float_sort(data, data + local_n);
    } else {
        MPI_Request request;
        MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_File_read_at(input_file, sizeof(float) * rank * t_divided_s, tmp_data, local_n / 2, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_iread_at(input_file, sizeof(float) * (rank * t_divided_s + local_n / 2), merge_data, (local_n + 1) / 2, MPI_FLOAT, &request);
        MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
        boost::sort::spreadsort::float_sort(tmp_data, tmp_data + local_n / 2);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        boost::sort::spreadsort::float_sort(merge_data, merge_data + (local_n + 1) / 2);
        mergeSortedArrays(tmp_data, local_n / 2, merge_data, (local_n + 1) / 2, data, local_n);
        // MPI_File_close(&input_file);
    }

    // ===================odd-even sort===================
    int odd_neighbor, even_neighbor;
    if (rank % 2 == 0) {
        odd_neighbor = rank - 1;
        even_neighbor = rank + 1;
    } else {
        odd_neighbor = rank + 1;
        even_neighbor = rank - 1;
    }
    int cnt = (rank == size - 2) ? (local_n + t_mod_s) : local_n;
    float recv_val;
    double start_comm;
    float *tmp_ptr;

    for (int i = 0; i < size + 1; ++i) {
        if (i & 1) {  // odd phase
            if (odd_neighbor == -1 || odd_neighbor == size) continue;
            if (rank & 1) {
                MPI_Sendrecv(&data[local_n - 1], 1, MPI_FLOAT, odd_neighbor, 0, &recv_val, 1, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[local_n - 1] > recv_val) {
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, odd_neighbor, 0, merge_data, cnt, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    mergeSortedArrays(data, local_n, merge_data, cnt, tmp_data, local_n);
                    tmp_ptr = data;
                    data = tmp_data;
                    tmp_data = tmp_ptr;
                }
            } else {
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, odd_neighbor, 0, &recv_val, 1, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[0] < recv_val) {
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, odd_neighbor, 0, merge_data, t_divided_s, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    mergeSortedArrays2(data, local_n, merge_data, t_divided_s, tmp_data, local_n);
                    tmp_ptr = data;
                    data = tmp_data;
                    tmp_data = tmp_ptr;
                }
            }
        } else {  // even phase
            if (even_neighbor == -1 || even_neighbor == size) continue;
            if (rank & 1) {
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, even_neighbor, 0, &recv_val, 1, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[0] < recv_val) {
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, even_neighbor, 0, merge_data, t_divided_s, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    mergeSortedArrays2(data, local_n, merge_data, t_divided_s, tmp_data, local_n);
                    tmp_ptr = data;
                    data = tmp_data;
                    tmp_data = tmp_ptr;
                }
            } else {
                MPI_Sendrecv(&data[local_n - 1], 1, MPI_FLOAT, even_neighbor, 0, &recv_val, 1, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[local_n - 1] > recv_val) {
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, even_neighbor, 0, merge_data, cnt, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    mergeSortedArrays(data, local_n, merge_data, cnt, tmp_data, local_n);
                    tmp_ptr = data;
                    data = tmp_data;
                    tmp_data = tmp_ptr;
                }
            }
        }
    }
    MPI_File_write_at(output_file, sizeof(float) * rank * t_divided_s, data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);

    // MPI_Finalize();
    return 0;
}