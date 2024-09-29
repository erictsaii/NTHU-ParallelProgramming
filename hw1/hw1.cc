#include <math.h>
#include <mpi.h>

#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cstdio>
#include <cstdlib>
using namespace std;

void merge(float nums1[], int m, float nums2[], int n) {
    int i = m - 1;
    int j = n - 1;
    int k = m + n - 1;
    while (j >= 0) {
        if (i >= 0 && nums1[i] > nums2[j])
            nums1[k--] = nums1[i--];
        else
            nums1[k--] = nums2[j--];
    }
}
void merge_front(float nums1[], int m, float nums2[], int n) {
    int i = 0;
    int j = 0;
    int k = m + n - 1;
    while (k >= m) {
        if (i < m && nums1[i] < nums2[j])
            nums1[k--] = nums1[i++];
        else
            nums1[k--] = nums2[j++];
    }
}
void merge_back(float nums1[], int m, float nums2[], int n) {
    int i = m - 1;
    int j = n - 1;
    int k = m + n - 1;
    while (k >= m) {
        if (i >= 0 && nums1[i] > nums2[j])
            nums1[k--] = nums1[i--];
        else
            nums1[k--] = nums2[j--];
    }
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    double total_start = MPI_Wtime();
    double communicate_time = 0.0;
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
    float *merge_data = (float *)malloc(sizeof(float) * (local_n * 2 + size));

    double read_start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * rank * t_divided_s, data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    double read_time = MPI_Wtime() - read_start;

    boost::sort::spreadsort::spreadsort(data, data + local_n);

    int odd_neighbor, even_neighbor;
    if (rank % 2 == 0) {
        odd_neighbor = rank - 1;
        even_neighbor = rank + 1;
    } else {
        odd_neighbor = rank + 1;
        even_neighbor = rank - 1;
    }
    int cnt = (rank == size - 2) ? (local_n + t_mod_s) : local_n;
    int test = cnt + local_n - 1;
    float recv_val;
    double start_comm;
    for (int i = 0; i < size + 1; ++i) {
        if (i & 1) {  // odd phase
            if (odd_neighbor == -1 || odd_neighbor == size) continue;
            if (rank & 1) {
                start_comm = MPI_Wtime();
                MPI_Sendrecv(&data[local_n - 1], 1, MPI_FLOAT, odd_neighbor, 0, &recv_val, 1, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                communicate_time += MPI_Wtime() - start_comm;
                if (data[local_n - 1] > recv_val) {
                    start_comm = MPI_Wtime();
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, odd_neighbor, 0, merge_data, cnt, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    communicate_time += MPI_Wtime() - start_comm;
                    merge_front(merge_data, cnt, data, local_n);
                    for (int j = 0; j < local_n; ++j) {
                        data[j] = merge_data[test - j];
                    }
                }
            } else {
                start_comm = MPI_Wtime();
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, odd_neighbor, 0, &recv_val, 1, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                communicate_time += MPI_Wtime() - start_comm;
                if (data[0] < recv_val) {
                    start_comm = MPI_Wtime();
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, odd_neighbor, 0, merge_data, t_divided_s, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    communicate_time += MPI_Wtime() - start_comm;
                    merge_back(merge_data, t_divided_s, data, local_n);
                    for (int j = 0; j < local_n; ++j) {
                        data[j] = merge_data[j + t_divided_s];
                    }
                }
            }
        } else {  // even phase
            if (even_neighbor == -1 || even_neighbor == size) continue;
            if (rank & 1) {
                start_comm = MPI_Wtime();
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, even_neighbor, 0, &recv_val, 1, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                communicate_time += MPI_Wtime() - start_comm;
                if (data[0] < recv_val) {
                    start_comm = MPI_Wtime();
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, even_neighbor, 0, merge_data, t_divided_s, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    communicate_time += MPI_Wtime() - start_comm;
                    merge_back(merge_data, t_divided_s, data, local_n);
                    for (int j = 0; j < local_n; ++j) {
                        data[j] = merge_data[j + t_divided_s];
                    }
                }
            } else {
                start_comm = MPI_Wtime();
                MPI_Sendrecv(&data[local_n - 1], 1, MPI_FLOAT, even_neighbor, 0, &recv_val, 1, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                communicate_time += MPI_Wtime() - start_comm;
                if (data[local_n - 1] > recv_val) {
                    start_comm = MPI_Wtime();
                    MPI_Sendrecv(data, local_n, MPI_FLOAT, even_neighbor, 0, merge_data, cnt, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    communicate_time += MPI_Wtime() - start_comm;
                    merge_front(merge_data, cnt, data, local_n);
                    for (int j = 0; j < local_n; ++j) {
                        data[j] = merge_data[test - j];
                    }
                }
            }
        }
    }

    // for (int i = 0;i < size+1;i++){
    //     if (i % 2){ //odd phase
    //         if (odd_neighbor==-1 ||odd_neighbor==size) continue;
    //         if (rank % 2){
    //             MPI_Recv(merge_data, cnt, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             merge(merge_data, cnt, data, local_n);
    //             MPI_Send(merge_data+local_n, cnt, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD);
    //             for (int j=0;j < local_n;j++){
    //                 data[j] = merge_data[j];
    //             }
    //         }else{
    //             MPI_Send(data, local_n, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD);
    //             MPI_Recv(data, local_n, MPI_FLOAT, odd_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         }
    //     }else{ //even
    //         if (even_neighbor==-1 ||even_neighbor==size) continue;
    //         if (rank % 2){
    //             MPI_Send(data, local_n, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD);
    //             MPI_Recv(data, local_n, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         }else{
    //             MPI_Recv(merge_data, cnt, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //             merge(merge_data, cnt, data, local_n);
    //             MPI_Send(merge_data+local_n, cnt, MPI_FLOAT, even_neighbor, 0, MPI_COMM_WORLD);
    //             for (int j=0;j < local_n;j++){
    //                 data[j] = merge_data[j];
    //             }
    //         }
    //     }
    // }
    double write_start = MPI_Wtime();
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * rank * t_divided_s, data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    double write_time = MPI_Wtime() - write_start;

    double io_time = read_time + write_time;
    double compute_time = MPI_Wtime() - total_start - io_time;
    double io_sum, compute_sum, communicate_sum;
    MPI_Reduce(&io_time, &io_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &compute_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&communicate_time, &communicate_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        io_time = io_sum / size;
        compute_time = compute_sum / size;
        communicate_time = communicate_sum / size;
        printf("IO time: %f\n", io_time);
        printf("compute time: %f\n", compute_time);
        printf("communicate time: %f\n", communicate_time);
    }

    MPI_Finalize();
    return 0;
}
