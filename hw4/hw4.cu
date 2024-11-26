#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#define d_max 64
#define bc 32
#define br 32

void input(char *input_filename);
void output(char *output_filename);
__global__ void flash_attention(float *q, float *k, float *v, float *o, int d, int tc, int N);

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    input(argv[1]);

    int tr = N / br, tc = N / bc;
    float *d_q, *d_k, *d_v, *d_o;
    size_t BND = B * N * d * sizeof(float);

    // Q
    cudaHostRegister(Q, BND, cudaHostRegisterDefault);
    cudaMalloc(&d_q, BND);
    cudaMemcpy(d_q, Q, BND, cudaMemcpyHostToDevice);
    // K
    cudaHostRegister(K, BND, cudaHostRegisterDefault);
    cudaMalloc(&d_k, BND);
    cudaMemcpy(d_k, K, BND, cudaMemcpyHostToDevice);
    // V
    cudaHostRegister(V, BND, cudaHostRegisterDefault);
    cudaMalloc(&d_v, BND);
    cudaMemcpy(d_v, V, BND, cudaMemcpyHostToDevice);
    // O
    // cudaHostRegister(O, BND, cudaHostRegisterDefault);
    cudaMalloc(&d_o, BND);
    // cudaMemcpy(d_o, O, BND, cudaMemcpyHostToDevice);
    
    // grid size and block size
    dim3 grid_size(tr, B);
    dim3 block_size(32, 32); // 32 * 32 threads

    // kernel function
    flash_attention<<<grid_size, block_size>>>(d_q, d_k, d_v, d_o, d, tc, N);

    // copy the output to host
    cudaMemcpy(O, d_o, BND, cudaMemcpyDeviceToHost);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    size_t BND = B * N * d * sizeof(float);
    Q = (float *)malloc(BND);
    K = (float *)malloc(BND);
    V = (float *)malloc(BND);
    O = (float *)malloc(BND);

    int Nd = N * d;
    int offset = 0;

    for (int i = 0; i < B; ++i) {
        fread(Q + offset, sizeof(float), Nd, file);
        fread(K + offset, sizeof(float), Nd, file);
        fread(V + offset, sizeof(float), Nd, file);
        offset += Nd;
    }

    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}


__global__ void flash_attention(float *q, float *k, float *v, float *o, int d, int tc, int N) {
    __shared__ float kj[bc * d_max]; 
    __shared__ float vj[bc * d_max];
    __shared__ float qi[br * d_max];
    __shared__ float oi[br * d_max];
    __shared__ float li[br];
    __shared__ float li_new[br];
    __shared__ float pij[br * bc];

    float mij = 3.0;
    int qo_offset = blockIdx.y * N * d + blockIdx.x * br * d;
    float sqrt_d = 1.0 / sqrtf(d);
    float pv = 0.0F;
    float tmp;

    // load qi
    for (int i = 0; i < d; i += 32) {
        qi[threadIdx.y * d + threadIdx.x + i] = q[qo_offset + threadIdx.y * d + threadIdx.x + i];
    }
    
    // load oi
    for (int i = 0; i < d; i += 32) {
        oi[threadIdx.y * d + threadIdx.x + i] = 0.0;
    }
   
    // load li
    if (threadIdx.y == 0) {
        li[threadIdx.x] = 0.0;
    }


    // start for-loop
    for (int j = 0; j < tc; ++j) {
        // load kj 
        int kjvj_offset = blockIdx.y * N * d + j * bc * d;
        for (int i = 0; i < d; i += 32) {
            kj[threadIdx.y * d + threadIdx.x + i] = k[kjvj_offset + threadIdx.y * d + threadIdx.x + i];
        }

        // load vj
        for (int i = 0; i < d; i += 32) {
            vj[threadIdx.y * d + threadIdx.x + i] = v[kjvj_offset + threadIdx.y * d + threadIdx.x + i];
        }
        __syncthreads();
        
        // QKDotAndScalar
        tmp = 0.0F;
        for (int t = 0; t < d; t++) {
            tmp += qi[threadIdx.y * d + t] * kj[threadIdx.x * d + t];
        }
        tmp *= sqrt_d;

        // MinusMaxAndExp
        pij[threadIdx.y * bc + threadIdx.x] = expf(tmp - mij);
        __syncthreads();

        // RowSum
        if (threadIdx.y == 0) {
            tmp = 0.0F;
            for (int i = 0; i < bc; ++i) {
                tmp += pij[threadIdx.x * bc + i];
            }
        }

        // UpdateMiLiOi
        if (threadIdx.y == 0) {
            li_new[threadIdx.x] = li[threadIdx.x] + tmp;
        }
        __syncthreads();
        for (int i = 0; i < d; i += 32) {
            pv = 0.0F;
            for (int t = 0; t < bc; ++t) {
                pv += pij[threadIdx.y * bc + t] * vj[t * d + threadIdx.x + i];
            } 
            oi[threadIdx.y * d + threadIdx.x + i] = (li[threadIdx.y]  * oi[threadIdx.y * d + threadIdx.x + i] + pv) / li_new[threadIdx.y];
        }
        if (threadIdx.y == 0) {
            li[threadIdx.x] = li_new[threadIdx.x];
        }
    }
  
    // update o
    for (int i = 0; i < d; i += 32) {
        o[qo_offset + threadIdx.y * d + threadIdx.x + i] = oi[threadIdx.y * d + threadIdx.x + i];
    }
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    fclose(file);
}