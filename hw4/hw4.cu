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
void launch_flash_attention(float *q, float *k, float *v, float *o);

__global__ void flash_attention(float *q, float *k, float *v, float *o, float *l, float *m, int d, int tc);

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    input(argv[1]);

    for (int i = 0; i < B; i++) {
        launch_flash_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d)
        );
    }

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

void launch_flash_attention(float *q, float *k, float *v, float *o) {
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    int tr = N / br, tc = N / bc;
    float *d_q, *d_k, *d_v, *d_o, *d_l, *d_m;
    // Q
    cudaMalloc(&d_q, N*d*sizeof(float));
    cudaMemcpy(d_q, q, N*d*sizeof(float), cudaMemcpyHostToDevice);
    // K
    cudaMalloc(&d_k, N*d*sizeof(float));
    cudaMemcpy(d_k, k, N*d*sizeof(float), cudaMemcpyHostToDevice);
    // V
    cudaMalloc(&d_v, N*d*sizeof(float));
    cudaMemcpy(d_v, v, N*d*sizeof(float), cudaMemcpyHostToDevice);
    // O
    cudaMalloc(&d_o, N*d*sizeof(float));
    cudaMemcpy(d_o, o, N*d*sizeof(float), cudaMemcpyHostToDevice);
    // l
    cudaMalloc(&d_l, N*sizeof(float));
    cudaMemcpy(d_l, l, N*sizeof(float), cudaMemcpyHostToDevice);
    // m
    cudaMalloc(&d_m, N*sizeof(float));
    cudaMemcpy(d_m, m, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // grid size and block size
    dim3 grid_size(tr);
    dim3 block_size(32, 32); // 32 * 32 threads

    // kernel function
    flash_attention<<<grid_size, block_size>>>(d_q, d_k, d_v, d_o, d_l, d_m, d, tc);

    // copy the output to host
    cudaMemcpy(o, d_o, N*d*sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void flash_attention(float *q, float *k, float *v, float *o, float *l, float *m, int d, int tc) {
    __shared__ float kj[bc * d_max]; 
    __shared__ float vj[bc * d_max];
    __shared__ float qi[br * d_max];
    __shared__ float oi[br * d_max];
    __shared__ float li[br];
    __shared__ float mi[br];
    __shared__ float li_new[br];
    __shared__ float mi_new[br];

    __shared__ float sij[br * bc];
    __shared__ float pij[br * bc];
    __shared__ float mij[br];
    __shared__ float lij[br];

    int qo_offset = blockIdx.x * br * d;
    int lm_offset = blockIdx.x * br;
    float sqrt_d = 1.0 / sqrtf(d);

    // load qi
    for (int i = 0; i < d; i += 32) {
        qi[threadIdx.y * d + threadIdx.x + i] = q[qo_offset + threadIdx.y * d + threadIdx.x + i];
    }
    
    // load oi
    for (int i = 0; i < d; i += 32) {
        oi[threadIdx.y * d + threadIdx.x + i] = o[qo_offset + threadIdx.y * d + threadIdx.x + i];
    }
   
    // load li
    if (threadIdx.y == 0) {
        li[threadIdx.x] = l[lm_offset + threadIdx.x];
    }
  
    // load mi
    if (threadIdx.y == 0){
        mi[threadIdx.x] = m[lm_offset + threadIdx.x];
    }
    __syncthreads();

    // start for-loop
    for (int j = 0; j < tc; ++j) {
        // load kj 
        int kjvj_offset = j * bc * d;
        for (int i = 0; i < d; i += 32) {
            kj[threadIdx.y * d + threadIdx.x + i] = k[kjvj_offset + threadIdx.y * d + threadIdx.x + i];
        }

        // load vj
        for (int i = 0; i < d; i += 32) {
            vj[threadIdx.y * d + threadIdx.x + i] = v[kjvj_offset + threadIdx.y * d + threadIdx.x + i];
        }
        __syncthreads();
        
        // QKDotAndScalar
        sij[threadIdx.y * bc + threadIdx.x] = 0.0F;
        for (int t = 0; t < d; t++) {
            sij[threadIdx.y * bc + threadIdx.x] += qi[threadIdx.y * d + t] * kj[threadIdx.x * d + t];
        }
        sij[threadIdx.y * bc + threadIdx.x] *= sqrt_d;
        __syncthreads();

        // RowMax (turn to threadIdx.y==0 ?)
        if (threadIdx.x == 0) {
            mij[threadIdx.y] = sij[threadIdx.y * bc];
            for (int i = 0; i < bc; ++i) {
                mij[threadIdx.y] = fmaxf(mij[threadIdx.y], sij[threadIdx.y * bc + i]);
            }
        }
        __syncthreads();

        // MinusMaxAndExp
        pij[threadIdx.y * bc + threadIdx.x] = expf(sij[threadIdx.y  * bc + threadIdx.x] - mij[threadIdx.y]);
        __syncthreads();

        // RowSum
        if (threadIdx.x == 0) {
            lij[threadIdx.y] = 0.0F;
            for (int i = 0; i < bc; ++i) {
                lij[threadIdx.y] += pij[threadIdx.y * bc + i];
            }
        }
        __syncthreads();

        // UpdateMiLiOi
        if (threadIdx.y == 0) {
            mi_new[threadIdx.x] = fmaxf(mi[threadIdx.x], mij[threadIdx.x]);
            li_new[threadIdx.x] = expf(mi[threadIdx.x] - mi_new[threadIdx.x]) * li[threadIdx.x] + expf(mij[threadIdx.x] - mi_new[threadIdx.x]) * lij[threadIdx.x];
        }
        __syncthreads();
        for (int i = 0; i < d; i += 32) {
            float pv = 0.0F;
            for (int t = 0; t < bc; ++t) {
                pv += pij[threadIdx.y * bc + t] * vj[t * d + threadIdx.x + i];
            } 
            oi[threadIdx.y * d + threadIdx.x + i] = (li[threadIdx.y] * expf(mi[threadIdx.y] - mi_new[threadIdx.y]) * oi[threadIdx.y * d + threadIdx.x + i] + expf(mij[threadIdx.y] - mi_new[threadIdx.y]) * pv) / li_new[threadIdx.y];
        }
        if (threadIdx.y == 0) {
            mi[threadIdx.x] = mi_new[threadIdx.x];
            li[threadIdx.x] = li_new[threadIdx.x];
        }
        __syncthreads();
    }
  
    // update o
    for (int i = 0; i < d; i += 32) {
        o[qo_offset + threadIdx.y * d + threadIdx.x + i] = oi[threadIdx.y * d + threadIdx.x + i];
    }
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}