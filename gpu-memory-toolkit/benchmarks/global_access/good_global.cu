/**
 * good_global.cu
 * ============================================================
 * OPTIMIZED global memory access — fully coalesced.
 *
 * Fix:
 *   Thread tid accesses element tid (stride = 1).
 *   All 32 warp threads access 32 consecutive floats = one 128-byte cache line.
 *   For the matrix case: shared-memory tiling makes both reads and writes coalesced.
 *
 * Detection target:
 *   - Static:   index expression  tid  (stride = 1)
 *   - Runtime:  low sector overhead, high memory efficiency (NCU)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N    (1 << 22)
#define ROWS 4096
#define COLS 4096
#define TILE 16

// GOOD: stride-1 write, fully coalesced
__global__ void good_stride_write(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid] * 2.0f;  // GOOD: coalesced
    }
}

// GOOD: tiled transpose — coalesced read AND coalesced write via shared memory
__global__ void good_row_major_copy(const float* __restrict__ matrix,
                                    float* __restrict__ out,
                                    int rows, int cols)
{
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts on store

    int col_in = blockIdx.x * TILE + threadIdx.x;
    int row_in = blockIdx.y * TILE + threadIdx.y;

    // Coalesced read: consecutive threads read consecutive columns
    if (row_in < rows && col_in < cols)
        tile[threadIdx.y][threadIdx.x] = matrix[row_in * cols + col_in];

    __syncthreads();

    int col_out = blockIdx.y * TILE + threadIdx.x;
    int row_out = blockIdx.x * TILE + threadIdx.y;

    // Coalesced write: consecutive threads write consecutive columns
    if (row_out < cols && col_out < rows)
        out[row_out * rows + col_out] = tile[threadIdx.x][threadIdx.y];
}

__global__ void warmup(float* buf, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = (float)tid;
}

static void check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    printf("=== GOOD GLOBAL MEMORY ACCESS BENCHMARK ===\n");
    printf("Stride-1 write + shared-memory tiled transpose (coalesced)\n\n");

    // ---- Part 1: coalesced write ----
    {
        size_t bytes = (size_t)N * sizeof(float);
        float *d_in, *d_out;
        check(cudaMalloc(&d_in,  bytes), "malloc d_in");
        check(cudaMalloc(&d_out, bytes), "malloc d_out");

        float* h_in = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_in[i] = (float)i;
        check(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "H2D");
        check(cudaMemset(d_out, 0, bytes), "memset");

        int threads = 256;
        int blocks  = (N + threads - 1) / threads;

        warmup<<<blocks, threads>>>(d_out, N);
        check(cudaDeviceSynchronize(), "warmup sync");

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        for (int i = 0; i < 10; i++)
            good_stride_write<<<blocks, threads>>>(d_in, d_out, N);

        cudaEventRecord(t1);
        check(cudaDeviceSynchronize(), "kernel sync");

        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("[good_stride_write]     avg time = %.3f ms  (10 runs)\n", ms / 10.0f);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_in); cudaFree(d_out);
        free(h_in);
    }

    // ---- Part 2: tiled transpose ----
    {
        size_t bytes = (size_t)ROWS * COLS * sizeof(float);
        float *d_mat, *d_out;
        check(cudaMalloc(&d_mat, bytes), "malloc matrix");
        check(cudaMalloc(&d_out, bytes), "malloc out");
        check(cudaMemset(d_mat, 0, bytes), "memset mat");
        check(cudaMemset(d_out, 0, bytes), "memset out");

        dim3 threads(TILE, TILE);
        dim3 blocks((COLS + TILE - 1) / TILE, (ROWS + TILE - 1) / TILE);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        for (int i = 0; i < 5; i++)
            good_row_major_copy<<<blocks, threads>>>(d_mat, d_out, ROWS, COLS);

        cudaEventRecord(t1);
        check(cudaDeviceSynchronize(), "tiled sync");

        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("[good_row_major_copy]   avg time = %.3f ms  (5 runs)\n", ms / 5.0f);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_mat); cudaFree(d_out);
    }

    printf("\nDone. Compare with bad_global to observe speedup.\n");
    return 0;
}
