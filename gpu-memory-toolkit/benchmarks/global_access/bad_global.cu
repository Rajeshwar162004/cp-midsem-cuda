/**
 * bad_global.cu
 * ============================================================
 * Intentionally BAD global memory access pattern.
 *
 * Problem:
 *   Each thread accesses memory with a stride of STRIDE (default 32).
 *   Thread 0 -> element 0, Thread 1 -> element 32, Thread 2 -> element 64 ...
 *   Adjacent threads in a warp do NOT access adjacent memory locations.
 *   This causes uncoalesced (scattered) global memory transactions,
 *   wasting memory bandwidth.
 *
 * Detection target:
 *   - Static:   index expression  tid * STRIDE  (stride > 1)
 *   - Runtime:  high global memory sector overhead (NCU)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N       (1 << 22)   // 4 M elements
#define STRIDE  32          // intentionally bad stride

// BAD kernel: thread tid writes to output[tid * STRIDE]
// Warp of 32 threads produces 32 scattered stores, each 128 bytes apart.
__global__ void bad_stride_write(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n / stride) {
        output[tid * stride] = input[tid] * 2.0f;  // BAD: strided store
    }
}

// BAD: column-major read from a row-major matrix (uncoalesced read)
#define ROWS 4096
#define COLS 4096

__global__ void bad_column_read(const float* __restrict__ matrix,
                                 float* __restrict__ out,
                                 int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        // threads in a warp (same row, consecutive cols) read cols apart in memory
        out[row * cols + col] = matrix[col * rows + row];  // UNCOALESCED READ
    }
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
    printf("=== BAD GLOBAL MEMORY ACCESS BENCHMARK ===\n");
    printf("Strided write (stride=%d) + column-major read\n\n", STRIDE);

    // ---- Part 1: strided write ----
    {
        size_t bytes = (size_t)N * sizeof(float);
        float *d_in, *d_out;
        check(cudaMalloc(&d_in,  bytes), "malloc d_in");
        check(cudaMalloc(&d_out, bytes), "malloc d_out");

        float* h_in = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) h_in[i] = (float)i;
        check(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "H2D");
        check(cudaMemset(d_out, 0, bytes), "memset out");

        int threads = 256;
        int work_n  = N / STRIDE;
        int blocks  = (work_n + threads - 1) / threads;

        warmup<<<(N + threads - 1) / threads, threads>>>(d_out, N);
        check(cudaDeviceSynchronize(), "warmup sync");

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        for (int i = 0; i < 10; i++)
            bad_stride_write<<<blocks, threads>>>(d_in, d_out, N, STRIDE);

        cudaEventRecord(t1);
        check(cudaDeviceSynchronize(), "kernel sync");

        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("[bad_stride_write]  avg time = %.3f ms  (10 runs)\n", ms / 10.0f);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_in); cudaFree(d_out);
        free(h_in);
    }

    // ---- Part 2: column-major read ----
    {
        size_t bytes = (size_t)ROWS * COLS * sizeof(float);
        float *d_mat, *d_out;
        check(cudaMalloc(&d_mat, bytes), "malloc matrix");
        check(cudaMalloc(&d_out, bytes), "malloc out");
        check(cudaMemset(d_mat, 0, bytes), "memset mat");
        check(cudaMemset(d_out, 0, bytes), "memset out");

        dim3 threads(16, 16);
        dim3 blocks((COLS + 15) / 16, (ROWS + 15) / 16);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        for (int i = 0; i < 5; i++)
            bad_column_read<<<blocks, threads>>>(d_mat, d_out, ROWS, COLS);

        cudaEventRecord(t1);
        check(cudaDeviceSynchronize(), "col-read sync");

        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        printf("[bad_column_read]   avg time = %.3f ms  (5 runs)\n", ms / 5.0f);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(d_mat); cudaFree(d_out);
    }

    printf("\nDone. Profile with ncu to observe uncoalesced access metrics.\n");
    return 0;
}
