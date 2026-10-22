/**
 * good_spill.cu
 * ============================================================
 * OPTIMIZED register usage — no spilling.
 *
 * Fix:
 *   Restructure the kernel to keep only a small number of live variables
 *   at any given time. Instead of keeping 64 floats live simultaneously,
 *   process data in sequential passes with minimal live state.
 *
 * Detection target:
 *   - nvcc/ptxas: "spill stores: 0", "spill loads: 0"
 *   - ncu: local memory traffic near 0
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 65536

// ------------------------------------------------------------
// GOOD kernel: same logical computation, but restructured so only
// O(1) live variables exist at any point — fits comfortably in registers.
// No local array; accumulate into a single scalar.
// ------------------------------------------------------------
__global__ void good_no_spill(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float base = in[tid];
    float acc  = 0.0f;
    float prev = base;
    float curr = base;

    // Sequential reduction: only 3 live floats at a time (prev, curr, acc)
    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        float next = curr * prev + base * (float)(i + 1);
        acc  += next;
        prev  = curr;
        curr  = next * 0.5f + base * 0.001f;
    }

    // One more pass, minimal live state
    float result = acc;
    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        result = result * 0.999f + base * (float)i * 0.001f;
    }

    out[tid] = result;
}

static void check(cudaError_t e, const char* m)
{
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main(void)
{
    printf("=== GOOD REGISTER USAGE BENCHMARK (NO SPILLING) ===\n");
    printf("Kernel uses O(1) live variables — no local array, no spills.\n\n");

    size_t bytes = N * sizeof(float);
    float *d_in, *d_out;
    check(cudaMalloc(&d_in,  bytes), "in");
    check(cudaMalloc(&d_out, bytes), "out");

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i * 0.001f;
    check(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "H2D");
    check(cudaMemset(d_out, 0, bytes), "memset out");

    int threads = 128;
    int blocks  = (N + threads - 1) / threads;

    // warmup
    good_no_spill<<<blocks, threads>>>(d_in, d_out, N);
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int i = 0; i < 20; i++)
        good_no_spill<<<blocks, threads>>>(d_in, d_out, N);

    cudaEventRecord(t1);
    check(cudaDeviceSynchronize(), "good_spill sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[good_no_spill]  avg time = %.3f ms  (20 runs)\n", ms / 20.0f);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in);

    printf("\nDone. Compare ptxas output with bad_spill — spill counts should be 0.\n");
    return 0;
}
