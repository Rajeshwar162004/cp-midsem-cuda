/**
 * bad_spill.cu
 * ============================================================
 * Intentionally BAD register usage — causes register spilling to local memory.
 *
 * Problem:
 *   Each SM has a limited register file. When a kernel uses too many registers
 *   per thread, the compiler spills excess to "local memory" (off-chip DRAM).
 *   This shows up in ptxas output as:
 *       "N bytes stack frame"  (or spill stores/loads)
 *   and in NCU as local memory traffic.
 *
 * Technique:
 *   Use a large, runtime-indexed local array to force the compiler to keep
 *   it on the stack (local memory), since it can't index a register array
 *   with a runtime index. This is guaranteed to produce local memory traffic.
 *
 * Detection target:
 *   - ptxas:  "N bytes stack frame" > 0  OR  spill stores > 0
 *   - nvcc:   --resource-usage flag shows local memory per thread
 *   - NCU:    l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st > 0
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N          65536
#define ARRAY_SIZE 128   // local array — runtime-indexed → goes to local memory

// ------------------------------------------------------------
// BAD kernel: large runtime-indexed local array forces local memory usage.
// The compiler CANNOT store a runtime-indexed array in registers,
// so it places it in local memory (per-thread, off-chip DRAM).
// ------------------------------------------------------------
__global__ void bad_register_spill(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // This local array goes to local memory because it is indexed at runtime.
    float local_buf[ARRAY_SIZE];

    float base = in[tid];

    // Fill with runtime-index: compiler MUST store in local memory
    for (int i = 0; i < ARRAY_SIZE; i++) {
        local_buf[i] = base * (float)(i + 1) + (float)(tid % 17);
    }

    // Second pass: read back and compute — generates local memory loads
    float acc = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        int idx = (tid + i) % ARRAY_SIZE;   // runtime index — can't be compile-time resolved
        acc += local_buf[idx] * (float)(i + 1);
    }

    // Third pass: write back through runtime index — more local memory stores
    for (int i = 0; i < ARRAY_SIZE; i++) {
        int idx = (i * 3 + tid) % ARRAY_SIZE;
        local_buf[idx] = acc * 0.001f + local_buf[i];
    }

    // Final reduce
    float result = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        result += local_buf[i];
    }

    out[tid] = result;
}

static void check(cudaError_t e, const char* m)
{
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main(void)
{
    printf("=== BAD REGISTER SPILL BENCHMARK ===\n");
    printf("Kernel uses a runtime-indexed local array (%d floats = %d bytes/thread)\n",
           ARRAY_SIZE, (int)(ARRAY_SIZE * sizeof(float)));
    printf("This GUARANTEES local memory usage (spilling).\n\n");

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
    bad_register_spill<<<blocks, threads>>>(d_in, d_out, N);
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int i = 0; i < 20; i++)
        bad_register_spill<<<blocks, threads>>>(d_in, d_out, N);

    cudaEventRecord(t1);
    check(cudaDeviceSynchronize(), "spill sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[bad_register_spill]  avg time = %.3f ms  (20 runs)\n", ms / 20.0f);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in);

    printf("\nDone. Check ptxas output for 'stack frame' or 'spill' bytes.\n");
    return 0;
}
