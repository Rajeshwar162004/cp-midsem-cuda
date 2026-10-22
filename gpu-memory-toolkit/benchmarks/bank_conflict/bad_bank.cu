/**
 * bad_bank.cu
 * ============================================================
 * Intentionally BAD shared memory access — bank conflicts.
 *
 * Problem:
 *   Shared memory on sm_89 (RTX 4050) is divided into 32 banks,
 *   each 4 bytes wide. Bank of a 32-bit word at byte offset B is:
 *       bank = (B / 4) % 32
 *
 *   When multiple threads in a warp access the SAME bank (different
 *   addresses), the hardware serializes those accesses (bank conflict).
 *
 *   BAD pattern: tile[32][32], access tile[threadIdx.x][threadIdx.y]
 *   Within a warp (threadIdx.x varies 0..31, threadIdx.y fixed):
 *       each thread accesses row threadIdx.x, same column threadIdx.y
 *       → addresses stride by 32 floats = 128 bytes = 32 banks apart
 *       → all 32 threads hit bank (threadIdx.y % 32) — 32-way conflict!
 *
 * Detection target:
 *   - Static:  __shared__ float tile[32][32], access [threadIdx.x][threadIdx.y]
 *   - Runtime: ncu l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM  32
#define BLOCK_DIM 32
#define N_TILES   128   // how many tiles to process (for measurable runtime)

// ------------------------------------------------------------
// BAD kernel: column-first shared memory access causes 32-way bank conflicts
// Thread warp: threadIdx.x = 0..31, threadIdx.y = 0
//   tile[0][0], tile[1][0], tile[2][0] ... tile[31][0]
//   addresses: 0, 128, 256, ... 3968 bytes  — all map to BANK 0
// ------------------------------------------------------------
__global__ void bad_bank_conflict(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int width)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];  // 32x32, NO padding

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced global read (not the problem here)
    if (x < width && y < width)
        tile[threadIdx.x][threadIdx.y] = in[y * width + x];  // store: row=tx, col=ty

    __syncthreads();

    // BAD: read from shared with transposed indices
    // thread (tx, ty) reads tile[tx][ty]  — BUT with 1D warp layout (ty=0 for all
    // threads in same warp), all 32 threads read tile[0][0], tile[1][0]... tile[31][0]
    // which are in the same column → 32-way bank conflict on read
    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y] * 2.0f;
}

// Variant: explicit stride-32 access pattern to guarantee conflict
__global__ void bad_stride32_shared(float* __restrict__ data, int n)
{
    __shared__ float smem[TILE_DIM * TILE_DIM];  // 1024 floats

    int tid = threadIdx.x;  // 0..31 within a warp

    // BAD: every thread in warp accesses elements 32 apart
    // smem[0], smem[32], smem[64] ... smem[31*32]  → all map to bank 0
    if (tid < TILE_DIM) {
        smem[tid * TILE_DIM] = data[blockIdx.x * blockDim.x + tid];  // stride-32 store
        __syncthreads();
        data[blockIdx.x * blockDim.x + tid] = smem[tid * TILE_DIM] + 1.0f;  // stride-32 load
    }
}

static void check(cudaError_t e, const char* m)
{
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main(void)
{
    printf("=== BAD SHARED MEMORY (BANK CONFLICTS) BENCHMARK ===\n\n");

    const int W = TILE_DIM * N_TILES;  // 4096
    size_t bytes = (size_t)W * W * sizeof(float);

    float *d_in, *d_out;
    check(cudaMalloc(&d_in,  bytes), "in");
    check(cudaMalloc(&d_out, bytes), "out");
    check(cudaMemset(d_in,  0, bytes), "memset in");
    check(cudaMemset(d_out, 0, bytes), "memset out");

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(N_TILES, N_TILES);

    // warmup
    bad_bank_conflict<<<blocks, threads>>>(d_in, d_out, W);
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int i = 0; i < 10; i++)
        bad_bank_conflict<<<blocks, threads>>>(d_in, d_out, W);

    cudaEventRecord(t1);
    check(cudaDeviceSynchronize(), "bad_bank sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[bad_bank_conflict]   avg time = %.3f ms  (10 runs)\n", ms / 10.0f);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);

    printf("\nDone. Profile with ncu to observe shared memory bank conflicts.\n");
    return 0;
}
