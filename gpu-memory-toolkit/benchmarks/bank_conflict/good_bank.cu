/**
 * good_bank.cu
 * ============================================================
 * OPTIMIZED shared memory access — bank conflicts eliminated via padding.
 *
 * Fix:
 *   Pad the shared memory array by 1 element per row:
 *       tile[TILE_DIM][TILE_DIM + 1]
 *
 *   With +1 padding, row stride becomes 33 floats = 132 bytes.
 *   thread tx reads tile[tx][ty]:
 *       address = tx * 33 * 4  bytes
 *       bank    = (tx * 33) % 32  → different bank for each tx ✓
 *   No bank conflicts!
 *
 * Detection target:
 *   - Static:  __shared__ float tile[32][33]  ← padding present
 *   - Runtime: bank conflict counter near 0
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM  32
#define BLOCK_DIM 32
#define N_TILES   128

// ------------------------------------------------------------
// GOOD kernel: +1 padding eliminates bank conflicts
// ------------------------------------------------------------
__global__ void good_no_bank_conflict(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int width)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // GOOD: +1 padding

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < width)
        tile[threadIdx.x][threadIdx.y] = in[y * width + x];

    __syncthreads();

    // Now all 32 threads in the warp read tile[0..31][ty]
    // With +1 padding: addresses are 0, 132, 264 ... bytes
    // bank[tx] = (tx * 33) % 32 = 0,1,2,...30,31,0,...  all unique ✓
    if (x < width && y < width)
        out[y * width + x] = tile[threadIdx.x][threadIdx.y] * 2.0f;
}

// Variant: fix stride-32 by using stride-1
__global__ void good_stride1_shared(float* __restrict__ data, int n)
{
    __shared__ float smem[TILE_DIM * TILE_DIM];

    int tid = threadIdx.x;

    // GOOD: stride-1 access, each thread hits a unique bank
    if (tid < TILE_DIM) {
        smem[tid] = data[blockIdx.x * blockDim.x + tid];  // stride-1
        __syncthreads();
        data[blockIdx.x * blockDim.x + tid] = smem[tid] + 1.0f;
    }
}

static void check(cudaError_t e, const char* m)
{
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

int main(void)
{
    printf("=== GOOD SHARED MEMORY (NO BANK CONFLICTS) BENCHMARK ===\n\n");

    const int W = TILE_DIM * N_TILES;
    size_t bytes = (size_t)W * W * sizeof(float);

    float *d_in, *d_out;
    check(cudaMalloc(&d_in,  bytes), "in");
    check(cudaMalloc(&d_out, bytes), "out");
    check(cudaMemset(d_in,  0, bytes), "memset in");
    check(cudaMemset(d_out, 0, bytes), "memset out");

    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks(N_TILES, N_TILES);

    // warmup
    good_no_bank_conflict<<<blocks, threads>>>(d_in, d_out, W);
    check(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int i = 0; i < 10; i++)
        good_no_bank_conflict<<<blocks, threads>>>(d_in, d_out, W);

    cudaEventRecord(t1);
    check(cudaDeviceSynchronize(), "good_bank sync");

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("[good_no_bank_conflict]  avg time = %.3f ms  (10 runs)\n", ms / 10.0f);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_in); cudaFree(d_out);

    printf("\nDone. Compare with bad_bank to see bank-conflict reduction.\n");
    return 0;
}
