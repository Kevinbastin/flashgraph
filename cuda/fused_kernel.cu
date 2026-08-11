/**
 * @file fused_kernel.cu
 * @brief Fused RMSNorm → MatMul (INT8 dequant) → GELU CUDA kernel.
 *
 * KERNEL ARCHITECTURE OVERVIEW
 * ============================
 * This kernel fuses three operations into a single GPU launch to eliminate
 * intermediate global memory round-trips. In a non-fused implementation:
 *
 *   RMSNorm:  Read input (M×K)  → Write normed (M×K)    [2×M×K×2 bytes]
 *   MatMul:   Read normed (M×K) + Read weight (N×K)      [+ write M×N×2]
 *   GELU:     Read matmul_out   → Write final            [2×M×N×2 bytes]
 *
 *   Total global memory traffic: ~2×M×K×2 + M×K×2 + N×K + M×N×2 + 2×M×N×2
 *   For M=128, K=256, N=512: ~1.3 MB
 *
 * Fused kernel:
 *   Read input (M×K) + Read weight (N×K as INT8) → Write final (M×N×2)
 *   Total: ~M×K×2 + N×K×1 + M×N×2 = ~0.45 MB  (3× bandwidth reduction)
 *
 * The fused kernel processes tiles in shared memory (SRAM), keeping
 * intermediate results on-chip and never writing them to global memory.
 *
 * TILE CONFIGURATION
 * ==================
 * We use 2D tiling with dimensions:
 *   TILE_M = 64   — rows of output per thread block (along M)
 *   TILE_N = 64   — columns of output per thread block (along N)
 *   TILE_K = 32   — reduction dimension chunk size
 *
 * SHARED MEMORY BUDGET
 * ====================
 *   smem_input:  TILE_M × TILE_K × sizeof(half) = 64 × 32 × 2 = 4,096 bytes
 *   smem_weight: TILE_K × TILE_N × sizeof(int8) = 32 × 64 × 1 = 2,048 bytes
 *   smem_accum:  TILE_M × TILE_N × sizeof(float)= 64 × 64 × 4 = 16,384 bytes
 *   smem_rms:    TILE_M × sizeof(float)          = 64 × 4      = 256 bytes
 *   ─────────────────────────────────────────────────────────────────────
 *   TOTAL: 22,784 bytes ≈ 22.25 KB  ✓ (well within 48 KB limit)
 *
 * This leaves ~25 KB of shared memory headroom for register spills and
 * future per-channel scale vectors if we upgrade to per-channel quantization.
 *
 * THREAD BLOCK ORGANIZATION
 * =========================
 * Block dimensions: (TILE_N / 2, TILE_M / 4) = (32, 16) = 512 threads
 *
 * Each thread is responsible for computing a 2×4 output tile (using half2
 * vectorization and loop unrolling over 4 rows). This maps to:
 *   512 threads × 8 outputs/thread = 4,096 outputs = 64 × 64 = TILE_M × TILE_N  ✓
 *
 * WARP-LEVEL CONSIDERATIONS
 * =========================
 * 512 threads = 16 warps. Each warp (32 threads) processes one row of the
 * TILE_N dimension. All threads in a warp access consecutive columns,
 * enabling coalesced global memory loads (128-byte transactions aligned
 * to warp boundaries).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "quantization.cuh"
#include "stream_manager.cuh"

namespace flashgraph {

// ═══════════════════════════════════════════════════════════════════
// TILE SIZE CONFIGURATION (tunable for different GPU architectures)
// ═══════════════════════════════════════════════════════════════════

/**
 * Tile dimensions. These are the primary tuning knobs:
 *   - Increasing TILE_K → higher arithmetic intensity (more FMAs per byte)
 *   - Increasing TILE_M/TILE_N → more output reuse, but more shared memory
 *
 * Current config targets: V100 (48KB smem), A100 (48KB default config),
 * T4 (48KB), L4 (48KB). All have >= 48KB configurable shared memory.
 */
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

/**
 * Thread block dimensions, derived from tile sizes.
 *   BLOCK_X = TILE_N / 2 = 32  (each thread handles 2 output columns via half2)
 *   BLOCK_Y = TILE_M / 4 = 16  (each thread handles 4 output rows via unrolling)
 *   Total: 32 × 16 = 512 threads = 16 warps
 */
#define BLOCK_X (TILE_N / 2)
#define BLOCK_Y (TILE_M / 4)

// ═══════════════════════════════════════════════════════════════════
// FUSED KERNEL
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Fused RMSNorm → MatMul (INT8 dequant) → GELU kernel.
 *
 * EXECUTION FLOW PER THREAD BLOCK
 * ================================
 * Each thread block computes a TILE_M × TILE_N output tile:
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │ Stage 1: LOAD input tile [TILE_M × TILE_K] → shared memory │
 *   │          (cooperative load: all threads participate)         │
 *   │                     __syncthreads()                         │
 *   ├──────────────────────────────────────────────────────────────┤
 *   │ Stage 2: RMSNORM in shared memory                           │
 *   │          Each warp normalizes one row of the input tile.    │
 *   │          Result stays in shared memory (no global write).   │
 *   │                     __syncthreads()                         │
 *   ├──────────────────────────────────────────────────────────────┤
 *   │ Stage 3: For each K-tile (kk = 0, TILE_K, 2*TILE_K, ...): │
 *   │   3a. Load INT8 weight tile [TILE_K × TILE_N] → shared mem │
 *   │                     __syncthreads()                         │
 *   │   3b. Dequant INT8 → half2 + FMA into accumulator          │
 *   │                     __syncthreads()                         │
 *   ├──────────────────────────────────────────────────────────────┤
 *   │ Stage 4: GELU activation on accumulator (registers)         │
 *   │          Write result to global memory.                     │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * PARAMETERS
 * ----------
 * @param input       [M × K] FP16 input activations in global memory.
 * @param weight      [N × K] INT8 weights in global memory.
 * @param rms_weight  [K] FP16 RMSNorm gamma (scale) vector.
 * @param scale       Scalar half: INT8 de-quantization scale factor.
 * @param output      [M × N] FP16 output in global memory (pre-allocated).
 * @param M           Number of input rows.
 * @param K           Hidden dimension (shared dimension).
 * @param N           Output dimension (number of weight rows).
 * @param eps         RMSNorm epsilon.
 */
__global__ void fused_rmsnorm_matmul_gelu(
    const half* __restrict__ input,
    const int8_t* __restrict__ weight,
    const half* __restrict__ rms_weight,
    const half scale,
    half* __restrict__ output,
    int M, int K, int N,
    float eps
) {
    /**
     * SHARED MEMORY LAYOUT
     * ====================
     * We manually partition a single shared memory array into regions:
     *
     *   Offset 0:           smem_input  [TILE_M][TILE_K] as half
     *   Offset 4096:        smem_wt     [TILE_K][TILE_N] as int8_t
     *   Offset 6144:        smem_rms    [TILE_M] as float (RMS values per row)
     *
     * The accumulator lives in REGISTERS (not shared memory) to maximize
     * throughput — each thread owns its own 4×2 accumulator tile.
     */
    __shared__ half    smem_input[TILE_M][TILE_K];
    __shared__ int8_t  smem_wt[TILE_K][TILE_N];
    __shared__ float   smem_rms[TILE_M];

    // Thread and block indices
    const int tx = threadIdx.x;  // [0, BLOCK_X) = [0, 32)
    const int ty = threadIdx.y;  // [0, BLOCK_Y) = [0, 16)
    const int bx = blockIdx.x;   // output tile column index
    const int by = blockIdx.y;   // output tile row index

    // Global row/col base for this tile
    const int row_base = by * TILE_M;
    const int col_base = bx * TILE_N;

    /**
     * REGISTER ACCUMULATOR
     * ====================
     * Each thread accumulates a 4-row × 2-column sub-tile of the output.
     * Using float (FP32) accumulators prevents overflow during large
     * dot-product sums — half can only represent up to 65504, which is
     * easily exceeded by a dot product of length K=256 with half values.
     *
     * 4 rows × 2 cols = 8 float registers per thread.
     * 512 threads × 8 regs = 4096 accumulators = TILE_M × TILE_N  ✓
     */
    float acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc[i][0] = 0.0f;
        acc[i][1] = 0.0f;
    }

    /**
     * ═══════════════════════════════════════════════════════════════
     * STAGE 1+2: LOAD INPUT + RMSNORM (first K-tile)
     * ═══════════════════════════════════════════════════════════════
     *
     * We handle RMSNorm for the FULL row (all K elements) before the
     * tiled MatMul. This requires loading the entire input row to
     * compute the RMS statistic, then re-loading tiles for MatMul.
     *
     * Approach: Each warp computes RMS for one row using a warp-level
     * reduction, then we process K-tiles for the MatMul.
     */

    // Flat thread ID within the block
    const int tid = ty * BLOCK_X + tx;
    const int num_threads = BLOCK_X * BLOCK_Y;  // 512

    /**
     * STAGE 1: Compute RMS values for each row in our tile.
     *
     * Each thread processes multiple rows to cover TILE_M rows total.
     * For each row, we accumulate sum-of-squares across ALL K elements
     * (not just one tile), because RMSNorm needs the full-row statistic.
     *
     * __syncthreads() BARRIER: Guards smem_rms[] — all threads must finish
     * writing their row's RMS value before any thread reads it in Stage 2.
     */
    for (int row_offset = tid; row_offset < TILE_M; row_offset += num_threads) {
        int global_row = row_base + row_offset;
        float sum_sq = 0.0f;

        if (global_row < M) {
            for (int k = 0; k < K; ++k) {
                float val = __half2float(input[global_row * K + k]);
                sum_sq += val * val;
            }
            sum_sq /= static_cast<float>(K);
            smem_rms[row_offset] = rsqrtf(sum_sq + eps);
        } else {
            smem_rms[row_offset] = 0.0f;
        }
    }
    __syncthreads();

    /**
     * ═══════════════════════════════════════════════════════════════
     * STAGE 3: TILED MATMUL WITH INT8 DEQUANT
     * ═══════════════════════════════════════════════════════════════
     *
     * Loop over K in chunks of TILE_K. For each chunk:
     *   (a) Cooperatively load input tile (with RMSNorm applied on-the-fly)
     *   (b) Cooperatively load INT8 weight tile
     *   (c) Compute partial MatMul using half2 FMA
     *
     * The two __syncthreads() barriers per iteration are necessary:
     *   1st: After loading → ensures all data is in shared memory before compute
     *   2nd: After compute → ensures no thread starts loading the next tile
     *        while another thread is still reading the current tile
     */
    for (int kk = 0; kk < K; kk += TILE_K) {
        /**
         * STAGE 3a: Load input tile with RMSNorm applied.
         *
         * Instead of storing the raw input and normalizing separately,
         * we fuse the normalization into the load:
         *   smem_input[row][k] = input[row][kk+k] * rms_inv * rms_weight[kk+k]
         *
         * This saves one full pass over the shared memory tile.
         */
        for (int idx = tid; idx < TILE_M * TILE_K; idx += num_threads) {
            int row = idx / TILE_K;
            int k   = idx % TILE_K;
            int global_row = row_base + row;
            int global_k   = kk + k;

            if (global_row < M && global_k < K) {
                float val = __half2float(input[global_row * K + global_k]);
                float normed = val * smem_rms[row];
                float scaled = normed * __half2float(rms_weight[global_k]);
                smem_input[row][k] = __float2half(scaled);
            } else {
                smem_input[row][k] = __float2half(0.0f);
            }
        }

        /**
         * STAGE 3b: Load INT8 weight tile.
         *
         * Weight layout in global memory: weight[n][k] where n ∈ [0,N), k ∈ [0,K)
         * We load weight[col_base + n][kk + k] → smem_wt[k][n]
         *
         * Note the TRANSPOSITION: global is [N,K] row-major, shared is [K,N].
         * This gives us contiguous access along N in shared memory during
         * the MatMul, matching the output column layout.
         */
        for (int idx = tid; idx < TILE_K * TILE_N; idx += num_threads) {
            int k = idx / TILE_N;
            int n = idx % TILE_N;
            int global_n = col_base + n;
            int global_k = kk + k;

            if (global_n < N && global_k < K) {
                smem_wt[k][n] = weight[global_n * K + global_k];
            } else {
                smem_wt[k][n] = 0;
            }
        }

        /**
         * __syncthreads() BARRIER #1 (per kk iteration):
         * Ensures all threads have finished loading both smem_input and
         * smem_wt before any thread starts the MatMul compute phase.
         * Without this, a fast thread could read uninitialized shared memory.
         */
        __syncthreads();

        /**
         * STAGE 3c: MatMul accumulation with INT8 de-quantization.
         *
         * Each thread computes a 4×2 sub-tile of the output:
         *   acc[i][j] += dot(smem_input[row+i][:], dequant(smem_wt[:][col+j]))
         *
         * The inner loop over k de-quantizes INT8 → half2 on-the-fly using
         * the dequant_int8_to_half2 device function. We use half2 FMA
         * (fused multiply-add) for 2× throughput on Volta+ GPUs.
         *
         * Register usage per thread:
         *   8 float accumulators (acc[4][2])
         *   ~4 half2 temporaries (input, weight, products)
         *   Total: ~12 registers → well within the 255-register limit
         */
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // Each thread handles 4 rows (ty*4 + 0..3) and 2 cols (tx*2 + 0..1)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = ty * 4 + i;
                half input_val = smem_input[row][k];
                float input_f = __half2float(input_val);

                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int col = tx * 2 + j;
                    /**
                     * De-quantize the INT8 weight value to half, then to float
                     * for accumulation. We accumulate in FP32 to avoid overflow.
                     *
                     * The dequant operation: float_val = int8_val * scale
                     * is fused into the accumulation multiply-add.
                     */
                    half wt_half = dequant_int8_to_half(smem_wt[k][col], scale);
                    float wt_f = __half2float(wt_half);
                    acc[i][j] += input_f * wt_f;
                }
            }
        }

        /**
         * __syncthreads() BARRIER #2 (per kk iteration):
         * Ensures all threads have finished reading smem_input and smem_wt
         * before we overwrite them with the next tile's data.
         * Without this, a fast thread could start loading the next tile
         * while a slow thread is still reading the current tile.
         */
        __syncthreads();
    }

    /**
     * ═══════════════════════════════════════════════════════════════
     * STAGE 4: GELU ACTIVATION + WRITE TO GLOBAL MEMORY
     * ═══════════════════════════════════════════════════════════════
     *
     * Apply GELU to each accumulated value and write to global memory.
     *
     * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     *
     * We compute GELU in FP32 (using the float accumulator) and convert
     * to FP16 only at the final store. This avoids precision loss in the
     * tanh computation, which is sensitive to input magnitude.
     */
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_COEFF = 0.044715f;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int global_row = row_base + ty * 4 + i;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int global_col = col_base + tx * 2 + j;

            if (global_row < M && global_col < N) {
                float x = acc[i][j];

                // GELU activation (tanh approximation)
                float x3 = x * x * x;
                float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x3);
                float gelu_val = 0.5f * x * (1.0f + tanhf(inner));

                // Convert FP32 → FP16 and write to global memory
                output[global_row * N + global_col] = __float2half(gelu_val);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// HOST-SIDE LAUNCH WRAPPER
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Launch the fused RMSNorm→MatMul→GELU kernel.
 *
 * Computes grid and block dimensions from tile sizes, validates inputs,
 * and launches the kernel on the provided CUDA stream.
 *
 * GRID DIMENSIONS
 * ===============
 *   grid.x = ceil(N / TILE_N)  — tiles along output columns
 *   grid.y = ceil(M / TILE_M)  — tiles along output rows
 *
 * For M=128, K=256, N=512:
 *   grid = (512/64, 128/64) = (8, 2) = 16 thread blocks
 *   Each block: 512 threads = 16 warps
 *   Total: 16 × 16 = 256 warps = 8,192 threads
 *
 * @param input       [M, K] FP16 input tensor (device memory).
 * @param weight      [N, K] INT8 weight tensor (device memory).
 * @param rms_weight  [K] FP16 RMSNorm gamma vector (device memory).
 * @param scale       INT8 de-quantization scale factor.
 * @param output      [M, N] FP16 output tensor (device memory, pre-allocated).
 * @param M           Rows in input.
 * @param K           Hidden dimension.
 * @param N           Output dimension.
 * @param eps         RMSNorm epsilon.
 * @param stream      CUDA stream for async execution.
 */
void launch_fused_kernel(
    const half* input,
    const int8_t* weight,
    const half* rms_weight,
    half scale,
    half* output,
    int M, int K, int N,
    float eps,
    cudaStream_t stream
) {
    // Compute grid dimensions: ceil division
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    dim3 block(BLOCK_X, BLOCK_Y);  // (32, 16) = 512 threads

    /**
     * Shared memory is statically allocated via __shared__ declarations
     * inside the kernel, so we don't need dynamic shared memory here.
     * If we switch to dynamic shared memory (for runtime-tunable tile sizes),
     * we would pass the size as the third kernel launch parameter:
     *   kernel<<<grid, block, shared_mem_bytes, stream>>>(...)
     */
    fused_rmsnorm_matmul_gelu<<<grid, block, 0, stream>>>(
        input, weight, rms_weight, scale, output, M, K, N, eps
    );

    // Check for launch errors (kernel execution errors are caught at sync)
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace flashgraph
