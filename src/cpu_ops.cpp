/**
 * @file cpu_ops.cpp
 * @brief CPU baseline implementations of RMSNorm, MatMul (tiled), and GELU.
 *
 * These serve as the numerical ground truth for validating the CUDA fused kernel.
 * All operations use FP32 precision with no approximations beyond the standard
 * tanh-based GELU formula.
 */

#include "cpu_ops.h"

#include <cmath>
#include <cstring>

namespace flashgraph {
namespace cpu {

/**
 * RMSNORM IMPLEMENTATION
 * ======================
 *
 * Two-pass algorithm per row:
 *
 *   Pass 1 — Sum of squares:
 *     ss = (1/hidden_dim) * Σ(x_i²)
 *     This is the "mean square" in Root Mean Square.
 *
 *   Pass 2 — Normalize and scale:
 *     rms = 1 / sqrt(ss + eps)
 *     y_i = x_i * rms * weight_i
 *
 * The multiplication by (1/sqrt) is faster than division by sqrt, and the
 * compiler will optimize the reciprocal-sqrt pattern on modern x86.
 *
 * We use a single accumulator `ss` — for hidden_dim <= 16384, the floating
 * point error from naive summation is within acceptable bounds (< 1e-6 relative).
 * For production use with very large hidden dims, Kahan or pairwise summation
 * would be advisable.
 */
void rmsnorm(const float* x,
             const float* weight,
             float* y,
             int rows,
             int hidden_dim,
             float eps) {
    for (int r = 0; r < rows; ++r) {
        const float* x_row = x + r * hidden_dim;
        float* y_row = y + r * hidden_dim;

        // Pass 1: compute mean of squares
        float ss = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            ss += x_row[i] * x_row[i];
        }
        ss /= static_cast<float>(hidden_dim);

        // Pass 2: normalize and scale
        float rms_inv = 1.0f / std::sqrt(ss + eps);
        for (int i = 0; i < hidden_dim; ++i) {
            y_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

/**
 * MATMUL (B-TRANSPOSED) WITH CACHE-AWARE LOOP BLOCKING
 * =====================================================
 *
 * Computes C = A @ B^T where:
 *   A: [M, K] — input activations
 *   B: [N, K] — weight matrix (stored row-major)
 *   C: [M, N] — output activations
 *
 * C[i][j] = Σ_k A[i][k] * B[j][k]
 *
 * TILING STRATEGY
 * ---------------
 * We tile along all three dimensions (M, N, K) with BLOCK = 64.
 *
 * For a 64×64 float tile:
 *   64 × 64 × 4 bytes = 16,384 bytes = 16 KB
 *
 * Cache budget (typical Zen 3 / Skylake):
 *   L1 data cache: 32 KB → fits 2 tiles (A_tile + accumulator)
 *   L2 cache: 256–512 KB → fits B tiles for multiple jj iterations
 *
 * The key insight: by iterating kk in the innermost tiling loop, we
 * accumulate partial sums into C[i][j] across multiple B tiles. The
 * A_tile and C_tile stay hot in L1 while we stream through B.
 *
 * REGISTER-LEVEL OPTIMIZATION
 * ---------------------------
 * Inside the micro-kernel (the innermost 3 loops after tiling), we
 * accumulate `A[i][k] * B[j][k]` into a scalar `sum`. The compiler
 * will keep `sum` in a register, and the sequential access to A[i][k]
 * and B[j][k] triggers hardware prefetching.
 *
 * NOTE: The caller MUST zero-initialize C before calling this function.
 */

/// Tile size for loop blocking. 64 elements = 256 bytes (1/4 of a 4 KB page).
/// This is a compile-time constant so the compiler can optimize loop bounds.
constexpr int BLOCK = 64;

void matmul_bt(const float* A,
               const float* B,
               float* C,
               int M, int K, int N) {
    /**
     * Zero-initialize C. We do this here rather than requiring the caller
     * to do it, because the blocking algorithm accumulates into C across
     * multiple kk tiles. If C isn't zero, we'd get garbage.
     */
    std::memset(C, 0, static_cast<size_t>(M) * N * sizeof(float));

    /**
     * Six-deep loop nest (3 tiling loops × 2 element loops):
     *
     * for ii in [0, M, BLOCK):          ← tile over output rows
     *   for jj in [0, N, BLOCK):        ← tile over output cols (B rows)
     *     for kk in [0, K, BLOCK):      ← tile over reduction dim
     *       for i in [ii, min(ii+BLOCK, M)):  ← element within tile
     *         for j in [jj, min(jj+BLOCK, N)):
     *           sum = C[i][j]           ← load partial accumulation
     *           for k in [kk, min(kk+BLOCK, K)):
     *             sum += A[i][k] * B[j][k]
     *           C[i][j] = sum           ← store back
     *
     * The access pattern for A[i][k] is sequential (good).
     * The access pattern for B[j][k] is sequential within a tile (good).
     * C[i][j] is touched once per kk tile (registers handle it).
     */
    for (int ii = 0; ii < M; ii += BLOCK) {
        int i_end = (ii + BLOCK < M) ? ii + BLOCK : M;
        for (int jj = 0; jj < N; jj += BLOCK) {
            int j_end = (jj + BLOCK < N) ? jj + BLOCK : N;
            for (int kk = 0; kk < K; kk += BLOCK) {
                int k_end = (kk + BLOCK < K) ? kk + BLOCK : K;

                // Micro-kernel: accumulate one tile
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        float sum = C[i * N + j];
                        for (int k = kk; k < k_end; ++k) {
                            /**
                             * A[i][k] is at offset i*K + k — sequential in k ✓
                             * B[j][k] is at offset j*K + k — sequential in k ✓
                             * Both access patterns stride by 1 float (4 bytes),
                             * which triggers hardware prefetching on x86.
                             */
                            sum += A[i * K + k] * B[j * K + k];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

/**
 * GELU IMPLEMENTATION
 * ===================
 *
 * Uses the tanh approximation (standard in GPT-2, BERT, and PyTorch's
 * approximate='tanh' mode):
 *
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * Constants:
 *   sqrt(2/π) ≈ 0.7978845608028654
 *   0.044715  — empirical coefficient from the GELU paper (Hendrycks & Gimpel, 2016)
 *
 * The computation per element is:
 *   1. x³ = x * x * x                    (2 FP muls)
 *   2. inner = 0.7978... * (x + 0.044715 * x³)  (1 FMA + 1 mul)
 *   3. tanh(inner)                        (1 libm call — ~20 cycles on x86)
 *   4. 0.5 * x * (1 + tanh_result)        (1 add + 2 muls)
 *
 * Total: ~7 FP ops + 1 tanh call per element. Purely compute-bound for
 * large n (no cross-element data dependency).
 */
void gelu(const float* x, float* y, int n) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_COEFF = 0.044715f;

    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        float inner = SQRT_2_OVER_PI * (xi + GELU_COEFF * x3);
        y[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
}

}  // namespace cpu
}  // namespace flashgraph
