/**
 * @file cpu_ops.h
 * @brief CPU baseline implementations of RMSNorm, MatMul, and GELU.
 *
 * PURPOSE
 * =======
 * These are reference implementations for numerical ground-truth validation.
 * They are NOT performance-optimized for production CPU inference — their role
 * is to produce bit-exact FP32 results against which the CUDA FP16 kernel
 * is compared.
 *
 * That said, the MatMul implementation does use cache-aware loop blocking
 * (tiling) to demonstrate the optimization technique and keep execution
 * time reasonable for test-sized tensors.
 *
 * CONVENTIONS
 * ===========
 * - All functions operate on raw float* pointers (no std::vector, no heap).
 * - Dimensions are passed explicitly as integers.
 * - All outputs must be pre-allocated by the caller (from the arena).
 * - Functions do NOT allocate memory internally.
 */

#ifndef FLASHGRAPH_CPU_OPS_H
#define FLASHGRAPH_CPU_OPS_H

#include <cstddef>

namespace flashgraph {
namespace cpu {

/**
 * @brief RMSNorm: Root Mean Square Layer Normalization.
 *
 * FORMULA
 * -------
 *   rms = sqrt(mean(x^2) + eps)
 *   y_i = (x_i / rms) * weight_i
 *
 * This is a simpler alternative to LayerNorm that omits the mean-subtraction
 * step. Used in LLaMA, Gemma, and other modern transformer architectures
 * because it's cheaper (one reduction instead of two) and empirically
 * equivalent in quality.
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * Processes one row at a time (row = one token's hidden state). Each row
 * fits in L1 cache for hidden_dim <= 4096 (16 KB for float). Two passes
 * over each row:
 *   Pass 1: sum of squares (read x)
 *   Pass 2: normalize and scale (read x, read weight, write y)
 *
 * @param x       Input tensor, shape [rows, hidden_dim], row-major.
 * @param weight  Scale vector, shape [hidden_dim].
 * @param y       Output tensor, shape [rows, hidden_dim], row-major (pre-allocated).
 * @param rows    Number of rows (batch * seq_len).
 * @param hidden_dim  Hidden dimension size.
 * @param eps     Epsilon for numerical stability (typically 1e-6).
 */
void rmsnorm(const float* x,
             const float* weight,
             float* y,
             int rows,
             int hidden_dim,
             float eps = 1e-6f);

/**
 * @brief Matrix multiplication: C = A @ B^T (with cache-aware loop blocking).
 *
 * FORMULA
 * -------
 *   C[i][j] = sum_k( A[i][k] * B[j][k] )
 *
 * Note: B is stored in row-major but accessed as B^T. That is, B has shape
 * [N, K] and we compute C[i][j] = dot(A_row_i, B_row_j). This is equivalent
 * to C = A @ B^T.
 *
 * WHY B^T INSTEAD OF B?
 * ---------------------
 * In transformer inference, weight matrices are stored as [output_dim, input_dim].
 * Computing x @ W^T with W in this layout means both x and W are accessed
 * row-wise, which is cache-friendly for row-major storage.
 *
 * LOOP BLOCKING (TILING) STRATEGY
 * --------------------------------
 * The naive triple loop has poor cache behavior: the inner product over K
 * streams through A and B, evicting useful data from L1/L2.
 *
 * Blocking with tile size BLOCK=64 (256 bytes for float) ensures:
 *   - The A tile (BLOCK floats from row i) stays in L1 (256 B)
 *   - The B tile (BLOCK × BLOCK floats) stays in L2 (16 KB)
 *   - The C tile (BLOCK × BLOCK floats) stays in registers/L1
 *
 * Loop structure:
 *   for ii in [0, M, BLOCK):       // tile over output rows
 *     for jj in [0, N, BLOCK):     // tile over output columns
 *       for kk in [0, K, BLOCK):   // tile over reduction dimension
 *         // micro-kernel: accumulate BLOCK×BLOCK×BLOCK
 *
 * @param A   Input matrix, shape [M, K], row-major.
 * @param B   Weight matrix, shape [N, K], row-major (accessed as B^T).
 * @param C   Output matrix, shape [M, N], row-major (pre-allocated, zero-initialized by caller).
 * @param M   Number of rows in A / rows in C.
 * @param K   Shared dimension (columns of A, columns of B).
 * @param N   Number of rows in B / columns in C.
 */
void matmul_bt(const float* A,
               const float* B,
               float* C,
               int M, int K, int N);

/**
 * @brief GELU activation function (exact formulation).
 *
 * FORMULA
 * -------
 *   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * This is the "exact" GELU as used in GPT-2 and BERT. The approximation
 * with tanh is already fast on CPU (single call to libm tanh per element).
 *
 * PyTorch equivalent: torch.nn.functional.gelu(x, approximate='tanh')
 *
 * MEMORY ACCESS PATTERN
 * ---------------------
 * Purely element-wise — no cross-element dependencies. Memory access is
 * sequential and perfectly prefetch-friendly. No blocking needed.
 *
 * @param x   Input tensor (flat), length `n`.
 * @param y   Output tensor (flat), length `n` (pre-allocated). May alias x for in-place.
 * @param n   Number of elements.
 */
void gelu(const float* x, float* y, int n);

}  // namespace cpu
}  // namespace flashgraph

#endif  // FLASHGRAPH_CPU_OPS_H
