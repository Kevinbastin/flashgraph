/**
 * @file quantization.cuh
 * @brief INT8 symmetric quantization and FP16 de-quantization utilities.
 *
 * QUANTIZATION SCHEME: SYMMETRIC PER-TENSOR
 * ==========================================
 * We use the simplest quantization scheme that yields acceptable accuracy
 * for inference-only weight compression:
 *
 *   scale = max(|W_fp32|) / 127.0
 *   W_int8 = round(W_fp32 / scale)       // clamp to [-128, 127]
 *   W_fp16 = W_int8 * scale              // de-quantization
 *
 * WHY SYMMETRIC (vs. asymmetric)?
 *   - Zero point is always 0 → no zero-point offset in de-quant (1 fewer mul)
 *   - Simpler kernel code with half the register pressure
 *   - Acceptable for weights (which are roughly symmetric around 0)
 *   - NOT ideal for activations (which can be skewed) — but we don't quantize activations
 *
 * WHY PER-TENSOR (vs. per-channel)?
 *   - Single scalar scale → broadcasts trivially in the kernel
 *   - Sufficient for small models; per-channel adds complexity for marginal gain
 *   - Can upgrade to per-channel later by making scale a vector[N]
 *
 * QUANTIZATION NOISE ANALYSIS
 * ===========================
 * Maximum quantization error per element:
 *   |error| ≤ scale / 2 = max(|W|) / 254
 *
 * For typical transformer weights with max(|W|) ≈ 1.0:
 *   |error| ≤ 1/254 ≈ 0.004
 *
 * This is within the noise floor of FP16 representation (which has ~1e-3
 * relative error for values near 1.0), so INT8 quantization is the
 * dominant error source but still acceptable for inference.
 *
 * DE-QUANTIZATION ON GPU
 * ======================
 * The key insight: we store weights as int8 in GLOBAL memory (saving 4×
 * bandwidth vs FP32, 2× vs FP16), then de-quantize to half2 in SHARED
 * memory or registers. The de-quantization cost (2 int-to-half conversions
 * + 2 half multiplications) is negligible compared to the bandwidth saved.
 *
 * De-quantization uses half2 vectorization:
 *   - Load 2 int8 values (2 bytes)
 *   - Convert each to half via __int2half_rn (round-to-nearest)
 *   - Multiply each by scale (also half)
 *   - Pack into half2 for use in half2 FMA operations
 */

#ifndef FLASHGRAPH_QUANTIZATION_CUH
#define FLASHGRAPH_QUANTIZATION_CUH

#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstdio>

namespace flashgraph {

// ═══════════════════════════════════════════════════════════════════
// GPU DEVICE FUNCTIONS (run inside CUDA kernels)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief De-quantize two INT8 values to a half2 (FP16 pair).
 *
 * INSTRUCTION SEQUENCE
 * --------------------
 *   1. int8  → int32  (sign extension, free on GPU — just a register widen)
 *   2. int32 → half   (__int2half_rn — 1 special function unit cycle)
 *   3. half  × scale  (__hmul — 1 FP16 multiply)
 *   4. Pack two halfs into half2 (__halves2half2)
 *
 * Total: ~4 instructions for 2 de-quantized values = 2 instructions/element.
 * This is negligible compared to the FMA chain in the MatMul accumulation.
 *
 * WHY half2?
 * ----------
 * NVIDIA GPUs (Volta+) have half2 FMA instructions (hfma2) that process
 * two FP16 values in a single instruction. Using half2 throughout the
 * kernel doubles the throughput vs. scalar half operations.
 *
 * @param a     First INT8 weight value.
 * @param b     Second INT8 weight value.
 * @param scale De-quantization scale factor (as half).
 * @return half2 containing {a * scale, b * scale} in FP16.
 */
__device__ __forceinline__ half2 dequant_int8_to_half2(
    int8_t a, int8_t b, half scale
) {
    // Step 1: Widen int8 → int32 (sign-extended)
    int a_int = static_cast<int>(a);
    int b_int = static_cast<int>(b);

    // Step 2: Convert int32 → half (round to nearest even)
    half a_half = __int2half_rn(a_int);
    half b_half = __int2half_rn(b_int);

    // Step 3: Multiply by scale to recover approximate FP16 value
    a_half = __hmul(a_half, scale);
    b_half = __hmul(b_half, scale);

    // Step 4: Pack into half2 for vectorized FMA operations
    return __halves2half2(a_half, b_half);
}

/**
 * @brief De-quantize a single INT8 value to half.
 *
 * Used when we have an odd number of elements and can't vectorize
 * the last element with half2.
 *
 * @param val   INT8 weight value.
 * @param scale De-quantization scale factor (as half).
 * @return De-quantized half value.
 */
__device__ __forceinline__ half dequant_int8_to_half(
    int8_t val, half scale
) {
    return __hmul(__int2half_rn(static_cast<int>(val)), scale);
}

// ═══════════════════════════════════════════════════════════════════
// HOST FUNCTIONS (run on CPU for offline weight preparation)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Quantize an FP32 weight tensor to INT8 (host-side, offline).
 *
 * This runs ONCE during model loading, not during inference.
 *
 * ALGORITHM
 * ---------
 *   1. Find max(|W|) across all N elements (single pass)
 *   2. Compute scale = max_abs / 127.0
 *   3. For each element: W_int8 = clamp(round(W / scale), -128, 127)
 *
 * The clamp to [-128, 127] handles edge cases where rounding could
 * produce 128 (which doesn't fit in int8_t). In practice with symmetric
 * quantization to 127, the max quantized value is exactly 127.
 *
 * @param src        Input FP32 weight array, length N.
 * @param dst        Output INT8 array, length N (pre-allocated by caller).
 * @param out_scale  Output: the computed scale factor.
 * @param N          Number of elements.
 */
inline void quantize_weights_cpu(const float* src,
                                  int8_t* dst,
                                  float* out_scale,
                                  int N) {
    // Pass 1: find maximum absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
        float abs_val = std::fabs(src[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    /**
     * Handle the degenerate case where all weights are zero.
     * Scale must be non-zero to avoid division by zero in de-quant.
     * We set scale=1.0 and all quantized values will be 0.
     */
    if (max_abs == 0.0f) {
        *out_scale = 1.0f;
        for (int i = 0; i < N; ++i) dst[i] = 0;
        return;
    }

    float scale = max_abs / 127.0f;
    *out_scale = scale;

    // Pass 2: quantize each element
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < N; ++i) {
        /**
         * Round to nearest, then clamp to int8 range.
         * std::round rounds half-to-away-from-zero, which matches
         * PyTorch's default quantization behavior.
         */
        float scaled = src[i] * inv_scale;
        int rounded = static_cast<int>(std::round(scaled));

        // Clamp to [-128, 127]
        if (rounded < -128) rounded = -128;
        if (rounded >  127) rounded =  127;

        dst[i] = static_cast<int8_t>(rounded);
    }
}

/**
 * @brief De-quantize INT8 weights back to FP32 (host-side, for validation).
 *
 * Used in test_baseline.py to verify the quantization roundtrip error.
 *
 * @param src    INT8 weight array, length N.
 * @param dst    Output FP32 array, length N (pre-allocated).
 * @param scale  The scale factor from quantize_weights_cpu.
 * @param N      Number of elements.
 */
inline void dequantize_weights_cpu(const int8_t* src,
                                    float* dst,
                                    float scale,
                                    int N) {
    for (int i = 0; i < N; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

}  // namespace flashgraph

#endif  // FLASHGRAPH_QUANTIZATION_CUH
