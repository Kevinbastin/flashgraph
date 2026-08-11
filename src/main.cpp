/**
 * @file main.cpp
 * @brief CPU-only smoke test for the FlashGraph inference pipeline.
 *
 * Allocates a 16 MB arena, creates tensors, fills with deterministic
 * pseudo-random data, and runs: RMSNorm → MatMul → GELU.
 * Prints first 8 output values and runs NaN/zero sanity checks.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "arena_allocator.h"
#include "tensor.h"
#include "cpu_ops.h"

/**
 * Simple LCG pseudo-random float in [-1, 1].
 * Parameters from Numerical Recipes: a=1664525, c=1013904223, m=2^32.
 */
static float lcg_float(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state) / 4294967296.0f) * 2.0f - 1.0f;
}

int main() {
    std::printf("=== FlashGraph CPU Smoke Test ===\n\n");

    constexpr int M = 128;   // rows (batch * seq)
    constexpr int K = 256;   // hidden dim
    constexpr int N = 512;   // output dim

    // Arena: single allocation, 16 MB
    constexpr size_t ARENA_SIZE = 16 * 1024 * 1024;
    flashgraph::Arena arena(ARENA_SIZE);
    std::printf("Arena: %zu bytes, base=%p\n\n",
                arena.capacity(), static_cast<void*>(arena.base()));

    // Allocate tensors from arena (zero heap allocations after this)
    auto A     = flashgraph::arena_tensor(arena, {M, K}, sizeof(float));
    auto B     = flashgraph::arena_tensor(arena, {N, K}, sizeof(float));
    auto gamma = flashgraph::arena_tensor(arena, {K},    sizeof(float));
    auto A_norm = flashgraph::arena_tensor(arena, {M, K}, sizeof(float));
    auto C     = flashgraph::arena_tensor(arena, {M, N}, sizeof(float));
    auto out   = flashgraph::arena_tensor(arena, {M, N}, sizeof(float));

    std::printf("Tensors allocated — arena used: %zu / %zu bytes\n\n",
                arena.used(), arena.capacity());

    // Fill with deterministic data
    uint32_t seed = 42;
    for (int i = 0; i < M * K; ++i) A.as<float>()[i] = lcg_float(seed);
    for (int i = 0; i < N * K; ++i) B.as<float>()[i] = lcg_float(seed);
    for (int i = 0; i < K; ++i) gamma.as<float>()[i] = 0.5f + lcg_float(seed) * 0.1f;

    // Pipeline: RMSNorm → MatMul → GELU
    std::printf("Running pipeline: RMSNorm -> MatMul -> GELU\n");

    std::printf("  [1/3] RMSNorm [%d x %d]\n", M, K);
    flashgraph::cpu::rmsnorm(A.as<float>(), gamma.as<float>(),
                              A_norm.as<float>(), M, K, 1e-6f);

    std::printf("  [2/3] MatMul  [%d x %d] @ [%d x %d]^T -> [%d x %d]\n",
                M, K, N, K, M, N);
    flashgraph::cpu::matmul_bt(A_norm.as<float>(), B.as<float>(),
                                C.as<float>(), M, K, N);

    std::printf("  [3/3] GELU    [%d x %d]\n", M, N);
    flashgraph::cpu::gelu(C.as<float>(), out.as<float>(), M * N);

    // Print results
    std::printf("\nFirst 8 output values:\n  ");
    for (int i = 0; i < 8; ++i) {
        std::printf("%.6f ", out.as<float>()[i]);
    }
    std::printf("\n");

    // Sanity checks
    bool has_nan = false, all_zero = true;
    for (int i = 0; i < M * N; ++i) {
        float v = out.as<float>()[i];
        if (v != v) has_nan = true;
        if (v != 0.0f) all_zero = false;
    }

    std::printf("\nSanity: NaN=%s  AllZero=%s\n",
                has_nan ? "YES(FAIL)" : "no(OK)",
                all_zero ? "YES(FAIL)" : "no(OK)");
    std::printf("=== Smoke test %s ===\n",
                (!has_nan && !all_zero) ? "PASSED" : "FAILED");

    return (has_nan || all_zero) ? 1 : 0;
}
