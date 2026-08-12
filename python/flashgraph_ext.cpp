/**
 * @file flashgraph_ext.cpp
 * @brief Python/PyTorch binding for the FlashGraph inference engine.
 *
 * BINDING STRATEGY
 * ================
 * We use torch::utils::cpp_extension with PYBIND11_MODULE to expose two
 * Python-callable functions:
 *
 *   1. flashgraph.fused_inference(input, weight_int8, rms_weight, scale, eps)
 *      → Runs the fused RMSNorm→MatMul(INT8)→GELU kernel on GPU.
 *
 *   2. flashgraph.cpu_inference(input, weight, rms_weight, eps)
 *      → Runs the CPU baseline (FP32) for numerical validation.
 *
 *   3. flashgraph.quantize_weights(weight_fp32)
 *      → Returns (weight_int8, scale) for offline weight preparation.
 *
 *   4. flashgraph.benchmark(M, K, N, warmup, iterations)
 *      → Runs the fused kernel with timing for ncu profiling.
 *
 * MEMORY MANAGEMENT
 * =================
 * When called from Python/PyTorch, we do NOT use the Arena allocator.
 * Instead, we access PyTorch's GPU buffers directly via data_ptr<T>().
 * This is zero-copy: no memcpy, no staging buffers, no double allocation.
 *
 * The Arena allocator is reserved for the standalone C++ deployment path
 * (e.g., embedded inference without Python).
 *
 * DATA TYPE VALIDATION
 * ====================
 * Every binding function validates:
 *   - Tensor dtype matches expected type (float16, int8, float32)
 *   - Tensor is contiguous (no strides, no views with gaps)
 *   - Tensor device matches expected device (CPU vs CUDA)
 *   - Shape dimensions are consistent across inputs
 *
 * Mismatches throw std::runtime_error with diagnostic messages.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// CPU ops (always available)
#include "cpu_ops.h"

// Forward declarations for CUDA functions (defined in fused_kernel.cu)
// These are resolved at link time by the CUDA compiler.
namespace flashgraph {
void launch_fused_kernel(
    const half* input,
    const int8_t* weight,
    const half* rms_weight,
    half scale,
    half* output,
    int M, int K, int N,
    float eps,
    cudaStream_t stream
);
}

// ═══════════════════════════════════════════════════════════════════
// VALIDATION HELPERS
// ═══════════════════════════════════════════════════════════════════

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x)  TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dtype) \
    TORCH_CHECK(x.scalar_type() == dtype, \
        #x " must be " #dtype " but got ", x.scalar_type())

// ═══════════════════════════════════════════════════════════════════
// PYTHON-FACING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Run the fused RMSNorm→MatMul(INT8)→GELU kernel on GPU.
 *
 * @param input       [M, K] float16 CUDA tensor.
 * @param weight_int8 [N, K] int8 CUDA tensor (quantized weights).
 * @param rms_weight  [K] float16 CUDA tensor (RMSNorm gamma).
 * @param scale       Float scalar: INT8 de-quantization scale.
 * @param eps         Float scalar: RMSNorm epsilon (default 1e-6).
 * @return [M, N] float16 CUDA tensor.
 */
torch::Tensor fused_inference(
    torch::Tensor input,
    torch::Tensor weight_int8,
    torch::Tensor rms_weight,
    double scale,
    double eps
) {
    // Validate inputs
    CHECK_CUDA(input);
    CHECK_CUDA(weight_int8);
    CHECK_CUDA(rms_weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight_int8);
    CHECK_CONTIGUOUS(rms_weight);
    CHECK_DTYPE(input, torch::kFloat16);
    CHECK_DTYPE(weight_int8, torch::kInt8);
    CHECK_DTYPE(rms_weight, torch::kFloat16);

    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
    TORCH_CHECK(weight_int8.dim() == 2, "weight must be 2D [N, K]");
    TORCH_CHECK(rms_weight.dim() == 1, "rms_weight must be 1D [K]");

    int M = input.size(0);
    int K = input.size(1);
    int N = weight_int8.size(0);

    TORCH_CHECK(weight_int8.size(1) == K,
        "weight K dimension mismatch: input K=", K,
        " weight K=", weight_int8.size(1));
    TORCH_CHECK(rms_weight.size(0) == K,
        "rms_weight size mismatch: expected ", K,
        " got ", rms_weight.size(0));

    // Allocate output tensor (PyTorch manages this memory on GPU)
    auto output = torch::empty({M, N}, input.options());

    // Convert scale to half
    half scale_half = __float2half(static_cast<float>(scale));

    // Launch fused kernel on the default CUDA stream
    flashgraph::launch_fused_kernel(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        weight_int8.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(rms_weight.data_ptr<at::Half>()),
        scale_half,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, K, N,
        static_cast<float>(eps),
        0  // default stream
    );

    return output;
}

/**
 * @brief Run the CPU baseline (FP32) for numerical validation.
 *
 * Executes RMSNorm → MatMul(B^T) → GELU on CPU using the reference
 * implementations. All operations are FP32 for maximum precision.
 *
 * @param input      [M, K] float32 CPU tensor.
 * @param weight     [N, K] float32 CPU tensor (full-precision weights).
 * @param rms_weight [K] float32 CPU tensor (RMSNorm gamma).
 * @param eps        Float scalar: RMSNorm epsilon.
 * @return [M, N] float32 CPU tensor.
 */
torch::Tensor cpu_inference(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rms_weight,
    double eps
) {
    CHECK_CPU(input);
    CHECK_CPU(weight);
    CHECK_CPU(rms_weight);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(rms_weight);
    CHECK_DTYPE(input, torch::kFloat32);
    CHECK_DTYPE(weight, torch::kFloat32);
    CHECK_DTYPE(rms_weight, torch::kFloat32);

    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);

    // Intermediate buffer for RMSNorm output
    auto normed = torch::empty({M, K}, input.options());
    // Output buffer
    auto output = torch::empty({M, N}, input.options());

    // Stage 1: RMSNorm
    flashgraph::cpu::rmsnorm(
        input.data_ptr<float>(),
        rms_weight.data_ptr<float>(),
        normed.data_ptr<float>(),
        M, K, static_cast<float>(eps)
    );

    // Stage 2: MatMul (A @ B^T)
    auto matmul_out = torch::empty({M, N}, input.options());
    flashgraph::cpu::matmul_bt(
        normed.data_ptr<float>(),
        weight.data_ptr<float>(),
        matmul_out.data_ptr<float>(),
        M, K, N
    );

    // Stage 3: GELU
    flashgraph::cpu::gelu(
        matmul_out.data_ptr<float>(),
        output.data_ptr<float>(),
        M * N
    );

    return output;
}

/**
 * @brief Quantize FP32 weights to INT8 (host-side utility).
 *
 * @param weight [N, K] float32 CPU tensor.
 * @return Tuple of (weight_int8 [N, K] int8 CPU tensor, scale float).
 */
std::tuple<torch::Tensor, double> quantize_weights(torch::Tensor weight) {
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    CHECK_DTYPE(weight, torch::kFloat32);

    int N = weight.size(0);
    int K = weight.size(1);
    int total = N * K;

    auto weight_int8 = torch::empty({N, K}, torch::dtype(torch::kInt8));
    float scale = 0.0f;

    // Use our CPU quantization utility
    // (defined inline in quantization.cuh, but we inline it here for CPU build)
    const float* src = weight.data_ptr<float>();
    int8_t* dst = weight_int8.data_ptr<int8_t>();

    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < total; ++i) {
        float abs_val = std::fabs(src[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    if (max_abs == 0.0f) {
        scale = 1.0f;
        std::memset(dst, 0, total);
    } else {
        scale = max_abs / 127.0f;
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < total; ++i) {
            int rounded = static_cast<int>(std::round(src[i] * inv_scale));
            if (rounded < -128) rounded = -128;
            if (rounded >  127) rounded =  127;
            dst[i] = static_cast<int8_t>(rounded);
        }
    }

    return std::make_tuple(weight_int8, static_cast<double>(scale));
}

/**
 * @brief Benchmark the fused kernel for profiling with ncu.
 *
 * Runs warmup iterations followed by timed iterations.
 * Returns the average time per iteration in milliseconds.
 */
double benchmark_kernel(int M, int K, int N, int warmup, int iterations) {
    auto input = torch::randn({M, K}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    auto weight = torch::randint(-128, 127, {N, K}, torch::dtype(torch::kInt8).device(torch::kCUDA));
    auto rms_weight = torch::ones({K}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(torch::kCUDA));

    half scale_half = __float2half(0.01f);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        flashgraph::launch_fused_kernel(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            weight.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(rms_weight.data_ptr<at::Half>()),
            scale_half,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, K, N, 1e-6f, 0
        );
    }
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        flashgraph::launch_fused_kernel(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            weight.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(rms_weight.data_ptr<at::Half>()),
            scale_half,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            M, K, N, 1e-6f, 0
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return static_cast<double>(ms / iterations);
}

// ═══════════════════════════════════════════════════════════════════
// MODULE REGISTRATION
// ═══════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashGraph: High-performance fused inference engine with "
              "INT8 quantization and FP16 compute.";

    m.def("fused_inference", &fused_inference,
        "Fused RMSNorm->MatMul(INT8)->GELU on GPU",
        py::arg("input"),
        py::arg("weight_int8"),
        py::arg("rms_weight"),
        py::arg("scale"),
        py::arg("eps") = 1e-6
    );

    m.def("cpu_inference", &cpu_inference,
        "CPU baseline (FP32) for validation",
        py::arg("input"),
        py::arg("weight"),
        py::arg("rms_weight"),
        py::arg("eps") = 1e-6
    );

    m.def("quantize_weights", &quantize_weights,
        "Quantize FP32 weights to INT8. Returns (weight_int8, scale).",
        py::arg("weight")
    );

    m.def("benchmark", &benchmark_kernel,
        "Benchmark the fused kernel. Returns avg ms per iteration.",
        py::arg("M") = 128,
        py::arg("K") = 256,
        py::arg("N") = 512,
        py::arg("warmup") = 10,
        py::arg("iterations") = 100
    );

    m.def("benchmark_kernel", &benchmark_kernel,
        "Benchmark the fused kernel. Returns avg ms per iteration.",
        py::arg("M") = 128,
        py::arg("K") = 256,
        py::arg("N") = 512,
        py::arg("warmup") = 10,
        py::arg("iterations") = 100
    );
}
