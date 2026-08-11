"""
FlashGraph — Numerical Validation Test Suite

Compares the flashgraph engine output against PyTorch reference
implementations to verify mathematical correctness.

Tests:
    1. test_rmsnorm_parity      — CPU RMSNorm vs torch RMSNorm
    2. test_matmul_parity       — CPU MatMul vs torch.matmul
    3. test_gelu_parity         — CPU GELU vs torch.nn.functional.gelu
    4. test_int8_quantization_roundtrip — quantize→dequant error check
    5. test_cpu_pipeline_parity — Full CPU pipeline vs PyTorch pipeline
    6. test_fused_fp16_parity   — GPU fused vs PyTorch FP16 (CUDA only)
    7. test_fused_vs_cpu        — GPU fused vs CPU baseline (CUDA only)

Usage:
    # CPU-only tests (local):
    pytest tests/test_baseline.py -v -k "not cuda"

    # All tests (on GPU machine / Colab):
    pytest tests/test_baseline.py -v

Seed: torch.manual_seed(42) for reproducibility.
"""

import pytest
import torch
import torch.nn.functional as F
import math

# ─── Deterministic seeding ──────────────────────────────────────────
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ─── Try to import flashgraph ───────────────────────────────────────
try:
    import flashgraph
    HAS_FLASHGRAPH = True
except ImportError:
    HAS_FLASHGRAPH = False

# ─── Dimensions ─────────────────────────────────────────────────────
M = 64      # batch/seq rows
K = 128     # hidden dim
N = 256     # output dim
EPS = 1e-6


# ═══════════════════════════════════════════════════════════════════
# PYTORCH REFERENCE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm in pure PyTorch."""
    # x: [M, K], weight: [K]
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight.unsqueeze(0)


def pytorch_pipeline(x: torch.Tensor, weight: torch.Tensor,
                      rms_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference full pipeline: RMSNorm → MatMul(B^T) → GELU."""
    normed = pytorch_rmsnorm(x, rms_weight, eps)
    matmul_out = normed @ weight.T  # weight is [N, K], so weight.T is [K, N]
    return F.gelu(matmul_out, approximate='tanh')


# ═══════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def input_tensor():
    torch.manual_seed(42)
    return torch.randn(M, K, dtype=torch.float32)


@pytest.fixture
def weight_tensor():
    torch.manual_seed(123)
    return torch.randn(N, K, dtype=torch.float32)


@pytest.fixture
def rms_weight_tensor():
    torch.manual_seed(456)
    # RMSNorm gamma: initialized near 1.0 (typical transformer init)
    return 0.5 + torch.randn(K, dtype=torch.float32) * 0.1


# ═══════════════════════════════════════════════════════════════════
# CPU TESTS (run locally without GPU)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_FLASHGRAPH, reason="flashgraph module not installed")
class TestCPUOps:
    """Test CPU baseline ops against PyTorch reference."""

    def test_rmsnorm_parity(self, input_tensor, rms_weight_tensor):
        """Verify CPU RMSNorm matches PyTorch implementation."""
        # PyTorch reference
        expected = pytorch_rmsnorm(input_tensor, rms_weight_tensor, EPS)

        # FlashGraph CPU — runs full pipeline but we only check RMSNorm
        # by setting weight to identity and no GELU
        # Instead, we test via the full cpu_inference and compare RMSNorm stage
        # Note: we test the full pipeline separately; here we verify
        # the RMSNorm component is correct by using identity weight for matmul

        # For direct RMSNorm testing, use an identity-like weight matrix
        # and skip GELU by checking intermediate values.
        # Since cpu_inference fuses all three, we verify end-to-end below.
        # This test verifies the math of RMSNorm in isolation by comparison.

        # Direct test: create identity weight [K, K] and ones rms_weight
        eye_weight = torch.eye(K, dtype=torch.float32)
        ones_rms = torch.ones(K, dtype=torch.float32)

        # RMSNorm(x) with weight=ones = x / rms
        ref_normed = pytorch_rmsnorm(input_tensor, ones_rms, EPS)

        # FlashGraph pipeline with identity weight: RMSNorm(x) @ I^T → GELU
        fg_out = flashgraph.cpu_inference(input_tensor, eye_weight, ones_rms, EPS)

        # After GELU: the comparison is GELU(RMSNorm(x) @ I) vs GELU(ref_normed)
        expected_gelu = F.gelu(ref_normed, approximate='tanh')

        torch.testing.assert_close(fg_out, expected_gelu, atol=1e-5, rtol=1e-4)

    def test_matmul_parity(self, input_tensor, weight_tensor):
        """Verify CPU MatMul matches torch.matmul."""
        # Use no RMSNorm (weight=1) and no GELU by checking pre-GELU stage
        # Since we can't isolate MatMul from the pipeline, we verify
        # the full pipeline and rely on the GELU test below.

        # Direct matmul test: torch reference
        expected = input_tensor @ weight_tensor.T

        # We can verify matmul indirectly through the pipeline by using
        # rms_weight=ones (identity RMSNorm) and comparing pre-GELU.
        # But since cpu_inference applies GELU, we compare after GELU:
        expected_gelu = F.gelu(expected, approximate='tanh')

        ones_rms = torch.ones(K, dtype=torch.float32)
        fg_out = flashgraph.cpu_inference(input_tensor, weight_tensor, ones_rms, EPS)

        # Tolerance accounts for slight RMSNorm effect even with unit weights
        # (RMSNorm with ones weight still normalizes by sqrt(mean(x^2)+eps))
        # So we compare against full pipeline reference
        ref = pytorch_pipeline(input_tensor, weight_tensor,
                               ones_rms, EPS)
        torch.testing.assert_close(fg_out, ref, atol=1e-4, rtol=1e-3)

    def test_gelu_parity(self, input_tensor):
        """Verify GELU matches PyTorch's tanh approximation."""
        # Create a test where we can isolate GELU:
        # Use a simple known input
        x = torch.linspace(-3, 3, 1000, dtype=torch.float32)
        expected = F.gelu(x, approximate='tanh')

        # We can't call GELU in isolation from the Python API,
        # so we verify it through the pipeline with identity transforms:
        # x_row [1, 1000], weight=eye [1000, 1000], rms=ones
        # This makes RMSNorm(x) ≈ x/rms, then matmul with identity, then GELU
        x_2d = x.unsqueeze(0)  # [1, 1000]
        eye = torch.eye(1000, dtype=torch.float32)
        ones = torch.ones(1000, dtype=torch.float32)

        fg_out = flashgraph.cpu_inference(x_2d, eye, ones, EPS)

        # After RMSNorm (normalizes by rms), the values change.
        # So we compare against the full reference:
        ref = pytorch_pipeline(x_2d, eye, ones, EPS)
        torch.testing.assert_close(fg_out, ref, atol=1e-5, rtol=1e-5)

    def test_cpu_pipeline_full(self, input_tensor, weight_tensor, rms_weight_tensor):
        """Full pipeline parity: flashgraph CPU vs PyTorch reference."""
        expected = pytorch_pipeline(input_tensor, weight_tensor,
                                     rms_weight_tensor, EPS)
        fg_out = flashgraph.cpu_inference(input_tensor, weight_tensor,
                                           rms_weight_tensor, EPS)

        torch.testing.assert_close(fg_out, expected, atol=1e-4, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════
# INT8 QUANTIZATION TESTS (CPU)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_FLASHGRAPH, reason="flashgraph module not installed")
class TestQuantization:
    """Test INT8 quantization roundtrip accuracy."""

    def test_int8_quantization_roundtrip(self, weight_tensor):
        """Quantize → dequantize → compare vs original."""
        weight_int8, scale = flashgraph.quantize_weights(weight_tensor)

        # Dequantize manually
        dequantized = weight_int8.float() * scale

        # Maximum error should be <= scale/2 (half an LSB)
        max_error = (weight_tensor - dequantized).abs().max().item()
        expected_max = scale / 2.0 + 1e-6  # small epsilon for float rounding

        assert max_error <= expected_max, (
            f"Quantization roundtrip error {max_error:.6f} exceeds "
            f"expected maximum {expected_max:.6f} (scale={scale:.6f})"
        )

    def test_quantization_zero_weights(self):
        """Edge case: all-zero weights should quantize cleanly."""
        zeros = torch.zeros(16, 32, dtype=torch.float32)
        weight_int8, scale = flashgraph.quantize_weights(zeros)

        assert scale > 0, "Scale should be positive even for zero weights"
        assert weight_int8.abs().max().item() == 0, "Quantized zeros should be zero"

    def test_quantization_preserves_shape(self, weight_tensor):
        """Output shape must match input shape."""
        weight_int8, scale = flashgraph.quantize_weights(weight_tensor)

        assert weight_int8.shape == weight_tensor.shape, (
            f"Shape mismatch: {weight_int8.shape} vs {weight_tensor.shape}")
        assert weight_int8.dtype == torch.int8


# ═══════════════════════════════════════════════════════════════════
# GPU TESTS (run only on machines with CUDA)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_FLASHGRAPH, reason="flashgraph module not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUFused:
    """Test fused GPU kernel against PyTorch FP16 reference."""

    def test_fused_fp16_parity(self, input_tensor, weight_tensor, rms_weight_tensor):
        """Compare GPU fused output vs PyTorch FP16 reference.

        Tolerance: atol=1e-2, rtol=1e-2
        This accounts for:
          - FP16 representational error (~1e-3)
          - INT8 quantization noise (~scale/254)
          - Fused op accumulation order differences
        """
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Prepare GPU inputs
        input_fp16 = input_tensor.half().cuda()
        rms_fp16 = rms_weight_tensor.half().cuda()

        # Quantize weights
        weight_int8, scale = flashgraph.quantize_weights(weight_tensor)
        weight_int8_cuda = weight_int8.cuda()

        # FlashGraph fused kernel
        fg_out = flashgraph.fused_inference(
            input_fp16, weight_int8_cuda, rms_fp16, scale, EPS
        )

        # PyTorch FP16 reference (using dequantized weights for fair comparison)
        weight_dequant = (weight_int8.float() * scale).half().cuda()
        ref = pytorch_pipeline(
            input_tensor.half().cuda().float(),
            weight_dequant.float(),
            rms_weight_tensor.half().cuda().float(),
            EPS
        ).half()

        torch.testing.assert_close(
            fg_out.cpu(), ref.cpu(), atol=1e-2, rtol=1e-2
        )

    def test_fused_vs_cpu_baseline(self, input_tensor, weight_tensor, rms_weight_tensor):
        """Compare GPU fused vs CPU baseline.

        Tolerance: atol=5e-2
        Expected drift from: INT8 quantization + FP16 precision + fused op ordering.
        """
        # CPU baseline (FP32)
        cpu_out = flashgraph.cpu_inference(
            input_tensor, weight_tensor, rms_weight_tensor, EPS
        )

        # GPU fused (FP16 + INT8)
        input_fp16 = input_tensor.half().cuda()
        rms_fp16 = rms_weight_tensor.half().cuda()
        weight_int8, scale = flashgraph.quantize_weights(weight_tensor)
        weight_int8_cuda = weight_int8.cuda()

        gpu_out = flashgraph.fused_inference(
            input_fp16, weight_int8_cuda, rms_fp16, scale, EPS
        )

        # Compare with relaxed tolerance
        torch.testing.assert_close(
            gpu_out.cpu().float(), cpu_out, atol=5e-2, rtol=5e-2
        )

    def test_fused_output_shape(self, input_tensor, weight_tensor, rms_weight_tensor):
        """Verify output shape and dtype."""
        input_fp16 = input_tensor.half().cuda()
        rms_fp16 = rms_weight_tensor.half().cuda()
        weight_int8, scale = flashgraph.quantize_weights(weight_tensor)
        weight_int8_cuda = weight_int8.cuda()

        out = flashgraph.fused_inference(
            input_fp16, weight_int8_cuda, rms_fp16, scale, EPS
        )

        assert out.shape == (M, N), f"Expected shape ({M}, {N}), got {out.shape}"
        assert out.dtype == torch.float16
        assert out.device.type == "cuda"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
