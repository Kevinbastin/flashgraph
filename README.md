# FlashGraph — High-Performance Fused Inference Engine

A bare-metal C++/CUDA inference engine optimized for a micro-transformer block, featuring zero-allocation memory management, INT8 weight quantization, and fused FP16 GPU kernels.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                             │
│  import flashgraph                                              │
│  output = flashgraph.fused_inference(input, weight, gamma, ...) │
└────────────────────────────┬────────────────────────────────────┘
                             │ torch::Tensor::data_ptr<T>()
                             │ (zero-copy)
┌────────────────────────────▼────────────────────────────────────┐
│                    C++ Binding Layer                             │
│  flashgraph_ext.cpp — input validation, PyTorch ↔ raw pointers │
├─────────────────────────────────────────────────────────────────┤
│                     CUDA Fused Kernel                           │
│  ┌─────────┐   ┌──────────────────┐   ┌──────┐                │
│  │ RMSNorm │ → │ MatMul (INT8→FP16)│ → │ GELU │  all in SRAM  │
│  └─────────┘   └──────────────────┘   └──────┘                │
│  Shared Memory Tiling: 64×64×32, FP32 accumulators             │
├─────────────────────────────────────────────────────────────────┤
│                   Host Memory Arena                             │
│  64-byte aligned bump allocator (posix_memalign)               │
│  Zero malloc after init — standalone C++ deployment path        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Detail |
|---------|--------|
| **Fused Kernel** | RMSNorm → MatMul → GELU in a single GPU launch (3× bandwidth reduction) |
| **INT8 Quantization** | Symmetric per-tensor; INT8 weights in global memory, on-the-fly FP16 dequant |
| **Zero Allocation** | 64-byte aligned arena allocator; zero `malloc` after initialization |
| **FP16 Compute** | `half2` vectorized operations with FP32 accumulators (no overflow) |
| **Async Streams** | Pinned host memory + `cudaStreamNonBlocking` for DMA overlap |

## Quick Start

### Local Build (CPU-only)

```bash
# Clone
git clone <your-repo-url> flashgraph && cd flashgraph

# Build with Make
make all    # → ./flashgraph_main
make test   # → ./test_arena (4 tests)

# Or with CMake
mkdir build && cd build
cmake -DENABLE_CUDA=OFF ..
make -j$(nproc)
```

### Cloud Build (GPU — Colab)

```bash
# In a Colab notebook with GPU runtime:
!git clone <your-repo-url> /content/flashgraph
!cd /content/flashgraph/python && pip install -e .

import flashgraph
output = flashgraph.fused_inference(input_fp16, weight_int8, rms_weight, scale)
```

Or use the provided notebook: `notebooks/colab_setup.ipynb`

## Python API

```python
import flashgraph
import torch

# Prepare inputs
input = torch.randn(128, 256, dtype=torch.float16, device='cuda')
weight_fp32 = torch.randn(512, 256)  # full precision weights
rms_weight = torch.ones(256, dtype=torch.float16, device='cuda')

# Quantize weights (offline, once)
weight_int8, scale = flashgraph.quantize_weights(weight_fp32)
weight_int8 = weight_int8.cuda()

# Run fused inference
output = flashgraph.fused_inference(input, weight_int8, rms_weight, scale, eps=1e-6)
# output: [128, 512] float16 on CUDA

# CPU baseline (for validation)
output_cpu = flashgraph.cpu_inference(
    input.float().cpu(), weight_fp32, rms_weight.float().cpu(), eps=1e-6
)

# Benchmark
avg_ms = flashgraph.benchmark(M=128, K=256, N=512, warmup=50, iterations=1000)
```

## Profiling with Nsight Compute

```bash
# Roofline model
ncu --set roofline \
    --kernel-name "fused_rmsnorm_matmul_gelu" \
    --launch-skip 5 --launch-count 3 \
    -o flashgraph_roofline \
    python -c "import flashgraph; flashgraph.benchmark()"

# Memory bandwidth metrics
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name "fused_rmsnorm_matmul_gelu" \
    python -c "import flashgraph; flashgraph.benchmark()"
```

## Project Structure

```
flashgraph/
├── include/
│   ├── arena_allocator.h    # Bump-pointer arena API
│   ├── tensor.h             # Lightweight tensor descriptor
│   └── cpu_ops.h            # CPU op declarations
├── src/
│   ├── arena_allocator.cpp  # Arena implementation
│   ├── cpu_ops.cpp          # CPU RMSNorm, MatMul, GELU
│   └── main.cpp             # CPU smoke test
├── cuda/
│   ├── fused_kernel.cu      # Fused GPU kernel
│   ├── quantization.cuh     # INT8 ↔ FP16 utilities
│   └── stream_manager.cuh   # Async stream management
├── python/
│   ├── flashgraph_ext.cpp   # PyTorch binding
│   └── setup.py             # Build script
├── tests/
│   ├── test_arena.cpp       # C++ arena unit tests
│   └── test_baseline.py     # Python validation suite
├── notebooks/
│   └── colab_setup.ipynb    # Cloud GPU setup
├── Makefile                 # CPU-only local build
└── CMakeLists.txt           # Full build system
```

## Testing

```bash
# C++ arena tests (local)
make test

# Python validation (requires flashgraph module + GPU)
pytest tests/test_baseline.py -v

# CPU-only Python tests
pytest tests/test_baseline.py -v -k "not cuda"
```

## License

MIT
