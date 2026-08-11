"""
FlashGraph — Build script for the Python extension module.

Compiles the C++/CUDA inference engine into a Python module named
'flashgraph' using torch.utils.cpp_extension.CUDAExtension.

Usage:
    pip install -e .         (editable install, recommended for development)
    python setup.py install  (standard install)

Requirements:
    - PyTorch with CUDA support
    - CUDA toolkit (nvcc) accessible in PATH
    - g++ >= 9.0 (for C++17 support)
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Resolve paths relative to this file's location
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROOT_DIR)  # parent of python/

setup(
    name="flashgraph",
    version="0.1.0",
    description="High-performance fused inference engine with INT8 quantization",
    author="Kevin",
    ext_modules=[
        CUDAExtension(
            name="flashgraph",
            sources=[
                # Python binding glue
                os.path.join(ROOT_DIR, "flashgraph_ext.cpp"),
                # CPU core (arena + baseline ops)
                os.path.join(PROJECT_DIR, "src", "arena_allocator.cpp"),
                os.path.join(PROJECT_DIR, "src", "cpu_ops.cpp"),
                # CUDA fused kernel
                os.path.join(PROJECT_DIR, "cuda", "fused_kernel.cu"),
            ],
            include_dirs=[
                os.path.join(PROJECT_DIR, "include"),
                os.path.join(PROJECT_DIR, "cuda"),
            ],
            extra_compile_args={
                # C++ flags
                "cxx": ["-O2", "-std=c++17"],
                # CUDA flags
                "nvcc": [
                    "--use_fast_math",
                    "-std=c++17",
                    # Generate code for common GPU architectures
                    # T4 (Turing, sm_75) — common in Colab
                    # A100 (Ampere, sm_80) — Colab Pro
                    # V100 (Volta, sm_70) — older Colab
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0.0"],
)
