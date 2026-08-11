# ============================================================================
# FlashGraph — Full CMake Build System (CPU + optional CUDA)
# ============================================================================
#
# Usage:
#   LOCAL (CPU-only):
#     mkdir build && cd build
#     cmake -DENABLE_CUDA=OFF ..
#     make -j$(nproc)
#
#   CLOUD (with CUDA):
#     mkdir build && cd build
#     cmake -DENABLE_CUDA=ON ..
#     make -j$(nproc)
#
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(FlashGraph LANGUAGES CXX)

# C++17 required for std::initializer_list in constexpr contexts,
# structured bindings, and if-constexpr.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ────────────────────────────────────────────────────────
option(ENABLE_CUDA "Enable CUDA GPU support" OFF)

# ─── Compiler flags ────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wpedantic")

# ─── Include paths ─────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda)

# ─── CPU-only static library ───────────────────────────────────────
# Contains: arena allocator + CPU baseline ops
add_library(flashgraph_core STATIC
    src/arena_allocator.cpp
    src/cpu_ops.cpp
)
target_include_directories(flashgraph_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# ─── CPU test binary ───────────────────────────────────────────────
add_executable(flashgraph_main src/main.cpp)
target_link_libraries(flashgraph_main flashgraph_core m)

add_executable(test_arena tests/test_arena.cpp)
target_link_libraries(test_arena flashgraph_core m)

# ─── CUDA targets (only when ENABLE_CUDA=ON) ───────────────────────
if(ENABLE_CUDA)
    enable_language(CUDA)

    # Target architectures:
    #   70 = V100 (Volta)
    #   75 = T4 (Turing)
    #   80 = A100 (Ampere)
    #   86 = RTX 3090 (Ampere consumer)
    #   89 = L4/RTX 4090 (Ada Lovelace)
    #   90 = H100 (Hopper)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

    # CUDA compile flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

    # CUDA static library: fused kernel
    add_library(flashgraph_cuda STATIC
        cuda/fused_kernel.cu
    )
    target_include_directories(flashgraph_cuda PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/cuda
    )

    # Link CUDA library to core for combined usage
    target_link_libraries(flashgraph_cuda flashgraph_core)

    message(STATUS "CUDA enabled — building for architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA disabled — building CPU-only targets")
endif()

# ─── CTest support ─────────────────────────────────────────────────
enable_testing()
add_test(NAME arena_tests COMMAND test_arena)
