/**
 * @file tensor.h
 * @brief Lightweight tensor descriptor — a non-owning view into arena memory.
 *
 * DESIGN PHILOSOPHY
 * =================
 * TensorView is a Plain Old Data (POD) struct. It does NOT own memory.
 * It stores:
 *   - A raw pointer to data (lives inside the Arena)
 *   - Shape (up to 4 dimensions)
 *   - Strides (in elements, not bytes)
 *
 * This is intentionally minimal. No reference counting, no move semantics,
 * no dynamic allocation. TensorView is meant to be passed by value or const
 * reference — it's only 80 bytes on the stack.
 *
 * STRIDE CONVENTION
 * =================
 * Strides are stored in ROW-MAJOR order (C-contiguous), measured in ELEMENTS.
 * For a tensor of shape [D0, D1, D2, D3]:
 *   stride[3] = 1
 *   stride[2] = D3
 *   stride[1] = D3 * D2
 *   stride[0] = D3 * D2 * D1
 *
 * To compute the byte offset of element [i, j, k, l]:
 *   byte_offset = (i*stride[0] + j*stride[1] + k*stride[2] + l*stride[3]) * elem_size
 *
 * ELEMENT ACCESS
 * ==============
 * For a float tensor t:
 *   float val = ((float*)t.data)[i * t.stride[0] + j * t.stride[1]];
 *
 * This is equivalent to t[i][j] in a 2D array, but works with any memory layout.
 */

#ifndef FLASHGRAPH_TENSOR_H
#define FLASHGRAPH_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "arena_allocator.h"

namespace flashgraph {

/**
 * @brief Maximum number of tensor dimensions supported.
 *
 * 4 covers all common cases:
 *   - 1D: [hidden_dim]        → bias, scale vectors
 *   - 2D: [M, K]              → weight matrices, activations
 *   - 3D: [batch, seq, hidden] → transformer inputs
 *   - 4D: [batch, heads, seq, dim] → multi-head attention
 */
constexpr int MAX_DIMS = 4;

struct TensorView {
    void*   data;             ///< Raw pointer into arena memory (non-owning)
    int     dims[MAX_DIMS];   ///< Shape of the tensor (unused dims = 1)
    int     ndim;             ///< Number of active dimensions (1–4)
    size_t  stride[MAX_DIMS]; ///< Strides in elements (row-major, C-contiguous)
    size_t  elem_size;        ///< Size of one element in bytes (e.g., 4 for float)

    /**
     * @brief Total number of elements in the tensor.
     */
    size_t numel() const {
        size_t n = 1;
        for (int i = 0; i < ndim; ++i) {
            n *= static_cast<size_t>(dims[i]);
        }
        return n;
    }

    /**
     * @brief Total size in bytes.
     */
    size_t size_bytes() const {
        return numel() * elem_size;
    }

    /**
     * @brief Typed data access — returns a pointer cast to T*.
     *
     * Usage: float* ptr = tensor.as<float>();
     * WARNING: No type safety — caller is responsible for matching T to elem_size.
     */
    template<typename T>
    T* as() const {
        return static_cast<T*>(data);
    }
};

/**
 * @brief Allocate a tensor from the arena and return a populated TensorView.
 *
 * Computes row-major strides automatically.
 *
 * @param arena  The arena to allocate from.
 * @param shape  Dimension sizes (e.g., {128, 256} for a 128×256 matrix).
 * @param elem_size  Size of one element in bytes (e.g., sizeof(float) = 4).
 * @return A TensorView pointing into arena memory with computed strides.
 *
 * EXAMPLE
 * -------
 *   Arena arena(16 * 1024 * 1024);
 *   TensorView A = arena_tensor(arena, {128, 256}, sizeof(float));
 *   // A.dims = {128, 256, 1, 1}, A.ndim = 2
 *   // A.stride = {256, 1, 0, 0}
 *   // A.data points to 128*256*4 = 131072 bytes inside the arena
 */
inline TensorView arena_tensor(Arena& arena,
                                std::initializer_list<int> shape,
                                size_t elem_size) {
    TensorView t{};
    t.ndim = static_cast<int>(shape.size());
    t.elem_size = elem_size;

    // Copy shape into dims, padding unused dimensions with 1
    int i = 0;
    for (int d : shape) {
        t.dims[i++] = d;
    }
    for (; i < MAX_DIMS; ++i) {
        t.dims[i] = 1;
    }

    /**
     * Compute row-major (C-contiguous) strides.
     *
     * Starting from the innermost dimension:
     *   stride[ndim-1] = 1
     *   stride[i] = stride[i+1] * dims[i+1]
     *
     * For shape [128, 256]:
     *   stride[1] = 1
     *   stride[0] = 1 * 256 = 256
     *
     * Element [i][j] is at offset: i * 256 + j * 1
     */
    for (int d = 0; d < MAX_DIMS; ++d) {
        t.stride[d] = 0;
    }
    if (t.ndim > 0) {
        t.stride[t.ndim - 1] = 1;
        for (int d = t.ndim - 2; d >= 0; --d) {
            t.stride[d] = t.stride[d + 1] * static_cast<size_t>(t.dims[d + 1]);
        }
    }

    // Allocate from arena — aligned to 64 bytes by default
    size_t total_bytes = t.numel() * elem_size;
    t.data = arena.alloc(total_bytes);

    return t;
}

}  // namespace flashgraph

#endif  // FLASHGRAPH_TENSOR_H
