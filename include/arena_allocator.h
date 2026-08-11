/**
 * @file arena_allocator.h
 * @brief Zero-allocation bump-pointer memory arena with 64-byte cache-line alignment.
 *
 * DESIGN RATIONALE
 * ================
 * This arena provides a single contiguous memory block allocated at construction
 * time via posix_memalign. All subsequent "allocations" are pointer bumps — zero
 * syscalls, zero fragmentation, zero metadata overhead per allocation.
 *
 * WHY 64-BYTE ALIGNMENT?
 * ----------------------
 * Modern x86-64 CPUs (Intel Skylake+, AMD Zen+) use 64-byte cache lines. Aligning
 * tensor data to cache-line boundaries prevents false sharing between CPU cores and
 * ensures that SIMD loads (AVX-512 = 64 bytes) are naturally aligned, avoiding
 * split-line penalties that can cost 2–10 cycles per misaligned access.
 *
 * ALIGNMENT MATH
 * --------------
 * To align an offset to boundary `A` (must be power of 2):
 *   aligned = (offset + A - 1) & ~(A - 1)
 *
 * Example: offset=100, A=64
 *   (100 + 63) & ~63 = 163 & 0xFFFFFFC0 = 128  ✓
 *
 * RUNTIME GUARANTEE
 * -----------------
 * After Arena construction, ZERO calls to malloc/free/new/delete occur.
 * If the arena is exhausted, we abort — no fallback, no exception.
 * This is intentional: in a latency-critical inference engine, an OOM
 * condition is a configuration error that must be caught during development,
 * not silently handled at runtime.
 *
 * USAGE PATTERN
 * -------------
 *   Arena arena(16 * 1024 * 1024);  // 16 MB, allocated once
 *   float* weights = (float*)arena.alloc(4096 * sizeof(float));
 *   float* activations = (float*)arena.alloc(2048 * sizeof(float));
 *   // ... run inference ...
 *   arena.reset();  // "free" everything — pointer goes back to zero
 *   // ... next batch reuses the same memory ...
 */

#ifndef FLASHGRAPH_ARENA_ALLOCATOR_H
#define FLASHGRAPH_ARENA_ALLOCATOR_H

#include <cstddef>
#include <cstdint>

namespace flashgraph {

/**
 * @brief Default alignment in bytes — matches x86-64 cache line width.
 */
constexpr size_t DEFAULT_ALIGNMENT = 64;

class Arena {
public:
    /**
     * @brief Construct an arena with the given capacity.
     *
     * Allocates a single contiguous block of `capacity` bytes aligned to
     * DEFAULT_ALIGNMENT using posix_memalign. This is the ONLY dynamic
     * allocation that will ever occur.
     *
     * @param capacity Total arena size in bytes.
     */
    explicit Arena(size_t capacity);

    /**
     * @brief Destructor — frees the underlying memory block.
     */
    ~Arena();

    // Non-copyable, non-movable — the arena owns a unique memory block.
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;

    /**
     * @brief Allocate `bytes` from the arena with the given alignment.
     *
     * Bumps the internal pointer forward. Returns a pointer into the arena.
     * If the arena cannot satisfy the request, calls std::abort().
     *
     * @param bytes Number of bytes to allocate.
     * @param align Alignment requirement (must be power of 2, default 64).
     * @return Pointer to the allocated region.
     */
    void* alloc(size_t bytes, size_t align = DEFAULT_ALIGNMENT);

    /**
     * @brief Reset the arena — reclaims all allocations.
     *
     * Sets the bump pointer back to zero. Does NOT zero the memory.
     * All previously returned pointers become invalid.
     */
    void reset();

    /**
     * @brief Returns the total capacity of the arena in bytes.
     */
    size_t capacity() const { return capacity_; }

    /**
     * @brief Returns the number of bytes currently in use.
     */
    size_t used() const { return offset_; }

    /**
     * @brief Returns the number of bytes remaining.
     */
    size_t remaining() const { return capacity_ - offset_; }

    /**
     * @brief Returns the base pointer of the arena.
     *
     * Useful for computing offsets: offset = (uint8_t*)ptr - arena.base()
     */
    uint8_t* base() const { return base_; }

private:
    uint8_t* base_;      ///< Base pointer of the arena (aligned).
    size_t   capacity_;  ///< Total capacity in bytes.
    size_t   offset_;    ///< Current bump-pointer offset from base_.
};

}  // namespace flashgraph

#endif  // FLASHGRAPH_ARENA_ALLOCATOR_H
