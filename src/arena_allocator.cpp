/**
 * @file arena_allocator.cpp
 * @brief Implementation of the bump-pointer memory arena.
 *
 * MEMORY LAYOUT
 * =============
 * After construction, the arena looks like this:
 *
 *   base_ (64-byte aligned)
 *   |
 *   v
 *   [================ capacity_ bytes ================]
 *   ^
 *   |
 *   offset_ = 0  (bump pointer starts here)
 *
 * After two allocations of 128 and 256 bytes:
 *
 *   [AAAA...128B...AAAA|pppp|BBBB...256B...BBBB|......free......]
 *                       ^padding (to align B)
 *   offset_ points here ──────────────────────────────^
 *
 * Each alloc():
 *   1. Align the current offset up to the requested boundary
 *   2. Check capacity
 *   3. Return base_ + aligned_offset
 *   4. Advance offset_ past the allocation
 */

#include "arena_allocator.h"

#include <cstdio>
#include <cstdlib>

namespace flashgraph {

Arena::Arena(size_t capacity) : base_(nullptr), capacity_(capacity), offset_(0) {
    /**
     * posix_memalign(&ptr, alignment, size)
     *
     * Allocates `size` bytes aligned to `alignment`. Alignment MUST be:
     *   1. A power of two
     *   2. A multiple of sizeof(void*) — always true for 64
     *
     * Returns 0 on success. Unlike malloc, the returned pointer must be
     * freed with free(), not with any special deallocator.
     *
     * We use posix_memalign instead of aligned_alloc because:
     *   - aligned_alloc requires size to be a multiple of alignment (C11 §7.22.3.1)
     *   - posix_memalign has no such restriction
     *   - Both are available on Linux; posix_memalign is more portable to older glibc
     */
    void* raw = nullptr;
    int result = posix_memalign(&raw, DEFAULT_ALIGNMENT, capacity);
    if (result != 0 || raw == nullptr) {
        std::fprintf(stderr,
            "[FlashGraph] FATAL: posix_memalign failed for %zu bytes "
            "(alignment=%zu, error=%d)\n",
            capacity, DEFAULT_ALIGNMENT, result);
        std::abort();
    }
    base_ = static_cast<uint8_t*>(raw);
}

Arena::~Arena() {
    /**
     * free() is the correct deallocator for posix_memalign.
     * (Unlike _aligned_malloc on Windows, which requires _aligned_free.)
     */
    std::free(base_);
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
}

void* Arena::alloc(size_t bytes, size_t align) {
    /**
     * STEP 1: Align the current offset upward.
     *
     * Formula: aligned = (offset + align - 1) & ~(align - 1)
     *
     * This works because for any power-of-two A:
     *   ~(A - 1) creates a bitmask that zeros the lower log2(A) bits.
     *   Adding (A - 1) before masking ensures we round UP, not down.
     *
     * Example with align=64 (binary: 0b1000000):
     *   ~(63)     = ~(0b0111111) = 0b...11000000
     *   offset=100 → (100+63) & mask = 163 & mask = 128
     *   offset=128 → (128+63) & mask = 191 & mask = 128  (already aligned, no waste)
     */
    size_t aligned_offset = (offset_ + align - 1) & ~(align - 1);

    /**
     * STEP 2: Check if the allocation fits.
     *
     * We deliberately abort rather than returning nullptr because:
     *   - A null return would require every caller to check for null
     *   - In inference, an OOM means the arena was mis-sized at startup
     *   - Aborting immediately gives a clear stack trace for debugging
     */
    if (aligned_offset + bytes > capacity_) {
        std::fprintf(stderr,
            "[FlashGraph] FATAL: Arena exhausted. Requested %zu bytes at offset %zu, "
            "capacity is %zu bytes (%zu bytes remaining).\n",
            bytes, aligned_offset, capacity_, capacity_ - offset_);
        std::abort();
    }

    /**
     * STEP 3: Bump the pointer and return.
     *
     * The returned pointer is base_ + aligned_offset, which is guaranteed
     * to satisfy the alignment requirement because:
     *   - base_ is aligned to DEFAULT_ALIGNMENT (64)
     *   - aligned_offset is aligned to `align`
     *   - If align <= DEFAULT_ALIGNMENT, the sum is aligned to `align`
     *   - If align > DEFAULT_ALIGNMENT, the user must ensure the base is
     *     also aligned to that boundary (or accept the risk)
     */
    void* ptr = base_ + aligned_offset;
    offset_ = aligned_offset + bytes;
    return ptr;
}

void Arena::reset() {
    /**
     * Reset does NOT zero memory. This is intentional:
     *   - Zeroing is O(capacity), which defeats the purpose of O(1) reset.
     *   - In inference, we overwrite every byte before reading, so zeroing
     *     is wasted work.
     *   - If you need zeroed memory, call memset on the returned pointer
     *     after alloc().
     */
    offset_ = 0;
}

}  // namespace flashgraph
