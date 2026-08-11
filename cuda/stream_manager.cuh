/**
 * @file stream_manager.cuh
 * @brief CUDA stream and pinned memory management utilities.
 *
 * ASYNC DMA OVERLAP — WHY PINNED MEMORY MATTERS
 * ===============================================
 * CUDA DMA (cudaMemcpyAsync) can overlap with kernel execution ONLY when
 * the host memory is "pinned" (page-locked via cudaMallocHost). Here's why:
 *
 *   Normal (pageable) memory:
 *     1. GPU DMA engine cannot access pageable host memory directly.
 *     2. The driver must first copy data to an internal pinned staging buffer.
 *     3. This staging copy blocks the CPU and serializes with kernel execution.
 *
 *   Pinned memory:
 *     1. cudaMallocHost pins physical pages (prevents OS from swapping them).
 *     2. GPU DMA engine can read/write directly via PCIe BAR mapping.
 *     3. cudaMemcpyAsync returns immediately — the DMA runs concurrently
 *        with kernel execution on a different CUDA stream.
 *
 * TRADEOFF: Pinned memory reduces available host RAM for the OS page cache
 * and other processes. Over-allocating pinned memory can cause the OS to
 * thrash. Rule of thumb: pin only what you need for active DMA transfers,
 * not your entire working set.
 *
 * STREAM SEMANTICS
 * ================
 * Operations on the same stream execute in order (FIFO).
 * Operations on different streams may execute concurrently.
 *
 * Typical double-buffering pattern:
 *   Stream 0: [copy batch 0 H→D] [kernel batch 0] [copy result 0 D→H]
 *   Stream 1:                     [copy batch 1 H→D] [kernel batch 1] [copy result 1 D→H]
 *
 * The overlap between "kernel batch 0" and "copy batch 1 H→D" is what
 * hides the PCIe transfer latency.
 */

#ifndef FLASHGRAPH_STREAM_MANAGER_CUH
#define FLASHGRAPH_STREAM_MANAGER_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace flashgraph {

/**
 * @brief Check a CUDA API return code and abort on failure.
 *
 * In an inference engine, CUDA errors are unrecoverable (corrupted state,
 * OOM on device). We abort immediately with a diagnostic message rather
 * than propagating error codes that nobody checks.
 */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::fprintf(stderr,                                            \
                "[FlashGraph] CUDA error at %s:%d — %s: %s\n",             \
                __FILE__, __LINE__,                                         \
                #call, cudaGetErrorString(err));                            \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

/**
 * @brief Manages a pool of CUDA streams and pinned host memory buffers.
 *
 * Owns `num_streams` cudaStream_t handles and provides pinned memory
 * allocation for async DMA transfers.
 *
 * LIFECYCLE
 * ---------
 *   StreamManager mgr(2);                  // Create 2 streams
 *   void* buf = mgr.pinned_alloc(4096);    // Allocate 4KB pinned
 *   cudaStream_t s = mgr.get_stream(0);    // Get stream 0
 *   cudaMemcpyAsync(d_ptr, buf, 4096,
 *                   cudaMemcpyHostToDevice, s);
 *   mgr.sync_all();                        // Wait for all streams
 *   // mgr destructor frees streams + pinned buffers
 */
class StreamManager {
public:
    /**
     * @brief Create `num_streams` CUDA streams.
     *
     * Each stream is created with cudaStreamNonBlocking so it does NOT
     * implicitly synchronize with the default stream. This is essential
     * for true concurrency between streams.
     *
     * @param num_streams Number of streams to create (typically 2–4).
     */
    explicit StreamManager(int num_streams)
        : num_streams_(num_streams),
          streams_(nullptr),
          pinned_buffers_(nullptr),
          num_pinned_(0),
          pinned_capacity_(8) {

        streams_ = new cudaStream_t[num_streams_];
        for (int i = 0; i < num_streams_; ++i) {
            /**
             * cudaStreamNonBlocking:
             *   - Does NOT synchronize with stream 0 (the default stream)
             *   - Required for concurrent kernel execution across streams
             *   - Without this flag, any operation on stream 0 would
             *     serialize with operations on our named streams
             */
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &streams_[i], cudaStreamNonBlocking));
        }

        // Pre-allocate tracking array for pinned buffers (freed in destructor)
        pinned_buffers_ = new void*[pinned_capacity_];
    }

    ~StreamManager() {
        // Free all pinned host memory buffers
        for (int i = 0; i < num_pinned_; ++i) {
            cudaFreeHost(pinned_buffers_[i]);  // safe even if CUDA is shut down
        }
        delete[] pinned_buffers_;

        // Destroy all streams
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamDestroy(streams_[i]);
        }
        delete[] streams_;
    }

    // Non-copyable
    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;

    /**
     * @brief Get the CUDA stream at the given index.
     *
     * @param idx Stream index in [0, num_streams).
     * @return The cudaStream_t handle.
     */
    cudaStream_t get_stream(int idx) const {
        if (idx < 0 || idx >= num_streams_) {
            std::fprintf(stderr,
                "[FlashGraph] Stream index %d out of range [0, %d)\n",
                idx, num_streams_);
            std::abort();
        }
        return streams_[idx];
    }

    /**
     * @brief Allocate pinned (page-locked) host memory.
     *
     * This memory is suitable for cudaMemcpyAsync. The buffer is tracked
     * and will be freed in the destructor.
     *
     * @param bytes Number of bytes to allocate.
     * @return Pointer to pinned host memory.
     */
    void* pinned_alloc(size_t bytes) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, bytes));

        // Track for cleanup — grow array if needed
        if (num_pinned_ >= pinned_capacity_) {
            int new_cap = pinned_capacity_ * 2;
            void** new_arr = new void*[new_cap];
            for (int i = 0; i < num_pinned_; ++i) {
                new_arr[i] = pinned_buffers_[i];
            }
            delete[] pinned_buffers_;
            pinned_buffers_ = new_arr;
            pinned_capacity_ = new_cap;
        }
        pinned_buffers_[num_pinned_++] = ptr;

        return ptr;
    }

    /**
     * @brief Synchronize all managed streams.
     *
     * Blocks the host thread until all operations on all streams
     * have completed. Use this at pipeline boundaries.
     */
    void sync_all() const {
        for (int i = 0; i < num_streams_; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
        }
    }

    /**
     * @brief Number of managed streams.
     */
    int num_streams() const { return num_streams_; }

private:
    int           num_streams_;
    cudaStream_t* streams_;
    void**        pinned_buffers_;
    int           num_pinned_;
    int           pinned_capacity_;
};

}  // namespace flashgraph

#endif  // FLASHGRAPH_STREAM_MANAGER_CUH
